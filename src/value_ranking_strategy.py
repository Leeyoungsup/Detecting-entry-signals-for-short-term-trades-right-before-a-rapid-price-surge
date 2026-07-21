from __future__ import annotations

import json
import math
import sys
from dataclasses import replace
from decimal import Decimal
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset

from src.event_centric_entry import build_event_feature_arrays
from src.immediate_fill_labeling import ImmediateTradingConfig, calculate_trade_result
from src.walk_forward_oof import find_project_root, load_config, seed_everything


RAW_COLUMNS = [
    "symbol", "timestamp_utc", "open", "close", "quote_count", "last_bid", "last_ask",
]


class SequenceBackbone(nn.Module):
    def __init__(
        self,
        sequence_feature_count: int,
        context_feature_count: int,
        conv_channels: int,
        context_hidden: int,
        joint_hidden: int,
        dropout: float,
        extra_feature_count: int = 0,
        extra_hidden: int = 0,
    ) -> None:
        super().__init__()
        self.sequence_encoder = nn.Sequential(
            nn.Conv1d(sequence_feature_count, conv_channels, 3, padding=1),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(conv_channels, conv_channels, 3, padding=1),
            nn.ReLU(),
        )
        self.context_encoder = nn.Sequential(
            nn.Linear(context_feature_count, context_hidden), nn.ReLU(),
        )
        self.extra_encoder: nn.Module | None = None
        if extra_feature_count:
            self.extra_encoder = nn.Sequential(nn.Linear(extra_feature_count, extra_hidden), nn.ReLU())
        input_count = conv_channels * 2 + context_hidden + (extra_hidden if extra_feature_count else 0)
        self.joint = nn.Sequential(
            nn.Linear(input_count, joint_hidden), nn.ReLU(), nn.Dropout(dropout),
        )

    def forward(
        self,
        sequence: torch.Tensor,
        context: torch.Tensor,
        extra: torch.Tensor | None = None,
    ) -> torch.Tensor:
        encoded = self.sequence_encoder(sequence.transpose(1, 2))
        parts = [encoded[:, :, -1], encoded.amax(dim=2), self.context_encoder(context)]
        if self.extra_encoder is not None:
            if extra is None:
                raise ValueError("extra feature가 필요합니다.")
            parts.append(self.extra_encoder(extra))
        return self.joint(torch.cat(parts, dim=1))


def monotonic_quantiles(raw: torch.Tensor) -> torch.Tensor:
    median = raw[:, 0]
    q10 = median - F.softplus(raw[:, 1])
    q90 = median + F.softplus(raw[:, 2])
    return torch.stack([q10, median, q90], dim=1)


class EntryRankModel(nn.Module):
    def __init__(self, sequence_count: int, context_count: int, config: dict[str, Any], surface_count: int):
        super().__init__()
        self.backbone = SequenceBackbone(
            sequence_count,
            context_count,
            int(config["conv_channels"]),
            int(config["context_hidden"]),
            int(config["joint_hidden"]),
            float(config["dropout"]),
        )
        hidden = int(config["joint_hidden"])
        self.quantile_head = nn.Linear(hidden, 3)
        self.rank_head = nn.Linear(hidden, 1)
        self.surface_head = nn.Linear(hidden, surface_count)
        nn.init.zeros_(self.quantile_head.bias)
        nn.init.zeros_(self.rank_head.bias)
        nn.init.zeros_(self.surface_head.bias)

    def forward(self, sequence: torch.Tensor, context: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        hidden = self.backbone(sequence, context)
        return monotonic_quantiles(self.quantile_head(hidden)), self.rank_head(hidden).squeeze(1), self.surface_head(hidden)


class ExitValueModel(nn.Module):
    def __init__(self, sequence_count: int, context_count: int, config: dict[str, Any]):
        super().__init__()
        self.backbone = SequenceBackbone(
            sequence_count,
            context_count,
            int(config["conv_channels"]),
            int(config["context_hidden"]),
            int(config["joint_hidden"]),
            float(config["dropout"]),
            extra_feature_count=4,
            extra_hidden=int(config["position_hidden"]),
        )
        self.quantile_head = nn.Linear(int(config["joint_hidden"]), 3)
        nn.init.zeros_(self.quantile_head.bias)

    def forward(self, sequence: torch.Tensor, context: torch.Tensor, position: torch.Tensor) -> torch.Tensor:
        return monotonic_quantiles(self.quantile_head(self.backbone(sequence, context, position)))


def pinball_loss(predictions: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    quantiles = torch.tensor([0.1, 0.5, 0.9], device=predictions.device, dtype=predictions.dtype)
    error = target[:, None] - predictions
    return torch.maximum(quantiles * error, (quantiles - 1.0) * error).mean(dim=1)


def _load_quote_file(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, usecols=RAW_COLUMNS, encoding="utf-8-sig")
    frame["symbol"] = frame["symbol"].astype(str).str.upper()
    frame["timestamp_utc"] = pd.to_datetime(frame["timestamp_utc"], utc=True, errors="coerce")
    for column in ["open", "close", "quote_count", "last_bid", "last_ask"]:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame = frame.sort_values(["symbol", "timestamp_utc"]).reset_index(drop=True)
    delta = frame.groupby("symbol")["timestamp_utc"].diff().dt.total_seconds().div(60)
    consecutive = pd.Series(np.isclose(delta, 1.0, rtol=0.0, atol=1e-6), index=frame.index)
    new_run = (~consecutive) | frame["symbol"].ne(frame["symbol"].shift())
    frame["run_number"] = new_run.cumsum().astype(np.int64)
    frame["run_id"] = path.stem + "::" + frame["run_number"].astype(str)
    return frame


def executable_return_surface(
    entry_ask: float,
    future_bids: np.ndarray,
    surface_config: dict[str, Any],
) -> dict[str, float | int]:
    bids = np.asarray(future_bids, dtype=np.float64)
    horizons = [int(value) for value in surface_config["horizons_minutes"]]
    maximum_horizon = int(surface_config["maximum_horizon_minutes"])
    if entry_ask <= 0 or len(bids) < maximum_horizon or not np.isfinite(bids[:maximum_horizon]).all():
        raise ValueError("return surface에 유효한 ask와 future bid가 필요합니다.")
    cost = float(surface_config["estimated_round_trip_cost_rate"])
    result: dict[str, float | int] = {}
    gross = bids[:maximum_horizon] / entry_ask - 1.0
    for horizon in horizons:
        horizon_gross = gross[:horizon]
        result[f"future_bid_{horizon}m"] = float(bids[horizon - 1])
        result[f"gross_return_{horizon}m"] = float(horizon_gross[-1])
        result[f"net_return_{horizon}m"] = float(horizon_gross[-1] - cost)
        result[f"maximum_net_return_{horizon}m"] = float(horizon_gross.max() - cost)
        result[f"minimum_net_return_{horizon}m"] = float(horizon_gross.min() - cost)
    for minute in range(1, maximum_horizon + 1):
        result[f"future_bid_path_{minute}m"] = float(bids[minute - 1])
    maximum_index = int(np.argmax(gross))
    time_to_max = maximum_index + 1
    drawdown_before_max = float(gross[:maximum_index + 1].min())
    utility_config = surface_config["utility"]
    maximum_net = float(gross.max() - cost)
    terminal_3m_net = float(gross[2] - cost)
    downside = abs(min(drawdown_before_max, 0.0))
    utility = (
        float(utility_config["maximum_net_return_weight"]) * maximum_net
        + float(utility_config["terminal_3m_net_return_weight"]) * terminal_3m_net
        - float(utility_config["downside_penalty"]) * downside
        - float(utility_config["time_to_max_penalty_per_minute"]) * (time_to_max - 1)
    )
    result.update({
        "time_to_max_return_5m": time_to_max,
        "drawdown_before_max_5m": drawdown_before_max,
        "entry_utility_5m": float(utility),
    })
    return result


def build_return_surface_labels(
    sequence_index: pd.DataFrame,
    surface_config: dict[str, Any],
) -> pd.DataFrame:
    maximum_horizon = int(surface_config["maximum_horizon_minutes"])
    rows: list[dict[str, Any]] = []
    exclusions = {"invalid_current_quote": 0, "invalid_future_quote": 0, "insufficient_future": 0}
    for source_path, metadata in sequence_index.groupby("source_path", sort=False):
        raw = _load_quote_file(Path(source_path))
        for run_id, run_metadata in metadata.groupby("run_id", sort=False):
            run = raw[raw["run_id"].eq(run_id)].reset_index(drop=True)
            if run.empty:
                raise ValueError(f"raw run을 찾지 못했습니다: {run_id}")
            positions = pd.Series(np.arange(len(run)), index=run["timestamp_utc"]).to_dict()
            for meta in run_metadata.itertuples(index=False):
                timestamp = pd.Timestamp(meta.input_end_timestamp)
                position = positions.get(timestamp)
                if position is None or position + maximum_horizon >= len(run):
                    exclusions["insufficient_future"] += 1
                    continue
                current = run.iloc[position]
                current_valid = (
                    current["quote_count"] >= 1
                    and current["last_bid"] > 0
                    and current["last_ask"] >= current["last_bid"]
                )
                if not current_valid:
                    exclusions["invalid_current_quote"] += 1
                    continue
                future = run.iloc[position + 1:position + 1 + maximum_horizon]
                valid = future["quote_count"].ge(1) & future["last_bid"].gt(0) & future["last_ask"].ge(future["last_bid"])
                if len(future) != maximum_horizon or not valid.all():
                    exclusions["invalid_future_quote"] += 1
                    continue
                rows.append({
                    "source_path": source_path,
                    "session": meta.session,
                    "symbol": meta.symbol,
                    "run_id": run_id,
                    "input_end_timestamp": timestamp,
                    "current_last_bid": float(current["last_bid"]),
                    "entry_last_ask": float(current["last_ask"]),
                    "current_is_bearish": int(current["close"] < current["open"]),
                    **executable_return_surface(
                        float(current["last_ask"]),
                        future["last_bid"].to_numpy(dtype=np.float64),
                        surface_config,
                    ),
                })
    labels = pd.DataFrame(rows)
    labels.attrs["exclusions"] = exclusions
    labels["cross_sectional_utility_rank"] = labels.groupby(
        ["session", "input_end_timestamp"], sort=False,
    )["entry_utility_5m"].rank(method="average", pct=True)
    if labels.duplicated(["source_path", "symbol", "input_end_timestamp"]).any():
        raise AssertionError("return surface key가 중복됩니다.")
    return labels


def build_value_surface_dataset(
    project_root: Path | None = None,
    config_path: Path | None = None,
) -> dict[str, Any]:
    project_root = find_project_root(project_root)
    config_path = config_path or project_root / "configs/value_ranking_strategy.yaml"
    config = load_config(project_root, config_path)
    data_root = (project_root / config["data"]["root"]).resolve()
    processed = data_root / "processed"
    sequence_index = pd.read_parquet(data_root / config["data"]["sequence_index_artifact"])
    feature_rows = pd.read_parquet(data_root / config["data"]["feature_rows_artifact"])
    labels = build_return_surface_labels(sequence_index, config["surface"])
    exclusions = dict(labels.attrs["exclusions"])
    metadata, sequence, context = build_event_feature_arrays(feature_rows, sequence_index, labels, config["features"])
    metadata.insert(0, "row_id", np.arange(len(metadata), dtype=np.int64))
    version = config["artifacts"]["version"]
    paths = {
        "surface": processed / f"{version}_return_surface.parquet",
        "metadata": processed / f"{version}_metadata.parquet",
        "features": processed / f"{version}_features.npz",
        "schema": processed / f"{version}_schema.json",
    }
    labels.to_parquet(paths["surface"], index=False, compression="zstd")
    metadata.to_parquet(paths["metadata"], index=False, compression="zstd")
    np.savez_compressed(paths["features"], sequence=sequence, context=context)
    paths["schema"].write_text(json.dumps({
        "version": version,
        "surface": config["surface"],
        "sequence_length": config["features"]["sequence_length"],
        "sequence_features": config["features"]["sequence"],
        "context_features": config["features"]["context"],
        "quote_columns_are_label_only": True,
        "exclusions": exclusions,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"config": config, "metadata": metadata, "sequence": sequence, "context": context, "exclusions": exclusions, "paths": paths}


def _load_surface_dataset(project_root: Path, config: dict[str, Any]) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, dict[str, Path]]:
    data_root = (project_root / config["data"]["root"]).resolve()
    processed = data_root / "processed"
    version = config["artifacts"]["version"]
    paths = {
        "metadata": processed / f"{version}_metadata.parquet",
        "features": processed / f"{version}_features.npz",
        "schema": processed / f"{version}_schema.json",
    }
    if not all(path.exists() for path in paths.values()):
        build_value_surface_dataset(project_root, Path(config["_config_path"]))
    metadata = pd.read_parquet(paths["metadata"]).reset_index(drop=True)
    metadata["input_end_timestamp"] = pd.to_datetime(metadata["input_end_timestamp"], utc=True)
    arrays = np.load(paths["features"])
    return metadata, arrays["sequence"], arrays["context"], paths


def _fit_scalers(sequence: np.ndarray, context: np.ndarray, fit_mask: np.ndarray) -> tuple[np.ndarray, ...]:
    flattened = sequence[fit_mask].reshape(-1, sequence.shape[2]).astype(np.float64)
    sequence_center = np.median(flattened, axis=0)
    sequence_scale = np.quantile(flattened, 0.75, axis=0) - np.quantile(flattened, 0.25, axis=0)
    sequence_scale = np.where(sequence_scale > 1e-8, sequence_scale, 1.0)
    context_fit = context[fit_mask].astype(np.float64)
    context_center = np.median(context_fit, axis=0)
    context_scale = np.quantile(context_fit, 0.75, axis=0) - np.quantile(context_fit, 0.25, axis=0)
    context_scale = np.where(context_scale > 1e-8, context_scale, 1.0)
    return tuple(value.astype(np.float32) for value in (sequence_center, sequence_scale, context_center, context_scale))


def _transform(sequence: np.ndarray, context: np.ndarray, scalers: tuple[np.ndarray, ...], clip: float) -> tuple[np.ndarray, np.ndarray]:
    sequence_center, sequence_scale, context_center, context_scale = scalers
    transformed_sequence = np.clip(
        (sequence - sequence_center[None, None, :]) / sequence_scale[None, None, :], -clip, clip,
    ).astype(np.float32)
    transformed_context = np.clip(
        (context - context_center[None, :]) / context_scale[None, :], -clip, clip,
    ).astype(np.float32)
    if not np.isfinite(transformed_sequence).all() or not np.isfinite(transformed_context).all():
        raise ValueError("value strategy input에 non-finite 값이 있습니다.")
    return transformed_sequence, transformed_context


def _group_balanced_weights(frame: pd.DataFrame, mask: np.ndarray) -> np.ndarray:
    weights = np.zeros(len(frame), dtype=np.float32)
    selected = frame.loc[mask, ["session", "input_end_timestamp"]].copy()
    group_size = selected.groupby(["session", "input_end_timestamp"])["session"].transform("size")
    weights[mask] = (1.0 / group_size.to_numpy(dtype=np.float32))
    totals = pd.Series(weights[mask], index=frame.index[mask]).groupby(frame.loc[mask, "session"]).sum()
    target_total = float(totals.mean())
    for session, total in totals.items():
        session_mask = mask & frame["session"].eq(session).to_numpy()
        weights[session_mask] *= target_total / float(total)
    weights[mask] /= float(weights[mask].mean())
    return weights


def _training_timestamp_mask(frame: pd.DataFrame, sessions: list[str], stride: int) -> np.ndarray:
    session_mask = frame["session"].isin(sessions).to_numpy()
    if stride <= 1:
        return session_mask
    minute = frame["input_end_timestamp"].astype("int64").to_numpy() // (60 * 1_000_000_000)
    return session_mask & (minute % stride == 0)


def _predict_entry(
    model: EntryRankModel,
    sequence: np.ndarray,
    context: np.ndarray,
    device: torch.device,
    batch_size: int = 4096,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    quantiles, ranks, surfaces = [], [], []
    with torch.no_grad():
        for start in range(0, len(sequence), batch_size):
            s = torch.from_numpy(sequence[start:start + batch_size]).to(device)
            c = torch.from_numpy(context[start:start + batch_size]).to(device)
            q, rank_logit, surface = model(s, c)
            quantiles.append(q.cpu().numpy())
            ranks.append(torch.sigmoid(rank_logit).cpu().numpy())
            surfaces.append(surface.cpu().numpy())
    return np.concatenate(quantiles), np.concatenate(ranks), np.concatenate(surfaces)


def _fit_entry_model(
    sequence: np.ndarray,
    context: np.ndarray,
    frame: pd.DataFrame,
    fit_sessions: list[str],
    evaluation_mask: np.ndarray,
    config: dict[str, Any],
    device: torch.device,
    seed: int,
    capture_epochs: list[int],
) -> dict[str, Any]:
    model_config = config["entry_model"]
    train_mask = _training_timestamp_mask(
        frame, fit_sessions, int(model_config["training_timestamp_stride_minutes"]),
    )
    fit_mask = frame["session"].isin(fit_sessions).to_numpy()
    weights = _group_balanced_weights(frame, train_mask)[train_mask]
    scalers = _fit_scalers(sequence, context, fit_mask)
    train_sequence, train_context = _transform(
        sequence[train_mask], context[train_mask], scalers, float(config["features"]["scaler_clip"]),
    )
    evaluation_sequence, evaluation_context = _transform(
        sequence[evaluation_mask], context[evaluation_mask], scalers, float(config["features"]["scaler_clip"]),
    )
    target_scale = float(model_config["target_scale"])
    utility = np.clip(
        frame.loc[train_mask, "entry_utility_5m"].to_numpy(dtype=np.float32) * target_scale,
        float(model_config["target_clip_min"]), float(model_config["target_clip_max"]),
    )
    rank = frame.loc[train_mask, "cross_sectional_utility_rank"].to_numpy(dtype=np.float32)
    horizons = [int(value) for value in config["surface"]["horizons_minutes"]]
    surface_targets = np.column_stack([
        frame.loc[train_mask, f"net_return_{horizon}m"].to_numpy(dtype=np.float32)
        for horizon in horizons
    ]) * target_scale
    surface_targets = np.clip(
        surface_targets, float(model_config["target_clip_min"]), float(model_config["target_clip_max"]),
    ).astype(np.float32)
    seed_everything(seed)
    model = EntryRankModel(sequence.shape[2], context.shape[1], model_config, len(horizons)).to(device)
    loader = DataLoader(
        TensorDataset(
            torch.from_numpy(train_sequence), torch.from_numpy(train_context),
            torch.from_numpy(utility), torch.from_numpy(rank), torch.from_numpy(surface_targets),
            torch.from_numpy(weights),
        ),
        batch_size=int(model_config["batch_size"]), shuffle=True,
        generator=torch.Generator().manual_seed(seed + 1), pin_memory=device.type == "cuda",
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(model_config["learning_rate"]), weight_decay=float(model_config["weight_decay"]),
    )
    captures: dict[int, dict[str, np.ndarray]] = {}
    capture_set = set(capture_epochs)
    history = []
    for epoch in range(1, max(capture_set) + 1):
        model.train()
        sums = np.zeros(4, dtype=np.float64)
        for s, c, utility_batch, rank_batch, surface_batch, weight_batch in loader:
            s, c = s.to(device), c.to(device)
            utility_batch, rank_batch = utility_batch.to(device), rank_batch.to(device)
            surface_batch, weight_batch = surface_batch.to(device), weight_batch.to(device)
            optimizer.zero_grad(set_to_none=True)
            quantiles, rank_logit, surface_prediction = model(s, c)
            quantile_row = pinball_loss(quantiles, utility_batch)
            rank_row = F.mse_loss(torch.sigmoid(rank_logit), rank_batch, reduction="none")
            surface_row = F.smooth_l1_loss(surface_prediction, surface_batch, reduction="none").mean(dim=1)
            row_loss = quantile_row + float(model_config["rank_loss_weight"]) * rank_row + float(model_config["surface_loss_weight"]) * surface_row
            loss = (row_loss * weight_batch).sum() / weight_batch.sum()
            loss.backward()
            optimizer.step()
            sums += [float((row_loss.detach() * weight_batch).sum()), float((quantile_row.detach() * weight_batch).sum()), float((rank_row.detach() * weight_batch).sum()), float(weight_batch.sum())]
        history.append({"epoch": epoch, "weighted_loss": sums[0] / sums[3], "quantile_loss": sums[1] / sums[3], "rank_loss": sums[2] / sums[3]})
        if epoch in capture_set:
            q, rank_prediction, surface_prediction = _predict_entry(model, evaluation_sequence, evaluation_context, device)
            captures[epoch] = {"quantiles": q, "rank": rank_prediction, "surface": surface_prediction}
    return {"model": model, "scalers": scalers, "captures": captures, "history": pd.DataFrame(history), "train_rows": int(train_mask.sum())}


def _entry_prediction_frame(
    frame: pd.DataFrame,
    mask: np.ndarray,
    capture: dict[str, np.ndarray],
    config: dict[str, Any],
    evaluation_group: str,
) -> pd.DataFrame:
    horizons = [int(value) for value in config["surface"]["horizons_minutes"]]
    columns = [
        "row_id", "source_path", "session", "symbol", "input_end_timestamp", "entry_last_ask",
        "current_last_bid", "current_is_bearish", "entry_utility_5m", "cross_sectional_utility_rank",
        *[f"net_return_{horizon}m" for horizon in horizons],
    ]
    output = frame.loc[mask, columns].copy()
    scale = float(config["entry_model"]["target_scale"])
    output["predicted_utility_q10"] = capture["quantiles"][:, 0] / scale
    output["predicted_utility_q50"] = capture["quantiles"][:, 1] / scale
    output["predicted_utility_q90"] = capture["quantiles"][:, 2] / scale
    output["predicted_rank"] = capture["rank"]
    for index, horizon in enumerate(horizons):
        output[f"predicted_net_return_{horizon}m"] = capture["surface"][:, index] / scale
    output["evaluation_group"] = evaluation_group
    return output


def _spearman(left: pd.Series, right: pd.Series) -> float:
    return float(left.corr(right, method="spearman"))


def entry_prediction_metrics(predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for evaluation_group, group in predictions.groupby("evaluation_group", sort=False):
        for session, part in [("ALL", group), *list(group.groupby("session", sort=False))]:
            ordered = part.sort_values("predicted_rank", ascending=False)
            top_count = max(1, math.ceil(len(ordered) * 0.10))
            top = ordered.head(top_count)
            rows.append({
                "evaluation_group": evaluation_group,
                "session": session,
                "samples": len(part),
                "rank_spearman": _spearman(part["predicted_rank"], part["cross_sectional_utility_rank"]),
                "utility_spearman": _spearman(part["predicted_utility_q50"], part["entry_utility_5m"]),
                "utility_mae": float((part["predicted_utility_q50"] - part["entry_utility_5m"]).abs().mean()),
                "top10_mean_utility": float(top["entry_utility_5m"].mean()),
                "top10_mean_net_return_3m": float(top["net_return_3m"].mean()),
                "top10_positive_utility_share": float(top["entry_utility_5m"].gt(0).mean()),
            })
    return pd.DataFrame(rows)


def run_entry_ranker(
    project_root: Path | None = None,
    config_path: Path | None = None,
) -> dict[str, Any]:
    project_root = find_project_root(project_root)
    config_path = config_path or project_root / "configs/value_ranking_strategy.yaml"
    config = load_config(project_root, config_path)
    if "envs/urban" not in str(Path(sys.executable).resolve()):
        raise AssertionError(f"urban 환경이 아닙니다: {sys.executable}")
    frame, sequence, context, _ = _load_surface_dataset(project_root, config)
    train_sessions = list(config["data"]["train_sessions"])
    test_sessions = list(config["data"]["test_sessions"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seed = int(config["project"]["seed"])
    model_config = config["entry_model"]
    epochs = [int(value) for value in model_config["candidate_epochs"]]
    minimum_prior = int(model_config["minimum_oof_prior_sessions"])
    oof_parts: dict[int, list[pd.DataFrame]] = {epoch: [] for epoch in epochs}
    fold_rows, history_rows = [], []
    for evaluation_index in range(minimum_prior, len(train_sessions)):
        fit_sessions = train_sessions[:evaluation_index]
        evaluation_session = train_sessions[evaluation_index]
        evaluation_mask = frame["session"].eq(evaluation_session).to_numpy()
        fitted = _fit_entry_model(
            sequence, context, frame, fit_sessions, evaluation_mask, config, device,
            seed + evaluation_index * 100, epochs,
        )
        for epoch in epochs:
            oof_parts[epoch].append(_entry_prediction_frame(
                frame, evaluation_mask, fitted["captures"][epoch], config, "oof",
            ))
        history = fitted["history"].copy()
        history["fold"] = len(fold_rows) + 1
        history["evaluation_session"] = evaluation_session
        history_rows.append(history)
        fold_rows.append({
            "fold": len(fold_rows) + 1,
            "fit_sessions": ",".join(fit_sessions),
            "evaluation_session": evaluation_session,
            "train_rows": fitted["train_rows"],
        })
        del fitted
        if device.type == "cuda":
            torch.cuda.empty_cache()
    epoch_rows = []
    oof_by_epoch = {}
    for epoch in epochs:
        predictions = pd.concat(oof_parts[epoch], ignore_index=True).sort_values(
            ["input_end_timestamp", "symbol"],
        ).reset_index(drop=True)
        oof_by_epoch[epoch] = predictions
        overall = entry_prediction_metrics(predictions).query("session == 'ALL'").iloc[0]
        epoch_rows.append({"epoch": epoch, **overall.drop(["evaluation_group", "session"]).to_dict()})
    epoch_selection = pd.DataFrame(epoch_rows).sort_values(
        ["rank_spearman", "top10_mean_utility", "epoch"], ascending=[False, False, True],
    ).reset_index(drop=True)
    selected_epoch = int(epoch_selection.iloc[0]["epoch"])
    oof_predictions = oof_by_epoch[selected_epoch]
    train_mask = frame["session"].isin(train_sessions).to_numpy()
    test_mask = frame["session"].isin(test_sessions).to_numpy()
    final_fit = _fit_entry_model(
        sequence, context, frame, train_sessions, train_mask | test_mask, config, device,
        seed + 999, [selected_epoch],
    )
    capture = final_fit["captures"][selected_epoch]
    combined_positions = np.flatnonzero(train_mask | test_mask)
    train_selection = np.isin(combined_positions, np.flatnonzero(train_mask))
    test_selection = np.isin(combined_positions, np.flatnonzero(test_mask))
    final_train_capture = {name: value[train_selection] for name, value in capture.items()}
    test_capture = {name: value[test_selection] for name, value in capture.items()}
    predictions = pd.concat([
        oof_predictions,
        _entry_prediction_frame(frame, train_mask, final_train_capture, config, "final_train"),
        _entry_prediction_frame(frame, test_mask, test_capture, config, "test"),
    ], ignore_index=True)
    metrics = entry_prediction_metrics(predictions)
    data_root = (project_root / config["data"]["root"]).resolve()
    processed, model_root = data_root / "processed", data_root / "models"
    version = config["artifacts"]["version"]
    model_root.mkdir(parents=True, exist_ok=True)
    checkpoint = model_root / f"{version}_entry_ranker.pt"
    torch.save({
        "model_state_dict": {name: value.detach().cpu().clone() for name, value in final_fit["model"].state_dict().items()},
        "scalers": [torch.from_numpy(value) for value in final_fit["scalers"]],
        "selected_epoch": selected_epoch,
        "config_path": config["_config_path"],
        "train_sessions": train_sessions,
        "test_sessions": test_sessions,
        "test_is_pristine": config["data"]["test_is_pristine"],
    }, checkpoint)
    paths = {
        "predictions": processed / f"{version}_entry_predictions.parquet",
        "metrics": processed / f"{version}_entry_metrics.parquet",
        "epoch_selection": processed / f"{version}_entry_epoch_selection.parquet",
        "folds": processed / f"{version}_entry_folds.parquet",
        "oof_history": processed / f"{version}_entry_oof_history.parquet",
        "final_history": processed / f"{version}_entry_final_history.parquet",
        "checkpoint": checkpoint,
    }
    for name, output in {
        "predictions": predictions,
        "metrics": metrics,
        "epoch_selection": epoch_selection,
        "folds": pd.DataFrame(fold_rows),
        "oof_history": pd.concat(history_rows, ignore_index=True),
        "final_history": final_fit["history"],
    }.items():
        output.to_parquet(paths[name], index=False, compression="zstd")
    return {
        "config": config, "device": str(device), "selected_epoch": selected_epoch,
        "parameter_count": int(sum(parameter.numel() for parameter in final_fit["model"].parameters())),
        "epoch_selection": epoch_selection, "metrics": metrics, "predictions": predictions, "paths": paths,
    }


def build_exit_states(frame: pd.DataFrame) -> pd.DataFrame:
    key_to_row = pd.Series(
        frame["row_id"].to_numpy(dtype=np.int64),
        index=pd.MultiIndex.from_frame(frame[["source_path", "symbol", "input_end_timestamp"]]),
    ).to_dict()
    rows = []
    for entry in frame.itertuples(index=False):
        entry_timestamp = pd.Timestamp(entry.input_end_timestamp)
        bids = np.asarray([getattr(entry, f"future_bid_path_{minute}m") for minute in range(1, 6)], dtype=np.float64)
        gross_path = bids / float(entry.entry_last_ask) - 1.0
        for elapsed in range(1, 5):
            state_timestamp = entry_timestamp + pd.Timedelta(minutes=elapsed)
            state_row_id = key_to_row.get((entry.source_path, entry.symbol, state_timestamp))
            if state_row_id is None:
                continue
            current_bid = float(bids[elapsed - 1])
            next_bid = float(bids[elapsed])
            rows.append({
                "entry_row_id": int(entry.row_id),
                "state_row_id": int(state_row_id),
                "session": entry.session,
                "symbol": entry.symbol,
                "entry_timestamp": entry_timestamp,
                "state_timestamp": state_timestamp,
                "elapsed_minutes": elapsed,
                "entry_last_ask": float(entry.entry_last_ask),
                "current_last_bid": current_bid,
                "next_last_bid": next_bid,
                "unrealized_gross_return": float(gross_path[elapsed - 1]),
                "maximum_favorable_excursion": float(gross_path[:elapsed].max()),
                "maximum_adverse_excursion": float(gross_path[:elapsed].min()),
                "next_bid_return": float(next_bid / current_bid - 1.0),
            })
    states = pd.DataFrame(rows)
    if states.duplicated(["entry_row_id", "elapsed_minutes"]).any():
        raise AssertionError("exit state key가 중복됩니다.")
    return states


def _exit_position_features(states: pd.DataFrame) -> np.ndarray:
    return np.column_stack([
        states["elapsed_minutes"].to_numpy(dtype=np.float32) / 5.0,
        states["unrealized_gross_return"].to_numpy(dtype=np.float32),
        states["maximum_favorable_excursion"].to_numpy(dtype=np.float32),
        states["maximum_adverse_excursion"].to_numpy(dtype=np.float32),
    ]).astype(np.float32)


def _fit_position_scaler(position: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    values = position[mask].astype(np.float64)
    center = np.median(values, axis=0)
    scale = np.quantile(values, 0.75, axis=0) - np.quantile(values, 0.25, axis=0)
    return center.astype(np.float32), np.where(scale > 1e-8, scale, 1.0).astype(np.float32)


def _transform_position(position: np.ndarray, scaler: tuple[np.ndarray, np.ndarray], clip: float) -> np.ndarray:
    return np.clip((position - scaler[0]) / scaler[1], -clip, clip).astype(np.float32)


def _predict_exit(
    model: ExitValueModel,
    sequence: np.ndarray,
    context: np.ndarray,
    position: np.ndarray,
    device: torch.device,
    batch_size: int = 4096,
) -> np.ndarray:
    model.eval()
    parts = []
    with torch.no_grad():
        for start in range(0, len(sequence), batch_size):
            parts.append(model(
                torch.from_numpy(sequence[start:start + batch_size]).to(device),
                torch.from_numpy(context[start:start + batch_size]).to(device),
                torch.from_numpy(position[start:start + batch_size]).to(device),
            ).cpu().numpy())
    return np.concatenate(parts)


def _fit_exit_model(
    sequence: np.ndarray,
    context: np.ndarray,
    states: pd.DataFrame,
    position: np.ndarray,
    fit_sessions: list[str],
    evaluation_mask: np.ndarray,
    config: dict[str, Any],
    device: torch.device,
    seed: int,
    capture_epochs: list[int],
) -> dict[str, Any]:
    model_config = config["exit_model"]
    fit_mask = states["session"].isin(fit_sessions).to_numpy()
    stride = int(config["entry_model"]["training_timestamp_stride_minutes"])
    minute = states["entry_timestamp"].astype("int64").to_numpy() // (60 * 1_000_000_000)
    train_mask = fit_mask & (minute % stride == 0)
    state_rows = states["state_row_id"].to_numpy(dtype=np.int64)
    feature_fit_rows = state_rows[fit_mask]
    feature_fit_mask = np.zeros(len(sequence), dtype=bool)
    feature_fit_mask[np.unique(feature_fit_rows)] = True
    scalers = _fit_scalers(sequence, context, feature_fit_mask)
    train_sequence, train_context = _transform(
        sequence[state_rows[train_mask]], context[state_rows[train_mask]], scalers, float(config["features"]["scaler_clip"]),
    )
    evaluation_sequence, evaluation_context = _transform(
        sequence[state_rows[evaluation_mask]], context[state_rows[evaluation_mask]], scalers, float(config["features"]["scaler_clip"]),
    )
    position_scaler = _fit_position_scaler(position, fit_mask)
    train_position = _transform_position(position[train_mask], position_scaler, float(config["features"]["scaler_clip"]))
    evaluation_position = _transform_position(position[evaluation_mask], position_scaler, float(config["features"]["scaler_clip"]))
    scale = float(model_config["target_scale"])
    target = np.clip(
        states.loc[train_mask, "next_bid_return"].to_numpy(dtype=np.float32) * scale,
        float(model_config["target_clip_min"]), float(model_config["target_clip_max"]),
    )
    weights = np.ones(int(train_mask.sum()), dtype=np.float32)
    session_counts = states.loc[train_mask, "session"].value_counts()
    for session, count in session_counts.items():
        weights[states.loc[train_mask, "session"].eq(session).to_numpy()] = len(weights) / len(session_counts) / count
    seed_everything(seed)
    model = ExitValueModel(sequence.shape[2], context.shape[1], model_config).to(device)
    loader = DataLoader(
        TensorDataset(
            torch.from_numpy(train_sequence), torch.from_numpy(train_context), torch.from_numpy(train_position),
            torch.from_numpy(target), torch.from_numpy(weights),
        ), batch_size=int(model_config["batch_size"]), shuffle=True,
        generator=torch.Generator().manual_seed(seed + 1), pin_memory=device.type == "cuda",
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(model_config["learning_rate"]), weight_decay=float(model_config["weight_decay"]),
    )
    captures, history = {}, []
    capture_set = set(capture_epochs)
    for epoch in range(1, max(capture_set) + 1):
        model.train()
        loss_sum, weight_sum = 0.0, 0.0
        for s, c, p, target_batch, weight_batch in loader:
            s, c, p = s.to(device), c.to(device), p.to(device)
            target_batch, weight_batch = target_batch.to(device), weight_batch.to(device)
            optimizer.zero_grad(set_to_none=True)
            row_loss = pinball_loss(model(s, c, p), target_batch)
            loss = (row_loss * weight_batch).sum() / weight_batch.sum()
            loss.backward()
            optimizer.step()
            loss_sum += float((row_loss.detach() * weight_batch).sum())
            weight_sum += float(weight_batch.sum())
        history.append({"epoch": epoch, "weighted_quantile_loss": loss_sum / weight_sum})
        if epoch in capture_set:
            captures[epoch] = _predict_exit(model, evaluation_sequence, evaluation_context, evaluation_position, device)
    return {"model": model, "scalers": scalers, "position_scaler": position_scaler, "captures": captures, "history": pd.DataFrame(history), "train_rows": int(train_mask.sum())}


def _exit_prediction_frame(
    states: pd.DataFrame,
    mask: np.ndarray,
    quantiles: np.ndarray,
    config: dict[str, Any],
    evaluation_group: str,
) -> pd.DataFrame:
    columns = [
        "entry_row_id", "state_row_id", "session", "symbol", "entry_timestamp", "state_timestamp",
        "elapsed_minutes", "entry_last_ask", "current_last_bid", "next_last_bid", "next_bid_return",
        "unrealized_gross_return", "maximum_favorable_excursion", "maximum_adverse_excursion",
    ]
    output = states.loc[mask, columns].copy()
    scale = float(config["exit_model"]["target_scale"])
    output["predicted_next_return_q10"] = quantiles[:, 0] / scale
    output["predicted_next_return_q50"] = quantiles[:, 1] / scale
    output["predicted_next_return_q90"] = quantiles[:, 2] / scale
    output["hold_signal"] = (
        output["predicted_next_return_q50"].gt(float(config["exit_model"]["hold_minimum_q50_return"]))
        & output["predicted_next_return_q10"].gt(float(config["exit_model"]["hold_minimum_q10_return"]))
    )
    output["evaluation_group"] = evaluation_group
    return output


def exit_prediction_metrics(predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for evaluation_group, group in predictions.groupby("evaluation_group", sort=False):
        for session, part in [("ALL", group), *list(group.groupby("session", sort=False))]:
            hold = part["hold_signal"]
            rows.append({
                "evaluation_group": evaluation_group,
                "session": session,
                "samples": len(part),
                "q50_spearman": _spearman(part["predicted_next_return_q50"], part["next_bid_return"]),
                "q50_mae": float((part["predicted_next_return_q50"] - part["next_bid_return"]).abs().mean()),
                "hold_share": float(hold.mean()),
                "hold_mean_next_return": float(part.loc[hold, "next_bid_return"].mean()) if hold.any() else np.nan,
                "sell_mean_next_return_avoided": float(part.loc[~hold, "next_bid_return"].mean()) if (~hold).any() else np.nan,
                "hold_positive_share": float(part.loc[hold, "next_bid_return"].gt(0).mean()) if hold.any() else np.nan,
            })
    return pd.DataFrame(rows)


def run_exit_value_model(
    project_root: Path | None = None,
    config_path: Path | None = None,
) -> dict[str, Any]:
    project_root = find_project_root(project_root)
    config_path = config_path or project_root / "configs/value_ranking_strategy.yaml"
    config = load_config(project_root, config_path)
    if "envs/urban" not in str(Path(sys.executable).resolve()):
        raise AssertionError(f"urban 환경이 아닙니다: {sys.executable}")
    frame, sequence, context, _ = _load_surface_dataset(project_root, config)
    states = build_exit_states(frame).reset_index(drop=True)
    position = _exit_position_features(states)
    train_sessions = list(config["data"]["train_sessions"])
    test_sessions = list(config["data"]["test_sessions"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seed = int(config["project"]["seed"])
    model_config = config["exit_model"]
    epochs = [int(value) for value in model_config["candidate_epochs"]]
    minimum_prior = int(model_config["minimum_oof_prior_sessions"])
    oof_parts: dict[int, list[pd.DataFrame]] = {epoch: [] for epoch in epochs}
    fold_rows, history_rows = [], []
    for evaluation_index in range(minimum_prior, len(train_sessions)):
        fit_sessions = train_sessions[:evaluation_index]
        evaluation_session = train_sessions[evaluation_index]
        evaluation_mask = states["session"].eq(evaluation_session).to_numpy()
        fitted = _fit_exit_model(
            sequence, context, states, position, fit_sessions, evaluation_mask, config, device,
            seed + evaluation_index * 100, epochs,
        )
        for epoch in epochs:
            oof_parts[epoch].append(_exit_prediction_frame(
                states, evaluation_mask, fitted["captures"][epoch], config, "oof",
            ))
        history = fitted["history"].copy()
        history["fold"] = len(fold_rows) + 1
        history["evaluation_session"] = evaluation_session
        history_rows.append(history)
        fold_rows.append({
            "fold": len(fold_rows) + 1, "fit_sessions": ",".join(fit_sessions),
            "evaluation_session": evaluation_session, "train_rows": fitted["train_rows"],
        })
        del fitted
        if device.type == "cuda":
            torch.cuda.empty_cache()
    epoch_rows, oof_by_epoch = [], {}
    for epoch in epochs:
        predictions = pd.concat(oof_parts[epoch], ignore_index=True).sort_values(
            ["state_timestamp", "symbol", "entry_row_id"],
        ).reset_index(drop=True)
        oof_by_epoch[epoch] = predictions
        overall = exit_prediction_metrics(predictions).query("session == 'ALL'").iloc[0]
        epoch_rows.append({"epoch": epoch, **overall.drop(["evaluation_group", "session"]).to_dict()})
    epoch_selection = pd.DataFrame(epoch_rows).sort_values(
        ["q50_spearman", "q50_mae", "epoch"], ascending=[False, True, True],
    ).reset_index(drop=True)
    selected_epoch = int(epoch_selection.iloc[0]["epoch"])
    oof_predictions = oof_by_epoch[selected_epoch]
    train_mask = states["session"].isin(train_sessions).to_numpy()
    test_mask = states["session"].isin(test_sessions).to_numpy()
    final_fit = _fit_exit_model(
        sequence, context, states, position, train_sessions, train_mask | test_mask, config, device,
        seed + 999, [selected_epoch],
    )
    quantiles = final_fit["captures"][selected_epoch]
    combined_positions = np.flatnonzero(train_mask | test_mask)
    train_selection = np.isin(combined_positions, np.flatnonzero(train_mask))
    test_selection = np.isin(combined_positions, np.flatnonzero(test_mask))
    predictions = pd.concat([
        oof_predictions,
        _exit_prediction_frame(states, train_mask, quantiles[train_selection], config, "final_train"),
        _exit_prediction_frame(states, test_mask, quantiles[test_selection], config, "test"),
    ], ignore_index=True)
    metrics = exit_prediction_metrics(predictions)
    data_root = (project_root / config["data"]["root"]).resolve()
    processed, model_root = data_root / "processed", data_root / "models"
    version = config["artifacts"]["version"]
    model_root.mkdir(parents=True, exist_ok=True)
    states_path = processed / f"{version}_exit_states.parquet"
    states.to_parquet(states_path, index=False, compression="zstd")
    checkpoint = model_root / f"{version}_exit_value.pt"
    torch.save({
        "model_state_dict": {name: value.detach().cpu().clone() for name, value in final_fit["model"].state_dict().items()},
        "scalers": [torch.from_numpy(value) for value in final_fit["scalers"]],
        "position_scaler": [torch.from_numpy(value) for value in final_fit["position_scaler"]],
        "selected_epoch": selected_epoch,
        "config_path": config["_config_path"],
        "test_is_pristine": config["data"]["test_is_pristine"],
    }, checkpoint)
    paths = {
        "states": states_path,
        "predictions": processed / f"{version}_exit_predictions.parquet",
        "metrics": processed / f"{version}_exit_metrics.parquet",
        "epoch_selection": processed / f"{version}_exit_epoch_selection.parquet",
        "folds": processed / f"{version}_exit_folds.parquet",
        "oof_history": processed / f"{version}_exit_oof_history.parquet",
        "final_history": processed / f"{version}_exit_final_history.parquet",
        "checkpoint": checkpoint,
    }
    for name, output in {
        "predictions": predictions, "metrics": metrics, "epoch_selection": epoch_selection,
        "folds": pd.DataFrame(fold_rows), "oof_history": pd.concat(history_rows, ignore_index=True),
        "final_history": final_fit["history"],
    }.items():
        output.to_parquet(paths[name], index=False, compression="zstd")
    return {
        "config": config, "device": str(device), "selected_epoch": selected_epoch,
        "parameter_count": int(sum(parameter.numel() for parameter in final_fit["model"].parameters())),
        "state_count": len(states), "epoch_selection": epoch_selection, "metrics": metrics,
        "predictions": predictions, "paths": paths,
    }


def _trading_config(config: dict[str, Any]) -> ImmediateTradingConfig:
    fees = config["fees"]
    return replace(
        ImmediateTradingConfig(),
        order_notional_usd=Decimal(str(fees["order_notional_usd"])),
        buy_commission_rate=Decimal(str(fees["buy_commission_rate"])),
        sell_commission_rate=Decimal(str(fees["sell_commission_rate"])),
        sec_fee_rate=Decimal(str(fees["sec_fee_rate"])),
        taf_per_share=Decimal(str(fees["taf_per_share"])),
        taf_max_per_trade=Decimal(str(fees["taf_max_per_trade"])),
        sell_slippage_pct=Decimal(str(fees["sell_slippage_pct"])),
    )


def _max_drawdown(equity: pd.Series) -> float:
    if equity.empty:
        return 0.0
    running_max = equity.cummax()
    return float((equity / running_max - 1.0).min())


def run_timing_backtest(
    entry_predictions: pd.DataFrame,
    exit_predictions: pd.DataFrame,
    surface: pd.DataFrame,
    policy: dict[str, float | int],
    config: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if entry_predictions.empty:
        return pd.DataFrame(), pd.DataFrame()
    trading = _trading_config(config)
    initial_capital = float(config["policy"]["initial_capital_usd"])
    maximum_positions = int(config["policy"]["max_concurrent_positions"])
    maximum_holding = int(config["policy"]["max_holding_minutes"])
    top_k = int(policy["top_k"])
    minimum_q50 = float(policy["minimum_q50"])
    minimum_q10 = float(policy["minimum_q10"])
    surface_by_row = surface.set_index("row_id", drop=False)
    exit_lookup = {
        (int(row.entry_row_id), int(row.elapsed_minutes)): row
        for row in exit_predictions.itertuples(index=False)
    }
    cash = initial_capital
    positions: dict[str, dict[str, Any]] = {}
    ledgers, equity_rows = [], []
    entry_predictions = entry_predictions.sort_values(["input_end_timestamp", "predicted_rank", "symbol"], ascending=[True, False, True])
    for timestamp, candidates_at_time in entry_predictions.groupby("input_end_timestamp", sort=True):
        timestamp = pd.Timestamp(timestamp)
        for symbol in list(positions):
            position_state = positions[symbol]
            elapsed = int((timestamp - position_state["entry_timestamp"]).total_seconds() // 60)
            if elapsed < 1:
                continue
            elapsed = min(elapsed, maximum_holding)
            entry_surface = surface_by_row.loc[position_state["entry_row_id"]]
            current_bid = float(entry_surface[f"future_bid_path_{elapsed}m"])
            exit_prediction = exit_lookup.get((position_state["entry_row_id"], elapsed))
            should_sell = elapsed >= maximum_holding or exit_prediction is None or not bool(exit_prediction.hold_signal)
            if should_sell:
                result = calculate_trade_result(
                    position_state["entry_ask"], current_bid, position_state["shares"], trading,
                )
                cash += position_state["capital_required"] + result["net_pnl"]
                ledgers.append({
                    "session": position_state["session"], "symbol": symbol,
                    "entry_timestamp": position_state["entry_timestamp"], "exit_timestamp": timestamp,
                    "holding_minutes": elapsed, "entry_row_id": position_state["entry_row_id"],
                    "entry_price": position_state["entry_ask"], "exit_bid": current_bid,
                    "predicted_rank": position_state["predicted_rank"],
                    "predicted_utility_q10": position_state["predicted_q10"],
                    "predicted_utility_q50": position_state["predicted_q50"],
                    **result,
                })
                del positions[symbol]
        candidates = candidates_at_time[
            candidates_at_time["predicted_utility_q50"].ge(minimum_q50)
            & candidates_at_time["predicted_utility_q10"].ge(minimum_q10)
        ].head(top_k)
        for candidate in candidates.itertuples(index=False):
            if len(positions) >= maximum_positions or candidate.symbol in positions:
                continue
            entry_ask = float(candidate.entry_last_ask)
            shares = int(float(config["fees"]["order_notional_usd"]) // entry_ask)
            if shares < 1:
                continue
            buy_notional = entry_ask * shares
            buy_commission = round(buy_notional * float(config["fees"]["buy_commission_rate"]), 2)
            capital_required = buy_notional + buy_commission
            if cash < capital_required:
                continue
            cash -= capital_required
            positions[candidate.symbol] = {
                "session": candidate.session, "entry_timestamp": timestamp,
                "entry_row_id": int(candidate.row_id), "entry_ask": entry_ask, "shares": shares,
                "capital_required": capital_required, "predicted_rank": float(candidate.predicted_rank),
                "predicted_q10": float(candidate.predicted_utility_q10),
                "predicted_q50": float(candidate.predicted_utility_q50),
            }
        marked_equity = cash
        for position_state in positions.values():
            elapsed = int((timestamp - position_state["entry_timestamp"]).total_seconds() // 60)
            if elapsed < 1:
                marked_equity += position_state["capital_required"]
                continue
            elapsed = min(elapsed, maximum_holding)
            entry_surface = surface_by_row.loc[position_state["entry_row_id"]]
            current_bid = float(entry_surface[f"future_bid_path_{elapsed}m"])
            result = calculate_trade_result(position_state["entry_ask"], current_bid, position_state["shares"], trading)
            marked_equity += position_state["capital_required"] + result["net_pnl"]
        equity_rows.append({"timestamp": timestamp, "equity": marked_equity})
    if positions:
        last_timestamp = entry_predictions["input_end_timestamp"].max()
        for symbol, position_state in list(positions.items()):
            entry_surface = surface_by_row.loc[position_state["entry_row_id"]]
            current_bid = float(entry_surface[f"future_bid_path_{maximum_holding}m"])
            result = calculate_trade_result(position_state["entry_ask"], current_bid, position_state["shares"], trading)
            cash += position_state["capital_required"] + result["net_pnl"]
            ledgers.append({
                "session": position_state["session"], "symbol": symbol,
                "entry_timestamp": position_state["entry_timestamp"], "exit_timestamp": last_timestamp,
                "holding_minutes": maximum_holding, "entry_row_id": position_state["entry_row_id"],
                "entry_price": position_state["entry_ask"], "exit_bid": current_bid,
                "predicted_rank": position_state["predicted_rank"],
                "predicted_utility_q10": position_state["predicted_q10"],
                "predicted_utility_q50": position_state["predicted_q50"],
                **result,
            })
    ledger = pd.DataFrame(ledgers)
    equity = pd.DataFrame(equity_rows)
    return ledger, equity


def _ledger_summary(ledger: pd.DataFrame, equity: pd.DataFrame, initial_capital: float) -> dict[str, Any]:
    if ledger.empty:
        return {
            "trades": 0, "mean_net_return": np.nan, "median_net_return": np.nan,
            "win_rate": np.nan, "profit_factor": np.nan, "total_net_pnl": 0.0,
            "portfolio_return": 0.0, "max_drawdown": 0.0,
        }
    gains = ledger.loc[ledger["net_pnl"].gt(0), "net_pnl"].sum()
    losses = -ledger.loc[ledger["net_pnl"].lt(0), "net_pnl"].sum()
    return {
        "trades": len(ledger),
        "mean_net_return": float(ledger["net_return"].mean()),
        "median_net_return": float(ledger["net_return"].median()),
        "win_rate": float(ledger["net_return"].gt(0).mean()),
        "profit_factor": float(gains / losses) if losses > 0 else np.inf,
        "total_net_pnl": float(ledger["net_pnl"].sum()),
        "portfolio_return": float(ledger["net_pnl"].sum() / initial_capital),
        "max_drawdown": _max_drawdown(equity["equity"]) if not equity.empty else 0.0,
    }


def _evaluate_policy_by_session(
    entry_predictions: pd.DataFrame,
    exit_predictions: pd.DataFrame,
    surface: pd.DataFrame,
    policy: dict[str, Any],
    config: dict[str, Any],
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    initial = float(config["policy"]["initial_capital_usd"])
    ledgers, equities, session_rows = [], [], []
    for session, entries in entry_predictions.groupby("session", sort=False):
        exits = exit_predictions[exit_predictions["session"].eq(session)]
        ledger, equity = run_timing_backtest(entries, exits, surface[surface["session"].eq(session)], policy, config)
        summary = _ledger_summary(ledger, equity, initial)
        summary["session"] = session
        session_rows.append(summary)
        if not ledger.empty:
            ledgers.append(ledger)
        if not equity.empty:
            equity = equity.copy()
            equity["session"] = session
            equities.append(equity)
    ledger_all = pd.concat(ledgers, ignore_index=True) if ledgers else pd.DataFrame()
    equity_all = pd.concat(equities, ignore_index=True) if equities else pd.DataFrame()
    sessions = pd.DataFrame(session_rows)
    overall = _ledger_summary(ledger_all, equity_all, initial * max(len(sessions), 1))
    overall.update({
        "positive_session_share": float(sessions["mean_net_return"].gt(0).mean()) if len(sessions) else np.nan,
        "worst_session_mean_return": float(sessions["mean_net_return"].min()) if len(sessions) else np.nan,
    })
    return overall, sessions, ledger_all, equity_all
