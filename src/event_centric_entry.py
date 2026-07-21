from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, roc_auc_score
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset

from src.walk_forward_oof import find_project_root, load_config, seed_everything


RAW_COLUMNS = [
    "symbol",
    "timestamp_utc",
    "open",
    "close",
    "quote_count",
    "last_bid",
    "last_ask",
]


class EventSequenceModel(nn.Module):
    def __init__(
        self,
        sequence_feature_count: int,
        context_feature_count: int,
        target_count: int,
        conv_channels: int,
        context_hidden: int,
        joint_hidden: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.sequence_encoder = nn.Sequential(
            nn.Conv1d(sequence_feature_count, conv_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(conv_channels, conv_channels, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        self.context_encoder = nn.Sequential(
            nn.Linear(context_feature_count, context_hidden),
            nn.ReLU(),
        )
        self.joint = nn.Sequential(
            nn.Linear(conv_channels * 2 + context_hidden, joint_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.classifier = nn.Linear(joint_hidden, target_count)
        self.regressor = nn.Linear(joint_hidden, 2)
        nn.init.zeros_(self.classifier.bias)
        nn.init.zeros_(self.regressor.bias)

    def forward(self, sequence: torch.Tensor, context: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        encoded = self.sequence_encoder(sequence.transpose(1, 2))
        sequence_summary = torch.cat([encoded[:, :, -1], encoded.amax(dim=2)], dim=1)
        joint = self.joint(torch.cat([sequence_summary, self.context_encoder(context)], dim=1))
        return self.classifier(joint), self.regressor(joint)


def multi_horizon_quote_targets(
    entry_last_ask: float,
    future_last_bid: np.ndarray,
    target_specs: list[dict[str, Any]],
) -> dict[str, float | int]:
    bids = np.asarray(future_last_bid, dtype=np.float64)
    if entry_last_ask <= 0 or bids.size == 0 or not np.isfinite(bids).all():
        raise ValueError("유효한 entry ask와 future bid가 필요합니다.")
    maximum_horizon = max(int(spec["horizon_minutes"]) for spec in target_specs)
    if len(bids) < maximum_horizon:
        raise ValueError("future bid 길이가 최대 horizon보다 짧습니다.")
    result: dict[str, float | int] = {
        "maximum_future_bid_return_3m": float(bids[:maximum_horizon].max() / entry_last_ask - 1.0),
        "minimum_future_bid_return_3m": float(bids[:maximum_horizon].min() / entry_last_ask - 1.0),
    }
    for spec in target_specs:
        name = str(spec["name"])
        horizon = int(spec["horizon_minutes"])
        target_return = float(spec["target_return"])
        hits = np.flatnonzero(bids[:horizon] >= entry_last_ask * (1.0 + target_return))
        result[f"target_{name}"] = int(len(hits) > 0)
        result[f"time_to_{name}"] = int(hits[0] + 1) if len(hits) else 0
    return result


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


def build_multihorizon_labels(
    sequence_index: pd.DataFrame,
    label_config: dict[str, Any],
) -> pd.DataFrame:
    target_specs = list(label_config["targets"])
    maximum_horizon = max(int(spec["horizon_minutes"]) for spec in target_specs)
    rows: list[dict[str, Any]] = []
    exclusions = {
        "invalid_current_quote": 0,
        "invalid_future_quote": 0,
        "insufficient_future": 0,
    }
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
                future_valid = (
                    future["quote_count"].ge(1)
                    & future["last_bid"].gt(0)
                    & future["last_ask"].ge(future["last_bid"])
                )
                if len(future) != maximum_horizon or not future_valid.all():
                    exclusions["invalid_future_quote"] += 1
                    continue
                entry_ask = float(current["last_ask"])
                target_values = multi_horizon_quote_targets(
                    entry_ask,
                    future["last_bid"].to_numpy(dtype=np.float64),
                    target_specs,
                )
                rows.append({
                    "source_path": source_path,
                    "session": meta.session,
                    "symbol": meta.symbol,
                    "run_id": run_id,
                    "input_end_timestamp": timestamp,
                    "entry_last_ask": entry_ask,
                    "current_is_bearish": int(current["close"] < current["open"]),
                    "label_version": label_config["version"],
                    **target_values,
                })
    labels = pd.DataFrame(rows)
    labels.attrs["exclusions"] = exclusions
    keys = ["source_path", "symbol", "input_end_timestamp"]
    if labels.duplicated(keys).any():
        raise AssertionError("multi-horizon label key가 중복됩니다.")
    for spec in target_specs:
        column = f"target_{spec['name']}"
        if not labels[column].isin([0, 1]).all():
            raise AssertionError(f"{column}이 binary가 아닙니다.")
    return labels


def build_event_feature_arrays(
    feature_rows: pd.DataFrame,
    sequence_index: pd.DataFrame,
    labels: pd.DataFrame,
    feature_config: dict[str, Any],
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    keys = ["source_path", "symbol", "input_end_timestamp"]
    label_columns = [column for column in labels.columns if column not in {"session", "run_id"}]
    metadata = sequence_index.merge(labels[label_columns], on=keys, how="inner", validate="one_to_one")
    metadata = metadata.reset_index(drop=True)
    if len(metadata) != len(labels):
        raise AssertionError("sequence와 multi-horizon label 결합 수가 다릅니다.")
    sequence_features = list(feature_config["sequence"])
    context_features = list(feature_config["context"])
    missing = sorted(set(sequence_features + context_features) - set(feature_rows.columns))
    if missing:
        raise ValueError(f"feature_rows에 필요한 feature가 없습니다: {missing}")
    sequence_length = int(feature_config["sequence_length"])
    sequence_array = np.empty(
        (len(metadata), sequence_length, len(sequence_features)), dtype=np.float32,
    )
    context_array = np.empty((len(metadata), len(context_features)), dtype=np.float32)
    for source_path, group_indices in metadata.groupby("source_path", sort=False).groups.items():
        output_positions = np.asarray(list(group_indices), dtype=np.int64)
        group_metadata = metadata.loc[output_positions]
        rows = feature_rows[feature_rows["source_path"].eq(source_path)].sort_values("feature_row")
        feature_row = rows["feature_row"].to_numpy(dtype=np.int64)
        if not np.array_equal(feature_row, np.arange(len(rows))):
            raise AssertionError(f"feature_row가 연속적이지 않습니다: {source_path}")
        sequence_matrix = rows[sequence_features].to_numpy(dtype=np.float32)
        context_matrix = rows[context_features].to_numpy(dtype=np.float32)
        end = group_metadata["end_feature_row"].to_numpy(dtype=np.int64)
        offsets = np.arange(sequence_length - 1, -1, -1, dtype=np.int64)
        windows = end[:, None] - offsets[None, :]
        if (windows < 0).any():
            raise AssertionError("sequence window가 feature 시작 이전을 참조합니다.")
        sequence_array[output_positions] = sequence_matrix[windows]
        context_array[output_positions] = context_matrix[end]
    if not np.isfinite(sequence_array).all() or not np.isfinite(context_array).all():
        raise AssertionError("event-centric feature에 non-finite 값이 있습니다.")
    return metadata, sequence_array, context_array


def _cluster_inverse_weights(
    frame: pd.DataFrame,
    event_mask: np.ndarray,
    cluster_minutes: int,
) -> tuple[np.ndarray, int]:
    weights = np.ones(len(frame), dtype=np.float32)
    events = frame.loc[event_mask, ["session", "symbol", "input_end_timestamp"]].copy()
    if events.empty:
        return weights, 0
    events["original_position"] = events.index.to_numpy(dtype=np.int64)
    events = events.sort_values(["session", "symbol", "input_end_timestamp", "original_position"])
    gap = events.groupby(["session", "symbol"])["input_end_timestamp"].diff().dt.total_seconds().div(60)
    new_cluster = gap.isna() | gap.gt(cluster_minutes)
    events["cluster_id"] = new_cluster.cumsum().to_numpy(dtype=np.int64)
    sizes = events.groupby("cluster_id")["cluster_id"].transform("size").to_numpy(dtype=np.float32)
    weights[events["original_position"].to_numpy(dtype=np.int64)] = 1.0 / sizes
    return weights, int(events["cluster_id"].nunique())


def build_event_training_policy(
    frame: pd.DataFrame,
    primary_target_column: str,
    sampling_config: dict[str, Any],
) -> dict[str, Any]:
    primary = frame[primary_target_column].eq(1).to_numpy()
    hard_negative = (
        frame[primary_target_column].eq(0)
        & frame["maximum_future_bid_return_3m"].ge(float(sampling_config["hard_negative_min_return"]))
    ).to_numpy()
    selected = primary | hard_negative
    spacing = int(sampling_config["regular_negative_spacing_minutes"])
    for _, group in frame.groupby(["session", "symbol"], sort=False):
        regular = group.loc[~pd.Series(selected[group.index], index=group.index)].sort_values(
            "input_end_timestamp",
        )
        last_timestamp: pd.Timestamp | None = None
        for position, timestamp in regular["input_end_timestamp"].items():
            if last_timestamp is None or (timestamp - last_timestamp).total_seconds() >= spacing * 60:
                selected[int(position)] = True
                last_timestamp = timestamp
    positive_overlap, positive_clusters = _cluster_inverse_weights(
        frame, primary, int(sampling_config["event_cluster_minutes"]),
    )
    hard_overlap, hard_clusters = _cluster_inverse_weights(
        frame, hard_negative, int(sampling_config["event_cluster_minutes"]),
    )
    overlap_weight = np.ones(len(frame), dtype=np.float32)
    overlap_weight[primary] = positive_overlap[primary]
    overlap_weight[hard_negative] = hard_overlap[hard_negative]
    return {
        "selected_mask": selected,
        "overlap_weight": overlap_weight,
        "primary_mask": primary,
        "hard_negative_mask": hard_negative,
        "positive_clusters": positive_clusters,
        "hard_negative_clusters": hard_clusters,
    }


def _fold_training_weights(
    frame: pd.DataFrame,
    train_mask: np.ndarray,
    selected_mask: np.ndarray,
    overlap_weight: np.ndarray,
    equal_session_weights: bool,
) -> np.ndarray:
    active = train_mask & selected_mask
    weights = np.zeros(len(frame), dtype=np.float32)
    weights[active] = overlap_weight[active]
    if equal_session_weights:
        session_totals = pd.Series(weights[active], index=frame.index[active]).groupby(
            frame.loc[active, "session"],
        ).sum()
        target_total = float(session_totals.mean())
        for session, total in session_totals.items():
            session_mask = active & frame["session"].eq(session).to_numpy()
            weights[session_mask] *= target_total / float(total)
    mean_weight = float(weights[active].mean())
    if mean_weight <= 0:
        raise ValueError("학습 sample weight 합이 0입니다.")
    weights[active] /= mean_weight
    return weights


def _positive_class_weights(
    targets: np.ndarray,
    row_weights: np.ndarray,
    cap: float,
) -> np.ndarray:
    values = []
    for target_index in range(targets.shape[1]):
        positive = float(row_weights[targets[:, target_index] == 1].sum())
        negative = float(row_weights[targets[:, target_index] == 0].sum())
        if positive <= 0 or negative <= 0:
            raise ValueError("각 target에는 양성과 음성이 모두 필요합니다.")
        values.append(min(negative / positive, cap))
    return np.asarray(values, dtype=np.float32)


def _fit_scalers(
    sequence: np.ndarray,
    context: np.ndarray,
    fit_mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    flattened = sequence[fit_mask].reshape(-1, sequence.shape[2]).astype(np.float64)
    sequence_center = np.median(flattened, axis=0)
    sequence_scale = np.quantile(flattened, 0.75, axis=0) - np.quantile(flattened, 0.25, axis=0)
    sequence_scale = np.where(sequence_scale > 1e-8, sequence_scale, 1.0)
    context_fit = context[fit_mask].astype(np.float64)
    context_center = np.median(context_fit, axis=0)
    context_scale = np.quantile(context_fit, 0.75, axis=0) - np.quantile(context_fit, 0.25, axis=0)
    context_scale = np.where(context_scale > 1e-8, context_scale, 1.0)
    return tuple(array.astype(np.float32) for array in (
        sequence_center, sequence_scale, context_center, context_scale,
    ))


def _transform_inputs(
    sequence: np.ndarray,
    context: np.ndarray,
    scalers: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    clip: float,
) -> tuple[np.ndarray, np.ndarray]:
    sequence_center, sequence_scale, context_center, context_scale = scalers
    transformed_sequence = np.clip(
        (sequence - sequence_center[None, None, :]) / sequence_scale[None, None, :], -clip, clip,
    ).astype(np.float32)
    transformed_context = np.clip(
        (context - context_center[None, :]) / context_scale[None, :], -clip, clip,
    ).astype(np.float32)
    if not np.isfinite(transformed_sequence).all() or not np.isfinite(transformed_context).all():
        raise ValueError("scaled event input에 non-finite 값이 있습니다.")
    return transformed_sequence, transformed_context


def _predict(
    model: EventSequenceModel,
    sequence: np.ndarray,
    context: np.ndarray,
    device: torch.device,
    batch_size: int = 4096,
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    score_parts = []
    regression_parts = []
    with torch.no_grad():
        for start in range(0, len(sequence), batch_size):
            sequence_batch = torch.from_numpy(sequence[start:start + batch_size]).to(device)
            context_batch = torch.from_numpy(context[start:start + batch_size]).to(device)
            logits, regression = model(sequence_batch, context_batch)
            score_parts.append(torch.sigmoid(logits).cpu().numpy())
            regression_parts.append(regression.cpu().numpy())
    return np.concatenate(score_parts).astype(np.float32), np.concatenate(regression_parts).astype(np.float32)


def _fit_model(
    sequence: np.ndarray,
    context: np.ndarray,
    target_matrix: np.ndarray,
    regression_targets: np.ndarray,
    frame: pd.DataFrame,
    fit_mask: np.ndarray,
    evaluation_mask: np.ndarray,
    selected_mask: np.ndarray,
    overlap_weight: np.ndarray,
    config: dict[str, Any],
    device: torch.device,
    seed: int,
    capture_epochs: list[int],
) -> dict[str, Any]:
    model_config = config["model"]
    sampling_config = config["sampling"]
    feature_config = config["features"]
    row_weights = _fold_training_weights(
        frame,
        fit_mask,
        selected_mask,
        overlap_weight,
        bool(sampling_config["equal_session_weights"]),
    )
    train_mask = fit_mask & selected_mask
    train_weights = row_weights[train_mask]
    positive_weights = _positive_class_weights(
        target_matrix[train_mask], train_weights, float(sampling_config["max_positive_class_weight"]),
    )
    scalers = _fit_scalers(sequence, context, fit_mask)
    train_sequence, train_context = _transform_inputs(
        sequence[train_mask], context[train_mask], scalers, float(feature_config["scaler_clip"]),
    )
    evaluation_sequence, evaluation_context = _transform_inputs(
        sequence[evaluation_mask], context[evaluation_mask], scalers, float(feature_config["scaler_clip"]),
    )
    seed_everything(seed)
    model = EventSequenceModel(
        sequence.shape[2],
        context.shape[1],
        target_matrix.shape[1],
        int(model_config["conv_channels"]),
        int(model_config["context_hidden"]),
        int(model_config["joint_hidden"]),
        float(model_config["dropout"]),
    ).to(device)
    dataset = TensorDataset(
        torch.from_numpy(train_sequence),
        torch.from_numpy(train_context),
        torch.from_numpy(target_matrix[train_mask].astype(np.float32)),
        torch.from_numpy(regression_targets[train_mask].astype(np.float32)),
        torch.from_numpy(train_weights.astype(np.float32)),
    )
    loader = DataLoader(
        dataset,
        batch_size=int(model_config["batch_size"]),
        shuffle=True,
        generator=torch.Generator().manual_seed(seed + 1),
        pin_memory=device.type == "cuda",
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(model_config["learning_rate"]),
        weight_decay=float(model_config["weight_decay"]),
    )
    positive_weight_tensor = torch.from_numpy(positive_weights).to(device)
    auxiliary_weight = float(model_config["auxiliary_regression_weight"])
    capture_set = set(int(epoch) for epoch in capture_epochs)
    captures: dict[int, dict[str, np.ndarray]] = {}
    history = []
    for epoch in range(1, max(capture_set) + 1):
        model.train()
        weighted_loss_sum = 0.0
        weighted_classification_sum = 0.0
        weighted_regression_sum = 0.0
        weight_sum = 0.0
        for sequence_batch, context_batch, target_batch, regression_batch, weight_batch in loader:
            sequence_batch = sequence_batch.to(device, non_blocking=True)
            context_batch = context_batch.to(device, non_blocking=True)
            target_batch = target_batch.to(device, non_blocking=True)
            regression_batch = regression_batch.to(device, non_blocking=True)
            weight_batch = weight_batch.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            logits, regression = model(sequence_batch, context_batch)
            classification_loss = F.binary_cross_entropy_with_logits(
                logits, target_batch, pos_weight=positive_weight_tensor, reduction="none",
            ).mean(dim=1)
            regression_loss = F.smooth_l1_loss(regression, regression_batch, reduction="none").mean(dim=1)
            per_row_loss = classification_loss + auxiliary_weight * regression_loss
            loss = (per_row_loss * weight_batch).sum() / weight_batch.sum()
            loss.backward()
            optimizer.step()
            weighted_loss_sum += float((per_row_loss.detach() * weight_batch).sum())
            weighted_classification_sum += float((classification_loss.detach() * weight_batch).sum())
            weighted_regression_sum += float((regression_loss.detach() * weight_batch).sum())
            weight_sum += float(weight_batch.sum())
        history.append({
            "epoch": epoch,
            "weighted_loss": weighted_loss_sum / weight_sum,
            "weighted_classification_loss": weighted_classification_sum / weight_sum,
            "weighted_regression_loss": weighted_regression_sum / weight_sum,
        })
        if epoch in capture_set:
            scores, regression = _predict(model, evaluation_sequence, evaluation_context, device)
            captures[epoch] = {"scores": scores, "regression": regression}
    return {
        "model": model,
        "scalers": scalers,
        "positive_weights": positive_weights,
        "history": pd.DataFrame(history),
        "captures": captures,
        "train_rows": int(train_mask.sum()),
        "train_weight_sum": float(train_weights.sum()),
    }


def _prediction_frame(
    frame: pd.DataFrame,
    mask: np.ndarray,
    scores: np.ndarray,
    regression: np.ndarray,
    target_names: list[str],
    evaluation_group: str,
    regression_scale: float,
) -> pd.DataFrame:
    identity = [
        "source_path", "session", "symbol", "input_end_timestamp", "current_is_bearish",
        "entry_last_ask", "maximum_future_bid_return_3m", "minimum_future_bid_return_3m",
        *[f"target_{name}" for name in target_names],
    ]
    output = frame.loc[mask, identity].copy()
    for target_index, name in enumerate(target_names):
        output[f"score_{name}"] = scores[:, target_index]
    output["predicted_maximum_bid_return_3m"] = regression[:, 0] / regression_scale
    output["predicted_minimum_bid_return_3m"] = regression[:, 1] / regression_scale
    output["evaluation_group"] = evaluation_group
    return output


def _safe_metric(y: np.ndarray, score: np.ndarray, metric: str) -> float:
    if len(np.unique(y)) < 2:
        return float("nan")
    if metric == "pr":
        return float(average_precision_score(y, score))
    return float(roc_auc_score(y, score))


def event_metrics(predictions: pd.DataFrame, target_names: list[str]) -> pd.DataFrame:
    rows = []
    for evaluation_group, group in predictions.groupby("evaluation_group", sort=False):
        for subgroup, subgroup_mask in [
            ("all_candles", np.ones(len(group), dtype=bool)),
            ("bearish_only", group["current_is_bearish"].eq(1).to_numpy()),
        ]:
            subgroup_frame = group.loc[subgroup_mask]
            for session, part in [("ALL", subgroup_frame), *list(subgroup_frame.groupby("session", sort=False))]:
                for name in target_names:
                    y = part[f"target_{name}"].to_numpy(dtype=np.int8)
                    score = part[f"score_{name}"].to_numpy(dtype=np.float64)
                    prevalence = float(y.mean()) if len(y) else float("nan")
                    pr_auc = _safe_metric(y, score, "pr")
                    rows.append({
                        "evaluation_group": evaluation_group,
                        "subgroup": subgroup,
                        "session": session,
                        "target": name,
                        "samples": len(part),
                        "positives": int(y.sum()),
                        "positive_rate": prevalence,
                        "pr_auc": pr_auc,
                        "pr_lift": pr_auc / prevalence if prevalence > 0 else np.nan,
                        "roc_auc": _safe_metric(y, score, "roc"),
                        "mean_score": float(score.mean()) if len(score) else np.nan,
                    })
    return pd.DataFrame(rows)


def primary_decision_metrics(predictions: pd.DataFrame, primary_target: str) -> pd.DataFrame:
    rows = []
    for evaluation_group, group in predictions.groupby("evaluation_group", sort=False):
        for subgroup, subgroup_mask in [
            ("all_candles", np.ones(len(group), dtype=bool)),
            ("bearish_only", group["current_is_bearish"].eq(1).to_numpy()),
        ]:
            subgroup_frame = group.loc[subgroup_mask]
            for session, part in [("ALL", subgroup_frame), *list(subgroup_frame.groupby("session", sort=False))]:
                y = part[f"target_{primary_target}"].to_numpy(dtype=np.int8)
                score = part["decision_score"].to_numpy(dtype=np.float64)
                prevalence = float(y.mean()) if len(y) else float("nan")
                pr_auc = _safe_metric(y, score, "pr")
                rows.append({
                    "evaluation_group": evaluation_group,
                    "subgroup": subgroup,
                    "session": session,
                    "target": f"decision_for_{primary_target}",
                    "samples": len(part),
                    "positives": int(y.sum()),
                    "positive_rate": prevalence,
                    "pr_auc": pr_auc,
                    "pr_lift": pr_auc / prevalence if prevalence > 0 else np.nan,
                    "roc_auc": _safe_metric(y, score, "roc"),
                    "mean_score": float(score.mean()) if len(score) else np.nan,
                })
    return pd.DataFrame(rows)


def _cluster_count(frame: pd.DataFrame, mask: np.ndarray, minutes: int) -> int:
    selected = frame.loc[mask].sort_values(["session", "symbol", "input_end_timestamp"])
    if selected.empty:
        return 0
    gap = selected.groupby(["session", "symbol"])["input_end_timestamp"].diff().dt.total_seconds().div(60)
    return int((gap.isna() | gap.gt(minutes)).sum())


def event_threshold_metrics(
    predictions: pd.DataFrame,
    target_name: str,
    threshold: float,
    horizon_minutes: int,
    score_column: str | None = None,
) -> pd.DataFrame:
    score_column = score_column or f"score_{target_name}"
    rows = []
    for evaluation_group, group in predictions.groupby("evaluation_group", sort=False):
        for subgroup, subgroup_mask in [
            ("all_candles", np.ones(len(group), dtype=bool)),
            ("bearish_only", group["current_is_bearish"].eq(1).to_numpy()),
        ]:
            part = group.loc[subgroup_mask]
            predicted = part[score_column].ge(threshold).to_numpy()
            actual = part[f"target_{target_name}"].eq(1).to_numpy()
            true_positive = int((predicted & actual).sum())
            signals = int(predicted.sum())
            positives = int(actual.sum())
            rows.append({
                "evaluation_group": evaluation_group,
                "subgroup": subgroup,
                "threshold": threshold,
                "samples": len(part),
                "signals": signals,
                "signal_share": signals / len(part) if len(part) else np.nan,
                "true_positives": true_positive,
                "precision": true_positive / signals if signals else np.nan,
                "recall": true_positive / positives if positives else np.nan,
                "actual_positive_clusters": _cluster_count(part, actual, horizon_minutes),
                "predicted_signal_clusters": _cluster_count(part, predicted, horizon_minutes),
                "captured_positive_clusters": _cluster_count(part, predicted & actual, horizon_minutes),
            })
    return pd.DataFrame(rows)


def decision_score_candidates(
    predictions: pd.DataFrame,
    target_names: list[str],
) -> dict[str, np.ndarray]:
    candidates = {
        f"head_{name}": predictions[f"score_{name}"].to_numpy(dtype=np.float64)
        for name in target_names
    }
    candidates["mean_all_heads"] = predictions[
        [f"score_{name}" for name in target_names]
    ].mean(axis=1).to_numpy(dtype=np.float64)
    return candidates


def create_event_dataset(
    project_root: Path | None = None,
    config_path: Path | None = None,
) -> dict[str, Any]:
    project_root = find_project_root(project_root)
    config_path = config_path or project_root / "configs/event_centric_entry.yaml"
    config = load_config(project_root, config_path)
    data_root = (project_root / config["data"]["root"]).resolve()
    processed = data_root / "processed"
    sequence_index = pd.read_parquet(data_root / config["data"]["sequence_index_artifact"])
    feature_rows = pd.read_parquet(data_root / config["data"]["feature_rows_artifact"])
    labels = build_multihorizon_labels(sequence_index, config["label"])
    exclusions = dict(labels.attrs["exclusions"])
    metadata, sequence, context = build_event_feature_arrays(
        feature_rows, sequence_index, labels, config["features"],
    )
    version = config["artifacts"]["version"]
    paths = {
        "labels": processed / f"{version}_labels.parquet",
        "metadata": processed / f"{version}_metadata.parquet",
        "features": processed / f"{version}_features.npz",
        "schema": processed / f"{version}_schema.json",
    }
    labels.to_parquet(paths["labels"], index=False, compression="zstd")
    metadata.to_parquet(paths["metadata"], index=False, compression="zstd")
    np.savez_compressed(paths["features"], sequence=sequence, context=context)
    paths["schema"].write_text(json.dumps({
        "version": version,
        "label": config["label"],
        "sequence_length": config["features"]["sequence_length"],
        "sequence_features": config["features"]["sequence"],
        "context_features": config["features"]["context"],
        "quote_columns_are_label_only": True,
        "exclusions": exclusions,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    return {
        "config": config,
        "metadata": metadata,
        "sequence": sequence,
        "context": context,
        "exclusions": exclusions,
        "paths": paths,
    }


def run_event_centric_experiment(
    project_root: Path | None = None,
    config_path: Path | None = None,
    rebuild_dataset: bool = True,
) -> dict[str, Any]:
    project_root = find_project_root(project_root)
    config_path = config_path or project_root / "configs/event_centric_entry.yaml"
    config = load_config(project_root, config_path)
    if "envs/urban" not in str(Path(sys.executable).resolve()):
        raise AssertionError(f"urban 환경이 아닙니다: {sys.executable}")
    data_root = (project_root / config["data"]["root"]).resolve()
    processed = data_root / "processed"
    model_root = data_root / "models"
    version = config["artifacts"]["version"]
    dataset_paths = {
        "labels": processed / f"{version}_labels.parquet",
        "metadata": processed / f"{version}_metadata.parquet",
        "features": processed / f"{version}_features.npz",
        "schema": processed / f"{version}_schema.json",
    }
    if rebuild_dataset or not all(path.exists() for path in dataset_paths.values()):
        dataset = create_event_dataset(project_root, config_path)
        frame = dataset["metadata"]
        sequence = dataset["sequence"]
        context = dataset["context"]
        exclusions = dataset["exclusions"]
    else:
        frame = pd.read_parquet(dataset_paths["metadata"])
        arrays = np.load(dataset_paths["features"])
        sequence = arrays["sequence"]
        context = arrays["context"]
        schema = json.loads(dataset_paths["schema"].read_text(encoding="utf-8"))
        exclusions = schema["exclusions"]
    frame = frame.reset_index(drop=True)
    frame["input_end_timestamp"] = pd.to_datetime(frame["input_end_timestamp"], utc=True)
    train_sessions = list(config["data"]["train_sessions"])
    test_sessions = list(config["data"]["test_sessions"])
    if set(frame["session"]) != set(train_sessions + test_sessions):
        raise AssertionError("event dataset session과 config session이 다릅니다.")
    train_mask = frame["session"].isin(train_sessions).to_numpy()
    test_mask = frame["session"].isin(test_sessions).to_numpy()
    if frame.loc[train_mask, "input_end_timestamp"].max() >= frame.loc[test_mask, "input_end_timestamp"].min():
        raise AssertionError("Test는 Train보다 뒤여야 합니다.")
    target_names = [str(spec["name"]) for spec in config["label"]["targets"]]
    target_columns = [f"target_{name}" for name in target_names]
    primary_target = str(config["label"]["primary_target"])
    primary_column = f"target_{primary_target}"
    target_matrix = frame[target_columns].to_numpy(dtype=np.float32)
    model_config = config["model"]
    regression_scale = float(model_config["regression_scale"])
    regression_targets = np.column_stack([
        frame["maximum_future_bid_return_3m"].to_numpy(dtype=np.float32),
        frame["minimum_future_bid_return_3m"].to_numpy(dtype=np.float32),
    ]) * regression_scale
    regression_targets = np.clip(
        regression_targets,
        float(model_config["regression_clip_min"]),
        float(model_config["regression_clip_max"]),
    ).astype(np.float32)
    policy = build_event_training_policy(frame, primary_column, config["sampling"])
    selected_mask = policy["selected_mask"]
    overlap_weight = policy["overlap_weight"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seed = int(config["project"]["seed"])
    candidate_epochs = [int(epoch) for epoch in model_config["candidate_epochs"]]
    minimum_prior = int(model_config["minimum_oof_prior_sessions"])
    oof_parts: dict[int, list[pd.DataFrame]] = {epoch: [] for epoch in candidate_epochs}
    fold_rows = []
    history_parts = []
    for evaluation_index in range(minimum_prior, len(train_sessions)):
        fit_sessions = train_sessions[:evaluation_index]
        evaluation_session = train_sessions[evaluation_index]
        fit_mask = frame["session"].isin(fit_sessions).to_numpy()
        evaluation_mask = frame["session"].eq(evaluation_session).to_numpy()
        fitted = _fit_model(
            sequence,
            context,
            target_matrix,
            regression_targets,
            frame,
            fit_mask,
            evaluation_mask,
            selected_mask,
            overlap_weight,
            config,
            device,
            seed + evaluation_index * 100,
            candidate_epochs,
        )
        for epoch in candidate_epochs:
            capture = fitted["captures"][epoch]
            oof_parts[epoch].append(_prediction_frame(
                frame,
                evaluation_mask,
                capture["scores"],
                capture["regression"],
                target_names,
                "oof",
                regression_scale,
            ))
        history = fitted["history"].copy()
        history["fold"] = len(fold_rows) + 1
        history["evaluation_session"] = evaluation_session
        history_parts.append(history)
        fold_rows.append({
            "fold": len(fold_rows) + 1,
            "fit_sessions": ",".join(fit_sessions),
            "evaluation_session": evaluation_session,
            "train_rows": fitted["train_rows"],
            "positive_weights": json.dumps(fitted["positive_weights"].tolist()),
        })
        del fitted
        if device.type == "cuda":
            torch.cuda.empty_cache()
    epoch_rows = []
    oof_predictions_by_epoch = {}
    for epoch in candidate_epochs:
        predictions = pd.concat(oof_parts[epoch], ignore_index=True).sort_values(
            ["input_end_timestamp", "symbol"],
        ).reset_index(drop=True)
        oof_predictions_by_epoch[epoch] = predictions
        primary_y = predictions[primary_column].to_numpy(dtype=np.int8)
        bearish_mask = predictions["current_is_bearish"].eq(1).to_numpy()
        for score_source, score in decision_score_candidates(predictions, target_names).items():
            overall_pr = _safe_metric(primary_y, score, "pr")
            bearish_pr = _safe_metric(primary_y[bearish_mask], score[bearish_mask], "pr")
            overall_prevalence = float(primary_y.mean())
            bearish_prevalence = float(primary_y[bearish_mask].mean())
            epoch_rows.append({
                "epoch": epoch,
                "score_source": score_source,
                "oof_primary_pr_auc": overall_pr,
                "oof_primary_pr_lift": overall_pr / overall_prevalence,
                "oof_bearish_pr_auc": bearish_pr,
                "oof_bearish_pr_lift": bearish_pr / bearish_prevalence,
            })
    epoch_selection = pd.DataFrame(epoch_rows).sort_values(
        ["oof_primary_pr_auc", "epoch"], ascending=[False, True],
    ).reset_index(drop=True)
    selected_epoch = int(epoch_selection.iloc[0]["epoch"])
    selected_score_source = str(epoch_selection.iloc[0]["score_source"])
    oof_predictions = oof_predictions_by_epoch[selected_epoch]
    oof_predictions["decision_score"] = decision_score_candidates(
        oof_predictions, target_names,
    )[selected_score_source]
    final_fit = _fit_model(
        sequence,
        context,
        target_matrix,
        regression_targets,
        frame,
        train_mask,
        train_mask | test_mask,
        selected_mask,
        overlap_weight,
        config,
        device,
        seed + 999,
        [selected_epoch],
    )
    final_capture = final_fit["captures"][selected_epoch]
    all_evaluation_positions = np.flatnonzero(train_mask | test_mask)
    train_selection = np.isin(all_evaluation_positions, np.flatnonzero(train_mask))
    test_selection = np.isin(all_evaluation_positions, np.flatnonzero(test_mask))
    combined_scores = final_capture["scores"]
    combined_regression = final_capture["regression"]
    final_predictions = pd.concat([
        _prediction_frame(
            frame,
            train_mask,
            combined_scores[train_selection],
            combined_regression[train_selection],
            target_names,
            "final_train",
            regression_scale,
        ),
        _prediction_frame(
            frame,
            test_mask,
            combined_scores[test_selection],
            combined_regression[test_selection],
            target_names,
            "test",
            regression_scale,
        ),
    ], ignore_index=True)
    final_predictions["decision_score"] = decision_score_candidates(
        final_predictions, target_names,
    )[selected_score_source]
    all_predictions = pd.concat([oof_predictions, final_predictions], ignore_index=True)
    metrics = pd.concat([
        event_metrics(all_predictions, target_names),
        primary_decision_metrics(all_predictions, primary_target),
    ], ignore_index=True)
    threshold = float(
        oof_predictions["decision_score"].quantile(float(config["decision"]["oof_score_quantile"]))
    )
    threshold_frame = pd.DataFrame([{
        "threshold": threshold,
        "method": config["decision"]["method"],
        "oof_score_quantile": config["decision"]["oof_score_quantile"],
        "selected_epoch": selected_epoch,
        "selected_score_source": selected_score_source,
        "test_outcome_used": False,
    }])
    threshold_summary = event_threshold_metrics(
        pd.concat([oof_predictions, final_predictions[final_predictions["evaluation_group"].eq("test")]]),
        primary_target,
        threshold,
        3,
        "decision_score",
    )
    sampling_balance = pd.DataFrame({
        "session": frame["session"],
        "selected": selected_mask,
        "primary": policy["primary_mask"],
        "hard_negative": policy["hard_negative_mask"],
        "bearish": frame["current_is_bearish"].eq(1),
    }).groupby("session", as_index=False).agg(
        candidate_rows=("session", "size"),
        sampled_rows=("selected", "sum"),
        positives=("primary", "sum"),
        hard_negatives=("hard_negative", "sum"),
        bearish_rows=("bearish", "sum"),
    )
    model_root.mkdir(parents=True, exist_ok=True)
    checkpoint_path = model_root / f"{version}_final.pt"
    torch.save({
        "model_state_dict": {
            name: value.detach().cpu().clone() for name, value in final_fit["model"].state_dict().items()
        },
        "sequence_features": config["features"]["sequence"],
        "context_features": config["features"]["context"],
        "scalers": [torch.from_numpy(array) for array in final_fit["scalers"]],
        "target_names": target_names,
        "primary_target": primary_target,
        "positive_weights": final_fit["positive_weights"],
        "selected_epoch": selected_epoch,
        "selected_score_source": selected_score_source,
        "threshold": threshold,
        "train_sessions": train_sessions,
        "test_sessions": test_sessions,
        "test_is_pristine": config["data"]["test_is_pristine"],
        "probability_is_calibrated": False,
        "seed": seed,
    }, checkpoint_path)
    parameter_count = int(sum(parameter.numel() for parameter in final_fit["model"].parameters()))
    paths = {
        **dataset_paths,
        "predictions": processed / f"{version}_predictions.parquet",
        "metrics": processed / f"{version}_metrics.parquet",
        "epoch_selection": processed / f"{version}_epoch_selection.parquet",
        "threshold": processed / f"{version}_threshold.parquet",
        "threshold_summary": processed / f"{version}_threshold_summary.parquet",
        "sampling_balance": processed / f"{version}_sampling_balance.parquet",
        "oof_folds": processed / f"{version}_oof_folds.parquet",
        "oof_history": processed / f"{version}_oof_history.parquet",
        "final_history": processed / f"{version}_final_history.parquet",
        "checkpoint": checkpoint_path,
        "manifest": model_root / f"{version}_manifest.json",
    }
    outputs = {
        "predictions": all_predictions,
        "metrics": metrics,
        "epoch_selection": epoch_selection,
        "threshold": threshold_frame,
        "threshold_summary": threshold_summary,
        "sampling_balance": sampling_balance,
        "oof_folds": pd.DataFrame(fold_rows),
        "oof_history": pd.concat(history_parts, ignore_index=True),
        "final_history": final_fit["history"],
    }
    for name, output in outputs.items():
        output.to_parquet(paths[name], index=False, compression="zstd")
    manifest = {
        "version": version,
        "environment": "urban",
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "device": str(device),
        "config_path": config["_config_path"],
        "label": config["label"],
        "quote_columns_are_label_only": True,
        "all_candles_are_candidates": True,
        "sequence_shape": [int(config["features"]["sequence_length"]), len(config["features"]["sequence"])],
        "context_feature_count": len(config["features"]["context"]),
        "parameter_count": parameter_count,
        "selected_epoch": selected_epoch,
        "selected_score_source": selected_score_source,
        "positive_weights": final_fit["positive_weights"].tolist(),
        "train_rows": int(train_mask.sum()),
        "test_rows": int(test_mask.sum()),
        "sampled_train_rows": int((selected_mask & train_mask).sum()),
        "positive_event_clusters": int(policy["positive_clusters"]),
        "hard_negative_clusters": int(policy["hard_negative_clusters"]),
        "threshold": threshold_frame.iloc[0].to_dict(),
        "exclusions": exclusions,
        "validation_method": "train_session_walk_forward_oof",
        "test_is_pristine": bool(config["data"]["test_is_pristine"]),
        "probability_is_calibrated": False,
        "checkpoint": str(checkpoint_path),
        "artifacts": {name: str(path) for name, path in paths.items() if name != "manifest"},
    }
    paths["manifest"].write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, default=str), encoding="utf-8",
    )
    if not all(path.exists() for path in paths.values()):
        raise AssertionError("V7 필수 artifact가 누락됐습니다.")
    return {
        "config": config,
        "device": str(device),
        "parameter_count": parameter_count,
        "selected_epoch": selected_epoch,
        "selected_score_source": selected_score_source,
        "positive_weights": final_fit["positive_weights"],
        "exclusions": exclusions,
        "policy": policy,
        "sampling_balance": sampling_balance,
        "epoch_selection": epoch_selection,
        "metrics": metrics,
        "threshold": threshold_frame,
        "threshold_summary": threshold_summary,
        "predictions": all_predictions,
        "paths": paths,
    }


__all__ = [
    "EventSequenceModel",
    "build_event_feature_arrays",
    "build_event_training_policy",
    "build_multihorizon_labels",
    "create_event_dataset",
    "decision_score_candidates",
    "event_metrics",
    "event_threshold_metrics",
    "multi_horizon_quote_targets",
    "primary_decision_metrics",
    "run_event_centric_experiment",
]
