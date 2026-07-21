from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset

from src.walk_forward_oof import (
    find_project_root,
    fit_robust_scaler,
    load_config,
    seed_everything,
    session_balancing_weights,
    training_sample_mask,
    transform_features,
)


RAW_COLUMNS = [
    "symbol",
    "timestamp_utc",
    "open",
    "close",
    "quote_count",
    "last_bid",
    "last_ask",
]


class BinaryMLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: list[int], dropout: float):
        super().__init__()
        layers: list[nn.Module] = []
        previous = input_dim
        for hidden in hidden_dims:
            layers.extend([nn.Linear(previous, hidden), nn.ReLU(), nn.Dropout(dropout)])
            previous = hidden
        self.encoder = nn.Sequential(*layers)
        self.output = nn.Linear(previous, 1)
        nn.init.zeros_(self.output.bias)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.output(self.encoder(features)).squeeze(-1)


def quote_surge_target(
    entry_last_ask: float,
    future_last_bid: np.ndarray,
    target_return: float,
) -> tuple[int, int, float]:
    future_bid = np.asarray(future_last_bid, dtype=np.float64)
    if entry_last_ask <= 0 or future_bid.size == 0 or not np.isfinite(future_bid).all():
        raise ValueError("유효한 entry ask와 future bid가 필요합니다.")
    threshold_bid = entry_last_ask * (1.0 + target_return)
    hits = np.flatnonzero(future_bid >= threshold_bid)
    target = int(len(hits) > 0)
    first_hit_minute = int(hits[0] + 1) if target else 0
    maximum_return = float(future_bid.max() / entry_last_ask - 1.0)
    return target, first_hit_minute, maximum_return


def _load_quote_file(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, usecols=RAW_COLUMNS, encoding="utf-8-sig")
    frame["symbol"] = frame["symbol"].astype(str).str.upper()
    frame["timestamp_utc"] = pd.to_datetime(frame["timestamp_utc"], utc=True, errors="coerce")
    for column in ["open", "close", "quote_count", "last_bid", "last_ask"]:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame = frame.sort_values(["symbol", "timestamp_utc"]).reset_index(drop=True)
    delta_minutes = frame.groupby("symbol")["timestamp_utc"].diff().dt.total_seconds().div(60)
    consecutive = pd.Series(np.isclose(delta_minutes, 1.0, rtol=0.0, atol=1e-6), index=frame.index)
    new_run = (~consecutive) | frame["symbol"].ne(frame["symbol"].shift())
    frame["run_number"] = new_run.cumsum().astype(np.int64)
    frame["run_id"] = path.stem + "::" + frame["run_number"].astype(str)
    return frame


def build_quote_surge_labels(
    sequence_index: pd.DataFrame,
    label_config: dict[str, Any],
) -> pd.DataFrame:
    horizon = int(label_config["horizon_minutes"])
    target_return = float(label_config["target_return"])
    rows: list[dict[str, Any]] = []
    excluded_invalid_current_quote = 0
    excluded_invalid_future_quote = 0
    excluded_non_bearish = 0
    excluded_insufficient_future = 0
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
                if position is None or position + horizon >= len(run):
                    excluded_insufficient_future += 1
                    continue
                current = run.iloc[position]
                if not current["close"] < current["open"]:
                    excluded_non_bearish += 1
                    continue
                current_quote_valid = (
                    current["quote_count"] >= 1
                    and current["last_bid"] > 0
                    and current["last_ask"] >= current["last_bid"]
                )
                if not current_quote_valid:
                    excluded_invalid_current_quote += 1
                    continue
                future = run.iloc[position + 1:position + 1 + horizon]
                future_quote_valid = (
                    future["quote_count"].ge(1)
                    & future["last_bid"].gt(0)
                    & future["last_ask"].ge(future["last_bid"])
                )
                if len(future) != horizon or not future_quote_valid.all():
                    excluded_invalid_future_quote += 1
                    continue
                future_bid = future["last_bid"].to_numpy(dtype=np.float64)
                entry_ask = float(current["last_ask"])
                threshold_bid = entry_ask * (1.0 + target_return)
                target, first_hit_minute, maximum_return = quote_surge_target(
                    entry_ask, future_bid, target_return,
                )
                rows.append({
                    "source_path": source_path,
                    "session": meta.session,
                    "symbol": meta.symbol,
                    "run_id": run_id,
                    "input_end_timestamp": timestamp,
                    "entry_last_ask": entry_ask,
                    "target_bid": threshold_bid,
                    "maximum_future_last_bid_3m": float(future_bid.max()),
                    "maximum_future_bid_return_3m": maximum_return,
                    "first_hit_minute": first_hit_minute,
                    "target_surge_3m": target,
                    "label_version": label_config["version"],
                })
    labels = pd.DataFrame(rows)
    labels.attrs["exclusions"] = {
        "non_bearish": excluded_non_bearish,
        "invalid_current_quote": excluded_invalid_current_quote,
        "invalid_future_quote": excluded_invalid_future_quote,
        "insufficient_future": excluded_insufficient_future,
    }
    if labels.duplicated(["source_path", "symbol", "input_end_timestamp"]).any():
        raise AssertionError("quote surge label key가 중복됩니다.")
    if not labels["target_surge_3m"].isin([0, 1]).all():
        raise AssertionError("binary target이 0/1이 아닙니다.")
    return labels


def build_selected_tabular(
    feature_rows: pd.DataFrame,
    sequence_index: pd.DataFrame,
    labels: pd.DataFrame,
    selected_features: list[str],
) -> pd.DataFrame:
    merge_keys = ["source_path", "symbol", "input_end_timestamp"]
    label_columns = [
        *merge_keys,
        "session",
        "entry_last_ask",
        "target_bid",
        "maximum_future_last_bid_3m",
        "maximum_future_bid_return_3m",
        "first_hit_minute",
        "target_surge_3m",
        "label_version",
    ]
    label_view = labels[label_columns].rename(columns={"session": "label_session"})
    dataset = sequence_index.merge(label_view, on=merge_keys, how="inner", validate="one_to_one")
    if len(dataset) != len(labels):
        raise AssertionError("sequence와 quote label 결합 수가 다릅니다.")
    requested: list[tuple[str, str]] = []
    for name in selected_features:
        if "__" not in name:
            raise ValueError(f"집계 prefix가 없는 feature입니다: {name}")
        aggregation, base = name.split("__", 1)
        requested.append((aggregation, base))
    base_features = list(dict.fromkeys(base for _, base in requested))
    missing = sorted(set(base_features) - set(feature_rows.columns))
    if missing:
        raise ValueError(f"feature_rows에 base feature가 없습니다: {missing}")
    X = np.empty((len(dataset), len(selected_features)), dtype=np.float32)
    base_index = {name: index for index, name in enumerate(base_features)}
    for source_path, group_index in dataset.groupby("source_path", sort=False).groups.items():
        positions = np.asarray(list(group_index), dtype=np.int64)
        metadata = dataset.loc[positions]
        rows = feature_rows[feature_rows["source_path"].eq(source_path)].sort_values("feature_row")
        matrix = rows[base_features].to_numpy(dtype=np.float32)
        safe = np.nan_to_num(matrix, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float64)
        prefix = np.vstack([np.zeros((1, len(base_features))), np.cumsum(safe, axis=0)])
        prefix_sq = np.vstack([np.zeros((1, len(base_features))), np.cumsum(safe**2, axis=0)])
        start = metadata["start_feature_row"].to_numpy(dtype=np.int64)
        end = metadata["end_feature_row"].to_numpy(dtype=np.int64)
        mean60 = (prefix[end + 1] - prefix[start]) / 60.0
        second_moment = (prefix_sq[end + 1] - prefix_sq[start]) / 60.0
        std60 = np.sqrt(np.maximum(second_moment - mean60**2, 0.0))
        for output_index, (aggregation, base) in enumerate(requested):
            column_index = base_index[base]
            if aggregation == "last":
                values = matrix[end, column_index]
            elif aggregation == "mean60":
                values = mean60[:, column_index]
            elif aggregation == "std60":
                values = std60[:, column_index]
            elif aggregation == "delta5":
                values = matrix[end, column_index] - matrix[end - 5, column_index]
            elif aggregation == "delta20":
                values = matrix[end, column_index] - matrix[end - 20, column_index]
            else:
                raise ValueError(f"지원하지 않는 aggregation입니다: {aggregation}")
            X[positions, output_index] = values.astype(np.float32)
    if not np.isfinite(X).all():
        raise AssertionError("선택 tabular feature에 non-finite 값이 있습니다.")
    return pd.concat([dataset.reset_index(drop=True), pd.DataFrame(X, columns=selected_features)], axis=1)


def create_quote_surge_dataset(
    project_root: Path | None = None,
    config_path: Path | None = None,
) -> dict[str, Any]:
    project_root = find_project_root(project_root)
    config_path = config_path or project_root / "configs/quote_surge_3m_binary.yaml"
    config = load_config(project_root, config_path)
    data_root = (project_root / config["data"]["root"]).resolve()
    processed = data_root / "processed"
    sequence_path = data_root / config["data"]["sequence_index_artifact"]
    feature_rows_path = data_root / config["data"]["feature_rows_artifact"]
    selected_schema_path = data_root / config["data"]["selected_feature_schema"]
    selected_schema = json.loads(selected_schema_path.read_text(encoding="utf-8"))
    selected_features = list(selected_schema["selected_features"])
    forbidden = [
        "volume", "notional", "vwap", "quote", "bid", "ask", "spread", "trade_count",
        "imbalance", "aggressive", "trade_strength",
    ]
    if any(any(token in feature.lower() for token in forbidden) for feature in selected_features):
        raise AssertionError("모델 입력에 금지된 quote/trade feature가 있습니다.")
    sequence_index = pd.read_parquet(sequence_path)
    feature_rows = pd.read_parquet(feature_rows_path)
    labels = build_quote_surge_labels(sequence_index, config["label"])
    exclusions = dict(labels.attrs["exclusions"])
    tabular = build_selected_tabular(feature_rows, sequence_index, labels, selected_features)
    version = config["artifacts"]["version"]
    paths = {
        "labels": processed / f"{version}_labels.parquet",
        "tabular": processed / f"{version}_tabular.parquet",
        "schema": processed / f"{version}_schema.json",
    }
    labels.to_parquet(paths["labels"], index=False, compression="zstd")
    tabular.to_parquet(paths["tabular"], index=False, compression="zstd")
    paths["schema"].write_text(json.dumps({
        "version": version,
        "label": config["label"],
        "selected_feature_count": len(selected_features),
        "selected_features": selected_features,
        "quote_columns_are_label_only": True,
        "exclusions": exclusions,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    return {
        "config": config,
        "labels": labels,
        "tabular": tabular,
        "selected_features": selected_features,
        "exclusions": exclusions,
        "paths": paths,
    }


def _predict_score(
    model: nn.Module,
    features: np.ndarray,
    device: torch.device,
    batch_size: int = 4096,
) -> np.ndarray:
    model.eval()
    outputs = []
    with torch.no_grad():
        for start in range(0, len(features), batch_size):
            batch = torch.from_numpy(features[start:start + batch_size]).to(device)
            outputs.append(torch.sigmoid(model(batch)).cpu().numpy())
    return np.concatenate(outputs).astype(np.float32)


def _safe_auc(y: np.ndarray, probability: np.ndarray, metric: str) -> float:
    if len(np.unique(y)) < 2:
        return float("nan")
    if metric == "pr":
        return float(average_precision_score(y, probability))
    return float(roc_auc_score(y, probability))


def resolve_positive_weight(
    labels: np.ndarray,
    row_weights: np.ndarray,
    configured_weight: str | float,
    multiplier: float,
) -> tuple[float, float]:
    """Resolve the train-only BCE positive weight and its explicit multiplier."""
    labels = np.asarray(labels)
    row_weights = np.asarray(row_weights, dtype=np.float64)
    if len(labels) != len(row_weights) or len(labels) == 0:
        raise ValueError("label과 row weight 길이가 같고 비어 있지 않아야 합니다.")
    if not np.isfinite(row_weights).all() or (row_weights <= 0).any():
        raise ValueError("row weight는 유효한 양수여야 합니다.")
    if multiplier <= 0:
        raise ValueError("positive_weight_multiplier는 양수여야 합니다.")
    if configured_weight == "balanced":
        positive_total = float(row_weights[labels == 1].sum())
        negative_total = float(row_weights[labels == 0].sum())
        if positive_total <= 0 or negative_total <= 0:
            raise ValueError("balanced positive weight에는 양성과 음성이 모두 필요합니다.")
        base_weight = negative_total / positive_total
    else:
        base_weight = float(configured_weight)
        if not math.isfinite(base_weight) or base_weight <= 0:
            raise ValueError("positive_weight는 balanced 또는 유효한 양수여야 합니다.")
    return base_weight, base_weight * float(multiplier)


def binary_metrics(predictions: pd.DataFrame) -> pd.DataFrame:
    groups: list[tuple[str, str, pd.DataFrame]] = []
    for evaluation_group in ["train", "test"]:
        part = predictions[predictions["evaluation_group"].eq(evaluation_group)]
        groups.append((evaluation_group, "ALL", part))
    groups.extend(
        (evaluation_group, session, part)
        for (evaluation_group, session), part in predictions.groupby(["evaluation_group", "session"], sort=False)
    )
    rows = []
    for evaluation_group, session, part in groups:
        if part.empty:
            continue
        y = part["target_surge_3m"].to_numpy(dtype=np.int8)
        score = np.clip(part["surge_score"].to_numpy(dtype=np.float64), 1e-7, 1 - 1e-7)
        prevalence = float(y.mean())
        pr_auc = _safe_auc(y, score, "pr")
        rows.append({
            "evaluation_group": evaluation_group,
            "session": session,
            "samples": len(part),
            "positives": int(y.sum()),
            "positive_rate": prevalence,
            "pr_auc": pr_auc,
            "pr_lift": pr_auc / prevalence if prevalence > 0 else np.nan,
            "roc_auc": _safe_auc(y, score, "roc"),
            "uncalibrated_logloss": log_loss(y, score, labels=[0, 1]),
            "uncalibrated_brier": brier_score_loss(y, score),
            "mean_score": float(score.mean()),
        })
    return pd.DataFrame(rows)


def top_quantile_metrics(predictions: pd.DataFrame, quantiles: list[float]) -> pd.DataFrame:
    rows = []
    for group in ["train", "test"]:
        part = predictions[predictions["evaluation_group"].eq(group)].sort_values(
            "surge_score", ascending=False,
        )
        positives = int(part["target_surge_3m"].sum())
        for quantile in quantiles:
            count = max(1, math.ceil(len(part) * (1.0 - quantile)))
            selected = part.head(count)
            true_positive = int(selected["target_surge_3m"].sum())
            rows.append({
                "evaluation_group": group,
                "top_fraction": 1.0 - quantile,
                "signals": len(selected),
                "true_positives": true_positive,
                "precision": true_positive / len(selected),
                "recall": true_positive / positives if positives else np.nan,
                "minimum_score": float(selected["surge_score"].min()),
            })
    return pd.DataFrame(rows)


def _cluster_count(frame: pd.DataFrame, mask: pd.Series, minutes: int) -> int:
    selected = frame.loc[mask].sort_values(["session", "symbol", "input_end_timestamp"])
    if selected.empty:
        return 0
    gap = selected.groupby(["session", "symbol"])["input_end_timestamp"].diff().dt.total_seconds().div(60)
    return int((gap.isna() | gap.gt(minutes)).sum())


def threshold_metrics(predictions: pd.DataFrame, threshold: float, horizon: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    aggregate_rows = []
    session_rows = []
    for group in ["train", "test"]:
        group_frame = predictions[predictions["evaluation_group"].eq(group)]
        for session, part in [("ALL", group_frame), *list(group_frame.groupby("session", sort=False))]:
            predicted = part["surge_score"].ge(threshold)
            actual = part["target_surge_3m"].eq(1)
            true_positive = int((predicted & actual).sum())
            signals = int(predicted.sum())
            positives = int(actual.sum())
            row = {
                "evaluation_group": group,
                "session": session,
                "threshold": threshold,
                "samples": len(part),
                "signals": signals,
                "signal_share": signals / len(part) if len(part) else np.nan,
                "true_positives": true_positive,
                "precision": true_positive / signals if signals else np.nan,
                "recall": true_positive / positives if positives else np.nan,
                "actual_positive_clusters": _cluster_count(part, actual, horizon),
                "predicted_signal_clusters": _cluster_count(part, predicted, horizon),
                "captured_positive_clusters": _cluster_count(part, predicted & actual, horizon),
            }
            if session == "ALL":
                aggregate_rows.append(row)
            else:
                session_rows.append(row)
    return pd.DataFrame(aggregate_rows), pd.DataFrame(session_rows)


def run_quote_surge_experiment(
    project_root: Path | None = None,
    config_path: Path | None = None,
    rebuild_dataset: bool = True,
) -> dict[str, Any]:
    project_root = find_project_root(project_root)
    config_path = config_path or project_root / "configs/quote_surge_3m_binary.yaml"
    config = load_config(project_root, config_path)
    if "envs/urban" not in str(Path(sys.executable).resolve()):
        raise AssertionError(f"urban 환경이 아닙니다: {sys.executable}")
    data_root = (project_root / config["data"]["root"]).resolve()
    version = config["artifacts"]["version"]
    processed = data_root / "processed"
    model_root = data_root / "models"
    dataset_paths = {
        "labels": processed / f"{version}_labels.parquet",
        "tabular": processed / f"{version}_tabular.parquet",
        "schema": processed / f"{version}_schema.json",
    }
    if rebuild_dataset or not all(path.exists() for path in dataset_paths.values()):
        dataset_result = create_quote_surge_dataset(project_root, config_path)
        frame = dataset_result["tabular"]
        selected_features = dataset_result["selected_features"]
        exclusions = dataset_result["exclusions"]
    else:
        frame = pd.read_parquet(dataset_paths["tabular"])
        schema = json.loads(dataset_paths["schema"].read_text(encoding="utf-8"))
        selected_features = schema["selected_features"]
        exclusions = schema["exclusions"]
    frame["input_end_timestamp"] = pd.to_datetime(frame["input_end_timestamp"], utc=True)
    train_sessions = list(config["data"]["train_sessions"])
    test_sessions = list(config["data"]["test_sessions"])
    if set(frame["session"]) != set(train_sessions + test_sessions):
        raise AssertionError("dataset session과 config session이 다릅니다.")
    train_mask = frame["session"].isin(train_sessions).to_numpy()
    test_mask = frame["session"].isin(test_sessions).to_numpy()
    if frame.loc[train_mask, "input_end_timestamp"].max() >= frame.loc[test_mask, "input_end_timestamp"].min():
        raise AssertionError("test는 train보다 뒤여야 합니다.")

    seed = int(config["project"]["seed"])
    seed_everything(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    X_raw = frame[selected_features].to_numpy(dtype=np.float32)
    y = frame["target_surge_3m"].to_numpy(dtype=np.float32)
    sample_mask = training_sample_mask(frame, config["sampling"])
    selected_train_mask = train_mask & sample_mask
    center, scale = fit_robust_scaler(X_raw[train_mask])
    X_train = transform_features(
        X_raw[selected_train_mask], center, scale, float(config["sampling"]["scaler_clip"]),
    )
    y_train = y[selected_train_mask]
    row_weights = session_balancing_weights(
        frame, selected_train_mask, bool(config["sampling"]["equal_session_weights"]),
    )[selected_train_mask]
    model_config = config["model"]
    positive_weight_base, positive_weight = resolve_positive_weight(
        y_train,
        row_weights,
        model_config.get("positive_weight", "balanced"),
        float(model_config.get("positive_weight_multiplier", 1.0)),
    )
    positive_weight_multiplier = float(model_config.get("positive_weight_multiplier", 1.0))
    model = BinaryMLP(
        len(selected_features), list(model_config["hidden_dims"]), float(model_config["dropout"]),
    ).to(device)
    loader = DataLoader(
        TensorDataset(
            torch.from_numpy(X_train),
            torch.from_numpy(y_train),
            torch.from_numpy(row_weights),
        ),
        batch_size=int(model_config["batch_size"]),
        shuffle=True,
        generator=torch.Generator().manual_seed(seed + 100),
        pin_memory=device.type == "cuda",
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(model_config["learning_rate"]),
        weight_decay=float(model_config["weight_decay"]),
    )
    pos_weight_tensor = torch.tensor(positive_weight, dtype=torch.float32, device=device)
    history = []
    epochs = int(model_config["fixed_epochs"])
    for epoch in range(1, epochs + 1):
        model.train()
        weighted_loss_sum = 0.0
        weight_sum = 0.0
        for X_batch, y_batch, weight_batch in loader:
            X_batch = X_batch.to(device, non_blocking=True)
            y_batch = y_batch.to(device, non_blocking=True)
            weight_batch = weight_batch.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            per_row_loss = F.binary_cross_entropy_with_logits(
                model(X_batch), y_batch, pos_weight=pos_weight_tensor, reduction="none",
            )
            loss = (per_row_loss * weight_batch).sum() / weight_batch.sum()
            loss.backward()
            optimizer.step()
            weighted_loss_sum += float((per_row_loss.detach() * weight_batch).sum())
            weight_sum += float(weight_batch.sum())
        history.append({"epoch": epoch, "weighted_bce": weighted_loss_sum / weight_sum})
    history_frame = pd.DataFrame(history)

    prediction_frames = []
    for group, mask in [("train", train_mask), ("test", test_mask)]:
        transformed = transform_features(
            X_raw[mask], center, scale, float(config["sampling"]["scaler_clip"]),
        )
        score = _predict_score(model, transformed, device)
        prediction = frame.loc[mask, [
            "source_path", "session", "symbol", "input_end_timestamp", "entry_last_ask",
            "maximum_future_last_bid_3m", "maximum_future_bid_return_3m", "first_hit_minute",
            "target_surge_3m",
        ]].copy()
        prediction["surge_score"] = score
        prediction["evaluation_group"] = group
        prediction_frames.append(prediction)
    predictions = pd.concat(prediction_frames, ignore_index=True).sort_values(
        ["input_end_timestamp", "symbol"],
    ).reset_index(drop=True)
    metrics = binary_metrics(predictions)
    top_metrics = top_quantile_metrics(predictions, [0.99, 0.98, 0.95, 0.90])
    train_scores = predictions.loc[predictions["evaluation_group"].eq("train"), "surge_score"]
    threshold = float(train_scores.quantile(float(config["decision"]["train_score_quantile"])))
    threshold_frame = pd.DataFrame([{
        "threshold": threshold,
        "method": config["decision"]["method"],
        "train_score_quantile": config["decision"]["train_score_quantile"],
        "uses_train_outcome_for_selection": False,
        "validation_used": False,
    }])
    threshold_summary, threshold_sessions = threshold_metrics(
        predictions, threshold, int(config["label"]["horizon_minutes"]),
    )
    sampling_balance = pd.DataFrame({
        "session": frame["session"],
        "is_train": train_mask,
        "selected_for_training": selected_train_mask,
        "target": y,
        "sampled_positive": y * selected_train_mask.astype(np.float32),
    }).groupby("session", as_index=False).agg(
        candidate_rows=("session", "size"),
        is_train=("is_train", "max"),
        sampled_rows=("selected_for_training", "sum"),
        positives=("target", "sum"),
        sampled_positives=("sampled_positive", "sum"),
    )

    model_root.mkdir(parents=True, exist_ok=True)
    checkpoint_path = model_root / f"{version}_final.pt"
    torch.save({
        "model_state_dict": model.cpu().state_dict(),
        "feature_names": selected_features,
        "center": torch.from_numpy(center),
        "scale": torch.from_numpy(scale),
        "positive_weight_base": positive_weight_base,
        "positive_weight_multiplier": positive_weight_multiplier,
        "positive_weight": positive_weight,
        "probability_is_calibrated": False,
        "threshold": threshold,
        "label": config["label"],
        "train_sessions": train_sessions,
        "test_sessions": test_sessions,
        "test_is_pristine": config["data"]["test_is_pristine"],
        "epochs": epochs,
        "seed": seed,
    }, checkpoint_path)
    parameter_count = int(sum(parameter.numel() for parameter in model.parameters()))
    paths = {
        **dataset_paths,
        "predictions": processed / f"{version}_predictions.parquet",
        "metrics": processed / f"{version}_metrics.parquet",
        "top_metrics": processed / f"{version}_top_metrics.parquet",
        "threshold": processed / f"{version}_threshold.parquet",
        "threshold_summary": processed / f"{version}_threshold_summary.parquet",
        "threshold_sessions": processed / f"{version}_threshold_sessions.parquet",
        "sampling_balance": processed / f"{version}_sampling_balance.parquet",
        "history": processed / f"{version}_history.parquet",
        "checkpoint": checkpoint_path,
        "manifest": model_root / f"{version}_manifest.json",
    }
    outputs = {
        "predictions": predictions,
        "metrics": metrics,
        "top_metrics": top_metrics,
        "threshold": threshold_frame,
        "threshold_summary": threshold_summary,
        "threshold_sessions": threshold_sessions,
        "sampling_balance": sampling_balance,
        "history": history_frame,
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
        "candidate_rule": "close_t < open_t",
        "feature_count": len(selected_features),
        "parameter_count": parameter_count,
        "train_rows": int(train_mask.sum()),
        "test_rows": int(test_mask.sum()),
        "sampled_train_rows": int(selected_train_mask.sum()),
        "positive_weight_base": positive_weight_base,
        "positive_weight_multiplier": positive_weight_multiplier,
        "positive_weight": positive_weight,
        "probability_is_calibrated": False,
        "epochs": epochs,
        "validation_used": False,
        "test_is_pristine": bool(config["data"]["test_is_pristine"]),
        "threshold": threshold_frame.iloc[0].to_dict(),
        "exclusions": exclusions,
        "checkpoint": str(checkpoint_path),
        "artifacts": {name: str(path) for name, path in paths.items() if name != "manifest"},
    }
    paths["manifest"].write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, default=str), encoding="utf-8",
    )
    if predictions.duplicated(["source_path", "symbol", "input_end_timestamp"]).any():
        raise AssertionError("prediction key가 중복됩니다.")
    if not predictions["surge_score"].between(0, 1).all():
        raise AssertionError("sigmoid score가 0~1 범위를 벗어났습니다.")
    if not all(path.exists() for path in paths.values()):
        raise AssertionError("필수 artifact가 누락됐습니다.")
    return {
        "config": config,
        "device": str(device),
        "feature_count": len(selected_features),
        "parameter_count": parameter_count,
        "positive_weight_base": positive_weight_base,
        "positive_weight_multiplier": positive_weight_multiplier,
        "positive_weight": positive_weight,
        "exclusions": exclusions,
        "sampling_balance": sampling_balance,
        "history": history_frame,
        "predictions": predictions,
        "metrics": metrics,
        "top_metrics": top_metrics,
        "threshold": threshold_frame,
        "threshold_summary": threshold_summary,
        "threshold_sessions": threshold_sessions,
        "paths": paths,
    }


__all__ = [
    "BinaryMLP",
    "binary_metrics",
    "build_quote_surge_labels",
    "build_selected_tabular",
    "create_quote_surge_dataset",
    "quote_surge_target",
    "resolve_positive_weight",
    "run_quote_surge_experiment",
    "threshold_metrics",
    "top_quantile_metrics",
]
