from __future__ import annotations

import json
import math
import random
import sys
from copy import deepcopy
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.metrics import average_precision_score, log_loss, roc_auc_score
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset


def find_project_root(start: Path | None = None) -> Path:
    start = (start or Path.cwd()).resolve()
    for candidate in (start, *start.parents):
        if (candidate / "AGENT.md").exists() and (candidate / "README.md").exists():
            return candidate
    raise FileNotFoundError("AGENT.md와 README.md가 있는 프로젝트 루트를 찾지 못했습니다.")


def load_config(project_root: Path, config_path: Path | None = None) -> dict[str, Any]:
    path = config_path or project_root / "configs/walk_forward_oof.yaml"
    with path.open(encoding="utf-8") as file:
        config = yaml.safe_load(file)
    config["_config_path"] = str(path.resolve())
    return config


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def softmax_numpy(logits: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    scaled = logits.astype(np.float64) / float(temperature)
    scaled -= scaled.max(axis=1, keepdims=True)
    exp = np.exp(scaled)
    return (exp / exp.sum(axis=1, keepdims=True)).astype(np.float32)


def make_walk_forward_folds(sessions: list[str], minimum_prior_sessions: int) -> list[dict[str, Any]]:
    if minimum_prior_sessions < 3:
        raise ValueError("inner validation을 포함하려면 최소 3개 prior session이 필요합니다.")
    folds = []
    for evaluation_index in range(minimum_prior_sessions, len(sessions)):
        prior_sessions = sessions[:evaluation_index]
        folds.append({
            "fold": len(folds) + 1,
            "fit_sessions": prior_sessions[:-1],
            "inner_validation_session": prior_sessions[-1],
            "payoff_history_sessions": prior_sessions,
            "evaluation_session": sessions[evaluation_index],
        })
    if not folds:
        raise ValueError("walk-forward fold를 만들 세션이 부족합니다.")
    return folds


def fit_robust_scaler(X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    center = np.median(X.astype(np.float64), axis=0)
    q25 = np.quantile(X.astype(np.float64), 0.25, axis=0)
    q75 = np.quantile(X.astype(np.float64), 0.75, axis=0)
    scale = q75 - q25
    scale = np.where(scale > 1e-8, scale, 1.0)
    return center.astype(np.float32), scale.astype(np.float32)


def transform_features(X: np.ndarray, center: np.ndarray, scale: np.ndarray, clip: float) -> np.ndarray:
    transformed = np.clip((X - center) / scale, -clip, clip).astype(np.float32)
    if not np.isfinite(transformed).all():
        raise ValueError("scaled feature에 non-finite 값이 있습니다.")
    return transformed


def training_stride_mask(frame: pd.DataFrame, stride_minutes: int) -> np.ndarray:
    if stride_minutes <= 1:
        return np.ones(len(frame), dtype=bool)
    minute_number = frame["input_end_timestamp"].astype("int64").to_numpy() // (60 * 1_000_000_000)
    return minute_number % stride_minutes == 0


def training_sample_mask(frame: pd.DataFrame, walk_config: dict[str, Any]) -> np.ndarray:
    """Return the deterministic training sample selected by the configured policy.

    V1 configurations omit ``sampling_method`` and retain the original fixed-minute
    stride. Reduced V2 uses one earliest observation per session/symbol/10-minute
    bucket so overlapping labels do not masquerade as independent examples.
    """
    method = walk_config.get("sampling_method", "fixed_minute_stride")
    if method == "fixed_minute_stride":
        return training_stride_mask(frame, int(walk_config.get("training_stride_minutes", 1)))
    if method not in {"ten_minute_bucket", "event_bucket"}:
        raise ValueError(f"지원하지 않는 training sampling_method입니다: {method}")
    bucket_minutes = int(walk_config.get("bucket_minutes", 10))
    if bucket_minutes <= 0:
        raise ValueError("bucket_minutes는 양수여야 합니다.")
    ordered = pd.DataFrame({
        "session": frame["session"].astype(str).to_numpy(),
        "symbol": frame["symbol"].astype(str).to_numpy(),
        "timestamp": pd.to_datetime(frame["input_end_timestamp"], utc=True).to_numpy(),
        "bucket": pd.to_datetime(frame["input_end_timestamp"], utc=True).dt.floor(f"{bucket_minutes}min").to_numpy(),
        "position": np.arange(len(frame)),
    }).sort_values(["session", "symbol", "bucket", "timestamp", "position"])
    selected_positions = ordered.drop_duplicates(["session", "symbol", "bucket"], keep="first")["position"]
    mask = np.zeros(len(frame), dtype=bool)
    mask[selected_positions.to_numpy(dtype=np.int64)] = True
    return mask


def session_balancing_weights(
    frame: pd.DataFrame,
    train_mask: np.ndarray,
    enabled: bool,
) -> np.ndarray:
    """Give every selected session equal total loss weight (mean row weight is one)."""
    weights = np.zeros(len(frame), dtype=np.float32)
    selected_count = int(train_mask.sum())
    if selected_count == 0:
        return weights
    if not enabled:
        weights[train_mask] = 1.0
        return weights
    sessions = frame.loc[train_mask, "session"]
    counts = sessions.value_counts()
    session_count = len(counts)
    per_session_total = selected_count / session_count
    weights[train_mask] = sessions.map(lambda session: per_session_total / counts.loc[session]).to_numpy(dtype=np.float32)
    return weights


class OutcomeMLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: list[int], dropout: float, class_bias: np.ndarray):
        super().__init__()
        layers: list[nn.Module] = []
        previous = input_dim
        for hidden in hidden_dims:
            layers.extend([nn.Linear(previous, hidden), nn.ReLU(), nn.Dropout(dropout)])
            previous = hidden
        self.encoder = nn.Sequential(*layers)
        self.output = nn.Linear(previous, len(class_bias))
        with torch.no_grad():
            self.output.bias.copy_(torch.from_numpy(class_bias.astype(np.float32)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.output(self.encoder(x))


def predict_logits(model: nn.Module, X: np.ndarray, device: torch.device, batch_size: int = 4096) -> np.ndarray:
    model.eval()
    outputs = []
    with torch.no_grad():
        for start in range(0, len(X), batch_size):
            batch = torch.from_numpy(X[start:start + batch_size]).to(device)
            outputs.append(model(batch).detach().cpu().numpy())
    return np.concatenate(outputs).astype(np.float32)


def initial_class_bias(y: np.ndarray, class_count: int, weights: np.ndarray | None = None) -> np.ndarray:
    counts = np.bincount(y, weights=weights, minlength=class_count).astype(np.float64)
    probabilities = (counts + 1.0) / (counts.sum() + class_count)
    return np.log(probabilities).astype(np.float32)


def choose_temperature(logits: np.ndarray, y: np.ndarray, model_config: dict[str, Any]) -> tuple[float, float, float]:
    grid = np.linspace(
        model_config["temperature_grid_min"],
        model_config["temperature_grid_max"],
        model_config["temperature_grid_steps"],
    )
    labels = list(range(logits.shape[1]))
    raw_loss = log_loss(y, softmax_numpy(logits), labels=labels)
    losses = np.array([log_loss(y, softmax_numpy(logits, temperature), labels=labels) for temperature in grid])
    best_index = int(np.argmin(losses))
    return float(grid[best_index]), float(raw_loss), float(losses[best_index])


def fit_early_stopped_model(
    X_raw: np.ndarray,
    y: np.ndarray,
    frame: pd.DataFrame,
    fit_sessions: list[str],
    inner_validation_session: str,
    config: dict[str, Any],
    device: torch.device,
    seed: int,
) -> dict[str, Any]:
    model_config = config["model"]
    walk_config = config["walk_forward"]
    fit_mask = frame["session"].isin(fit_sessions).to_numpy()
    inner_mask = frame["session"].eq(inner_validation_session).to_numpy()
    sample_mask = training_sample_mask(frame, walk_config)
    train_mask = fit_mask & sample_mask
    if not train_mask.any() or not inner_mask.any():
        raise ValueError("fold train 또는 inner validation 표본이 없습니다.")

    center, scale = fit_robust_scaler(X_raw[fit_mask])
    X_train = transform_features(X_raw[train_mask], center, scale, walk_config["scaler_clip"])
    X_inner = transform_features(X_raw[inner_mask], center, scale, walk_config["scaler_clip"])
    y_train = y[train_mask]
    y_inner = y[inner_mask]
    train_weights = session_balancing_weights(
        frame, train_mask, bool(walk_config.get("equal_session_weights", False)),
    )[train_mask]

    seed_everything(seed)
    model = OutcomeMLP(
        X_raw.shape[1], model_config["hidden_dims"], model_config["dropout"],
        initial_class_bias(y_train, len(model_config["classes"]), train_weights),
    ).to(device)
    dataset = TensorDataset(
        torch.from_numpy(X_train), torch.from_numpy(y_train.astype(np.int64)), torch.from_numpy(train_weights),
    )
    loader = DataLoader(
        dataset,
        batch_size=model_config["batch_size"],
        shuffle=True,
        generator=torch.Generator().manual_seed(seed),
        pin_memory=device.type == "cuda",
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=model_config["learning_rate"], weight_decay=model_config["weight_decay"],
    )
    best_loss = math.inf
    best_state = deepcopy(model.state_dict())
    best_epoch = 0
    patience = 0
    history = []
    for epoch in range(1, model_config["max_epochs"] + 1):
        model.train()
        train_loss_sum = 0.0
        train_count = 0
        for X_batch, y_batch, weight_batch in loader:
            X_batch = X_batch.to(device, non_blocking=True)
            y_batch = y_batch.to(device, non_blocking=True)
            weight_batch = weight_batch.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            per_row_loss = F.cross_entropy(model(X_batch), y_batch, reduction="none")
            loss = (per_row_loss * weight_batch).sum() / weight_batch.sum()
            loss.backward()
            optimizer.step()
            train_loss_sum += float((per_row_loss.detach() * weight_batch).sum())
            train_count += float(weight_batch.sum())
        inner_logits = predict_logits(model, X_inner, device)
        inner_loss = log_loss(y_inner, softmax_numpy(inner_logits), labels=list(range(len(model_config["classes"]))))
        history.append({"epoch": epoch, "train_cross_entropy": train_loss_sum / train_count, "inner_cross_entropy": inner_loss})
        if inner_loss < best_loss - model_config["min_delta"]:
            best_loss = inner_loss
            best_state = deepcopy(model.state_dict())
            best_epoch = epoch
            patience = 0
        else:
            patience += 1
            if patience >= model_config["early_stopping_patience"]:
                break
    model.load_state_dict(best_state)
    inner_logits = predict_logits(model, X_inner, device)
    temperature, raw_loss, calibrated_loss = choose_temperature(inner_logits, y_inner, model_config)
    return {
        "model": model,
        "center": center,
        "scale": scale,
        "temperature": temperature,
        "best_epoch": best_epoch,
        "inner_raw_logloss": raw_loss,
        "inner_calibrated_logloss": calibrated_loss,
        "train_rows_before_stride": int(fit_mask.sum()),
        "train_rows_after_stride": int(train_mask.sum()),
        "train_rows_after_sampling": int(train_mask.sum()),
        "inner_rows": int(inner_mask.sum()),
        "history": pd.DataFrame(history),
    }


def fit_fixed_epoch_model(
    X_raw: np.ndarray,
    y: np.ndarray,
    frame: pd.DataFrame,
    sessions: list[str],
    epochs: int,
    config: dict[str, Any],
    device: torch.device,
    seed: int,
) -> dict[str, Any]:
    model_config = config["model"]
    walk_config = config["walk_forward"]
    fit_mask = frame["session"].isin(sessions).to_numpy()
    sample_mask = training_sample_mask(frame, walk_config)
    train_mask = fit_mask & sample_mask
    center, scale = fit_robust_scaler(X_raw[fit_mask])
    X_train = transform_features(X_raw[train_mask], center, scale, walk_config["scaler_clip"])
    y_train = y[train_mask]
    train_weights = session_balancing_weights(
        frame, train_mask, bool(walk_config.get("equal_session_weights", False)),
    )[train_mask]
    seed_everything(seed)
    model = OutcomeMLP(
        X_raw.shape[1], model_config["hidden_dims"], model_config["dropout"],
        initial_class_bias(y_train, len(model_config["classes"]), train_weights),
    ).to(device)
    loader = DataLoader(
        TensorDataset(
            torch.from_numpy(X_train), torch.from_numpy(y_train.astype(np.int64)), torch.from_numpy(train_weights),
        ),
        batch_size=model_config["batch_size"], shuffle=True,
        generator=torch.Generator().manual_seed(seed), pin_memory=device.type == "cuda",
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=model_config["learning_rate"], weight_decay=model_config["weight_decay"],
    )
    history = []
    for epoch in range(1, epochs + 1):
        model.train()
        loss_sum = 0.0
        count = 0
        for X_batch, y_batch, weight_batch in loader:
            X_batch = X_batch.to(device, non_blocking=True)
            y_batch = y_batch.to(device, non_blocking=True)
            weight_batch = weight_batch.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            per_row_loss = F.cross_entropy(model(X_batch), y_batch, reduction="none")
            loss = (per_row_loss * weight_batch).sum() / weight_batch.sum()
            loss.backward()
            optimizer.step()
            loss_sum += float((per_row_loss.detach() * weight_batch).sum())
            count += float(weight_batch.sum())
        history.append({"epoch": epoch, "train_cross_entropy": loss_sum / count})
    return {
        "model": model,
        "center": center,
        "scale": scale,
        "train_rows_before_stride": int(fit_mask.sum()),
        "train_rows_after_stride": int(train_mask.sum()),
        "train_rows_after_sampling": int(train_mask.sum()),
        "history": pd.DataFrame(history),
    }


def price_tier(entry_price: pd.Series, boundary: float) -> pd.Series:
    return pd.Series(np.where(entry_price.lt(boundary), "below_1", "at_or_above_1"), index=entry_price.index)


def estimate_payoffs(
    frame: pd.DataFrame,
    history_sessions: list[str],
    classes: list[str],
    config: dict[str, Any],
    fold_name: str,
) -> tuple[dict[str, dict[str, float]], pd.DataFrame]:
    ev_config = config["expected_value"]
    history = frame[frame["session"].isin(history_sessions)].copy()
    global_means = history.groupby("dual_outcome_10m")["expected_net_return_dual_10m"].mean().to_dict()
    payoff: dict[str, dict[str, float]] = {}
    rows = []
    for tier in ["below_1", "at_or_above_1"]:
        payoff[tier] = {}
        tier_history = history[history["price_tier"].eq(tier)]
        for outcome in classes:
            outcome_history = tier_history[tier_history["dual_outcome_10m"].eq(outcome)]["expected_net_return_dual_10m"]
            global_mean = float(global_means.get(outcome, 0.0))
            if outcome == "NO_FILL":
                value = 0.0
            elif len(outcome_history) >= ev_config["minimum_tier_outcome_samples"]:
                shrinkage = ev_config["tier_payoff_shrinkage_samples"]
                value = float((outcome_history.sum() + shrinkage * global_mean) / (len(outcome_history) + shrinkage))
            else:
                value = global_mean
            payoff[tier][outcome] = value
            rows.append({
                "fold": fold_name, "price_tier": tier, "outcome": outcome,
                "history_samples": len(outcome_history), "global_mean_return": global_mean,
                "payoff_return": value, "history_sessions": ",".join(history_sessions),
            })
    return payoff, pd.DataFrame(rows)


def expected_value_from_probabilities(
    probabilities: np.ndarray,
    tiers: pd.Series,
    classes: list[str],
    payoff: dict[str, dict[str, float]],
) -> np.ndarray:
    values = np.empty(len(probabilities), dtype=np.float32)
    for tier in payoff:
        mask = tiers.eq(tier).to_numpy()
        payoff_vector = np.array([payoff[tier][outcome] for outcome in classes], dtype=np.float32)
        values[mask] = probabilities[mask] @ payoff_vector
    return values


def calibration_error(y: np.ndarray, probability: np.ndarray, bins: int = 10) -> float:
    edges = np.linspace(0.0, 1.0, bins + 1)
    bin_id = np.clip(np.digitize(probability, edges[1:-1]), 0, bins - 1)
    error = 0.0
    for current in range(bins):
        mask = bin_id == current
        if mask.any():
            error += mask.mean() * abs(y[mask].mean() - probability[mask].mean())
    return float(error)


def prediction_metrics(predictions: pd.DataFrame, classes: list[str]) -> pd.DataFrame:
    rows = []
    groups = [("oof", "ALL", predictions[predictions["evaluation_group"].eq("oof")]),
              ("test", "ALL", predictions[predictions["evaluation_group"].eq("test")])]
    groups.extend((group, session, part) for (group, session), part in predictions.groupby(["evaluation_group", "session"], sort=False))
    for group, session, part in groups:
        if part.empty:
            continue
        y = part["outcome_id"].to_numpy(dtype=np.int64)
        probability = part[[f"probability_{outcome.lower()}" for outcome in classes]].to_numpy(dtype=np.float64)
        probability /= probability.sum(axis=1, keepdims=True)
        actual_return = part["expected_net_return_dual_10m"].to_numpy(dtype=np.float64)
        predicted_return = part["predicted_expected_net_return"].to_numpy(dtype=np.float64)
        class_ap = {}
        class_roc = {}
        for class_id, outcome in enumerate(classes):
            binary = (y == class_id).astype(np.int8)
            class_ap[outcome] = average_precision_score(binary, probability[:, class_id])
            class_roc[outcome] = roc_auc_score(binary, probability[:, class_id])
        order = np.argsort(-predicted_return)
        top1 = order[:max(1, math.ceil(len(order) * 0.01))]
        top5 = order[:max(1, math.ceil(len(order) * 0.05))]
        rows.append({
            "evaluation_group": group, "session": session, "samples": len(part),
            "multiclass_logloss": log_loss(y, probability, labels=list(range(len(classes)))),
            "multiclass_brier": float(np.mean(np.sum((probability - np.eye(len(classes))[y]) ** 2, axis=1))),
            "accuracy": float((probability.argmax(axis=1) == y).mean()),
            "macro_pr_auc": float(np.mean(list(class_ap.values()))),
            "tp_pr_auc": class_ap["TP"], "tp_roc_auc": class_roc["TP"],
            "tp_positive_rate": float((y == classes.index("TP")).mean()),
            "tp_ece": calibration_error((y == classes.index("TP")).astype(np.int8), probability[:, classes.index("TP")]),
            "return_spearman": pd.Series(actual_return).corr(pd.Series(predicted_return), method="spearman"),
            "return_mae": float(np.mean(np.abs(actual_return - predicted_return))),
            "top1_mean_actual_return": float(actual_return[top1].mean()),
            "top5_mean_actual_return": float(actual_return[top5].mean()),
        })
    return pd.DataFrame(rows)


@dataclass(frozen=True)
class SequentialConfig:
    initial_capital: float
    max_concurrent_positions: int
    order_notional_usd: float
    order_ttl_minutes: int
    max_holding_minutes: int


def build_equity_curve(ledger: pd.DataFrame, initial_capital: float) -> pd.DataFrame:
    if ledger.empty:
        return pd.DataFrame(columns=["release_timestamp", "net_pnl", "equity", "portfolio_return", "peak", "drawdown"])
    released = ledger.groupby("release_timestamp", as_index=False)["net_pnl"].sum().sort_values("release_timestamp")
    released["equity"] = initial_capital + released["net_pnl"].cumsum()
    released["portfolio_return"] = released["equity"] / initial_capital - 1.0
    released["peak"] = released["equity"].cummax().clip(lower=initial_capital)
    released["drawdown"] = released["equity"] / released["peak"] - 1.0
    return released


def summarize_backtest(
    ledger: pd.DataFrame,
    equity: pd.DataFrame,
    signal_count: int,
    cluster_count: int,
    skipped: dict[str, int],
    config: SequentialConfig,
) -> dict[str, Any]:
    filled = ledger[ledger["filled"]] if not ledger.empty else ledger
    positive = filled.loc[filled["net_pnl"] > 0, "net_pnl"].sum() if not filled.empty else 0.0
    negative = -filled.loc[filled["net_pnl"] < 0, "net_pnl"].sum() if not filled.empty else 0.0
    total_pnl = float(filled["net_pnl"].sum()) if not filled.empty else 0.0
    deployed = float(filled["capital_required"].sum()) if not filled.empty else 0.0
    return {
        "signals_above_threshold": int(signal_count), "ten_minute_signal_clusters": int(cluster_count),
        "order_attempts": len(ledger), "filled_trades": len(filled),
        "tp_trades": int(filled["outcome"].eq("TP").sum()) if not filled.empty else 0,
        "sl_trades": int(filled["outcome"].eq("SL").sum()) if not filled.empty else 0,
        "timeout_trades": int(filled["outcome"].eq("TIMEOUT").sum()) if not filled.empty else 0,
        "tp_precision_given_fill": float(filled["outcome"].eq("TP").mean()) if not filled.empty else np.nan,
        "mean_net_return_per_fill": float(filled["net_return"].mean()) if not filled.empty else np.nan,
        "median_net_return_per_fill": float(filled["net_return"].median()) if not filled.empty else np.nan,
        "net_return_on_deployed_capital": total_pnl / deployed if deployed else np.nan,
        "return_on_initial_capital": total_pnl / config.initial_capital,
        "total_net_pnl": total_pnl,
        "profit_factor": float(positive / negative) if negative > 0 else (np.inf if positive > 0 else np.nan),
        "max_drawdown": float(equity["drawdown"].min()) if not equity.empty else 0.0,
        "skipped_same_symbol": skipped["same_symbol"],
        "skipped_position_limit": skipped["position_limit"],
        "skipped_cash_limit": skipped["cash_limit"],
    }


def run_sequential_backtest(frame: pd.DataFrame, threshold: float, config: SequentialConfig) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
    eligible = frame.loc[
        frame["dual_agreement_10m"] & frame["predicted_expected_net_return"].ge(threshold)
    ].sort_values(["input_end_timestamp", "predicted_expected_net_return", "symbol"], ascending=[True, False, True])
    if eligible.empty:
        cluster_count = 0
    else:
        clustered = eligible.sort_values(["session", "symbol", "input_end_timestamp"])
        delta = clustered.groupby(["session", "symbol"])["input_end_timestamp"].diff().dt.total_seconds().div(60)
        cluster_count = int((delta.isna() | delta.gt(10)).sum())
    cash = config.initial_capital
    active: list[dict[str, Any]] = []
    rows = []
    skipped = {"same_symbol": 0, "position_limit": 0, "cash_limit": 0}
    for timestamp, candidates in eligible.groupby("input_end_timestamp", sort=True):
        remaining = []
        for position in active:
            if position["release_timestamp"] <= timestamp:
                cash += position["capital_required"] + position["net_pnl"]
            else:
                remaining.append(position)
        active = remaining
        for row in candidates.itertuples(index=False):
            if row.symbol in {position["symbol"] for position in active}:
                skipped["same_symbol"] += 1
                continue
            if len(active) >= config.max_concurrent_positions:
                skipped["position_limit"] += 1
                continue
            if cash + 1e-9 < row.capital_required:
                skipped["cash_limit"] += 1
                continue
            cash -= row.capital_required
            position = {
                "evaluation_group": row.evaluation_group, "session": row.session, "symbol": row.symbol,
                "signal_timestamp": timestamp, "release_timestamp": row.release_timestamp,
                "predicted_expected_net_return": float(row.predicted_expected_net_return), "threshold": float(threshold),
                "filled": bool(row.filled), "outcome": row.outcome_hf_10m,
                "holding_minutes": int(row.holding_minutes), "entry_price": float(row.entry_price),
                "entry_notional": float(row.entry_notional), "capital_required": float(row.capital_required),
                "net_pnl": float(row.net_pnl), "net_return": float(row.net_return), "price_tier": row.price_tier,
                "concurrent_after_order": len(active) + 1, "cash_after_reservation": cash,
            }
            active.append(position)
            rows.append(position.copy())
    for position in sorted(active, key=lambda item: item["release_timestamp"]):
        cash += position["capital_required"] + position["net_pnl"]
    ledger = pd.DataFrame(rows)
    equity = build_equity_curve(ledger, config.initial_capital)
    metrics = summarize_backtest(ledger, equity, len(eligible), cluster_count, skipped, config)
    metrics.update({"threshold": float(threshold), "ending_cash": float(cash)})
    if not ledger.empty:
        assert ledger["concurrent_after_order"].le(config.max_concurrent_positions).all()
        assert ledger["holding_minutes"].between(1, config.max_holding_minutes).all()
        assert ledger["cash_after_reservation"].ge(-1e-6).all()
    assert math.isclose(cash, config.initial_capital + metrics["total_net_pnl"], rel_tol=0.0, abs_tol=1e-6)
    return metrics, ledger, equity


def prepare_execution_frame(predictions: pd.DataFrame, labels: pd.DataFrame) -> pd.DataFrame:
    keys = ["source_path", "symbol", "input_end_timestamp"]
    execution_columns = [
        *keys, "reference_close", "entry_price", "shares", "dual_agreement_10m",
        "outcome_hf_10m", "outcome_lf_10m", "event_bar_hf_10m", "event_bar_lf_10m",
        "net_pnl_hf_10m", "net_pnl_lf_10m", "net_return_hf_10m", "net_return_lf_10m",
    ]
    result = predictions.merge(labels[execution_columns], on=keys, how="inner", validate="one_to_one")
    assert len(result) == len(predictions)
    result["entry_notional"] = result["entry_price"] * result["shares"]
    buy_commission = np.floor(result["entry_notional"] * 0.001 * 100 + 0.5) / 100
    result["capital_required"] = result["entry_notional"] + buy_commission
    result["net_pnl"] = (result["net_pnl_hf_10m"] + result["net_pnl_lf_10m"]) / 2
    result["net_return"] = (result["net_return_hf_10m"] + result["net_return_lf_10m"]) / 2
    result["filled"] = result["outcome_hf_10m"].ne("NO_FILL")
    event_bar = result[["event_bar_hf_10m", "event_bar_lf_10m"]].max(axis=1)
    result["holding_minutes"] = np.where(result["filled"], event_bar, 1).astype(int)
    result["release_timestamp"] = result["input_end_timestamp"] + pd.to_timedelta(result["holding_minutes"], unit="m")
    assert result["dual_agreement_10m"].all()
    return result


def threshold_search(
    execution: pd.DataFrame,
    config: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    backtest_config = config["backtest"]
    sequential = SequentialConfig(
        initial_capital=backtest_config["initial_capital_usd"],
        max_concurrent_positions=backtest_config["max_concurrent_positions"],
        order_notional_usd=backtest_config["order_notional_usd"],
        order_ttl_minutes=backtest_config["order_ttl_minutes"],
        max_holding_minutes=backtest_config["max_holding_minutes"],
    )
    oof = execution[execution["evaluation_group"].eq("oof")]
    quantiles = np.linspace(
        backtest_config["threshold_quantile_min"], backtest_config["threshold_quantile_max"],
        backtest_config["threshold_candidates"],
    )
    floor = backtest_config["minimum_predicted_expected_return"]
    nonnegative_scores = oof.loc[oof["predicted_expected_net_return"].ge(floor), "predicted_expected_net_return"]
    if nonnegative_scores.empty:
        thresholds = np.array([floor], dtype=float)
    else:
        thresholds = np.unique(np.append(np.quantile(nonnegative_scores, quantiles), floor))
    search_rows = []
    oof_sessions = sorted(oof["session"].unique())
    for threshold in thresholds:
        metrics, _, _ = run_sequential_backtest(oof, float(threshold), sequential)
        session_rows = []
        for session in oof_sessions:
            session_metrics, _, _ = run_sequential_backtest(oof[oof["session"].eq(session)], float(threshold), sequential)
            session_metrics["session"] = session
            session_rows.append(session_metrics)
        session_frame = pd.DataFrame(session_rows)
        profitable = (
            session_frame["filled_trades"].ge(backtest_config["minimum_filled_trades_per_session"])
            & session_frame["mean_net_return_per_fill"].gt(0)
        )
        metrics.update({
            "oof_sessions": len(session_frame),
            "min_session_filled_trades": int(session_frame["filled_trades"].min()),
            "profitable_sessions": int(profitable.sum()),
            "profitable_session_share": float(profitable.mean()),
            "worst_session_mean_net_return": float(session_frame["mean_net_return_per_fill"].min()),
        })
        metrics["meets_constraints"] = (
            metrics["signals_above_threshold"] >= backtest_config["minimum_signals"]
            and metrics["filled_trades"] >= backtest_config["minimum_filled_trades"]
            and metrics["oof_sessions"] >= backtest_config["minimum_oof_sessions"]
            and metrics["min_session_filled_trades"] >= backtest_config["minimum_filled_trades_per_session"]
            and metrics["profitable_session_share"] >= backtest_config["minimum_profitable_oof_session_share"]
            and metrics["worst_session_mean_net_return"] > 0
            and metrics["mean_net_return_per_fill"] > 0
            and metrics["return_on_initial_capital"] > 0
            and metrics["profit_factor"] > 1
        )
        search_rows.append(metrics)
    search = pd.DataFrame(search_rows)
    valid = search[search["meets_constraints"]]
    if not valid.empty:
        selected = valid.sort_values(
            ["worst_session_mean_net_return", "mean_net_return_per_fill", "return_on_initial_capital"],
            ascending=False,
        ).iloc[0]
        status = "VALID"
    else:
        fallback = search[search["filled_trades"].ge(backtest_config["minimum_filled_trades"])]
        if fallback.empty:
            fallback = search.sort_values("filled_trades", ascending=False).head(1)
        selected = fallback.sort_values(
            ["profitable_session_share", "worst_session_mean_net_return", "mean_net_return_per_fill"],
            ascending=False,
        ).iloc[0]
        status = "NO_VALID_THRESHOLD"
    selected_frame = pd.DataFrame([{
        "selected_threshold": float(selected["threshold"]), "selection_status": status,
        "oof_mean_net_return": float(selected["mean_net_return_per_fill"]),
        "oof_portfolio_return": float(selected["return_on_initial_capital"]),
        "oof_profit_factor": float(selected["profit_factor"]),
        "oof_filled_trades": int(selected["filled_trades"]),
        "oof_profitable_session_share": float(selected["profitable_session_share"]),
        "oof_worst_session_mean_net_return": float(selected["worst_session_mean_net_return"]),
    }])
    threshold = float(selected["threshold"])
    aggregate_rows = []
    ledgers = []
    equities = []
    session_rows = []
    for group in ["oof", "test"]:
        group_frame = execution[execution["evaluation_group"].eq(group)]
        metrics, ledger, equity = run_sequential_backtest(group_frame, threshold, sequential)
        metrics.update({"evaluation_group": group, "selection_status": status})
        aggregate_rows.append(metrics)
        if not ledger.empty:
            ledgers.append(ledger)
        if not equity.empty:
            equity["evaluation_group"] = group
            equities.append(equity)
        for session in sorted(group_frame["session"].unique()):
            session_metrics, _, _ = run_sequential_backtest(group_frame[group_frame["session"].eq(session)], threshold, sequential)
            session_metrics.update({"evaluation_group": group, "session": session})
            session_rows.append(session_metrics)
    aggregate = pd.DataFrame(aggregate_rows)
    session_metrics = pd.DataFrame(session_rows)
    test_sessions = session_metrics[session_metrics["evaluation_group"].eq("test")]
    test_profitable = (
        test_sessions["filled_trades"].ge(backtest_config["minimum_filled_trades_per_session"])
        & test_sessions["mean_net_return_per_fill"].gt(0)
    )
    test_row = aggregate[aggregate["evaluation_group"].eq("test")].iloc[0]
    test_pass = (
        len(test_sessions) >= backtest_config["minimum_test_sessions"]
        and float(test_profitable.mean()) >= backtest_config["minimum_profitable_test_session_share"]
        and test_row["mean_net_return_per_fill"] > 0
        and test_row["return_on_initial_capital"] > 0
        and test_row["profit_factor"] > 1
    )
    deployment = pd.DataFrame([{
        "selection_status": status, "oof_eligible": status == "VALID", "test_return_pass": bool(test_pass),
        "test_sessions": len(test_sessions), "test_profitable_sessions": int(test_profitable.sum()),
        "test_profitable_session_share": float(test_profitable.mean()),
        "test_worst_session_mean_net_return": float(test_sessions["mean_net_return_per_fill"].min()),
        "deployment_status": "PASS" if status == "VALID" and test_pass else "FAIL",
    }])
    ledger = pd.concat(ledgers, ignore_index=True) if ledgers else pd.DataFrame()
    equity = pd.concat(equities, ignore_index=True) if equities else pd.DataFrame()
    return search, selected_frame, aggregate, session_metrics, deployment, ledger, equity


def run_experiment(project_root: Path | None = None, config_path: Path | None = None) -> dict[str, Any]:
    project_root = find_project_root(project_root)
    config = load_config(project_root, config_path)
    if "envs/urban" not in str(Path(sys.executable).resolve()):
        raise AssertionError(f"urban 환경이 아닙니다: {sys.executable}")
    seed = int(config["project"]["seed"])
    seed_everything(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_root = (project_root / config["data"]["root"]).resolve()
    tabular_path = data_root / config["data"]["tabular_artifact"]
    schema_path = data_root / config["data"]["feature_schema"]
    labels_path = data_root / config["data"]["labels_artifact"]
    with schema_path.open(encoding="utf-8") as file:
        feature_names = json.load(file)["features"]
    frame = pd.read_parquet(tabular_path)
    labels = pd.read_parquet(labels_path)
    classes = list(config["model"]["classes"])
    frame = frame[frame["dual_outcome_10m"].isin(classes)].copy().reset_index(drop=True)
    frame["input_end_timestamp"] = pd.to_datetime(frame["input_end_timestamp"], utc=True)
    frame["outcome_id"] = frame["dual_outcome_10m"].map({outcome: index for index, outcome in enumerate(classes)}).astype(np.int64)
    frame["price_tier"] = price_tier(frame["entry_price"], config["expected_value"]["price_tier_boundary"])
    development_sessions = list(config["data"]["development_sessions"])
    test_sessions = list(config["data"]["test_sessions"])
    assert not set(development_sessions) & set(test_sessions)
    assert set(frame["session"]) == set(development_sessions + test_sessions)
    assert frame.loc[frame["session"].isin(development_sessions), "input_end_timestamp"].max() < frame.loc[frame["session"].isin(test_sessions), "input_end_timestamp"].min()
    X_raw = frame[feature_names].to_numpy(dtype=np.float32)
    y = frame["outcome_id"].to_numpy(dtype=np.int64)
    folds = make_walk_forward_folds(development_sessions, config["walk_forward"]["minimum_prior_sessions"])

    processed_root = data_root / "processed"
    model_root = data_root / "models"
    backtest_root = data_root / "backtests"
    for path in [processed_root, model_root, backtest_root]:
        path.mkdir(parents=True, exist_ok=True)

    prediction_frames = []
    fold_rows = []
    history_frames = []
    payoff_frames = []
    fold_checkpoint_paths = []
    for fold in folds:
        fold_seed = seed + fold["fold"] * 100
        fitted = fit_early_stopped_model(
            X_raw, y, frame, fold["fit_sessions"], fold["inner_validation_session"],
            config, device, fold_seed,
        )
        evaluation_mask = frame["session"].eq(fold["evaluation_session"]).to_numpy()
        X_evaluation = transform_features(
            X_raw[evaluation_mask], fitted["center"], fitted["scale"], config["walk_forward"]["scaler_clip"],
        )
        logits = predict_logits(fitted["model"], X_evaluation, device)
        probability = softmax_numpy(logits, fitted["temperature"])
        payoff, payoff_frame = estimate_payoffs(
            frame, fold["payoff_history_sessions"], classes, config, f"fold_{fold['fold']}",
        )
        payoff_frames.append(payoff_frame)
        evaluation = frame.loc[evaluation_mask, [
            "source_path", "session", "symbol", "input_end_timestamp", "dual_outcome_10m",
            "outcome_id", "expected_net_return_dual_10m", "price_tier",
        ]].copy()
        for class_id, outcome in enumerate(classes):
            evaluation[f"probability_{outcome.lower()}"] = probability[:, class_id]
        evaluation["predicted_expected_net_return"] = expected_value_from_probabilities(
            probability, evaluation["price_tier"], classes, payoff,
        )
        evaluation["evaluation_group"] = "oof"
        evaluation["fold"] = fold["fold"]
        evaluation["fit_sessions"] = ",".join(fold["fit_sessions"])
        evaluation["inner_validation_session"] = fold["inner_validation_session"]
        prediction_frames.append(evaluation)
        fold_rows.append({
            **fold, "best_epoch": fitted["best_epoch"], "temperature": fitted["temperature"],
            "inner_raw_logloss": fitted["inner_raw_logloss"],
            "inner_calibrated_logloss": fitted["inner_calibrated_logloss"],
            "train_rows_before_stride": fitted["train_rows_before_stride"],
            "train_rows_after_stride": fitted["train_rows_after_stride"], "inner_rows": fitted["inner_rows"],
        })
        history = fitted["history"].copy()
        history["fold"] = fold["fold"]
        history_frames.append(history)
        checkpoint_path = model_root / f"torch_multiclass_oof_fold_{fold['fold']}_{fold['evaluation_session'].removeprefix('session_')}.pt"
        torch.save({
            "model_state_dict": fitted["model"].cpu().state_dict(), "feature_names": feature_names,
            "classes": classes, "center": torch.from_numpy(fitted["center"]), "scale": torch.from_numpy(fitted["scale"]),
            "temperature": fitted["temperature"], "fold": fold, "best_epoch": fitted["best_epoch"], "seed": fold_seed,
        }, checkpoint_path)
        fold_checkpoint_paths.append(str(checkpoint_path))
        if device.type == "cuda":
            torch.cuda.empty_cache()

    fold_metrics = pd.DataFrame(fold_rows)
    final_epochs = max(1, int(np.median(fold_metrics["best_epoch"])))
    final_temperature = float(np.median(fold_metrics["temperature"]))
    final_fit = fit_fixed_epoch_model(
        X_raw, y, frame, development_sessions, final_epochs, config, device, seed + 900,
    )
    test_mask = frame["session"].isin(test_sessions).to_numpy()
    X_test = transform_features(
        X_raw[test_mask], final_fit["center"], final_fit["scale"], config["walk_forward"]["scaler_clip"],
    )
    test_logits = predict_logits(final_fit["model"], X_test, device)
    test_probability = softmax_numpy(test_logits, final_temperature)
    final_payoff, final_payoff_frame = estimate_payoffs(frame, development_sessions, classes, config, "final")
    payoff_frames.append(final_payoff_frame)
    test_prediction = frame.loc[test_mask, [
        "source_path", "session", "symbol", "input_end_timestamp", "dual_outcome_10m",
        "outcome_id", "expected_net_return_dual_10m", "price_tier",
    ]].copy()
    for class_id, outcome in enumerate(classes):
        test_prediction[f"probability_{outcome.lower()}"] = test_probability[:, class_id]
    test_prediction["predicted_expected_net_return"] = expected_value_from_probabilities(
        test_probability, test_prediction["price_tier"], classes, final_payoff,
    )
    test_prediction["evaluation_group"] = "test"
    test_prediction["fold"] = 0
    test_prediction["fit_sessions"] = ",".join(development_sessions)
    test_prediction["inner_validation_session"] = None
    prediction_frames.append(test_prediction)
    final_checkpoint_path = model_root / "torch_multiclass_walk_forward_final_10m_tp5_sl3_price_v2.pt"
    torch.save({
        "model_state_dict": final_fit["model"].cpu().state_dict(), "feature_names": feature_names,
        "classes": classes, "center": torch.from_numpy(final_fit["center"]), "scale": torch.from_numpy(final_fit["scale"]),
        "temperature": final_temperature, "epochs": final_epochs, "development_sessions": development_sessions,
        "payoff": final_payoff, "config": config["model"], "seed": seed + 900,
    }, final_checkpoint_path)
    if device.type == "cuda":
        torch.cuda.empty_cache()

    predictions = pd.concat(prediction_frames, ignore_index=True)
    predictions = predictions.sort_values(["input_end_timestamp", "symbol"]).reset_index(drop=True)
    metrics = prediction_metrics(predictions, classes)
    payoff_table = pd.concat(payoff_frames, ignore_index=True)
    execution = prepare_execution_frame(predictions, labels)
    search, selected, backtest_metrics, session_metrics, deployment, ledger, equity = threshold_search(execution, config)

    version = config["artifacts"]["version"]
    paths = {
        "predictions": processed_root / f"{version}_predictions.parquet",
        "metrics": processed_root / f"{version}_metrics.parquet",
        "folds": processed_root / f"{version}_folds.parquet",
        "history": processed_root / f"{version}_history.parquet",
        "payoffs": processed_root / f"{version}_payoffs.parquet",
        "threshold_search": backtest_root / f"{version}_threshold_search.parquet",
        "selected_threshold": backtest_root / f"{version}_selected_threshold.parquet",
        "backtest_metrics": backtest_root / f"{version}_backtest_metrics.parquet",
        "session_metrics": backtest_root / f"{version}_session_metrics.parquet",
        "deployment": backtest_root / f"{version}_deployment.parquet",
        "ledger": backtest_root / f"{version}_ledger.parquet",
        "equity": backtest_root / f"{version}_equity.parquet",
        "manifest": model_root / f"{version}_manifest.json",
        "final_checkpoint": final_checkpoint_path,
    }
    predictions.to_parquet(paths["predictions"], index=False, compression="zstd")
    metrics.to_parquet(paths["metrics"], index=False, compression="zstd")
    fold_metrics.to_parquet(paths["folds"], index=False, compression="zstd")
    pd.concat(history_frames, ignore_index=True).to_parquet(paths["history"], index=False, compression="zstd")
    payoff_table.to_parquet(paths["payoffs"], index=False, compression="zstd")
    search.to_parquet(paths["threshold_search"], index=False, compression="zstd")
    selected.to_parquet(paths["selected_threshold"], index=False, compression="zstd")
    backtest_metrics.to_parquet(paths["backtest_metrics"], index=False, compression="zstd")
    session_metrics.to_parquet(paths["session_metrics"], index=False, compression="zstd")
    deployment.to_parquet(paths["deployment"], index=False, compression="zstd")
    ledger.to_parquet(paths["ledger"], index=False, compression="zstd")
    equity.to_parquet(paths["equity"], index=False, compression="zstd")
    manifest = {
        "version": version, "environment": "urban", "python": sys.version.split()[0],
        "torch": torch.__version__, "device": str(device), "seed": seed,
        "config_path": config["_config_path"], "tabular_path": str(tabular_path), "labels_path": str(labels_path),
        "feature_count": len(feature_names), "classes": classes, "folds": fold_rows,
        "final_epochs": final_epochs, "final_temperature": final_temperature,
        "fold_checkpoints": fold_checkpoint_paths, "final_checkpoint": str(final_checkpoint_path),
        "selected_threshold": selected.iloc[0].to_dict(), "deployment": deployment.iloc[0].to_dict(),
        "artifacts": {name: str(path) for name, path in paths.items() if name != "manifest"},
    }
    paths["manifest"].write_text(json.dumps(manifest, ensure_ascii=False, indent=2, default=str), encoding="utf-8")

    assert predictions.duplicated(["source_path", "symbol", "input_end_timestamp"]).sum() == 0
    assert set(predictions.loc[predictions["evaluation_group"].eq("test"), "session"]) == set(test_sessions)
    assert not set(predictions.loc[predictions["evaluation_group"].eq("oof"), "session"]) & set(test_sessions)
    probability_columns = [f"probability_{outcome.lower()}" for outcome in classes]
    assert np.allclose(predictions[probability_columns].sum(axis=1), 1.0, atol=1e-5)
    assert predictions[probability_columns].apply(lambda column: column.between(0, 1).all()).all()
    assert all(path.exists() for path in paths.values())

    return {
        "config": config, "device": str(device), "folds": fold_metrics, "predictions": predictions,
        "metrics": metrics, "payoffs": payoff_table, "threshold_search": search,
        "selected_threshold": selected, "backtest_metrics": backtest_metrics,
        "session_metrics": session_metrics, "deployment": deployment, "ledger": ledger,
        "equity": equity, "paths": paths,
    }


__all__ = [
    "find_project_root", "load_config", "make_walk_forward_folds", "run_experiment",
    "softmax_numpy", "expected_value_from_probabilities", "run_sequential_backtest",
    "training_sample_mask", "session_balancing_weights",
]
