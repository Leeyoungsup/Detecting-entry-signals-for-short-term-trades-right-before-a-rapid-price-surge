from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from src.walk_forward_oof import (
    SequentialConfig,
    estimate_payoffs,
    expected_value_from_probabilities,
    find_project_root,
    fit_fixed_epoch_model,
    load_config,
    predict_logits,
    prediction_metrics,
    prepare_execution_frame,
    price_tier,
    run_sequential_backtest,
    seed_everything,
    session_balancing_weights,
    softmax_numpy,
    training_sample_mask,
    transform_features,
)


IDENTITY_COLUMNS = [
    "source_path",
    "session",
    "symbol",
    "input_end_timestamp",
    "dual_outcome_10m",
    "outcome_id",
    "expected_net_return_dual_10m",
    "price_tier",
]


def chronological_session_split(sessions: list[str], test_session_count: int = 2) -> tuple[list[str], list[str]]:
    ordered = sorted(sessions)
    if test_session_count <= 0 or len(ordered) <= test_session_count:
        raise ValueError("train과 test에 각각 하나 이상의 session이 필요합니다.")
    return ordered[:-test_session_count], ordered[-test_session_count:]


def score_quantile_threshold(scores: pd.Series, quantile: float, floor: float = 0.0) -> float:
    if not 0.0 <= quantile <= 1.0:
        raise ValueError("quantile은 0과 1 사이여야 합니다.")
    if scores.empty:
        return float(floor)
    return float(max(floor, scores.quantile(quantile)))


def _prediction_frame(
    frame: pd.DataFrame,
    mask: np.ndarray,
    probability: np.ndarray,
    predicted_ev: np.ndarray,
    classes: list[str],
    evaluation_group: str,
) -> pd.DataFrame:
    result = frame.loc[mask, IDENTITY_COLUMNS].copy()
    for class_id, outcome in enumerate(classes):
        result[f"probability_{outcome.lower()}"] = probability[:, class_id]
    result["predicted_expected_net_return"] = predicted_ev
    result["evaluation_group"] = evaluation_group
    return result


def _train_test_metrics(predictions: pd.DataFrame, classes: list[str]) -> pd.DataFrame:
    compatible = predictions.copy()
    compatible["evaluation_group"] = compatible["evaluation_group"].replace({"train": "oof"})
    metrics = prediction_metrics(compatible, classes)
    metrics["evaluation_group"] = metrics["evaluation_group"].replace({"oof": "train"})
    return metrics


def prepare_immediate_execution_frame(predictions: pd.DataFrame, labels: pd.DataFrame) -> pd.DataFrame:
    keys = ["source_path", "symbol", "input_end_timestamp"]
    execution_columns = [
        *keys,
        "reference_close",
        "entry_price",
        "shares",
        "dual_agreement_10m",
        "outcome_hf_10m",
        "outcome_lf_10m",
        "event_bar_hf_10m",
        "event_bar_lf_10m",
        "net_pnl_hf_10m",
        "net_pnl_lf_10m",
        "net_return_hf_10m",
        "net_return_lf_10m",
    ]
    result = predictions.merge(labels[execution_columns], on=keys, how="inner", validate="one_to_one")
    if len(result) != len(predictions):
        raise AssertionError("immediate prediction과 label 결합 건수가 다릅니다.")
    result["entry_notional"] = result["entry_price"] * result["shares"]
    buy_commission = np.floor(result["entry_notional"] * 0.001 * 100 + 0.5) / 100
    result["capital_required"] = result["entry_notional"] + buy_commission
    result["net_pnl"] = (result["net_pnl_hf_10m"] + result["net_pnl_lf_10m"]) / 2
    result["net_return"] = (result["net_return_hf_10m"] + result["net_return_lf_10m"]) / 2
    result["filled"] = True
    result["holding_minutes"] = result[["event_bar_hf_10m", "event_bar_lf_10m"]].max(axis=1).astype(int)
    result["release_timestamp"] = result["input_end_timestamp"] + pd.to_timedelta(
        result["holding_minutes"], unit="m",
    )
    if not result["dual_agreement_10m"].all():
        raise AssertionError("실행 데이터에는 dual-path 확정 라벨만 있어야 합니다.")
    if result[["outcome_hf_10m", "outcome_lf_10m"]].eq("NO_FILL").any().any():
        raise AssertionError("즉시체결 실행 데이터에 NO_FILL이 있습니다.")
    return result


def _backtest_groups(
    execution: pd.DataFrame,
    threshold: float,
    config: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    backtest = config["backtest"]
    sequential = SequentialConfig(
        initial_capital=backtest["initial_capital_usd"],
        max_concurrent_positions=backtest["max_concurrent_positions"],
        order_notional_usd=backtest["order_notional_usd"],
        order_ttl_minutes=backtest["order_ttl_minutes"],
        max_holding_minutes=backtest["max_holding_minutes"],
    )
    aggregate_rows = []
    session_rows = []
    ledgers = []
    equities = []
    for group in ["train", "test"]:
        group_frame = execution[execution["evaluation_group"].eq(group)]
        metrics, ledger, equity = run_sequential_backtest(group_frame, threshold, sequential)
        metrics["evaluation_group"] = group
        aggregate_rows.append(metrics)
        if not ledger.empty:
            ledgers.append(ledger)
        if not equity.empty:
            equity["evaluation_group"] = group
            equities.append(equity)
        for session in sorted(group_frame["session"].unique()):
            metrics, _, _ = run_sequential_backtest(
                group_frame[group_frame["session"].eq(session)], threshold, sequential,
            )
            metrics.update({"evaluation_group": group, "session": session})
            session_rows.append(metrics)
    return (
        pd.DataFrame(aggregate_rows),
        pd.DataFrame(session_rows),
        pd.concat(ledgers, ignore_index=True) if ledgers else pd.DataFrame(),
        pd.concat(equities, ignore_index=True) if equities else pd.DataFrame(),
    )


def run_fixed_train_test_experiment(
    project_root: Path | None = None,
    config_path: Path | None = None,
) -> dict[str, Any]:
    project_root = find_project_root(project_root)
    config_path = config_path or project_root / "configs/fixed_train_test_80_20.yaml"
    config = load_config(project_root, config_path)
    if "envs/urban" not in str(Path(sys.executable).resolve()):
        raise AssertionError(f"urban 환경이 아닙니다: {sys.executable}")
    seed = int(config["project"]["seed"])
    seed_everything(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data_root = (project_root / config["data"]["root"]).resolve()
    tabular_path = data_root / config["data"]["tabular_artifact"]
    feature_schema_path = data_root / config["data"]["selected_feature_schema"]
    labels_path = data_root / config["data"]["labels_artifact"]
    feature_schema = json.loads(feature_schema_path.read_text(encoding="utf-8"))
    feature_names = list(feature_schema["selected_features"])
    frame = pd.read_parquet(tabular_path)
    labels = pd.read_parquet(labels_path)
    classes = list(config["model"]["classes"])
    frame = frame[frame["dual_outcome_10m"].isin(classes)].copy().reset_index(drop=True)
    frame["input_end_timestamp"] = pd.to_datetime(frame["input_end_timestamp"], utc=True)
    frame["outcome_id"] = frame["dual_outcome_10m"].map(
        {outcome: index for index, outcome in enumerate(classes)},
    ).astype(np.int64)
    frame["price_tier"] = price_tier(frame["entry_price"], config["expected_value"]["price_tier_boundary"])
    train_sessions = list(config["data"]["train_sessions"])
    test_sessions = list(config["data"]["test_sessions"])
    discovered_train, discovered_test = chronological_session_split(frame["session"].unique().tolist(), 2)
    if train_sessions != discovered_train or test_sessions != discovered_test:
        raise AssertionError("config가 시간순 7:2 session split과 일치하지 않습니다.")
    if frame.loc[frame["session"].isin(train_sessions), "input_end_timestamp"].max() >= frame.loc[
        frame["session"].isin(test_sessions), "input_end_timestamp"
    ].min():
        raise AssertionError("test는 모든 train session 뒤에 있어야 합니다.")

    X_raw = frame[feature_names].to_numpy(dtype=np.float32)
    y = frame["outcome_id"].to_numpy(dtype=np.int64)
    epochs = int(config["model"]["fixed_epochs"])
    fitted = fit_fixed_epoch_model(
        X_raw, y, frame, train_sessions, epochs, config, device, seed + 100,
    )
    payoff, payoff_frame = estimate_payoffs(frame, train_sessions, classes, config, "fixed_train")
    predictions = []
    temperature = float(config["model"]["temperature"])
    for group, sessions in [("train", train_sessions), ("test", test_sessions)]:
        mask = frame["session"].isin(sessions).to_numpy()
        X = transform_features(
            X_raw[mask], fitted["center"], fitted["scale"], config["walk_forward"]["scaler_clip"],
        )
        probability = softmax_numpy(predict_logits(fitted["model"], X, device), temperature)
        tiers = frame.loc[mask, "price_tier"]
        predictions.append(_prediction_frame(
            frame,
            mask,
            probability,
            expected_value_from_probabilities(probability, tiers, classes, payoff),
            classes,
            group,
        ))
    predictions = pd.concat(predictions, ignore_index=True).sort_values(
        ["input_end_timestamp", "symbol"],
    ).reset_index(drop=True)
    metrics = _train_test_metrics(predictions, classes)
    overall = metrics[metrics["session"].eq("ALL")].set_index("evaluation_group")
    gap_columns = [
        "multiclass_logloss", "macro_pr_auc", "tp_pr_auc", "tp_roc_auc",
        "return_spearman", "return_mae", "top1_mean_actual_return", "top5_mean_actual_return",
    ]
    generalization_gap = pd.DataFrame({
        "metric": gap_columns,
        "train": [overall.loc["train", column] for column in gap_columns],
        "test": [overall.loc["test", column] for column in gap_columns],
    })
    generalization_gap["test_minus_train"] = generalization_gap["test"] - generalization_gap["train"]

    decision = config["decision"]
    train_scores = predictions.loc[predictions["evaluation_group"].eq("train"), "predicted_expected_net_return"]
    threshold = score_quantile_threshold(
        train_scores,
        float(decision["train_score_quantile"]),
        float(decision["minimum_predicted_expected_return"]),
    )
    threshold_frame = pd.DataFrame([{
        "threshold": threshold,
        "method": decision["method"],
        "train_score_quantile": decision["train_score_quantile"],
        "uses_outcome_for_threshold_selection": False,
        "temperature": temperature,
        "validation_used": False,
    }])
    entry_mode = config.get("execution", {}).get("entry_mode", "limit_close_minus_tick")
    if entry_mode == "immediate_market_proxy_at_close":
        execution = prepare_immediate_execution_frame(predictions, labels)
    else:
        execution = prepare_execution_frame(predictions, labels)
    backtest_metrics, session_metrics, ledger, equity = _backtest_groups(execution, threshold, config)

    full_sample_mask = training_sample_mask(frame, config["walk_forward"])
    train_mask = frame["session"].isin(train_sessions).to_numpy()
    selected_mask = train_mask & full_sample_mask
    weights = session_balancing_weights(
        frame, selected_mask, bool(config["walk_forward"]["equal_session_weights"]),
    )
    sampling_balance = pd.DataFrame({
        "session": frame["session"],
        "is_train": train_mask,
        "selected_for_training": selected_mask,
        "loss_weight": weights,
    }).groupby("session", as_index=False).agg(
        source_rows=("session", "size"),
        is_train=("is_train", "max"),
        sampled_rows=("selected_for_training", "sum"),
        total_loss_weight=("loss_weight", "sum"),
    )

    version = config["artifacts"]["version"]
    processed_root = data_root / "processed"
    model_root = data_root / "models"
    backtest_root = data_root / "backtests"
    for path in [processed_root, model_root, backtest_root]:
        path.mkdir(parents=True, exist_ok=True)
    checkpoint_path = model_root / f"{version}_final.pt"
    torch.save({
        "model_state_dict": fitted["model"].cpu().state_dict(),
        "feature_names": feature_names,
        "feature_groups": feature_schema["selected_groups"],
        "classes": classes,
        "center": torch.from_numpy(fitted["center"]),
        "scale": torch.from_numpy(fitted["scale"]),
        "temperature": temperature,
        "entry_mode": entry_mode,
        "epochs": epochs,
        "train_sessions": train_sessions,
        "test_sessions": test_sessions,
        "test_is_pristine": config["data"]["test_is_pristine"],
        "payoff": payoff,
        "threshold": threshold,
        "seed": seed + 100,
    }, checkpoint_path)
    parameter_count = int(sum(parameter.numel() for parameter in fitted["model"].parameters()))
    paths = {
        "predictions": processed_root / f"{version}_predictions.parquet",
        "metrics": processed_root / f"{version}_metrics.parquet",
        "generalization_gap": processed_root / f"{version}_generalization_gap.parquet",
        "training_history": processed_root / f"{version}_training_history.parquet",
        "sampling_balance": processed_root / f"{version}_sampling_balance.parquet",
        "payoffs": processed_root / f"{version}_payoffs.parquet",
        "threshold": backtest_root / f"{version}_threshold.parquet",
        "backtest_metrics": backtest_root / f"{version}_backtest_metrics.parquet",
        "session_metrics": backtest_root / f"{version}_session_metrics.parquet",
        "ledger": backtest_root / f"{version}_ledger.parquet",
        "equity": backtest_root / f"{version}_equity.parquet",
        "checkpoint": checkpoint_path,
        "manifest": model_root / f"{version}_manifest.json",
    }
    outputs = {
        "predictions": predictions,
        "metrics": metrics,
        "generalization_gap": generalization_gap,
        "training_history": fitted["history"],
        "sampling_balance": sampling_balance,
        "payoffs": payoff_frame,
        "threshold": threshold_frame,
        "backtest_metrics": backtest_metrics,
        "session_metrics": session_metrics,
        "ledger": ledger,
        "equity": equity,
    }
    for name, table in outputs.items():
        table.to_parquet(paths[name], index=False, compression="zstd")
    manifest = {
        "version": version,
        "environment": "urban",
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "device": str(device),
        "seed": seed,
        "config_path": config["_config_path"],
        "split_method": "chronological_session_7_train_2_test",
        "train_sessions": train_sessions,
        "test_sessions": test_sessions,
        "train_rows": int(train_mask.sum()),
        "test_rows": int((~train_mask).sum()),
        "train_row_share": float(train_mask.mean()),
        "test_is_pristine": bool(config["data"]["test_is_pristine"]),
        "validation_used": False,
        "early_stopping_used": False,
        "temperature_calibration_used": False,
        "feature_count": len(feature_names),
        "parameter_count": parameter_count,
        "fixed_epochs": epochs,
        "temperature": temperature,
        "entry_mode": entry_mode,
        "threshold": threshold_frame.iloc[0].to_dict(),
        "deployment_status": "EXPERIMENT_ONLY_NON_PRISTINE_TEST",
        "checkpoint": str(checkpoint_path),
        "artifacts": {name: str(path) for name, path in paths.items() if name != "manifest"},
    }
    paths["manifest"].write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, default=str), encoding="utf-8",
    )
    probability_columns = [f"probability_{outcome.lower()}" for outcome in classes]
    if not np.allclose(predictions[probability_columns].sum(axis=1), 1.0, atol=1e-5):
        raise AssertionError("class probability 합이 1이 아닙니다.")
    if predictions.duplicated(["source_path", "symbol", "input_end_timestamp"]).any():
        raise AssertionError("prediction key가 중복됩니다.")
    if not np.allclose(
        sampling_balance.loc[sampling_balance["is_train"], "total_loss_weight"],
        sampling_balance.loc[sampling_balance["is_train"], "total_loss_weight"].iloc[0],
        rtol=1e-5,
    ):
        raise AssertionError("날짜별 total loss weight가 같지 않습니다.")
    if not all(path.exists() for path in paths.values()):
        raise AssertionError("필수 artifact가 누락됐습니다.")
    return {
        "config": config,
        "device": str(device),
        "feature_count": len(feature_names),
        "parameter_count": parameter_count,
        "predictions": predictions,
        "metrics": metrics,
        "generalization_gap": generalization_gap,
        "training_history": fitted["history"],
        "sampling_balance": sampling_balance,
        "threshold": threshold_frame,
        "backtest_metrics": backtest_metrics,
        "session_metrics": session_metrics,
        "ledger": ledger,
        "equity": equity,
        "paths": paths,
    }


__all__ = [
    "chronological_session_split",
    "prepare_immediate_execution_frame",
    "run_fixed_train_test_experiment",
    "score_quantile_threshold",
]
