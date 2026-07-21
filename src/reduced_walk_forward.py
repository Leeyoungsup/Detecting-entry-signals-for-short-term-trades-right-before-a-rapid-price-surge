from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from src.walk_forward_oof import (
    estimate_payoffs,
    expected_value_from_probabilities,
    find_project_root,
    fit_early_stopped_model,
    fit_fixed_epoch_model,
    load_config,
    make_walk_forward_folds,
    predict_logits,
    prediction_metrics,
    prepare_execution_frame,
    price_tier,
    seed_everything,
    softmax_numpy,
    threshold_search,
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


def build_feature_groups(config: dict[str, Any], schema_features: list[str]) -> dict[str, list[str]]:
    """Expand the hand-curated aggregation/base-feature map and validate the schema."""
    available = set(schema_features)
    groups: dict[str, list[str]] = {}
    seen: set[str] = set()
    for group_name, aggregation_map in config["feature_selection"]["groups"].items():
        names = [
            f"{aggregation}__{base_feature}"
            for aggregation, base_features in aggregation_map.items()
            for base_feature in base_features
        ]
        missing = [name for name in names if name not in available]
        if missing:
            raise ValueError(f"{group_name} feature가 schema에 없습니다: {missing}")
        duplicated = [name for name in names if name in seen]
        if duplicated:
            raise ValueError(f"feature group 간 중복이 있습니다: {duplicated}")
        if len(names) != len(set(names)):
            raise ValueError(f"{group_name} 내부에 중복 feature가 있습니다.")
        groups[group_name] = names
        seen.update(names)
    return groups


def combine_feature_groups(groups: dict[str, list[str]], selected_groups: list[str]) -> list[str]:
    names = [name for group in selected_groups for name in groups[group]]
    if len(names) != len(set(names)):
        raise ValueError("선택된 feature에 중복이 있습니다.")
    return names


def selection_summary(metrics: pd.DataFrame) -> dict[str, float]:
    overall = metrics[(metrics["evaluation_group"].eq("oof")) & (metrics["session"].eq("ALL"))].iloc[0]
    sessions = metrics[(metrics["evaluation_group"].eq("oof")) & (~metrics["session"].eq("ALL"))]
    return {
        "return_spearman": float(overall["return_spearman"]),
        "worst_session_return_spearman": float(sessions["return_spearman"].min()),
        "multiclass_logloss": float(overall["multiclass_logloss"]),
        "tp_pr_auc": float(overall["tp_pr_auc"]),
    }


def decide_feature_restoration(
    current: dict[str, float],
    candidate: dict[str, float],
    selection_config: dict[str, Any],
) -> tuple[bool, dict[str, float | bool]]:
    spearman_improvement = candidate["return_spearman"] - current["return_spearman"]
    worst_session_drop = current["worst_session_return_spearman"] - candidate["worst_session_return_spearman"]
    logloss_increase = candidate["multiclass_logloss"] - current["multiclass_logloss"]
    checks = {
        "spearman_improvement": float(spearman_improvement),
        "worst_session_spearman_drop": float(worst_session_drop),
        "multiclass_logloss_increase": float(logloss_increase),
        "passes_spearman": bool(
            np.isfinite(spearman_improvement)
            and spearman_improvement >= selection_config["min_return_spearman_improvement"]
        ),
        "passes_worst_session": bool(
            np.isfinite(worst_session_drop)
            and worst_session_drop <= selection_config["max_worst_session_spearman_drop"]
        ),
        "passes_logloss": bool(
            np.isfinite(logloss_increase)
            and logloss_increase <= selection_config["max_multiclass_logloss_increase"]
        ),
    }
    accepted = bool(checks["passes_spearman"] and checks["passes_worst_session"] and checks["passes_logloss"])
    return accepted, checks


def _make_prediction_frame(
    frame: pd.DataFrame,
    mask: np.ndarray,
    probability: np.ndarray,
    predicted_ev: np.ndarray,
    classes: list[str],
    evaluation_group: str,
    fold: int,
    fit_sessions: list[str],
    inner_validation_session: str | None,
) -> pd.DataFrame:
    prediction = frame.loc[mask, IDENTITY_COLUMNS].copy()
    for class_id, outcome in enumerate(classes):
        prediction[f"probability_{outcome.lower()}"] = probability[:, class_id]
    prediction["predicted_expected_net_return"] = predicted_ev
    prediction["evaluation_group"] = evaluation_group
    prediction["fold"] = fold
    prediction["fit_sessions"] = ",".join(fit_sessions)
    prediction["inner_validation_session"] = inner_validation_session
    return prediction


def evaluate_oof_variant(
    frame: pd.DataFrame,
    y: np.ndarray,
    feature_names: list[str],
    folds: list[dict[str, Any]],
    config: dict[str, Any],
    device: torch.device,
    seed: int,
    variant_name: str,
) -> dict[str, Any]:
    X_raw = frame[feature_names].to_numpy(dtype=np.float32)
    classes = list(config["model"]["classes"])
    predictions = []
    fold_rows = []
    histories = []
    payoffs = []
    checkpoints = []
    for fold in folds:
        fold_seed = seed + fold["fold"] * 100
        fitted = fit_early_stopped_model(
            X_raw,
            y,
            frame,
            fold["fit_sessions"],
            fold["inner_validation_session"],
            config,
            device,
            fold_seed,
        )
        evaluation_mask = frame["session"].eq(fold["evaluation_session"]).to_numpy()
        X_evaluation = transform_features(
            X_raw[evaluation_mask], fitted["center"], fitted["scale"], config["walk_forward"]["scaler_clip"],
        )
        logits = predict_logits(fitted["model"], X_evaluation, device)
        probability = softmax_numpy(logits, fitted["temperature"])
        payoff, payoff_frame = estimate_payoffs(
            frame, fold["payoff_history_sessions"], classes, config, f"{variant_name}_fold_{fold['fold']}",
        )
        evaluation_tiers = frame.loc[evaluation_mask, "price_tier"]
        predictions.append(_make_prediction_frame(
            frame,
            evaluation_mask,
            probability,
            expected_value_from_probabilities(probability, evaluation_tiers, classes, payoff),
            classes,
            "oof",
            fold["fold"],
            fold["fit_sessions"],
            fold["inner_validation_session"],
        ))
        fold_rows.append({
            **fold,
            "variant": variant_name,
            "feature_count": len(feature_names),
            "best_epoch": fitted["best_epoch"],
            "temperature": fitted["temperature"],
            "inner_raw_logloss": fitted["inner_raw_logloss"],
            "inner_calibrated_logloss": fitted["inner_calibrated_logloss"],
            "train_rows_before_sampling": fitted["train_rows_before_stride"],
            "train_rows_after_sampling": fitted["train_rows_after_sampling"],
            "inner_rows": fitted["inner_rows"],
        })
        history = fitted["history"].copy()
        history["variant"] = variant_name
        history["fold"] = fold["fold"]
        histories.append(history)
        payoff_frame["variant"] = variant_name
        payoffs.append(payoff_frame)
        checkpoints.append({
            "model_state_dict": {
                name: value.detach().cpu().clone() for name, value in fitted["model"].state_dict().items()
            },
            "feature_names": feature_names,
            "classes": classes,
            "center": torch.from_numpy(fitted["center"]),
            "scale": torch.from_numpy(fitted["scale"]),
            "temperature": fitted["temperature"],
            "fold": fold,
            "best_epoch": fitted["best_epoch"],
            "seed": fold_seed,
        })
        del fitted
        if device.type == "cuda":
            torch.cuda.empty_cache()
    prediction_frame = pd.concat(predictions, ignore_index=True).sort_values(
        ["input_end_timestamp", "symbol"],
    ).reset_index(drop=True)
    metric_frame = prediction_metrics(prediction_frame, classes)
    metric_frame["variant"] = variant_name
    return {
        "variant": variant_name,
        "feature_names": feature_names,
        "predictions": prediction_frame,
        "metrics": metric_frame,
        "summary": selection_summary(metric_frame),
        "folds": pd.DataFrame(fold_rows),
        "history": pd.concat(histories, ignore_index=True),
        "payoffs": pd.concat(payoffs, ignore_index=True),
        "checkpoints": checkpoints,
    }


def _reference_safe_backtest(
    execution: pd.DataFrame,
    config: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    compatible = execution.copy()
    compatible["evaluation_group"] = compatible["evaluation_group"].replace({"reference_test": "test"})
    search, selected, aggregate, sessions, deployment, ledger, equity = threshold_search(compatible, config)
    for table in [aggregate, sessions, ledger, equity]:
        if "evaluation_group" in table:
            table["evaluation_group"] = table["evaluation_group"].replace({"test": "reference_test"})
    deployment = deployment.rename(columns={
        "test_return_pass": "reference_return_pass",
        "test_sessions": "reference_sessions",
        "test_profitable_sessions": "reference_profitable_sessions",
        "test_profitable_session_share": "reference_profitable_session_share",
        "test_worst_session_mean_net_return": "reference_worst_session_mean_net_return",
    })
    deployment["fresh_test_available"] = False
    deployment["deployment_status"] = np.where(
        deployment["selection_status"].eq("VALID"), "REQUIRES_NEW_TEST", "FAIL",
    )
    return search, selected, aggregate, sessions, deployment, ledger, equity


def run_reduced_experiment(
    project_root: Path | None = None,
    config_path: Path | None = None,
) -> dict[str, Any]:
    project_root = find_project_root(project_root)
    config_path = config_path or project_root / "configs/reduced_walk_forward_oof.yaml"
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
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    feature_groups = build_feature_groups(config, schema["features"])

    frame = pd.read_parquet(tabular_path)
    labels = pd.read_parquet(labels_path)
    classes = list(config["model"]["classes"])
    frame = frame[frame["dual_outcome_10m"].isin(classes)].copy().reset_index(drop=True)
    frame["input_end_timestamp"] = pd.to_datetime(frame["input_end_timestamp"], utc=True)
    frame["outcome_id"] = frame["dual_outcome_10m"].map(
        {outcome: index for index, outcome in enumerate(classes)},
    ).astype(np.int64)
    frame["price_tier"] = price_tier(frame["entry_price"], config["expected_value"]["price_tier_boundary"])
    development_sessions = list(config["data"]["development_sessions"])
    reference_sessions = list(config["data"]["reference_sessions"])
    expected_sessions = set(development_sessions + reference_sessions)
    if set(frame["session"]) != expected_sessions:
        raise AssertionError("tabular session과 reduced config session이 일치하지 않습니다.")
    if set(development_sessions) & set(reference_sessions):
        raise AssertionError("development/reference session이 겹칩니다.")
    if frame.loc[frame["session"].isin(development_sessions), "input_end_timestamp"].max() >= frame.loc[
        frame["session"].isin(reference_sessions), "input_end_timestamp"
    ].min():
        raise AssertionError("reference session은 development session 뒤여야 합니다.")
    y = frame["outcome_id"].to_numpy(dtype=np.int64)
    folds = make_walk_forward_folds(development_sessions, config["walk_forward"]["minimum_prior_sessions"])

    selection_config = config["feature_selection"]
    selected_groups = list(selection_config["base_groups"])
    selected_features = combine_feature_groups(feature_groups, selected_groups)
    if len(selected_features) > selection_config["maximum_features"]:
        raise ValueError("base feature 수가 maximum_features를 초과합니다.")
    all_results: list[dict[str, Any]] = []
    decisions: list[dict[str, Any]] = []
    base_name = "+".join(selected_groups)
    current_result = evaluate_oof_variant(
        frame, y, selected_features, folds, config, device, seed, base_name,
    )
    all_results.append(current_result)
    decisions.append({
        "step": 0,
        "candidate_group": "BASE",
        "variant": base_name,
        "feature_count": len(selected_features),
        "accepted": True,
        "reason": "PREDECLARED_BASE",
        **{f"candidate_{key}": value for key, value in current_result["summary"].items()},
    })

    for step, candidate_group in enumerate(selection_config["restore_order"], start=1):
        candidate_groups = [*selected_groups, candidate_group]
        candidate_features = combine_feature_groups(feature_groups, candidate_groups)
        candidate_name = "+".join(candidate_groups)
        if len(candidate_features) > selection_config["maximum_features"]:
            decisions.append({
                "step": step,
                "candidate_group": candidate_group,
                "variant": candidate_name,
                "feature_count": len(candidate_features),
                "accepted": False,
                "reason": "FEATURE_LIMIT",
                **{f"current_{key}": value for key, value in current_result["summary"].items()},
            })
            continue
        candidate_result = evaluate_oof_variant(
            frame, y, candidate_features, folds, config, device, seed, candidate_name,
        )
        all_results.append(candidate_result)
        accepted, checks = decide_feature_restoration(
            current_result["summary"], candidate_result["summary"], selection_config,
        )
        decisions.append({
            "step": step,
            "candidate_group": candidate_group,
            "variant": candidate_name,
            "feature_count": len(candidate_features),
            "accepted": accepted,
            "reason": "OOF_CRITERIA_PASS" if accepted else "OOF_CRITERIA_FAIL",
            **{f"current_{key}": value for key, value in current_result["summary"].items()},
            **{f"candidate_{key}": value for key, value in candidate_result["summary"].items()},
            **checks,
        })
        if accepted:
            selected_groups = candidate_groups
            selected_features = candidate_features
            current_result = candidate_result

    fold_metrics = current_result["folds"]
    final_epochs = max(1, min(config["model"]["max_epochs"], int(np.median(fold_metrics["best_epoch"]))))
    final_temperature = float(np.median(fold_metrics["temperature"]))
    X_raw = frame[selected_features].to_numpy(dtype=np.float32)
    final_fit = fit_fixed_epoch_model(
        X_raw, y, frame, development_sessions, final_epochs, config, device, seed + 900,
    )
    reference_mask = frame["session"].isin(reference_sessions).to_numpy()
    X_reference = transform_features(
        X_raw[reference_mask], final_fit["center"], final_fit["scale"], config["walk_forward"]["scaler_clip"],
    )
    reference_logits = predict_logits(final_fit["model"], X_reference, device)
    reference_probability = softmax_numpy(reference_logits, final_temperature)
    final_payoff, final_payoff_frame = estimate_payoffs(
        frame, development_sessions, classes, config, "final_development",
    )
    reference_prediction = _make_prediction_frame(
        frame,
        reference_mask,
        reference_probability,
        expected_value_from_probabilities(
            reference_probability, frame.loc[reference_mask, "price_tier"], classes, final_payoff,
        ),
        classes,
        "reference_test",
        0,
        development_sessions,
        None,
    )
    predictions = pd.concat([current_result["predictions"], reference_prediction], ignore_index=True).sort_values(
        ["input_end_timestamp", "symbol"],
    ).reset_index(drop=True)
    metric_compatible = predictions.copy()
    metric_compatible["evaluation_group"] = metric_compatible["evaluation_group"].replace({"reference_test": "test"})
    metrics = prediction_metrics(metric_compatible, classes)
    metrics["evaluation_group"] = metrics["evaluation_group"].replace({"test": "reference_test"})
    execution = prepare_execution_frame(predictions, labels)
    search, selected_threshold, backtest_metrics, session_metrics, deployment, ledger, equity = _reference_safe_backtest(
        execution, config,
    )

    processed_root = data_root / "processed"
    model_root = data_root / "models"
    backtest_root = data_root / "backtests"
    for path in [processed_root, model_root, backtest_root]:
        path.mkdir(parents=True, exist_ok=True)
    version = config["artifacts"]["version"]
    final_checkpoint_path = model_root / f"{version}_final.pt"
    fold_checkpoint_paths = []
    for checkpoint in current_result["checkpoints"]:
        fold = checkpoint["fold"]
        path = model_root / f"{version}_fold_{fold['fold']}_{fold['evaluation_session'].removeprefix('session_')}.pt"
        torch.save(checkpoint, path)
        fold_checkpoint_paths.append(str(path))
    torch.save({
        "model_state_dict": final_fit["model"].cpu().state_dict(),
        "feature_names": selected_features,
        "feature_groups": selected_groups,
        "classes": classes,
        "center": torch.from_numpy(final_fit["center"]),
        "scale": torch.from_numpy(final_fit["scale"]),
        "temperature": final_temperature,
        "epochs": final_epochs,
        "development_sessions": development_sessions,
        "reference_sessions": reference_sessions,
        "reference_only": True,
        "payoff": final_payoff,
        "config": config["model"],
        "seed": seed + 900,
    }, final_checkpoint_path)
    if device.type == "cuda":
        torch.cuda.empty_cache()

    decisions_frame = pd.DataFrame(decisions)
    variant_metrics = pd.concat([result["metrics"] for result in all_results], ignore_index=True)
    variant_folds = pd.concat([result["folds"] for result in all_results], ignore_index=True)
    variant_summaries = pd.DataFrame([
        {"variant": result["variant"], "feature_count": len(result["feature_names"]), **result["summary"]}
        for result in all_results
    ])
    selected_payoffs = pd.concat([current_result["payoffs"], final_payoff_frame], ignore_index=True)
    selected_history = current_result["history"]
    feature_schema = {
        "version": version,
        "source_feature_schema": str(schema_path),
        "base_groups": selection_config["base_groups"],
        "restore_order": selection_config["restore_order"],
        "selected_groups": selected_groups,
        "selected_feature_count": len(selected_features),
        "selected_features": selected_features,
        "group_features": feature_groups,
    }
    paths = {
        "predictions": processed_root / f"{version}_predictions.parquet",
        "metrics": processed_root / f"{version}_metrics.parquet",
        "variant_metrics": processed_root / f"{version}_variant_metrics.parquet",
        "variant_summaries": processed_root / f"{version}_variant_summaries.parquet",
        "variant_folds": processed_root / f"{version}_variant_folds.parquet",
        "selected_folds": processed_root / f"{version}_selected_folds.parquet",
        "selected_history": processed_root / f"{version}_selected_history.parquet",
        "feature_decisions": processed_root / f"{version}_feature_decisions.parquet",
        "feature_schema": processed_root / f"{version}_feature_schema.json",
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
    parquet_outputs = {
        "predictions": predictions,
        "metrics": metrics,
        "variant_metrics": variant_metrics,
        "variant_summaries": variant_summaries,
        "variant_folds": variant_folds,
        "selected_folds": fold_metrics,
        "selected_history": selected_history,
        "feature_decisions": decisions_frame,
        "payoffs": selected_payoffs,
        "threshold_search": search,
        "selected_threshold": selected_threshold,
        "backtest_metrics": backtest_metrics,
        "session_metrics": session_metrics,
        "deployment": deployment,
        "ledger": ledger,
        "equity": equity,
    }
    for name, table in parquet_outputs.items():
        table.to_parquet(paths[name], index=False, compression="zstd")
    paths["feature_schema"].write_text(
        json.dumps(feature_schema, ensure_ascii=False, indent=2), encoding="utf-8",
    )
    parameter_count = int(sum(parameter.numel() for parameter in final_fit["model"].parameters()))
    manifest = {
        "version": version,
        "environment": "urban",
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "device": str(device),
        "seed": seed,
        "config_path": config["_config_path"],
        "tabular_path": str(tabular_path),
        "labels_path": str(labels_path),
        "sampling_method": config["walk_forward"]["sampling_method"],
        "equal_session_weights": config["walk_forward"]["equal_session_weights"],
        "selected_groups": selected_groups,
        "feature_count": len(selected_features),
        "parameter_count": parameter_count,
        "classes": classes,
        "final_epochs": final_epochs,
        "final_temperature": final_temperature,
        "development_sessions": development_sessions,
        "reference_sessions": reference_sessions,
        "reference_only": True,
        "fresh_test_required": True,
        "fold_checkpoints": fold_checkpoint_paths,
        "final_checkpoint": str(final_checkpoint_path),
        "selected_threshold": selected_threshold.iloc[0].to_dict(),
        "deployment": deployment.iloc[0].to_dict(),
        "artifacts": {name: str(path) for name, path in paths.items() if name != "manifest"},
    }
    paths["manifest"].write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, default=str), encoding="utf-8",
    )

    probability_columns = [f"probability_{outcome.lower()}" for outcome in classes]
    if predictions.duplicated(["source_path", "symbol", "input_end_timestamp"]).any():
        raise AssertionError("prediction key가 중복됩니다.")
    if not np.allclose(predictions[probability_columns].sum(axis=1), 1.0, atol=1e-5):
        raise AssertionError("class probability 합이 1이 아닙니다.")
    if len(selected_features) > selection_config["maximum_features"]:
        raise AssertionError("최종 feature 수가 제한을 초과했습니다.")
    if deployment.iloc[0]["deployment_status"] == "PASS":
        raise AssertionError("reference-only 결과로 deployment PASS를 만들 수 없습니다.")
    if not all(path.exists() for path in paths.values()):
        raise AssertionError("필수 artifact가 누락됐습니다.")
    return {
        "config": config,
        "device": str(device),
        "selected_groups": selected_groups,
        "selected_features": selected_features,
        "feature_decisions": decisions_frame,
        "variant_summaries": variant_summaries,
        "folds": fold_metrics,
        "predictions": predictions,
        "metrics": metrics,
        "threshold_search": search,
        "selected_threshold": selected_threshold,
        "backtest_metrics": backtest_metrics,
        "session_metrics": session_metrics,
        "deployment": deployment,
        "ledger": ledger,
        "equity": equity,
        "paths": paths,
        "parameter_count": parameter_count,
    }


__all__ = [
    "build_feature_groups",
    "combine_feature_groups",
    "decide_feature_restoration",
    "evaluate_oof_variant",
    "run_reduced_experiment",
    "selection_summary",
]
