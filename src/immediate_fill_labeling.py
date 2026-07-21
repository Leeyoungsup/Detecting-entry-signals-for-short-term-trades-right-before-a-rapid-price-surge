from __future__ import annotations

import json
from dataclasses import dataclass
from decimal import Decimal, ROUND_CEILING, ROUND_HALF_UP
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.walk_forward_oof import find_project_root


D = Decimal
CENT = D("0.01")


@dataclass(frozen=True)
class ImmediateTradingConfig:
    order_notional_usd: Decimal = D("1000.00")
    take_profit_pct: Decimal = D("0.05")
    stop_loss_pct: Decimal = D("0.03")
    buy_slippage_pct: Decimal = D("0.0")
    sell_slippage_pct: Decimal = D("0.001")
    buy_commission_rate: Decimal = D("0.001")
    sell_commission_rate: Decimal = D("0.001")
    sec_fee_rate: Decimal = D("0.0000206")
    taf_per_share: Decimal = D("0.000195")
    taf_max_per_trade: Decimal = D("9.79")


@dataclass(frozen=True)
class ImmediateOrder:
    reference_price: Decimal
    entry_price: Decimal
    take_profit_price: Decimal
    stop_loss_price: Decimal
    shares: int


def _to_decimal(value: Any) -> Decimal:
    return value if isinstance(value, Decimal) else D(str(value))


def _round_cent(value: Decimal) -> Decimal:
    return value.quantize(CENT, rounding=ROUND_HALF_UP)


def _ceil_cent(value: Decimal) -> Decimal:
    return value.quantize(CENT, rounding=ROUND_CEILING)


def build_immediate_order(
    reference_close: Any,
    config: ImmediateTradingConfig = ImmediateTradingConfig(),
) -> ImmediateOrder | None:
    reference = _to_decimal(reference_close)
    entry = reference * (D("1") + config.buy_slippage_pct)
    if entry <= 0:
        return None
    shares = int(config.order_notional_usd // entry)
    if shares < 1:
        return None
    return ImmediateOrder(
        reference_price=reference,
        entry_price=entry,
        take_profit_price=entry * (D("1") + config.take_profit_pct),
        stop_loss_price=entry * (D("1") - config.stop_loss_pct),
        shares=shares,
    )


def calculate_trade_result(
    entry_price: Any,
    sell_reference_price: Any,
    shares: int,
    config: ImmediateTradingConfig = ImmediateTradingConfig(),
) -> dict[str, float]:
    buy_price = _to_decimal(entry_price)
    sell_reference = _to_decimal(sell_reference_price)
    sell_price = sell_reference * (D("1") - config.sell_slippage_pct)
    share_count = D(shares)
    buy_notional = buy_price * share_count
    sell_notional = sell_price * share_count
    buy_commission = _round_cent(buy_notional * config.buy_commission_rate)
    sell_commission = _round_cent(sell_notional * config.sell_commission_rate)
    sec_fee = _ceil_cent(sell_notional * config.sec_fee_rate)
    taf_fee = _ceil_cent(min(share_count * config.taf_per_share, config.taf_max_per_trade))
    total_fees = buy_commission + sell_commission + sec_fee + taf_fee
    gross_pnl = sell_notional - buy_notional
    net_pnl = gross_pnl - total_fees
    invested = buy_notional + buy_commission
    return {
        "sell_fill_price": float(sell_price),
        "total_fees": float(total_fees),
        "gross_return": float(gross_pnl / buy_notional),
        "net_return": float(net_pnl / invested),
        "net_pnl": float(net_pnl),
    }


def _barrier_event_at_price(price: float, take_profit: float, stop_loss: float) -> str | None:
    if price >= take_profit:
        return "TP"
    if price <= stop_loss:
        return "SL"
    return None


def _barrier_event_on_segment(start: float, end: float, take_profit: float, stop_loss: float) -> str | None:
    immediate = _barrier_event_at_price(start, take_profit, stop_loss)
    if immediate:
        return immediate
    if end > start and end >= take_profit:
        return "TP"
    if end < start and end <= stop_loss:
        return "SL"
    return None


def _bar_nodes(bar: Any, path_mode: str) -> list[float]:
    if path_mode == "high_first":
        return [float(bar.open), float(bar.high), float(bar.low), float(bar.close)]
    if path_mode == "low_first":
        return [float(bar.open), float(bar.low), float(bar.high), float(bar.close)]
    raise ValueError(f"unknown path mode: {path_mode}")


def evaluate_immediate_path(
    future_bars: pd.DataFrame,
    order: ImmediateOrder,
    path_mode: str,
    horizon: int = 10,
) -> dict[str, Any]:
    take_profit = float(order.take_profit_price)
    stop_loss = float(order.stop_loss_price)
    current_price = float(order.entry_price)
    for bar_number, bar in enumerate(future_bars.iloc[:horizon].itertuples(index=False), start=1):
        nodes = _bar_nodes(bar, path_mode)
        for target_price in nodes:
            event = _barrier_event_on_segment(current_price, target_price, take_profit, stop_loss)
            if event:
                return {
                    "outcome": event,
                    "exit_reference": take_profit if event == "TP" else stop_loss,
                    "event_bar": bar_number,
                }
            current_price = target_price
    return {
        "outcome": "TIMEOUT",
        "exit_reference": float(future_bars.iloc[horizon - 1]["close"]),
        "event_bar": horizon,
    }


def _load_raw_file(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(
        path,
        usecols=["symbol", "timestamp_utc", "open", "high", "low", "close"],
        encoding="utf-8-sig",
    )
    frame["symbol"] = frame["symbol"].astype(str).str.upper()
    frame["timestamp_utc"] = pd.to_datetime(frame["timestamp_utc"], errors="coerce", utc=True)
    for column in ["open", "high", "low", "close"]:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame = frame.sort_values(["symbol", "timestamp_utc"]).reset_index(drop=True)
    delta_minutes = frame.groupby("symbol")["timestamp_utc"].diff().dt.total_seconds().div(60)
    consecutive = pd.Series(np.isclose(delta_minutes, 1.0, rtol=0.0, atol=1e-6), index=frame.index)
    new_run = (~consecutive) | frame["symbol"].ne(frame["symbol"].shift())
    frame["run_number"] = new_run.cumsum().astype(np.int64)
    frame["run_id"] = path.stem + "::" + frame["run_number"].astype(str)
    return frame


def build_immediate_labels(
    sequence_index: pd.DataFrame,
    config: ImmediateTradingConfig = ImmediateTradingConfig(),
    horizon: int = 10,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for source_path, metadata in sequence_index.groupby("source_path", sort=False):
        path = Path(source_path)
        raw = _load_raw_file(path)
        for run_id, run_metadata in metadata.groupby("run_id", sort=False):
            run = raw[raw["run_id"].eq(run_id)].reset_index(drop=True)
            if run.empty:
                raise ValueError(f"sequence run을 raw에서 찾지 못했습니다: {run_id}")
            positions = pd.Series(np.arange(len(run)), index=run["timestamp_utc"]).to_dict()
            for meta in run_metadata.itertuples(index=False):
                decision_timestamp = pd.Timestamp(meta.input_end_timestamp)
                decision_position = positions.get(decision_timestamp)
                if decision_position is None or decision_position + horizon >= len(run):
                    continue
                decision = run.iloc[decision_position]
                future = run.iloc[decision_position + 1:decision_position + 1 + horizon].reset_index(drop=True)
                order = build_immediate_order(decision["close"], config)
                if order is None:
                    continue
                record: dict[str, Any] = {
                    "source_path": source_path,
                    "session": meta.session,
                    "split": meta.split,
                    "symbol": meta.symbol,
                    "run_id": run_id,
                    "input_end_timestamp": decision_timestamp,
                    "reference_close": float(order.reference_price),
                    "entry_price": float(order.entry_price),
                    "take_profit_price": float(order.take_profit_price),
                    "stop_loss_price": float(order.stop_loss_price),
                    "shares": order.shares,
                    "entry_mode": "immediate_market_proxy_at_close",
                    "label_version": "immediate_fill_dual_path_10m_tp5_sl3_v1",
                }
                path_results = {}
                for short_name, path_mode in [("hf", "high_first"), ("lf", "low_first")]:
                    path_result = evaluate_immediate_path(future, order, path_mode, horizon)
                    trade = calculate_trade_result(
                        order.entry_price, path_result["exit_reference"], order.shares, config,
                    )
                    result = {**path_result, **trade}
                    path_results[short_name] = result
                    for column in ["outcome", "event_bar", "gross_return", "net_return", "net_pnl", "total_fees"]:
                        record[f"{column}_{short_name}_{horizon}m"] = result[column]
                agreement = path_results["hf"]["outcome"] == path_results["lf"]["outcome"]
                dual_outcome = path_results["hf"]["outcome"] if agreement else "AMBIGUOUS"
                record[f"dual_agreement_{horizon}m"] = agreement
                record[f"dual_outcome_{horizon}m"] = dual_outcome
                record[f"binary_target_{horizon}m"] = (
                    1.0 if dual_outcome == "TP" else (0.0 if agreement else np.nan)
                )
                record[f"expected_net_return_dual_{horizon}m"] = (
                    path_results["hf"]["net_return"] + path_results["lf"]["net_return"]
                ) / 2
                rows.append(record)
    labels = pd.DataFrame(rows)
    if labels.duplicated(["source_path", "symbol", "input_end_timestamp"]).any():
        raise AssertionError("immediate label key가 중복됩니다.")
    valid_outcomes = {"TP", "SL", "TIMEOUT", "AMBIGUOUS"}
    if not set(labels[f"dual_outcome_{horizon}m"]).issubset(valid_outcomes):
        raise AssertionError("즉시체결 라벨에 허용되지 않은 outcome이 있습니다.")
    if labels[f"dual_outcome_{horizon}m"].eq("NO_FILL").any():
        raise AssertionError("즉시체결 라벨에는 NO_FILL이 있을 수 없습니다.")
    return labels


def build_tabular_artifact(
    feature_rows: pd.DataFrame,
    sequence_index: pd.DataFrame,
    labels: pd.DataFrame,
) -> tuple[pd.DataFrame, list[str]]:
    feature_metadata = [
        "source_path", "session", "split", "symbol", "run_id", "feature_row", "timestamp_utc",
    ]
    merge_keys = ["source_path", "symbol", "input_end_timestamp"]
    model_features = [column for column in feature_rows.columns if column not in feature_metadata]
    label_columns = [
        *merge_keys,
        "session",
        "split",
        "label_version",
        "entry_mode",
        "reference_close",
        "entry_price",
        "take_profit_price",
        "stop_loss_price",
        "shares",
        "dual_agreement_10m",
        "dual_outcome_10m",
        "binary_target_10m",
        "expected_net_return_dual_10m",
    ]
    confirmed_labels = labels[labels["dual_agreement_10m"]].copy()
    label_view = confirmed_labels[label_columns].rename(
        columns={"session": "label_session", "split": "label_split"},
    )
    dataset_index = sequence_index.merge(label_view, on=merge_keys, how="inner", validate="one_to_one")
    if len(dataset_index) != len(confirmed_labels):
        raise AssertionError("sequence와 immediate label 결합 건수가 다릅니다.")
    aggregate_names = [
        f"{prefix}__{feature}"
        for prefix in ["last", "mean60", "std60", "delta5", "delta20"]
        for feature in model_features
    ]
    X = np.empty((len(dataset_index), len(aggregate_names)), dtype=np.float32)
    for source_path, index in dataset_index.groupby("source_path", sort=False).groups.items():
        positions = np.asarray(list(index), dtype=np.int64)
        metadata = dataset_index.loc[positions]
        rows = feature_rows[feature_rows["source_path"].eq(source_path)].sort_values("feature_row")
        matrix = rows[model_features].to_numpy(dtype=np.float32)
        safe = np.nan_to_num(matrix, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float64)
        prefix_sum = np.vstack([np.zeros((1, len(model_features))), np.cumsum(safe, axis=0)])
        prefix_sq = np.vstack([np.zeros((1, len(model_features))), np.cumsum(safe**2, axis=0)])
        start = metadata["start_feature_row"].to_numpy(dtype=np.int64)
        end = metadata["end_feature_row"].to_numpy(dtype=np.int64)
        last = matrix[end]
        mean60 = (prefix_sum[end + 1] - prefix_sum[start]) / 60.0
        second_moment = (prefix_sq[end + 1] - prefix_sq[start]) / 60.0
        std60 = np.sqrt(np.maximum(second_moment - mean60**2, 0.0))
        delta5 = last - matrix[end - 5]
        delta20 = last - matrix[end - 20]
        block = np.concatenate([last, mean60, std60, delta5, delta20], axis=1).astype(np.float32)
        if not np.isfinite(block).all():
            raise AssertionError("tabular feature에 non-finite 값이 있습니다.")
        X[positions] = block
    tabular = pd.concat(
        [dataset_index.reset_index(drop=True), pd.DataFrame(X, columns=aggregate_names)], axis=1,
    )
    return tabular, aggregate_names


def create_immediate_artifacts(project_root: Path | None = None) -> dict[str, Any]:
    project_root = find_project_root(project_root)
    data_root = (project_root / "../../data/stock_data").resolve()
    processed = data_root / "processed"
    sequence_path = processed / "sequence_index_price_v2.parquet"
    feature_rows_path = processed / "feature_rows_price_v2.parquet"
    labels_path = processed / "labels_immediate_fill_dual_path_10m_tp5_sl3_v1.parquet"
    summary_path = processed / "labels_immediate_fill_dual_path_10m_tp5_sl3_v1_summary.parquet"
    tabular_path = processed / "baseline_tabular_immediate_fill_10m_tp5_sl3_price_v2.parquet"
    schema_path = processed / "baseline_tabular_immediate_fill_10m_tp5_sl3_price_v2_schema.json"
    sequence_index = pd.read_parquet(sequence_path)
    feature_rows = pd.read_parquet(feature_rows_path)
    labels = build_immediate_labels(sequence_index)
    tabular, aggregate_names = build_tabular_artifact(feature_rows, sequence_index, labels)
    summary = labels.groupby(["session", "dual_outcome_10m"], as_index=False).size().rename(
        columns={"size": "samples"},
    )
    labels.to_parquet(labels_path, index=False, compression="zstd")
    summary.to_parquet(summary_path, index=False, compression="zstd")
    tabular.to_parquet(tabular_path, index=False, compression="zstd")
    schema_path.write_text(json.dumps({
        "feature_version": "price_v2_no_weekday_tabular_60bar_v1",
        "label_version": "immediate_fill_dual_path_10m_tp5_sl3_v1",
        "entry_mode": "immediate_market_proxy_at_close",
        "base_feature_count": len(aggregate_names) // 5,
        "tabular_feature_count": len(aggregate_names),
        "aggregation": ["last", "mean60", "std60", "delta5", "delta20"],
        "features": aggregate_names,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    return {
        "labels": labels,
        "summary": summary,
        "tabular": tabular,
        "paths": {
            "labels": labels_path,
            "summary": summary_path,
            "tabular": tabular_path,
            "schema": schema_path,
        },
    }


__all__ = [
    "ImmediateOrder",
    "ImmediateTradingConfig",
    "build_immediate_labels",
    "build_immediate_order",
    "build_tabular_artifact",
    "calculate_trade_result",
    "create_immediate_artifacts",
    "evaluate_immediate_path",
]
