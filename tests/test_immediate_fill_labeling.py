from __future__ import annotations

import pandas as pd

from src.immediate_fill_labeling import build_immediate_order, evaluate_immediate_path


def _bars(open_: float, high: float, low: float, close: float, count: int = 10) -> pd.DataFrame:
    return pd.DataFrame([
        {"open": open_, "high": high, "low": low, "close": close}
        for _ in range(count)
    ])


def test_immediate_order_uses_reference_close_without_fill_threshold() -> None:
    order = build_immediate_order(100.0)
    assert order is not None
    assert float(order.entry_price) == 100.0
    assert float(order.take_profit_price) == 105.0
    assert float(order.stop_loss_price) == 97.0


def test_immediate_path_never_returns_no_fill() -> None:
    order = build_immediate_order(100.0)
    assert order is not None
    timeout = evaluate_immediate_path(_bars(100.0, 101.0, 99.0, 100.0), order, "high_first")
    tp = evaluate_immediate_path(_bars(100.0, 106.0, 99.0, 101.0), order, "high_first")
    sl = evaluate_immediate_path(_bars(100.0, 101.0, 96.0, 99.0), order, "low_first")
    assert timeout["outcome"] == "TIMEOUT"
    assert tp["outcome"] == "TP"
    assert sl["outcome"] == "SL"
    assert {timeout["outcome"], tp["outcome"], sl["outcome"]}.isdisjoint({"NO_FILL"})
