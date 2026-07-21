from __future__ import annotations

import numpy as np
import pandas as pd

from src.quote_surge_binary import quote_surge_target, resolve_positive_weight, top_quantile_metrics
from src.walk_forward_oof import training_sample_mask


def test_quote_surge_uses_current_ask_and_future_last_bid() -> None:
    target, first_hit, maximum_return = quote_surge_target(
        100.0, np.array([101.0, 103.0, 102.0]), 0.03,
    )
    assert target == 1
    assert first_hit == 2
    assert np.isclose(maximum_return, 0.03)
    negative, no_hit, _ = quote_surge_target(
        100.0, np.array([101.0, 102.99, 102.0]), 0.03,
    )
    assert negative == 0
    assert no_hit == 0


def test_three_minute_event_bucket_keeps_one_row_per_symbol_bucket() -> None:
    frame = pd.DataFrame({
        "session": ["d1"] * 4,
        "symbol": ["AAA"] * 4,
        "input_end_timestamp": pd.to_datetime([
            "2026-07-07 13:00Z", "2026-07-07 13:02Z", "2026-07-07 13:03Z", "2026-07-07 13:05Z",
        ]),
    })
    mask = training_sample_mask(frame, {"sampling_method": "event_bucket", "bucket_minutes": 3})
    assert mask.tolist() == [True, False, True, False]


def test_balanced_positive_weight_applies_explicit_multiplier() -> None:
    labels = np.array([0, 0, 0, 1], dtype=np.float32)
    row_weights = np.ones(4, dtype=np.float32)
    base, weighted = resolve_positive_weight(labels, row_weights, "balanced", 1.5)
    assert np.isclose(base, 3.0)
    assert np.isclose(weighted, 4.5)


def test_top_quantile_metrics_reports_precision_and_recall() -> None:
    predictions = pd.DataFrame({
        "evaluation_group": ["train"] * 4 + ["test"] * 4,
        "surge_score": [0.9, 0.8, 0.2, 0.1] * 2,
        "target_surge_3m": [1, 0, 1, 0] * 2,
    })
    result = top_quantile_metrics(predictions, [0.5])
    assert np.allclose(result["precision"], 0.5)
    assert np.allclose(result["recall"], 0.5)
