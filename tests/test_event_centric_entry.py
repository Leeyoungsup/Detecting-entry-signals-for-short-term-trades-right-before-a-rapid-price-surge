from __future__ import annotations

import numpy as np
import pandas as pd

from src.event_centric_entry import (
    build_event_training_policy,
    decision_score_candidates,
    multi_horizon_quote_targets,
)


TARGET_SPECS = [
    {"name": "up_1pct_1m", "horizon_minutes": 1, "target_return": 0.01},
    {"name": "up_2pct_2m", "horizon_minutes": 2, "target_return": 0.02},
    {"name": "up_3pct_3m", "horizon_minutes": 3, "target_return": 0.03},
]


def test_multihorizon_targets_use_only_each_declared_horizon() -> None:
    result = multi_horizon_quote_targets(100.0, np.array([101.0, 101.5, 103.0]), TARGET_SPECS)
    assert result["target_up_1pct_1m"] == 1
    assert result["target_up_2pct_2m"] == 0
    assert result["target_up_3pct_3m"] == 1
    assert result["time_to_up_3pct_3m"] == 3
    assert np.isclose(result["maximum_future_bid_return_3m"], 0.03)


def test_event_policy_deoverlaps_positive_cluster_and_spaces_regular_negatives() -> None:
    frame = pd.DataFrame({
        "session": ["d1"] * 6,
        "symbol": ["AAA"] * 6,
        "input_end_timestamp": pd.date_range("2026-07-07 13:00Z", periods=6, freq="min"),
        "target_up_3pct_3m": [1, 1, 0, 0, 0, 0],
        "maximum_future_bid_return_3m": [0.03, 0.03, 0.02, 0.0, 0.0, 0.0],
    })
    policy = build_event_training_policy(frame, "target_up_3pct_3m", {
        "hard_negative_min_return": 0.015,
        "regular_negative_spacing_minutes": 3,
        "event_cluster_minutes": 3,
    })
    assert policy["selected_mask"].tolist() == [True, True, True, True, False, False]
    assert np.allclose(policy["overlap_weight"][:2], [0.5, 0.5])
    assert policy["positive_clusters"] == 1


def test_decision_score_candidates_include_each_head_and_mean() -> None:
    predictions = pd.DataFrame({
        "score_up_1pct_1m": [0.2, 0.8],
        "score_up_2pct_2m": [0.4, 0.6],
        "score_up_3pct_3m": [0.6, 0.4],
    })
    candidates = decision_score_candidates(predictions, [
        "up_1pct_1m", "up_2pct_2m", "up_3pct_3m",
    ])
    assert set(candidates) == {
        "head_up_1pct_1m", "head_up_2pct_2m", "head_up_3pct_3m", "mean_all_heads",
    }
    assert np.allclose(candidates["mean_all_heads"], [0.4, 0.6])
