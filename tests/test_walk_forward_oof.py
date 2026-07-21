from __future__ import annotations

import numpy as np
import pandas as pd

from src.walk_forward_oof import (
    expected_value_from_probabilities,
    make_walk_forward_folds,
    session_balancing_weights,
    softmax_numpy,
    training_sample_mask,
)


def test_walk_forward_folds_use_only_past_sessions() -> None:
    sessions = [f"session_{index}" for index in range(6)]
    folds = make_walk_forward_folds(sessions, minimum_prior_sessions=3)
    assert [fold["evaluation_session"] for fold in folds] == sessions[3:]
    for fold in folds:
        evaluation_index = sessions.index(fold["evaluation_session"])
        assert all(sessions.index(session) < evaluation_index for session in fold["fit_sessions"])
        assert sessions.index(fold["inner_validation_session"]) < evaluation_index
        assert fold["inner_validation_session"] not in fold["fit_sessions"]


def test_softmax_temperature_returns_valid_probabilities() -> None:
    logits = np.array([[1.0, 2.0, -1.0, 0.5], [0.0, 0.0, 0.0, 0.0]], dtype=np.float32)
    probability = softmax_numpy(logits, temperature=1.25)
    assert np.all(probability >= 0)
    assert np.all(probability <= 1)
    assert np.allclose(probability.sum(axis=1), 1.0)


def test_expected_value_uses_outcome_probability_and_price_tier_payoff() -> None:
    classes = ["NO_FILL", "TP", "SL", "TIMEOUT"]
    probability = np.array([
        [0.5, 0.3, 0.1, 0.1],
        [0.2, 0.2, 0.4, 0.2],
    ], dtype=np.float32)
    tiers = pd.Series(["below_1", "at_or_above_1"])
    payoff = {
        "below_1": {"NO_FILL": 0.0, "TP": 0.05, "SL": -0.03, "TIMEOUT": -0.01},
        "at_or_above_1": {"NO_FILL": 0.0, "TP": 0.04, "SL": -0.02, "TIMEOUT": 0.0},
    }
    expected = np.array([0.3 * 0.05 - 0.1 * 0.03 - 0.1 * 0.01, 0.2 * 0.04 - 0.4 * 0.02])
    actual = expected_value_from_probabilities(probability, tiers, classes, payoff)
    assert np.allclose(actual, expected)


def test_ten_minute_bucket_sampling_keeps_one_row_per_session_symbol_bucket() -> None:
    frame = pd.DataFrame({
        "session": ["d1", "d1", "d1", "d1", "d2"],
        "symbol": ["AAA", "AAA", "AAA", "BBB", "AAA"],
        "input_end_timestamp": pd.to_datetime([
            "2026-07-07 13:01Z", "2026-07-07 13:09Z", "2026-07-07 13:10Z",
            "2026-07-07 13:05Z", "2026-07-08 13:01Z",
        ]),
    })
    mask = training_sample_mask(frame, {"sampling_method": "ten_minute_bucket", "bucket_minutes": 10})
    assert mask.tolist() == [True, False, True, True, True]


def test_session_balancing_weights_give_each_date_equal_total_weight() -> None:
    frame = pd.DataFrame({"session": ["d1", "d1", "d1", "d2", "d2"]})
    mask = np.array([True, True, True, True, False])
    weights = session_balancing_weights(frame, mask, enabled=True)
    assert np.isclose(weights[mask].mean(), 1.0)
    assert np.isclose(weights[:3].sum(), weights[3:4].sum())
    assert weights[4] == 0
