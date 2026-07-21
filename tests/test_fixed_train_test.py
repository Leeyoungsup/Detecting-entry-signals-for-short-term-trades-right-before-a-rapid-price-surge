from __future__ import annotations

import numpy as np
import pandas as pd

from src.fixed_train_test import chronological_session_split, score_quantile_threshold


def test_chronological_split_uses_last_two_complete_sessions_as_test() -> None:
    sessions = [f"session_2026-07-{day:02d}" for day in [17, 7, 16, 8, 9, 10, 13, 14, 15]]
    train, test = chronological_session_split(sessions, test_session_count=2)
    assert train == sorted(sessions)[:7]
    assert test == sorted(sessions)[-2:]
    assert max(train) < min(test)


def test_score_quantile_threshold_does_not_require_outcomes() -> None:
    scores = pd.Series([-0.2, 0.0, 0.01, 0.02, 0.03])
    threshold = score_quantile_threshold(scores, quantile=0.5, floor=0.0)
    assert np.isclose(threshold, 0.01)
