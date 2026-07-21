from __future__ import annotations

from src.reduced_walk_forward import (
    build_feature_groups,
    combine_feature_groups,
    decide_feature_restoration,
)


def test_feature_groups_use_only_declared_aggregation_pairs() -> None:
    config = {
        "feature_selection": {
            "groups": {
                "core": {"last": ["a", "b"], "mean60": ["a"]},
                "trend": {"last": ["trend"]},
            },
        },
    }
    schema = ["last__a", "last__b", "mean60__a", "last__trend", "std60__unused"]
    groups = build_feature_groups(config, schema)
    assert groups == {
        "core": ["last__a", "last__b", "mean60__a"],
        "trend": ["last__trend"],
    }
    assert combine_feature_groups(groups, ["core", "trend"]) == [
        "last__a", "last__b", "mean60__a", "last__trend",
    ]


def test_feature_restoration_requires_all_predeclared_oof_checks() -> None:
    selection_config = {
        "min_return_spearman_improvement": 0.005,
        "max_worst_session_spearman_drop": 0.020,
        "max_multiclass_logloss_increase": 0.010,
    }
    current = {
        "return_spearman": 0.10,
        "worst_session_return_spearman": 0.02,
        "multiclass_logloss": 0.90,
    }
    accepted, checks = decide_feature_restoration(
        current,
        {
            "return_spearman": 0.11,
            "worst_session_return_spearman": 0.01,
            "multiclass_logloss": 0.905,
        },
        selection_config,
    )
    assert accepted
    assert all(checks[key] for key in ["passes_spearman", "passes_worst_session", "passes_logloss"])

    rejected, rejected_checks = decide_feature_restoration(
        current,
        {
            "return_spearman": 0.11,
            "worst_session_return_spearman": -0.01,
            "multiclass_logloss": 0.905,
        },
        selection_config,
    )
    assert not rejected
    assert not rejected_checks["passes_worst_session"]
