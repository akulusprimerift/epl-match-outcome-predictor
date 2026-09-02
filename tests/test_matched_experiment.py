"""Phase 7 possession features, matched rows, and comparison-rule tests."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import unittest

import pandas as pd

from src.build_history import build_team_history_frame
from src.compare_models import (
    ModelComparisonError,
    _build_report,
    apply_model_b_acceptance_rule,
)
from src.constants import (
    FEATURE_COLUMNS,
    POSSESSION_FEATURE_ADDITIONS,
    POSSESSION_FEATURE_COLUMNS,
    SPLIT_ORDER,
)
from src.matched_experiment import (
    MatchedExperimentError,
    assert_identical_model_match_ids,
    build_matched_split_manifest,
    build_possession_complete_dataset,
    compute_possession_rolling_features,
    feature_columns_for_set,
    previous_possession_match_ids,
    split_matched_dataset,
)
from src.split_data import load_split_policy
from src.train_xgboost import XGBoostTrainingError, train_xgboost_model
from tests.test_leakage import synthetic_canonical


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def possession_canonical(match_count: int = 10) -> pd.DataFrame:
    """Return valid synthetic fixtures with complete, non-complementary values."""
    canonical = synthetic_canonical(match_count)
    for position in canonical.index:
        canonical.loc[position, "home_possession"] = 43.0 + position
        canonical.loc[position, "away_possession"] = 54.0 - position / 2
        canonical.loc[position, "api_fixture_id"] = 10_000 + position
    canonical["home_possession"] = pd.to_numeric(
        canonical["home_possession"]
    ).astype("Float64")
    canonical["away_possession"] = pd.to_numeric(
        canonical["away_possession"]
    ).astype("Float64")
    canonical["api_fixture_id"] = pd.to_numeric(
        canonical["api_fixture_id"]
    ).astype("Int64")
    return canonical


class PossessionRollingFeatureTests(unittest.TestCase):
    """Prove possession windows are complete, strict, and leakage-safe."""

    def setUp(self) -> None:
        self.canonical = possession_canonical()
        self.history = build_team_history_frame(self.canonical)

    def test_previous_five_complete_matches_are_used(self) -> None:
        current_date = self.canonical.loc[7, "date"]
        expected = tuple(self.canonical.loc[2:6, "match_id"])
        self.assertEqual(
            previous_possession_match_ids(
                self.history, "alpha", current_date
            ),
            expected,
        )
        features = compute_possession_rolling_features(self.history)
        row = features.loc[
            features["match_id"].eq(self.canonical.loc[7, "match_id"])
            & features["team_slug"].eq("alpha")
        ].iloc[0]
        source = self.history.loc[
            self.history["team_slug"].eq("alpha")
            & self.history["match_id"].isin(expected),
            "possession",
        ]
        self.assertAlmostEqual(row["possession_avg_5"], source.mean())

    def test_missing_history_is_skipped_not_zero_filled(self) -> None:
        canonical = deepcopy(self.canonical)
        canonical.loc[2, ["home_possession", "away_possession", "api_fixture_id"]] = pd.NA
        history = build_team_history_frame(canonical)
        current_date = canonical.loc[7, "date"]
        expected = tuple(canonical.loc[[1, 3, 4, 5, 6], "match_id"])
        self.assertEqual(
            previous_possession_match_ids(history, "alpha", current_date),
            expected,
        )

    def test_current_possession_cannot_change_current_features(self) -> None:
        current_position = 7
        match_id = self.canonical.loc[current_position, "match_id"]
        baseline = build_possession_complete_dataset(
            self.canonical,
            self.history,
            model_b_period_start="1011",
        ).set_index("match_id")
        mutated = deepcopy(self.canonical)
        mutated.loc[current_position, "home_possession"] = 99.0
        rebuilt = build_possession_complete_dataset(
            mutated,
            build_team_history_frame(mutated),
            model_b_period_start="1011",
        ).set_index("match_id")
        pd.testing.assert_series_equal(
            baseline.loc[match_id, list(POSSESSION_FEATURE_ADDITIONS)],
            rebuilt.loc[match_id, list(POSSESSION_FEATURE_ADDITIONS)],
        )

    def test_future_possession_cannot_change_earlier_features(self) -> None:
        baseline = build_possession_complete_dataset(
            self.canonical,
            self.history,
            model_b_period_start="1011",
        ).set_index("match_id")
        mutated = deepcopy(self.canonical)
        mutated.loc[9, "away_possession"] = 1.0
        rebuilt = build_possession_complete_dataset(
            mutated,
            build_team_history_frame(mutated),
            model_b_period_start="1011",
        ).set_index("match_id")
        earlier_ids = baseline.index[
            baseline["date"].lt(self.canonical.loc[9, "date"])
        ]
        pd.testing.assert_frame_equal(
            baseline.loc[earlier_ids, list(POSSESSION_FEATURE_ADDITIONS)],
            rebuilt.loc[earlier_ids, list(POSSESSION_FEATURE_ADDITIONS)],
        )


class MatchedCohortContractTests(unittest.TestCase):
    """Verify identical row IDs and the exact three-column feature difference."""

    @classmethod
    def setUpClass(cls) -> None:
        canonical = possession_canonical(12)
        cls.dataset = build_possession_complete_dataset(
            canonical,
            build_team_history_frame(canonical),
            model_b_period_start="1011",
        )

    def test_only_possession_columns_differ(self) -> None:
        self.assertEqual(
            feature_columns_for_set("baseline_matched"), FEATURE_COLUMNS
        )
        self.assertEqual(
            feature_columns_for_set("possession"), POSSESSION_FEATURE_COLUMNS
        )
        self.assertEqual(
            set(POSSESSION_FEATURE_COLUMNS) - set(FEATURE_COLUMNS),
            set(POSSESSION_FEATURE_ADDITIONS),
        )

    def test_frozen_manifest_is_shared_by_both_models_per_split(self) -> None:
        rows = self.dataset.iloc[:4].copy().reset_index(drop=True)
        rows["season"] = ["2223", "2324", "2425", "2526"]
        rows["date"] = [
            "2023-05-01",
            "2024-05-01",
            "2025-05-01",
            "2026-05-01",
        ]
        rows["match_id"] = [f"matched-{index}" for index in range(4)]
        policy = load_split_policy(
            PROJECT_ROOT / "config" / "model_config.json"
        )
        splits = split_matched_dataset(rows, policy)
        manifest = build_matched_split_manifest(splits)
        for split_name in SPLIT_ORDER:
            expected = set(
                manifest.loc[
                    manifest["split"].eq(split_name), "match_id"
                ]
            )
            self.assertEqual(
                set(splits.by_name()[split_name]["match_id"]), expected
            )
        ids = manifest["match_id"].tolist()
        assert_identical_model_match_ids(manifest, ids, ids)
        with self.assertRaisesRegex(MatchedExperimentError, "differ"):
            assert_identical_model_match_ids(manifest, ids[:-1], ids)

    def test_empty_possession_cohort_fails_explicitly(self) -> None:
        canonical = synthetic_canonical(8)
        with self.assertRaisesRegex(
            MatchedExperimentError, "Possession-complete model dataset is empty"
        ):
            build_possession_complete_dataset(
                canonical,
                build_team_history_frame(canonical),
                model_b_period_start="1011",
            )


class ModelBAcceptanceRuleTests(unittest.TestCase):
    """Verify all three declared gates are required and correctly directed."""

    def test_model_b_passes_only_when_every_rule_passes(self) -> None:
        result = apply_model_b_acceptance_rule(
            model_a_matched_log_loss=1.02,
            model_b_log_loss=1.01,
            model_a_matched_macro_f1=0.50,
            model_b_macro_f1=0.48,
            max_macro_f1_drop=0.02,
            integrity_and_probabilities=True,
        )
        self.assertEqual(result, (True, True, True))

        integrity_failure = apply_model_b_acceptance_rule(
            model_a_matched_log_loss=1.02,
            model_b_log_loss=1.01,
            model_a_matched_macro_f1=0.50,
            model_b_macro_f1=0.49,
            max_macro_f1_drop=0.02,
            integrity_and_probabilities=False,
        )
        self.assertEqual(integrity_failure, (True, True, False))

    def test_macro_f1_drop_beyond_guardrail_fails(self) -> None:
        result = apply_model_b_acceptance_rule(
            model_a_matched_log_loss=1.02,
            model_b_log_loss=1.01,
            model_a_matched_macro_f1=0.50,
            model_b_macro_f1=0.479,
            max_macro_f1_drop=0.02,
            integrity_and_probabilities=True,
        )
        self.assertEqual(result, (True, False, False))

    def test_invalid_guardrail_is_rejected(self) -> None:
        with self.assertRaises(ModelComparisonError):
            apply_model_b_acceptance_rule(
                model_a_matched_log_loss=1.02,
                model_b_log_loss=1.01,
                model_a_matched_macro_f1=0.50,
                model_b_macro_f1=0.50,
                max_macro_f1_drop=-0.01,
                integrity_and_probabilities=True,
            )

    def test_comparison_report_states_each_rule_without_causal_claim(self) -> None:
        rows = {
            (model_name, split_name): pd.Series(
                {
                    "row_count": 100,
                    "log_loss": 1.0,
                    "macro_f1": 0.5,
                    "accuracy": 0.55,
                }
            )
            for model_name in ("model_a_matched", "model_b")
            for split_name in ("validation", "test")
        }
        report = _build_report(
            rows,
            lower_log_loss=True,
            macro_guardrail=False,
            integrity=True,
            accepted=False,
            max_macro_f1_drop=0.02,
        )
        self.assertIn("PASS — Test log loss", report)
        self.assertIn("FAIL — Test macro F1", report)
        self.assertIn("PASS — Frozen-row", report)
        self.assertIn("does not establish", report)
        self.assertIn("holdout remains unopened", report)

    def test_model_feature_set_pairing_is_fixed(self) -> None:
        with self.assertRaisesRegex(
            XGBoostTrainingError, "requires --feature-set possession"
        ):
            train_xgboost_model("model_b", "baseline")


if __name__ == "__main__":
    unittest.main()
