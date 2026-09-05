"""Phase 7 possession features, matched rows, and comparison-rule tests."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import unittest

import pandas as pd

from src.build_history import build_team_history_frame
from src.collect_possession import TEAM_SEASON_POSSESSION_COLUMNS
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
    build_lagged_possession_features,
    build_matched_split_manifest,
    build_possession_complete_dataset,
    feature_columns_for_set,
    split_matched_dataset,
)
from src.split_data import load_split_policy
from src.train_xgboost import XGBoostTrainingError, train_xgboost_model
from tests.test_leakage import synthetic_canonical


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def team_season_possession(
    *,
    include_beta: bool = True,
    include_future_source: bool = False,
) -> pd.DataFrame:
    """Return a valid synthetic Phase 6 team-season table."""
    rows = [
        {
            "source_season": "0910",
            "target_season": "1011",
            "source_season_start_year": 2009,
            "sofascore_season_id": 90,
            "team": "Alpha",
            "team_slug": "alpha",
            "sofascore_team_name": "Alpha FC",
            "sofascore_team_id": 1,
            "average_possession_pct": 57.0,
            "matches_recorded": 38,
            "source_url": "https://www.sofascore.com/alpha-0910",
        }
    ]
    if include_beta:
        rows.append(
            {
                "source_season": "0910",
                "target_season": "1011",
                "source_season_start_year": 2009,
                "sofascore_season_id": 90,
                "team": "Beta",
                "team_slug": "beta",
                "sofascore_team_name": "Beta FC",
                "sofascore_team_id": 2,
                "average_possession_pct": 43.0,
                "matches_recorded": 38,
                "source_url": "https://www.sofascore.com/beta-0910",
            }
        )
    if include_future_source:
        for team_id, team, value in ((1, "Alpha", 61.0), (2, "Beta", 39.0)):
            rows.append(
                {
                    "source_season": "1011",
                    "target_season": "1112",
                    "source_season_start_year": 2010,
                    "sofascore_season_id": 101,
                    "team": team,
                    "team_slug": team.lower(),
                    "sofascore_team_name": f"{team} FC",
                    "sofascore_team_id": team_id,
                    "average_possession_pct": value,
                    "matches_recorded": 38,
                    "source_url": f"https://www.sofascore.com/{team.lower()}-1011",
                }
            )
    return pd.DataFrame(rows, columns=TEAM_SEASON_POSSESSION_COLUMNS)


class LaggedPossessionFeatureTests(unittest.TestCase):
    """Prove every possession input comes from the preceding EPL season."""

    def setUp(self) -> None:
        self.canonical = synthetic_canonical(10)
        self.history = build_team_history_frame(self.canonical)
        self.possession = team_season_possession()

    def test_immediately_preceding_season_is_joined_by_team_slug(self) -> None:
        dataset = build_possession_complete_dataset(
            self.canonical,
            self.history,
            team_season_possession=self.possession,
            model_b_period_start="1011",
        )
        self.assertEqual(len(dataset), len(self.canonical))
        alpha_home = dataset["home_team"].eq("Alpha")
        self.assertTrue(
            dataset.loc[alpha_home, "home_previous_season_possession"].eq(57.0).all()
        )
        self.assertTrue(
            dataset.loc[alpha_home, "away_previous_season_possession"].eq(43.0).all()
        )
        self.assertTrue(dataset.loc[alpha_home, "possession_edge"].eq(14.0).all())
        self.assertTrue(dataset.loc[~alpha_home, "possession_edge"].eq(-14.0).all())

    def test_current_fixture_possession_cannot_change_lagged_features(self) -> None:
        baseline = build_possession_complete_dataset(
            self.canonical,
            self.history,
            team_season_possession=self.possession,
            model_b_period_start="1011",
        )
        mutated = deepcopy(self.canonical)
        mutated.loc[:, "home_possession"] = 99.0
        mutated.loc[:, "away_possession"] = 1.0
        mutated.loc[:, "api_fixture_id"] = range(1, len(mutated) + 1)
        rebuilt = build_possession_complete_dataset(
            mutated,
            build_team_history_frame(mutated),
            team_season_possession=self.possession,
            model_b_period_start="1011",
        )
        pd.testing.assert_frame_equal(
            baseline.loc[:, list(POSSESSION_FEATURE_ADDITIONS)],
            rebuilt.loc[:, list(POSSESSION_FEATURE_ADDITIONS)],
        )

    def test_same_season_final_average_cannot_change_target_features(self) -> None:
        possession = team_season_possession(include_future_source=True)
        baseline = build_possession_complete_dataset(
            self.canonical,
            self.history,
            team_season_possession=possession,
            model_b_period_start="1011",
        )
        mutated = possession.copy()
        mutated.loc[mutated["source_season"].eq("1011"), "average_possession_pct"] = 99.0
        rebuilt = build_possession_complete_dataset(
            self.canonical,
            self.history,
            team_season_possession=mutated,
            model_b_period_start="1011",
        )
        pd.testing.assert_frame_equal(
            baseline.loc[:, list(POSSESSION_FEATURE_ADDITIONS)],
            rebuilt.loc[:, list(POSSESSION_FEATURE_ADDITIONS)],
        )

    def test_promoted_team_missing_value_is_retained_for_training_imputation(self) -> None:
        dataset = build_possession_complete_dataset(
            self.canonical,
            self.history,
            team_season_possession=team_season_possession(include_beta=False),
            model_b_period_start="1011",
        )
        self.assertEqual(len(dataset), len(self.canonical))
        beta_home = dataset["home_team"].eq("Beta")
        self.assertTrue(
            dataset.loc[beta_home, "home_previous_season_possession"].isna().all()
        )
        self.assertTrue(
            dataset.loc[~beta_home, "away_previous_season_possession"].isna().all()
        )
        self.assertTrue(dataset["possession_edge"].isna().all())

    def test_missing_established_team_average_fails_instead_of_imputing(self) -> None:
        canonical = synthetic_canonical(2)
        canonical.loc[0, "season"] = "0910"
        canonical.loc[1, "season"] = "1011"
        with self.assertRaisesRegex(
            MatchedExperimentError, "established EPL team 'beta'"
        ):
            build_lagged_possession_features(
                canonical, team_season_possession(include_beta=False)
            )

    def test_same_season_source_target_mapping_is_rejected(self) -> None:
        invalid = self.possession.copy()
        invalid.loc[:, "source_season"] = "1011"
        with self.assertRaisesRegex(MatchedExperimentError, "source season N-1"):
            build_lagged_possession_features(self.canonical, invalid)

    def test_eligible_season_without_preceding_table_is_rejected(self) -> None:
        future_only = team_season_possession(include_future_source=True).loc[
            lambda frame: frame["source_season"].eq("1011")
        ].reset_index(drop=True)
        with self.assertRaisesRegex(
            MatchedExperimentError, r"eligible target seasons: \['1011'\]"
        ):
            build_possession_complete_dataset(
                self.canonical,
                self.history,
                team_season_possession=future_only,
                model_b_period_start="1011",
            )


class MatchedCohortContractTests(unittest.TestCase):
    """Verify identical row IDs and the exact three-column feature difference."""

    @classmethod
    def setUpClass(cls) -> None:
        canonical = synthetic_canonical(12)
        cls.dataset = build_possession_complete_dataset(
            canonical,
            build_team_history_frame(canonical),
            team_season_possession=team_season_possession(),
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

    def test_noncanonical_model_period_fails_explicitly(self) -> None:
        canonical = synthetic_canonical(8)
        with self.assertRaisesRegex(
            MatchedExperimentError, "period start '1112' is not canonical"
        ):
            build_possession_complete_dataset(
                canonical,
                build_team_history_frame(canonical),
                team_season_possession=team_season_possession(),
                model_b_period_start="1112",
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
        self.assertIn("possession-eligible fixture cohort", report)
        self.assertIn("does not establish", report)
        self.assertIn("holdout remains unopened", report)

    def test_model_feature_set_pairing_is_fixed(self) -> None:
        with self.assertRaisesRegex(
            XGBoostTrainingError, "requires --feature-set possession"
        ):
            train_xgboost_model("model_b", "baseline")


if __name__ == "__main__":
    unittest.main()
