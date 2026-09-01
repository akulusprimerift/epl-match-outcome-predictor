"""Phase 3 feature, history, and preprocessing contract tests."""

from pathlib import Path
import unittest

import pandas as pd

from src.build_features import (
    apply_training_medians,
    fit_training_medians,
    previous_match_ids,
    validate_model_dataset,
)
from src.build_history import (
    points_from_score,
    read_canonical_matches,
    read_team_history,
    validate_team_history,
)
from src.constants import (
    CURRENT_MATCH_STAT_COLUMNS,
    FEATURE_COLUMNS,
    MODEL_DATASET_COLUMNS,
    TARGET_MAPPING,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CANONICAL_PATH = PROJECT_ROOT / "data" / "processed" / "canonical_matches.csv"
HISTORY_PATH = PROJECT_ROOT / "data" / "processed" / "team_match_history.csv"
MODEL_PATH = PROJECT_ROOT / "data" / "processed" / "model_dataset.csv"


class TeamHistoryContractTests(unittest.TestCase):
    """Verify the persisted two-row team perspective table."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.canonical = read_canonical_matches(CANONICAL_PATH)
        cls.history = read_team_history(HISTORY_PATH)
        cls.model = pd.read_csv(MODEL_PATH, low_memory=False)

    def test_every_fixture_has_exactly_two_history_rows(self) -> None:
        validate_team_history(self.history, self.canonical)
        self.assertEqual(len(self.history), 2 * len(self.canonical))
        self.assertTrue(self.history.groupby("match_id").size().eq(2).all())

    def test_team_perspective_values_match_canonical_fixture(self) -> None:
        fixture = self.canonical.iloc[0]
        rows = self.history.loc[self.history["match_id"].eq(fixture["match_id"])]
        home = rows.loc[rows["is_home"]].iloc[0]
        away = rows.loc[~rows["is_home"]].iloc[0]

        self.assertEqual(home["team_slug"], fixture["home_team_slug"])
        self.assertEqual(home["opponent_slug"], fixture["away_team_slug"])
        self.assertEqual(home["goals_for"], fixture["home_goals"])
        self.assertEqual(home["goals_against"], fixture["away_goals"])
        self.assertEqual(home["shots"], fixture["home_shots"])
        self.assertEqual(away["team_slug"], fixture["away_team_slug"])
        self.assertEqual(away["goals_for"], fixture["away_goals"])
        self.assertEqual(away["goals_against"], fixture["home_goals"])

    def test_points_mapping(self) -> None:
        self.assertEqual(points_from_score(2, 1), 3)
        self.assertEqual(points_from_score(1, 1), 1)
        self.assertEqual(points_from_score(0, 1), 0)

    def test_known_window_contains_exact_previous_match_ids(self) -> None:
        current_match_id = "1011|2010-09-25|arsenal|west-bromwich-albion"
        expected = (
            "1011|2010-08-15|liverpool|arsenal",
            "1011|2010-08-21|arsenal|blackpool",
            "1011|2010-08-28|blackburn-rovers|arsenal",
            "1011|2010-09-11|arsenal|bolton-wanderers",
            "1011|2010-09-18|sunderland|arsenal",
        )
        current_date = self.canonical.loc[
            self.canonical["match_id"].eq(current_match_id), "date"
        ].item()
        self.assertEqual(
            previous_match_ids(self.history, "arsenal", current_date), expected
        )
        sources = self.history.loc[
            self.history["team_slug"].eq("arsenal")
            & self.history["match_id"].isin(expected)
        ]
        current = self.model.loc[self.model["match_id"].eq(current_match_id)].iloc[0]
        self.assertAlmostEqual(
            current["home_goals_for_avg_5"], sources["goals_for"].mean()
        )
        self.assertAlmostEqual(
            current["home_goals_against_avg_5"], sources["goals_against"].mean()
        )
        self.assertAlmostEqual(current["home_shots_avg_5"], sources["shots"].mean())
        self.assertAlmostEqual(current["home_form_points_5"], sources["points"].sum())
        self.assertAlmostEqual(current["home_overall_ppg_5"], sources["points"].mean())


class ModelDatasetContractTests(unittest.TestCase):
    """Verify the persisted one-row baseline feature dataset."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.canonical = read_canonical_matches(CANONICAL_PATH)
        cls.model = pd.read_csv(
            MODEL_PATH,
            dtype={
                "match_id": "string",
                "season": "string",
                "date": "string",
                "home_team": "string",
                "away_team": "string",
            },
            low_memory=False,
        )

    def test_one_model_row_per_canonical_fixture(self) -> None:
        self.assertEqual(tuple(self.model.columns), MODEL_DATASET_COLUMNS)
        validate_model_dataset(self.model, self.canonical)
        self.assertEqual(len(self.model), len(self.canonical))
        self.assertTrue(self.model["match_id"].is_unique)

    def test_target_mapping_is_fixed(self) -> None:
        expected = self.canonical.set_index("match_id")["result_code"].map(
            TARGET_MAPPING
        )
        actual = self.model.set_index("match_id")["target"]
        pd.testing.assert_series_equal(
            actual.sort_index(),
            expected.astype("int64").sort_index(),
            check_names=False,
        )
        self.assertEqual(TARGET_MAPPING, {"A": 0, "D": 1, "H": 2})

    def test_feature_contract_excludes_current_match_statistics(self) -> None:
        self.assertFalse(set(FEATURE_COLUMNS).intersection(CURRENT_MATCH_STAT_COLUMNS))
        self.assertNotIn("home_advantage", FEATURE_COLUMNS)
        self.assertFalse(any("possession" in name for name in FEATURE_COLUMNS))

    def test_edges_use_home_positive_direction(self) -> None:
        expected = {
            "goals_scored_edge": self.model["home_goals_for_avg_5"]
            - self.model["away_goals_for_avg_5"],
            "defensive_edge": self.model["away_goals_against_avg_5"]
            - self.model["home_goals_against_avg_5"],
            "shots_edge": self.model["home_shots_avg_5"]
            - self.model["away_shots_avg_5"],
            "form_edge": self.model["home_form_points_5"]
            - self.model["away_form_points_5"],
            "venue_edge": self.model["home_venue_ppg_5"]
            - self.model["away_venue_ppg_5"],
            "history_edge": self.model["home_history_matches"]
            - self.model["away_history_matches"],
        }
        for name, values in expected.items():
            with self.subTest(edge=name):
                pd.testing.assert_series_equal(
                    self.model[name], values, check_names=False
                )

    def test_cold_start_rows_are_retained_with_counts(self) -> None:
        first = self.model.iloc[0]
        self.assertEqual(first["home_history_matches"], 0)
        self.assertEqual(first["away_history_matches"], 0)
        self.assertEqual(first["home_venue_history_matches"], 0)
        self.assertEqual(first["away_venue_history_matches"], 0)
        self.assertTrue(pd.isna(first["home_goals_for_avg_5"]))
        self.assertTrue(pd.isna(first["away_goals_for_avg_5"]))


class TrainingOnlyImputationTests(unittest.TestCase):
    """Verify reusable imputation learns only from an explicit training frame."""

    def test_fit_and_apply_training_medians(self) -> None:
        training = pd.DataFrame(
            {"feature_a": [1.0, pd.NA, 3.0], "feature_b": [2.0, 4.0, pd.NA]}
        )
        medians = fit_training_medians(
            training, feature_columns=("feature_a", "feature_b")
        )
        future = pd.DataFrame(
            {"feature_a": [pd.NA, 1_000.0], "feature_b": [pd.NA, -1_000.0]}
        )
        transformed = apply_training_medians(
            future,
            medians,
            feature_columns=("feature_a", "feature_b"),
        )

        self.assertEqual(medians, {"feature_a": 2.0, "feature_b": 3.0})
        self.assertEqual(transformed.loc[0, "feature_a"], 2.0)
        self.assertEqual(transformed.loc[0, "feature_b"], 3.0)
        self.assertEqual(transformed.loc[1, "feature_a"], 1_000.0)
        self.assertEqual(transformed.loc[1, "feature_b"], -1_000.0)


if __name__ == "__main__":
    unittest.main()
