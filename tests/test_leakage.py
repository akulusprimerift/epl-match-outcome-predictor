"""Phase 3 temporal leakage mutation and same-date tests."""

from copy import deepcopy
from datetime import date, timedelta
import unittest

import pandas as pd

from src.build_features import (
    build_model_dataset_frame,
    compute_team_rolling_features,
    previous_match_ids,
)
from src.build_history import build_team_history_frame
from src.clean_data import CANONICAL_MATCH_COLUMNS, expected_result_code, generate_match_id
from src.constants import FEATURE_COLUMNS


def synthetic_canonical(match_count: int = 8) -> pd.DataFrame:
    """Create a chronological, contract-valid sequence of EPL-only fixtures."""
    rows = []
    first_date = date(2010, 8, 1)
    for index in range(match_count):
        match_date = first_date + timedelta(days=7 * index)
        if index % 2 == 0:
            home_team, away_team = "Alpha", "Beta"
        else:
            home_team, away_team = "Beta", "Alpha"
        home_slug, away_slug = home_team.lower(), away_team.lower()
        home_goals = (index + 1) % 4
        away_goals = index % 3
        rows.append(
            {
                "match_id": generate_match_id(
                    "1011", match_date, home_slug, away_slug
                ),
                "season": "1011",
                "date": match_date.isoformat(),
                "kickoff_time": pd.NA,
                "home_team": home_team,
                "away_team": away_team,
                "home_team_slug": home_slug,
                "away_team_slug": away_slug,
                "home_goals": home_goals,
                "away_goals": away_goals,
                "result_code": expected_result_code(home_goals, away_goals),
                "home_shots": 8 + index,
                "away_shots": 6 + index,
                "home_possession": pd.NA,
                "away_possession": pd.NA,
                "football_data_source_file": "data/raw/football_data/E0_1011.csv",
                "api_fixture_id": pd.NA,
            }
        )
    return pd.DataFrame(rows, columns=CANONICAL_MATCH_COLUMNS)


def mutate_score(canonical: pd.DataFrame, position: int) -> pd.DataFrame:
    """Return a valid copy with one completed result changed."""
    mutated = deepcopy(canonical)
    home_goals = int(mutated.loc[position, "home_goals"])
    away_goals = int(mutated.loc[position, "away_goals"])
    if home_goals > away_goals:
        home_goals, away_goals = 0, 1
    else:
        home_goals, away_goals = 2, 0
    mutated.loc[position, "home_goals"] = home_goals
    mutated.loc[position, "away_goals"] = away_goals
    mutated.loc[position, "result_code"] = expected_result_code(
        home_goals, away_goals
    )
    return mutated


class TemporalLeakageTests(unittest.TestCase):
    """Prove current and future results cannot influence earlier features."""

    def setUp(self) -> None:
        self.canonical = synthetic_canonical()

    def test_current_result_mutation_does_not_change_current_features(self) -> None:
        position = 5
        match_id = self.canonical.loc[position, "match_id"]
        baseline = build_model_dataset_frame(self.canonical).set_index("match_id")
        mutated = build_model_dataset_frame(
            mutate_score(self.canonical, position)
        ).set_index("match_id")
        pd.testing.assert_series_equal(
            baseline.loc[match_id, list(FEATURE_COLUMNS)],
            mutated.loc[match_id, list(FEATURE_COLUMNS)],
            check_names=False,
        )

    def test_future_result_mutation_does_not_change_earlier_features(self) -> None:
        position = len(self.canonical) - 1
        future_date = self.canonical.loc[position, "date"]
        baseline = build_model_dataset_frame(self.canonical).set_index("match_id")
        mutated = build_model_dataset_frame(
            mutate_score(self.canonical, position)
        ).set_index("match_id")
        earlier_ids = self.canonical.loc[
            self.canonical["date"].lt(future_date), "match_id"
        ]
        pd.testing.assert_frame_equal(
            baseline.loc[earlier_ids, list(FEATURE_COLUMNS)],
            mutated.loc[earlier_ids, list(FEATURE_COLUMNS)],
        )

    def test_window_membership_is_exact_and_strictly_prior(self) -> None:
        history = build_team_history_frame(self.canonical)
        current_position = 7
        current_date = self.canonical.loc[current_position, "date"]
        expected = tuple(self.canonical.loc[2:6, "match_id"])
        self.assertEqual(
            previous_match_ids(history, "alpha", current_date), expected
        )
        source_dates = self.canonical.set_index("match_id").loc[
            list(expected), "date"
        ]
        self.assertTrue(source_dates.lt(current_date).all())

    def test_same_date_matches_are_contemporaneous(self) -> None:
        canonical = synthetic_canonical(6)
        same_date = canonical.loc[4, "date"]
        canonical.loc[5, "date"] = same_date
        canonical.loc[5, "match_id"] = generate_match_id(
            canonical.loc[5, "season"],
            same_date,
            canonical.loc[5, "home_team_slug"],
            canonical.loc[5, "away_team_slug"],
        )
        history = build_team_history_frame(canonical)
        features = compute_team_rolling_features(history)
        alpha = features.loc[
            features["team_slug"].eq("alpha") & features["date"].eq(same_date)
        ].sort_values("match_id")

        self.assertEqual(len(alpha), 2)
        self.assertEqual(alpha["history_matches"].tolist(), [4, 4])
        for column in (
            "goals_for_avg_5",
            "goals_against_avg_5",
            "shots_avg_5",
            "form_points_5",
            "overall_ppg_5",
        ):
            with self.subTest(feature=column):
                self.assertEqual(alpha[column].nunique(dropna=False), 1)
        self.assertEqual(
            previous_match_ids(history, "alpha", same_date),
            tuple(canonical.loc[0:3, "match_id"]),
        )


if __name__ == "__main__":
    unittest.main()
