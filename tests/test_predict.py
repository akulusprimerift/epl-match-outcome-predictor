"""Phase 10 feature parity, temporal boundaries, frozen inference and CLI tests."""

from contextlib import redirect_stderr, redirect_stdout
from datetime import date
from io import StringIO
import json
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from src.build_features import apply_training_medians, previous_match_ids
from src.build_history import read_canonical_matches, read_team_history
from src.constants import POSSESSION_FEATURE_COLUMNS
from src.freeze_model import PROJECT_ROOT, FreezeError, file_hash, read_json, verify_freeze
from src.predict import PredictionError, build_upcoming_features, canonical_teams, main, parse_prediction_date, predict_fixture, verify_prediction_extension


class PredictionFeatureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.canonical = read_canonical_matches(PROJECT_ROOT / "data/processed/canonical_matches.csv")
        cls.possession = pd.read_csv(PROJECT_ROOT / "data/processed/team_season_possession.csv", dtype={"source_season": str, "target_season": str})
        cls.teams = canonical_teams(PROJECT_ROOT)

    def build(self, *, canonical=None, possession=None, home="Arsenal", away="Chelsea", when="2026-09-12"):
        return build_upcoming_features(
            self.canonical if canonical is None else canonical,
            self.possession if possession is None else possession,
            home=home, away=away, home_slug=self.teams.get(home, "new-epl-club"), away_slug=self.teams[away],
            match_date=date.fromisoformat(when),
        )

    def test_exact_parity_with_frozen_training_feature_builder(self):
        # Feature-only comparisons on validation/test fixtures, never holdout inference.
        model = pd.read_csv(PROJECT_ROOT / "data/processed/matched_model_dataset.csv", dtype={"season": str})
        for index in (0, 25, 140, 379):
            fixture = model.loc[model.season.eq("2425")].iloc[index]
            with self.subTest(match_id=fixture.match_id):
                frame, as_of, _ = self.build(home=fixture.home_team, away=fixture.away_team, when=fixture.date)
                np.testing.assert_allclose(frame.iloc[0].to_numpy(dtype=float), fixture.loc[list(POSSESSION_FEATURE_COLUMNS)].to_numpy(dtype=float), rtol=0, atol=1e-12, equal_nan=True)
                self.assertEqual(tuple(frame.columns), POSSESSION_FEATURE_COLUMNS)
                self.assertLess(as_of, fixture.date)
                self.assertNotIn("target", frame)

    def test_same_day_and_future_results_cannot_affect_features(self):
        before = self.build(when="2024-10-05")[0]
        changed = self.canonical.copy()
        mask = changed.date.ge("2024-10-05")
        # Deliberately invalid future results prove they are excluded before feature validation.
        changed.loc[mask, ["home_goals", "away_goals", "home_shots", "away_shots"]] = 999
        changed.loc[mask, "result_code"] = "unavailable"
        after = self.build(canonical=changed, when="2024-10-05")[0]
        pd.testing.assert_frame_equal(before, after)

    def test_exact_prior_windows_and_venue_roles(self):
        frame, _, _ = self.build()
        history = read_team_history(PROJECT_ROOT / "data/processed/team_match_history.csv")
        ids = previous_match_ids(history, "arsenal", "2026-09-12")
        rows = history.loc[history.team_slug.eq("arsenal") & history.match_id.isin(ids)]
        self.assertEqual(len(rows), 5)
        self.assertAlmostEqual(frame.iloc[0].home_goals_for_avg_5, rows.goals_for.mean())
        venue_ids = previous_match_ids(history, "chelsea", "2026-09-12", is_home=False)
        venue = history.loc[history.team_slug.eq("chelsea") & history.match_id.isin(venue_ids)]
        self.assertAlmostEqual(frame.iloc[0].away_venue_ppg_5, venue.points.mean())

    def test_possession_uses_immediately_previous_season(self):
        frame, _, _ = self.build()
        expected = self.possession.loc[self.possession.source_season.eq("2526") & self.possession.team_slug.eq("arsenal"), "average_possession_pct"].item()
        self.assertEqual(frame.iloc[0].home_previous_season_possession, expected)
        other = self.possession.copy()
        other.loc[~other.source_season.eq("2526"), "average_possession_pct"] = 99
        pd.testing.assert_frame_equal(frame, self.build(possession=other)[0])

    def test_no_history_uses_zero_counts_and_saved_training_medians(self):
        frame, _, warnings = self.build(home="New EPL Club")
        row = frame.iloc[0]
        self.assertEqual(row.home_history_matches, 0)
        self.assertEqual(row.home_venue_history_matches, 0)
        self.assertTrue(pd.isna(row.home_goals_for_avg_5))
        self.assertTrue(pd.isna(row.home_previous_season_possession))
        self.assertTrue(any("Cold start" in warning for warning in warnings))
        medians = read_json(PROJECT_ROOT / "config/model_config.json")["frozen_candidate"]["preprocessing"]["median_values"]
        filled = apply_training_medians(frame, medians, POSSESSION_FEATURE_COLUMNS)
        self.assertEqual(filled.iloc[0].home_goals_for_avg_5, medians["home_goals_for_avg_5"])
        self.assertEqual(filled.iloc[0].home_history_matches, 0)

    def test_missing_possession_does_not_fall_back_to_older_season(self):
        frame, _, warnings = self.build(home="Luton Town")
        self.assertTrue(pd.isna(frame.iloc[0].home_previous_season_possession))
        self.assertTrue(any("Missing previous-season" in warning for warning in warnings))

    def test_staleness_is_reported(self):
        _, as_of, warnings = self.build()
        self.assertEqual(as_of, str(self.canonical.date.max()))
        self.assertTrue(any("Stale history for Arsenal" in warning for warning in warnings))

    def test_unsupported_future_season_fails_instead_of_imputing_whole_table(self):
        with self.assertRaisesRegex(PredictionError, "cannot support target season"):
            self.build(when="2027-09-12")

    def test_incomplete_possession_table_and_low_coverage_fail(self):
        for kind in ("missing_row", "partial_season", "low_coverage"):
            table = self.possession.copy()
            indices = table.index[table.source_season.eq("2526")]
            if kind == "missing_row":
                table = table.drop(indices[0])
            elif kind == "partial_season":
                table.loc[indices[0], "matches_recorded"] = 37
            else:
                table.loc[indices[:2], "average_possession_pct"] = np.nan
            with self.subTest(kind=kind), self.assertRaises(PredictionError):
                self.build(possession=table)


class PredictionCommandTests(unittest.TestCase):
    def test_strict_date_format(self):
        for value in ("2026-2-01", "2026-02-30", "2026-09-12T12:00:00", "tomorrow"):
            with self.subTest(value=value), self.assertRaises(PredictionError):
                parse_prediction_date(value)

    def test_unknown_alias_same_team_and_historical_dates_fail(self):
        for home, away, when, message in (
            ("Barcelona", "Chelsea", "2026-09-12", "Unknown canonical"),
            ("Man United", "Chelsea", "2026-09-12", "Unknown canonical"),
            ("Arsenal", "Arsenal", "2026-09-12", "different"),
            ("Arsenal", "Chelsea", "2025-09-12", "after the latest"),
        ):
            with self.subTest(home=home, when=when), self.assertRaisesRegex(PredictionError, message):
                predict_fixture(home, away, when)

    def test_predictions_are_deterministic_normalized_and_read_only(self):
        config = verify_freeze()
        protected = ["config/model_config.json", *config["frozen_candidate"]["artifacts_sha256"],
                     "config/phase9_protocol.json", "reports/final_holdout_receipt.json"]
        before = {path: file_hash(PROJECT_ROOT / path) for path in protected}
        with patch("xgboost.XGBClassifier.fit", side_effect=AssertionError("refitting forbidden")), \
             patch("src.build_features.fit_training_medians", side_effect=AssertionError("refitting forbidden")):
            first = predict_fixture("Arsenal", "Chelsea", "2026-09-12")
            second = predict_fixture("Arsenal", "Chelsea", "2026-09-12")
        self.assertEqual(first, second)
        self.assertEqual(set(first), set(config["frozen_candidate"]["prediction_schema"]["required"]))
        values = list(first["probabilities"].values())
        self.assertEqual(list(first["probabilities"]), ["away_win", "draw", "home_win"])
        self.assertTrue(all(np.isfinite(value) and 0 <= value <= 1 for value in values))
        self.assertAlmostEqual(sum(values), 1, places=6)
        self.assertEqual(first["predicted_outcome"], max(first["probabilities"], key=first["probabilities"].get))
        self.assertEqual(before, {path: file_hash(PROJECT_ROOT / path) for path in protected})

    def test_integrity_failure_blocks_prediction(self):
        with patch("src.predict.verify_freeze", side_effect=FreezeError("changed artifact")), \
             patch("src.evaluate.load_xgboost_model") as load:
            with self.assertRaisesRegex(FreezeError, "changed artifact"):
                predict_fixture("Arsenal", "Chelsea", "2026-09-12")
            load.assert_not_called()

    def test_cli_emits_only_json_on_stdout_and_clear_failure_on_stderr(self):
        out, err = StringIO(), StringIO()
        with redirect_stdout(out), redirect_stderr(err):
            code = main(["--home", "Arsenal", "--away", "Chelsea", "--date", "2026-09-12"])
        self.assertEqual(code, 0)
        self.assertEqual(json.loads(out.getvalue())["model_name"], "model_b")
        self.assertEqual(err.getvalue(), "")
        out, err = StringIO(), StringIO()
        with redirect_stdout(out), redirect_stderr(err):
            code = main(["--home", "Unknown", "--away", "Chelsea", "--date", "2026-09-12"])
        self.assertEqual(code, 1)
        self.assertEqual(out.getvalue(), "")
        self.assertIn("Unknown canonical", err.getvalue())

    def test_resealed_prediction_protocol_cannot_modify_feature_builder(self):
        config = read_json(PROJECT_ROOT / "config/model_config.json")
        protocol = read_json(PROJECT_ROOT / "config/phase10_protocol.json")
        phase9 = read_json(PROJECT_ROOT / "config/phase9_protocol.json")
        actual = dict(protocol["implementation_files_sha256"])
        actual["src/build_features.py"] = "0" * 64
        protocol["implementation_files_sha256"] = actual
        with patch("src.predict.read_json", return_value=protocol):
            with self.assertRaisesRegex(FreezeError, "implementation checksum"):
                verify_prediction_extension(PROJECT_ROOT, config, actual, phase9)


if __name__ == "__main__":
    unittest.main()
