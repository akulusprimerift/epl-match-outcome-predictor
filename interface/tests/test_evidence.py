"""Phase 12 additions; do not alter the frozen modeling test inventory."""

import json
import math
import unittest
from unittest.mock import Mock

import numpy as np
import pandas as pd

from interface.evidence import GROUPS, explain_match, model_attribution, team_evidence
from src.build_history import build_team_history_frame, read_canonical_matches
from src.constants import CLASS_NAMES, POSSESSION_FEATURE_COLUMNS
from src.freeze_model import PROJECT_ROOT


class RealEvidenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.result = explain_match("Arsenal", "Chelsea", "2026-09-12")

    def test_prediction_exactly_matches_frozen_sample(self):
        sample = json.loads((PROJECT_ROOT / "docs/sample_prediction.json").read_text())
        self.assertEqual(self.result["prediction"], sample)
        self.assertAlmostEqual(sum(sample["probabilities"].values()), 1, places=6)
        json.dumps(self.result, allow_nan=False)

    def test_real_statistics_and_provider_provenance(self):
        home, away = self.result["teams"].values()
        self.assertEqual((home["recent_points"], away["recent_points"]), (15, 4))
        self.assertEqual((home["goals_for_average"], away["goals_for_average"]), (1.6, 1.0))
        self.assertEqual((home["goals_against_average"], away["goals_against_average"]), (.2, 2.0))
        self.assertEqual((home["shots_average"], away["shots_average"]), (14.8, 10.4))
        self.assertEqual((home["possession"], away["possession"]), (56.1, 57.7))
        for team in (home, away):
            self.assertEqual(team["possession_source_season"], "2526")
            self.assertEqual(team["possession_matches"], 38)
            self.assertIn("unique-tournament/17/", team["possession_source_url"])
            self.assertEqual(len(team["recent_matches"]), 5)
            self.assertEqual(team["recent_points"], sum(m["points"] for m in team["recent_matches"]))
            self.assertEqual(team["goals_for_average"], sum(m["goals_for"] for m in team["recent_matches"]) / 5)
            self.assertEqual(team["shots_average"], sum(m["shots"] for m in team["recent_matches"]) / 5)
            self.assertEqual(team["venue_points_per_game"], sum(m["points"] for m in team["venue_matches"]) / 5)
            for match in team["recent_matches"] + team["venue_matches"]:
                self.assertLess(match["date"], "2026-09-12")
                self.assertIn("/2526/E0.csv", match["source_url"])
            self.assertTrue(all(m["venue"].lower() == team["role"] for m in team["venue_matches"]))

    def test_all_frozen_features_explained_once_and_score_reconstructed(self):
        attribution = self.result["attribution"]
        covered = [name for group in attribution["groups"] for name in group["features"]]
        self.assertCountEqual(covered, POSSESSION_FEATURE_COLUMNS)
        self.assertEqual(len(set(covered)), 25)
        self.assertEqual(attribution["leading_outcome"], "home_win")
        self.assertEqual(attribution["comparison_outcome"], "away_win")
        total = attribution["baseline_gap"] + sum(g["contribution"] for g in attribution["groups"])
        self.assertAlmostEqual(total, attribution["total_score_gap"], places=6)
        probabilities = self.result["prediction"]["probabilities"]
        self.assertAlmostEqual(total, math.log(probabilities["home_win"] / probabilities["away_win"]), places=5)
        self.assertTrue(any(g["direction"] == "supports" for g in attribution["groups"]))
        for group in attribution["groups"]:
            expected = "supports" if group["contribution"] > 0 else "opposes" if group["contribution"] < 0 else "neutral"
            self.assertEqual(group["direction"], expected)

    def test_missing_possession_is_not_presented_as_observed_data(self):
        result = explain_match("Luton Town", "Arsenal", "2026-09-12")
        self.assertIsNone(result["teams"]["home"]["possession"])
        row = next(s for s in result["statistics"] if s["group"] == "Previous-season possession")
        self.assertIsNone(row["home"])
        self.assertTrue(row["home_imputed"])
        self.assertIsInstance(row["home_model_input"], float)
        self.assertTrue(any("Luton Town" in warning for warning in result["prediction"]["warnings"]))

    def test_weak_leader_and_opposing_signals_are_disclosed(self):
        result = explain_match("Everton", "Manchester United", "2026-09-12")
        self.assertLess(max(result["prediction"]["probabilities"].values()), .5)
        self.assertIn("less likely than the other two outcomes combined", result["summary"])
        self.assertTrue(any(g["direction"] == "opposes" for g in result["attribution"]["groups"]))

    def test_invalid_fixtures_rejected(self):
        for home, away, day in (("Arsenal", "Arsenal", "2026-09-12"),
                               ("Barcelona", "Chelsea", "2026-09-12"),
                               ("Arsenal", "Chelsea", "2026-05-24"),
                               ("Arsenal", "Chelsea", "2026-02-30"),
                               ("Arsenal", "Chelsea", "2027-09-12")):
            with self.subTest(home=home, day=day), self.assertRaises((ValueError, RuntimeError)):
                explain_match(home, away, day)

    def test_empty_history_does_not_invent_observations(self):
        history = build_team_history_frame(read_canonical_matches(PROJECT_ROOT / "data/processed/canonical_matches.csv"))
        possession = pd.read_csv(PROJECT_ROOT / "data/processed/team_season_possession.csv", dtype={"source_season": str})
        result = team_evidence(history.iloc[:0], possession.iloc[:0], name="Arsenal", slug="arsenal", role="home", source_season="2526", sources={})
        self.assertEqual(result["history_count"], 0)
        self.assertEqual(result["recent_matches"], [])
        for key in ("recent_points", "shots_average", "goals_for_average", "possession", "venue_points_per_game"):
            self.assertIsNone(result[key])


class AttributionContractTests(unittest.TestCase):
    def fixture(self, probabilities):
        contributions = np.zeros((1, 3, 26))
        contributions[0, :, -1] = np.log(probabilities)
        model = Mock(classes_=np.array([0, 1, 2]))
        model.get_booster.return_value.predict.return_value = contributions
        frame = pd.DataFrame([[0.0] * 25], columns=POSSESSION_FEATURE_COLUMNS)
        return model, frame, dict(zip(CLASS_NAMES, probabilities))

    def test_draw_compares_with_strongest_win(self):
        model, frame, probabilities = self.fixture([.2, .5, .3])
        result = model_attribution(model, frame, probabilities, 267)
        self.assertEqual(result["leading_outcome"], "draw")
        self.assertEqual(result["comparison_outcome"], "home_win")
        kwargs = model.get_booster.return_value.predict.call_args.kwargs
        self.assertEqual(kwargs["iteration_range"], (0, 268))
        self.assertFalse(kwargs["approx_contribs"])

    def test_away_win_compares_with_home_win(self):
        model, frame, probabilities = self.fixture([.7, .2, .1])
        result = model_attribution(model, frame, probabilities, 267)
        self.assertEqual(result["comparison_outcome"], "home_win")

    def test_wrong_probability_reconstruction_fails_closed(self):
        model, frame, probabilities = self.fixture([.2, .5, .3])
        probabilities.update(away_win=.3, draw=.4)
        with self.assertRaisesRegex(RuntimeError, "reconstruct"):
            model_attribution(model, frame, probabilities, 267)

    def test_nonfinite_contribution_fails_closed(self):
        model, frame, probabilities = self.fixture([.2, .5, .3])
        model.get_booster.return_value.predict.return_value[0, 0, 0] = np.nan
        with self.assertRaisesRegex(RuntimeError, "valid per-class"):
            model_attribution(model, frame, probabilities, 267)


if __name__ == "__main__":
    unittest.main()
