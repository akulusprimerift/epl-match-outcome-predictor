"""Phase 8 decision, artifact mutation, and holdout boundary checks."""

from copy import deepcopy
import json
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

from src.freeze_model import (
    PROJECT_ROOT, FreezeError, choose_candidate, prediction_schema,
    read_json, record_hash, test_results, validate_candidates, verify_freeze,
)


class SelectionTests(unittest.TestCase):
    def setUp(self):
        self.results = test_results(PROJECT_ROOT)

    def test_current_results_select_b(self):
        self.assertEqual(choose_candidate(self.results, 0.02), "model_b")

    def test_lower_loss_cannot_override_f1_guardrail(self):
        self.results["model_b"].update(log_loss=0.8, macro_f1=0.1)
        self.assertEqual(choose_candidate(self.results, 0.02), "model_a")

    def test_b_requires_incremental_improvement(self):
        self.results["model_b"]["log_loss"] = 1.2
        self.assertEqual(choose_candidate(self.results, 0.02), "model_a")

    def test_invalid_guardrail_fails(self):
        for invalid in (-1, 2, float("nan")):
            with self.subTest(invalid=invalid), self.assertRaises(FreezeError):
                choose_candidate(self.results, invalid)

    def test_schema_preserves_specified_output_and_three_classes(self):
        schema = prediction_schema()
        self.assertEqual(set(schema["required"]), {
            "home_team", "away_team", "match_date", "model_name", "model_version",
            "probabilities", "predicted_outcome", "feature_as_of", "warnings",
        })
        self.assertEqual(schema["properties"]["probabilities"]["required"],
                         ["away_win", "draw", "home_win"])


class FrozenRepositoryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = read_json(PROJECT_ROOT / "config/model_config.json")
        if not cls.config.get("frozen_at_utc"):
            raise unittest.SkipTest("Create the Phase 8 freeze before repository freeze tests.")

    def test_freeze_and_saved_test_metrics_are_valid(self):
        verified = verify_freeze()
        validate_candidates(PROJECT_ROOT, test_results(PROJECT_ROOT))
        frozen = verified["frozen_candidate"]
        self.assertEqual(frozen["model_name"], "model_b")
        self.assertEqual(len(frozen["feature_columns"]), 25)
        self.assertEqual(frozen["iteration_range"], [0, 268])
        self.assertEqual(frozen["preprocessing"]["fitted_on_split"], "train")
        self.assertFalse(frozen["holdout_evaluated"])

    def test_verify_is_read_only_and_never_runs_inference(self):
        before = (PROJECT_ROOT / "config/model_config.json").read_bytes()
        with patch("src.evaluate.evaluate_saved_model", side_effect=AssertionError("inference forbidden")):
            verify_freeze()
        self.assertEqual(before, (PROJECT_ROOT / "config/model_config.json").read_bytes())

    def test_all_training_entry_points_stop_before_overwriting_freeze(self):
        from src.train_xgboost import XGBoostTrainingError, train_xgboost_model
        from src.train_baselines import BaselineError, train_and_evaluate_baselines
        for name, features in (("model_a", "baseline"), ("model_a_matched", "baseline_matched"), ("model_b", "possession")):
            with self.subTest(name=name), self.assertRaisesRegex(XGBoostTrainingError, "Phase 8 is frozen"):
                train_xgboost_model(name, features)
        with self.assertRaisesRegex(BaselineError, "Phase 8 is frozen"):
            train_and_evaluate_baselines("baseline")

    def test_holdout_evaluation_remains_unavailable(self):
        from src.evaluate import EvaluationError, evaluate_saved_model
        for name in ("model_a", "model_a_matched", "model_b"):
            with self.subTest(name=name), self.assertRaisesRegex(EvaluationError, "holdout"):
                evaluate_saved_model(name, "holdout")

    def test_changed_config_implementation_and_model_are_rejected(self):
        # Mutate copies only; the frozen repository and raw sources stay immutable.
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            frozen = self.config["frozen_candidate"]
            paths = set(frozen["artifacts_sha256"]) | set(frozen["implementation_files_sha256"])
            paths.add("config/model_config.json")
            from src.holdout import ADDITIONS, PROTOCOL, git_text
            paths.update(ADDITIONS)
            paths.add(PROTOCOL)
            paths.update({"config/phase10_protocol.json", "tests/test_predict.py"})
            for relative in paths:
                destination = root / relative
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(PROJECT_ROOT / relative, destination)
            git_patch = patch("src.holdout.git_text", side_effect=lambda _root, *args: git_text(PROJECT_ROOT, *args))
            git_patch.start()
            self.addCleanup(git_patch.stop)
            verify_freeze(root)
            for relative, message in (
                ("config/model_config.json", "configuration checksum"),
                ("src/build_features.py", "implementation checksum"),
                ("models/model_b_xgb.json", "artifact checksum"),
                ("models/model_b_preprocessing.json", "artifact checksum"),
            ):
                with self.subTest(relative=relative):
                    path = root / relative
                    original = path.read_bytes()
                    if relative == "config/model_config.json":
                        changed = deepcopy(self.config)
                        changed["frozen_candidate"]["feature_columns"].reverse()
                        path.write_text(json.dumps(changed), encoding="utf-8")
                    else:
                        path.write_bytes(original + b"\nchanged")
                    with self.assertRaisesRegex(FreezeError, message):
                        verify_freeze(root)
                    path.write_bytes(original)

    def test_configuration_checksum_covers_preprocessing_and_schema(self):
        for key in ("preprocessing", "prediction_schema", "parameters"):
            changed = deepcopy(self.config)
            changed["frozen_candidate"][key] = {}
            self.assertNotEqual(record_hash(changed), self.config["freeze_record_sha256"])


if __name__ == "__main__":
    unittest.main()
