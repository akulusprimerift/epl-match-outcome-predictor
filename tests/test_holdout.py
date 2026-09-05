"""Phase 9 checks use synthetic outcomes only; never rerun real holdout inference."""

from contextlib import ExitStack
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd

from src.constants import CLASS_NAMES, MODEL_RESULT_COLUMNS, POSSESSION_FEATURE_COLUMNS
from src.evaluate import main
from src.freeze_model import PROJECT_ROOT, FreezeError, read_json, verify_freeze
from src.holdout import (
    FIGURE, OUTPUTS, PREDICTIONS, PROTOCOL, RECEIPT, RESULTS, START,
    evaluate_final_holdout, holdout_metrics, load_frozen_inputs, preflight_commit,
    protected_ast, verify_evaluation_extension, verify_holdout_receipt,
)
from src.train_baselines import BaselineError, evaluate_probabilities


class HoldoutMetricTests(unittest.TestCase):
    def setUp(self):
        self.frame = pd.DataFrame({"target": [0, 1, 2, 2]})
        self.probabilities = np.array([[.7, .2, .1], [.3, .4, .3], [.1, .2, .7], [.5, .3, .2]])
        self.frozen = dict(model_name="model_b", feature_set="possession", best_iteration=267, parameters={})

    def test_metric_definitions_match_existing_evaluator(self):
        expected = evaluate_probabilities(
            model_name="model_b", model_family="xgboost", feature_set="possession",
            split_name="test", frame=self.frame, probabilities=self.probabilities, parameters={},
        )
        expected.update(split="holdout", best_iteration=267)
        actual = holdout_metrics(self.frame, self.probabilities, self.frozen)
        self.assertEqual(actual, expected)
        self.assertEqual(tuple(actual), MODEL_RESULT_COLUMNS)

    def test_bad_probabilities_fail(self):
        for values in (np.ones((4, 3)), np.ones((4, 2)), np.full((4, 3), np.nan),
                       np.array([[-.1, .5, .6]] * 4), self.probabilities[:3]):
            with self.subTest(shape=values.shape), self.assertRaises(BaselineError):
                holdout_metrics(self.frame, values, self.frozen)

    def test_bad_labels_fail(self):
        self.frame.loc[0, "target"] = 3
        with self.assertRaisesRegex(FreezeError, "target"):
            holdout_metrics(self.frame, self.probabilities, self.frozen)


class SyntheticOneTimeEvaluationTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        (self.root / "reports").mkdir()
        (self.root / "config").mkdir()
        self.config = read_json(PROJECT_ROOT / "config/model_config.json")
        self.protocol = {"freeze_commit": "0" * 40}
        (self.root / PROTOCOL).write_text(json.dumps(self.protocol), encoding="utf-8")
        self.frame = pd.DataFrame({column: [np.nan, 1., 2.] for column in POSSESSION_FEATURE_COLUMNS})
        self.frame["match_id"] = ["synthetic-away", "synthetic-draw", "synthetic-home"]
        self.frame["target"] = [0, 1, 2]
        self.model = Mock()
        self.model.classes_ = np.array([0, 1, 2])
        self.model.predict_proba.return_value = np.array([[.7, .2, .1], [.2, .6, .2], [.1, .2, .7]])
        self.model.fit.side_effect = AssertionError("Model fit forbidden")
        self.medians = self.config["frozen_candidate"]["preprocessing"]["median_values"]
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.stack.enter_context(patch("src.holdout.verify_freeze", return_value=self.config))
        self.stack.enter_context(patch("src.holdout.preflight_commit", return_value="1" * 40))
        self.stack.enter_context(patch("src.holdout.git_text", return_value=json.dumps(self.protocol)))
        self.inputs = self.stack.enter_context(patch("src.holdout.load_frozen_inputs", return_value=(self.model, self.frame, self.medians)))
        self.stack.enter_context(patch("src.build_features.fit_training_medians", side_effect=AssertionError("Imputer fit forbidden")))

    def test_single_prediction_saved_outputs_and_idempotent_repeat(self):
        result = evaluate_final_holdout(self.root)
        self.assertFalse(result["reused_saved_results"])
        self.assertTrue(result["holdout_evaluated"])
        self.assertEqual(result["row_count"], 3)
        self.model.fit.assert_not_called()
        self.model.predict_proba.assert_called_once()
        kwargs = self.model.predict_proba.call_args.kwargs
        self.assertEqual(kwargs["iteration_range"], (0, 268))
        features = self.model.predict_proba.call_args.args[0]
        self.assertEqual(list(features), list(POSSESSION_FEATURE_COLUMNS))
        self.assertEqual(features.iloc[0].to_dict(), self.medians)
        self.assertEqual(tuple(pd.read_csv(self.root / RESULTS).columns), MODEL_RESULT_COLUMNS)
        predictions = read_json(self.root / PREDICTIONS)
        self.assertEqual(predictions["probability_columns"], list(CLASS_NAMES))
        self.assertEqual(len(predictions["predictions"]), 3)
        self.assertTrue((self.root / FIGURE).read_bytes().startswith(b"\x89PNG"))
        before = {path: (self.root / path).read_bytes() for path in (*OUTPUTS, START, RECEIPT)}
        again = evaluate_final_holdout(self.root)
        self.assertTrue(again["reused_saved_results"])
        self.assertEqual(again["log_loss"], result["log_loss"])
        self.inputs.assert_called_once()
        self.model.predict_proba.assert_called_once()
        self.assertEqual(before, {path: (self.root / path).read_bytes() for path in before})

    def test_failed_inference_stays_locked(self):
        self.model.predict_proba.side_effect = ValueError("synthetic interruption")
        with self.assertRaisesRegex(RuntimeError, "synthetic interruption"):
            evaluate_final_holdout(self.root)
        self.assertTrue((self.root / START).exists())
        self.assertFalse((self.root / RECEIPT).exists())
        with self.assertRaisesRegex(FreezeError, "incomplete"):
            evaluate_final_holdout(self.root)
        self.model.predict_proba.assert_called_once()

    def test_output_tampering_rejected_without_inference(self):
        evaluate_final_holdout(self.root)
        for relative in (*OUTPUTS, START, PROTOCOL):
            with self.subTest(relative=relative):
                path = self.root / relative
                original = path.read_bytes()
                path.write_bytes(original + b"\n")
                with self.assertRaises(FreezeError):
                    evaluate_final_holdout(self.root)
                path.write_bytes(original)
        self.model.predict_proba.assert_called_once()

    def test_orphan_output_rejected_before_loading_model(self):
        (self.root / RESULTS).write_text("orphan", encoding="utf-8")
        with self.assertRaisesRegex(FreezeError, "without its one-time"):
            evaluate_final_holdout(self.root)
        self.inputs.assert_not_called()

    def test_integrity_failure_prevents_inference(self):
        with patch("src.holdout.verify_freeze", side_effect=FreezeError("changed model")):
            with self.assertRaisesRegex(FreezeError, "changed model"):
                evaluate_final_holdout(self.root)
        self.model.predict_proba.assert_not_called()
        self.assertFalse((self.root / START).exists())

    def test_missing_output_inventory_rejected(self):
        evaluate_final_holdout(self.root)
        receipt = read_json(self.root / RECEIPT)
        receipt["outputs_sha256"].pop(RESULTS)
        (self.root / RECEIPT).write_text(json.dumps(receipt), encoding="utf-8")
        with self.assertRaisesRegex(FreezeError, "inventory"):
            verify_holdout_receipt(self.root, self.config)


class HoldoutBoundaryTests(unittest.TestCase):
    def test_cli_only_allows_explicit_selected_frozen_holdout(self):
        for args in (
            ["--model-name", "model_b", "--split", "holdout", "--frozen"],
            ["--model-name", "selected", "--split", "holdout"],
            ["--model-name", "selected", "--split", "test", "--frozen"],
            ["--model-name", "model_b", "--split", "test", "--frozen"],
        ):
            with self.subTest(args=args), patch("src.holdout.evaluate_final_holdout") as evaluate:
                self.assertEqual(main(args), 1)
                evaluate.assert_not_called()
        with patch("src.holdout.evaluate_final_holdout", return_value={"synthetic": True}) as evaluate:
            self.assertEqual(main(["--model-name", "selected", "--split", "holdout", "--frozen"]), 0)
            evaluate.assert_called_once()

    def test_dirty_worktree_cannot_open_holdout(self):
        config = read_json(PROJECT_ROOT / "config/model_config.json")
        with patch("src.holdout.git_text", return_value=" M src/evaluate.py\n"):
            with self.assertRaisesRegex(FreezeError, "Commit the reviewed"):
                preflight_commit(PROJECT_ROOT, config)

    def test_original_prediction_logic_is_still_protected(self):
        original = "import math\ndef predict():\n return 1\ndef main():\n return 0\n"
        self.assertEqual(protected_ast(original, {"main"}), protected_ast(original.replace("return 0", "return 2"), {"main"}))
        self.assertNotEqual(protected_ast(original, {"main"}), protected_ast(original.replace("return 1", "return 2"), {"main"}))

    def test_resealing_protocol_cannot_change_frozen_features(self):
        config = read_json(PROJECT_ROOT / "config/model_config.json")
        protocol = read_json(PROJECT_ROOT / PROTOCOL)
        actual = dict(protocol["implementation_files_sha256"])
        actual["src/build_features.py"] = "0" * 64
        protocol["implementation_files_sha256"] = actual
        with patch("src.holdout.read_json", return_value=protocol):
            with self.assertRaisesRegex(FreezeError, "implementation checksum"):
                verify_evaluation_extension(PROJECT_ROOT, config, actual)

    def test_frozen_inputs_load_without_predicting_or_fitting(self):
        config = verify_freeze()
        with patch("xgboost.XGBClassifier.predict_proba", side_effect=AssertionError("Holdout inference forbidden in tests")), \
             patch("xgboost.XGBClassifier.fit", side_effect=AssertionError("Fit forbidden")), \
             patch("src.build_features.fit_training_medians", side_effect=AssertionError("Fit forbidden")):
            _, frame, medians = load_frozen_inputs(PROJECT_ROOT, config)
        self.assertEqual(len(frame), 380)
        self.assertEqual(set(frame["season"]), {"2526"})
        self.assertEqual(medians, config["frozen_candidate"]["preprocessing"]["median_values"])


if __name__ == "__main__":
    unittest.main()
