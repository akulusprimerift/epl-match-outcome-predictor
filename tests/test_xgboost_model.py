"""Phase 5 XGBoost search, artifact, prediction, and report tests."""

import hashlib
import json
from pathlib import Path
import subprocess
import sys
import unittest

import numpy as np
import pandas as pd

from src.constants import (
    CLASS_LABELS,
    FEATURE_COLUMNS,
    MODEL_RESULT_COLUMNS,
    TUNING_RESULT_COLUMNS,
)
from src.evaluate import (
    EvaluationError,
    evaluate_saved_model,
    load_json_object,
    load_preprocessing,
    load_xgboost_model,
    predict_xgboost_probabilities,
    validate_model_metadata,
)
from src.split_data import load_split_policy, read_model_dataset, split_model_dataset
from src.train_xgboost import (
    APPROVED_GRID,
    STARTING_PARAMETERS,
    load_search_config,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "config" / "model_config.json"
MODEL_PATH = PROJECT_ROOT / "models" / "model_a_xgb.json"
METADATA_PATH = PROJECT_ROOT / "models" / "model_metadata.json"
PREPROCESSING_PATH = PROJECT_ROOT / "models" / "preprocessing.json"
MODEL_RESULTS_PATH = PROJECT_ROOT / "reports" / "model_results.csv"
TUNING_RESULTS_PATH = PROJECT_ROOT / "reports" / "tuning_results.csv"


def sha256(path: Path) -> str:
    """Return the SHA-256 digest for one local artifact."""
    digest = hashlib.sha256()
    with path.open("rb") as input_file:
        for block in iter(lambda: input_file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class ApprovedSearchTests(unittest.TestCase):
    """Ensure tuning cannot leave the specification's approved search space."""

    def test_search_is_bounded_and_uses_only_approved_values(self) -> None:
        search = load_search_config(CONFIG_PATH)
        self.assertEqual(search.n_estimators, 500)
        self.assertEqual(search.early_stopping_rounds, 40)
        self.assertEqual(search.tie_tolerance, 0.0001)
        self.assertEqual(len(search.candidates), 7)
        starting_count = 0
        for candidate in search.candidates:
            parameters = {
                name: candidate[name] for name in STARTING_PARAMETERS
            }
            if parameters == STARTING_PARAMETERS:
                starting_count += 1
                continue
            for name, approved_values in APPROVED_GRID.items():
                with self.subTest(candidate=candidate["candidate_id"], parameter=name):
                    self.assertIn(candidate[name], approved_values)
        self.assertEqual(starting_count, 1)

    def test_tuning_report_records_every_candidate_without_future_metrics(self) -> None:
        results = pd.read_csv(TUNING_RESULTS_PATH)
        model_a_results = results.loc[results["model_name"].eq("model_a")]
        search = load_search_config(CONFIG_PATH)
        self.assertEqual(tuple(results.columns), TUNING_RESULT_COLUMNS)
        self.assertEqual(len(model_a_results), len(search.candidates))
        self.assertEqual(model_a_results["candidate_id"].tolist(), [
            candidate["candidate_id"] for candidate in search.candidates
        ])
        self.assertEqual(int(model_a_results["selected"].sum()), 1)
        self.assertEqual(set(model_a_results["feature_set"]), {"baseline"})
        self.assertFalse(
            any("test" in column or "holdout" in column for column in results.columns)
        )
        selected = model_a_results.loc[model_a_results["selected"]].iloc[0]
        self.assertLessEqual(
            selected["validation_log_loss"],
            model_a_results["validation_log_loss"].min() + search.tie_tolerance,
        )


class SavedModelATests(unittest.TestCase):
    """Validate the generated Model A artifact and its exact saved metadata."""

    @classmethod
    def setUpClass(cls) -> None:
        required = (MODEL_PATH, METADATA_PATH, PREPROCESSING_PATH)
        missing = [path for path in required if not path.is_file()]
        if missing:
            raise unittest.SkipTest(
                "Run python -m src.train_xgboost --model-name model_a "
                "--feature-set baseline before artifact tests."
            )
        cls.metadata = load_json_object(METADATA_PATH)
        validate_model_metadata(cls.metadata, expected_model_name="model_a")
        cls.medians = load_preprocessing(PREPROCESSING_PATH)
        cls.model = load_xgboost_model(MODEL_PATH)
        dataset = read_model_dataset()
        cls.splits = split_model_dataset(dataset, load_split_policy())

    def test_metadata_records_fixed_contract_and_checksums(self) -> None:
        self.assertEqual(self.metadata["model_name"], "model_a")
        self.assertEqual(self.metadata["feature_set"], "baseline")
        self.assertEqual(tuple(self.metadata["feature_columns"]), FEATURE_COLUMNS)
        self.assertEqual(tuple(self.metadata["classes"]), CLASS_LABELS)
        self.assertEqual(self.metadata["training_match_count"], 4_940)
        self.assertEqual(self.metadata["validation_match_count"], 380)
        self.assertEqual(self.metadata["test_match_count"], 380)
        self.assertFalse(self.metadata["holdout_evaluated"])
        self.assertEqual(self.metadata["parameters"]["objective"], "multi:softprob")
        self.assertEqual(self.metadata["parameters"]["num_class"], 3)
        self.assertEqual(self.metadata["model_sha256"], sha256(MODEL_PATH))
        self.assertEqual(
            self.metadata["preprocessing_sha256"], sha256(PREPROCESSING_PATH)
        )

    def test_probabilities_are_normalized_and_deterministic(self) -> None:
        frame = self.splits.test.head(50)
        keyword_arguments = {
            "feature_columns": tuple(self.metadata["feature_columns"]),
            "medians": self.medians,
            "best_iteration": int(self.metadata["best_iteration"]),
        }
        first = predict_xgboost_probabilities(
            self.model, frame, **keyword_arguments
        )
        second = predict_xgboost_probabilities(
            self.model, frame, **keyword_arguments
        )
        self.assertEqual(first.shape, (50, 3))
        self.assertTrue(np.isfinite(first).all())
        self.assertTrue((first >= 0).all())
        np.testing.assert_allclose(first.sum(axis=1), 1.0, atol=1e-6, rtol=0.0)
        np.testing.assert_array_equal(first, second)

    def test_model_loads_in_a_fresh_python_process(self) -> None:
        script = (
            "import sys; from xgboost import XGBClassifier; "
            "model=XGBClassifier(); model.load_model(sys.argv[1]); "
            "print(','.join(str(int(value)) for value in model.classes_))"
        )
        completed = subprocess.run(
            [sys.executable, "-c", script, str(MODEL_PATH)],
            cwd=PROJECT_ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
        self.assertEqual(completed.stdout.strip(), "0,1,2")

    def test_feature_order_mismatch_is_rejected(self) -> None:
        invalid = dict(self.metadata)
        invalid["feature_columns"] = list(reversed(FEATURE_COLUMNS))
        with self.assertRaisesRegex(EvaluationError, "feature order"):
            validate_model_metadata(invalid, expected_model_name="model_a")


class ModelAReportTests(unittest.TestCase):
    """Verify Model A is compared fairly and holdout remains unopened."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.results = pd.read_csv(MODEL_RESULTS_PATH)

    def test_model_results_compare_all_phase_five_models(self) -> None:
        self.assertEqual(tuple(self.results.columns), MODEL_RESULT_COLUMNS)
        observed = set(zip(self.results["model_name"], self.results["split"]))
        for model_name in (
            "majority_baseline",
            "logistic_regression",
            "model_a",
        ):
            for split_name in ("validation", "test"):
                self.assertIn((model_name, split_name), observed)
        self.assertNotIn("holdout", set(self.results["split"]))

    def test_model_a_beats_majority_test_log_loss(self) -> None:
        test = self.results.loc[self.results["split"].eq("test")].set_index(
            "model_name"
        )
        self.assertLess(
            test.loc["model_a", "log_loss"],
            test.loc["majority_baseline", "log_loss"],
        )
        self.assertTrue(np.isfinite(test.loc["logistic_regression", "log_loss"]))

    def test_saved_evaluation_matches_report(self) -> None:
        if not MODEL_PATH.is_file():
            self.skipTest("Generated Model A artifact is not present.")
        evaluated = evaluate_saved_model("model_a", "test")
        expected = self.results.loc[
            self.results["model_name"].eq("model_a")
            & self.results["split"].eq("test")
        ].iloc[0]
        for metric in ("log_loss", "macro_f1", "accuracy"):
            self.assertAlmostEqual(evaluated[metric], expected[metric], places=12)
        self.assertEqual(evaluated["best_iteration"], int(expected["best_iteration"]))

    def test_holdout_evaluation_is_rejected(self) -> None:
        with self.assertRaisesRegex(EvaluationError, "holdout"):
            evaluate_saved_model("model_a", "holdout")

    def test_required_phase_five_plots_exist(self) -> None:
        plot_paths = (
            PROJECT_ROOT / "reports" / "class_distribution.png",
            PROJECT_ROOT / "reports" / "confusion_matrix_model_a.png",
            PROJECT_ROOT / "reports" / "feature_importance_model_a.png",
        )
        for path in plot_paths:
            with self.subTest(path=path.name):
                self.assertTrue(path.is_file())
                self.assertGreater(path.stat().st_size, 10_000)
                with path.open("rb") as image_file:
                    self.assertEqual(image_file.read(8), b"\x89PNG\r\n\x1a\n")


if __name__ == "__main__":
    unittest.main()
