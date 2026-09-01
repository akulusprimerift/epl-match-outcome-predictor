"""Phase 4 baseline probability, preprocessing, and report tests."""

import json
from pathlib import Path
import unittest

import numpy as np
import pandas as pd

from src.build_features import fit_training_medians
from src.constants import (
    CLASS_LABELS,
    FEATURE_COLUMNS,
    MODEL_RESULT_COLUMNS,
    TRAIN_SEASONS,
)
from src.split_data import load_split_policy, read_model_dataset, split_model_dataset
from src.train_baselines import (
    MODEL_RESULTS_PATH,
    BaselineError,
    build_preprocessing_record,
    evaluate_baseline_models,
    fit_baseline_models,
    predict_ordered_probabilities,
    validate_probability_matrix,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class BaselineProbabilityTests(unittest.TestCase):
    """Exercise both estimators against the frozen repository splits."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.model = read_model_dataset()
        cls.policy = load_split_policy()
        cls.splits = split_model_dataset(cls.model, cls.policy)
        cls.models = fit_baseline_models(cls.splits.train)

    def test_both_baselines_return_normalized_three_class_probabilities(self) -> None:
        for model_name, model in self.models.by_name().items():
            for split_name in ("validation", "test"):
                with self.subTest(model=model_name, split=split_name):
                    frame = self.splits.by_name()[split_name]
                    probabilities = predict_ordered_probabilities(model, frame)
                    self.assertEqual(probabilities.shape, (len(frame), 3))
                    self.assertTrue(np.isfinite(probabilities).all())
                    self.assertTrue((probabilities >= 0).all())
                    np.testing.assert_allclose(
                        probabilities.sum(axis=1),
                        np.ones(len(frame)),
                        atol=1e-6,
                        rtol=0.0,
                    )

    def test_model_class_membership_is_fixed(self) -> None:
        for model_name, model in self.models.by_name().items():
            with self.subTest(model=model_name):
                self.assertEqual(tuple(int(value) for value in model.classes_), CLASS_LABELS)

    def test_majority_predictions_use_most_frequent_training_label(self) -> None:
        training_target = self.splits.train["target"].astype("int64")
        expected_label = int(training_target.value_counts().idxmax())
        predictions = self.models.majority.predict(
            self.splits.validation.loc[:, FEATURE_COLUMNS]
        )
        self.assertTrue((np.asarray(predictions) == expected_label).all())

    def test_probability_validator_rejects_invalid_outputs(self) -> None:
        invalid_matrices = (
            np.asarray([[0.2, 0.8]]),
            np.asarray([[0.2, -0.1, 0.9]]),
            np.asarray([[0.2, 0.2, 0.2]]),
            np.asarray([[np.nan, 0.5, 0.5]]),
        )
        for values in invalid_matrices:
            with self.subTest(values=values), self.assertRaises(BaselineError):
                validate_probability_matrix(values)


class TrainingOnlyPreprocessingTests(unittest.TestCase):
    """Prove imputation state comes exclusively from the training split."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.model = read_model_dataset()
        cls.policy = load_split_policy()
        cls.splits = split_model_dataset(cls.model, cls.policy)
        cls.models = fit_baseline_models(cls.splits.train)

    def test_saved_record_matches_independent_training_medians(self) -> None:
        record = build_preprocessing_record(self.models.logistic, self.splits.train)
        expected = fit_training_medians(self.splits.train, FEATURE_COLUMNS)
        self.assertEqual(record["fitted_on_split"], "train")
        self.assertEqual(record["training_seasons"], list(TRAIN_SEASONS))
        self.assertEqual(record["training_match_count"], 4_940)
        self.assertEqual(record["feature_columns"], list(FEATURE_COLUMNS))
        self.assertEqual(record["median_values"], expected)

    def test_nontraining_values_cannot_change_fitted_medians(self) -> None:
        original = build_preprocessing_record(self.models.logistic, self.splits.train)
        altered_model = self.model.copy()
        nontraining = ~altered_model["season"].astype(str).isin(TRAIN_SEASONS)
        altered_model.loc[nontraining, "home_goals_for_avg_5"] = 1_000_000.0
        altered_model.loc[nontraining, "goals_scored_edge"] = (
            altered_model.loc[nontraining, "home_goals_for_avg_5"]
            - altered_model.loc[nontraining, "away_goals_for_avg_5"]
        )
        altered_splits = split_model_dataset(altered_model, self.policy)
        refitted = fit_baseline_models(altered_splits.train)
        after = build_preprocessing_record(refitted.logistic, altered_splits.train)
        self.assertEqual(original["median_values"], after["median_values"])


class BaselineReportTests(unittest.TestCase):
    """Verify persisted metrics include only approved evaluation splits."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.path = PROJECT_ROOT / MODEL_RESULTS_PATH.relative_to(PROJECT_ROOT)
        cls.results = pd.read_csv(cls.path)

    def test_report_has_both_models_on_validation_and_test_only(self) -> None:
        self.assertTrue(self.path.is_file(), f"Missing model results: {self.path}")
        self.assertEqual(tuple(self.results.columns), MODEL_RESULT_COLUMNS)
        self.assertEqual(len(self.results), 4)
        observed = set(zip(self.results["model_name"], self.results["split"]))
        self.assertEqual(
            observed,
            {
                ("majority_baseline", "validation"),
                ("majority_baseline", "test"),
                ("logistic_regression", "validation"),
                ("logistic_regression", "test"),
            },
        )
        self.assertNotIn("holdout", set(self.results["split"]))
        for value in self.results["parameters"]:
            self.assertIsInstance(json.loads(value), dict)

    def test_report_metrics_are_finite_and_in_range(self) -> None:
        for column in ("log_loss", "macro_f1", "accuracy"):
            values = pd.to_numeric(self.results[column], errors="raise")
            self.assertTrue(np.isfinite(values).all())
            self.assertTrue(values.ge(0).all())
        self.assertTrue(self.results["macro_f1"].le(1).all())
        self.assertTrue(self.results["accuracy"].le(1).all())
        for row in self.results.itertuples(index=False):
            support = row.support_away_win + row.support_draw + row.support_home_win
            confusion = sum(
                getattr(row, column)
                for column in MODEL_RESULT_COLUMNS
                if column.startswith("confusion_")
            )
            self.assertEqual(support, row.row_count)
            self.assertEqual(confusion, row.row_count)

    def test_rebuilt_report_is_deterministic(self) -> None:
        model = read_model_dataset()
        splits = split_model_dataset(model, load_split_policy())
        models = fit_baseline_models(splits.train)
        rebuilt = evaluate_baseline_models(models, splits)
        persisted = self.results.copy()
        persisted["best_iteration"] = persisted["best_iteration"].fillna("")
        pd.testing.assert_frame_equal(
            rebuilt.reset_index(drop=True),
            persisted.loc[:, MODEL_RESULT_COLUMNS].reset_index(drop=True),
            check_dtype=False,
            atol=1e-12,
            rtol=0.0,
        )


if __name__ == "__main__":
    unittest.main()
