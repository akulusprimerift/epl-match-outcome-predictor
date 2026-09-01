"""Train and evaluate Phase 4 majority and logistic-regression baselines."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
import sys
import tempfile
from typing import Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    log_loss,
    precision_recall_fscore_support,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.build_features import FeatureError, fit_training_medians
from src.clean_data import PROJECT_ROOT
from src.constants import (
    CLASS_LABELS,
    CLASS_NAMES,
    FEATURE_COLUMNS,
    MODEL_RESULT_COLUMNS,
    RANDOM_SEED,
    TRAIN_SEASONS,
)
from src.split_data import (
    DatasetSplits,
    SplitError,
    build_split_manifest,
    load_split_policy,
    read_model_dataset,
    split_model_dataset,
    write_split_manifest_atomic,
)


MODEL_RESULTS_PATH = PROJECT_ROOT / "reports" / "model_results.csv"
PREPROCESSING_PATH = PROJECT_ROOT / "models" / "preprocessing.json"
SUPPORTED_FEATURE_SETS = ("baseline",)
EVALUATION_SPLITS = ("validation", "test")


class BaselineError(RuntimeError):
    """Raised when baseline fitting, prediction, or reporting fails."""


@dataclass(frozen=True)
class BaselineModels:
    """Fitted Phase 4 benchmark estimators."""

    majority: DummyClassifier
    logistic: Pipeline

    def by_name(self) -> dict[str, object]:
        """Return stable report names for both models."""
        return {
            "majority_baseline": self.majority,
            "logistic_regression": self.logistic,
        }


@dataclass(frozen=True)
class BaselineTrainingSummary:
    """Paths and counts emitted by a successful baseline run."""

    training_rows: int
    validation_rows: int
    test_rows: int
    result_rows: int
    split_manifest_path: Path
    results_path: Path
    preprocessing_path: Path


def _feature_matrix(frame: pd.DataFrame) -> pd.DataFrame:
    missing = set(FEATURE_COLUMNS).difference(frame.columns)
    if missing:
        raise BaselineError(f"Model frame is missing features: {sorted(missing)}")
    return frame.loc[:, FEATURE_COLUMNS]


def _target_vector(frame: pd.DataFrame) -> pd.Series:
    if "target" not in frame.columns:
        raise BaselineError("Model frame is missing the target column.")
    values = pd.to_numeric(frame["target"], errors="coerce")
    if values.isna().any() or not values.isin(CLASS_LABELS).all():
        raise BaselineError("Targets must use only the fixed classes 0, 1, and 2.")
    return values.astype("int64")


def fit_baseline_models(training_frame: pd.DataFrame) -> BaselineModels:
    """Fit both models using only the explicitly supplied training rows."""
    if training_frame.empty:
        raise BaselineError("Cannot fit baselines on an empty training split.")
    features = _feature_matrix(training_frame)
    target = _target_vector(training_frame)
    if set(target.unique()) != set(CLASS_LABELS):
        raise BaselineError("Training data must contain all three target classes.")

    majority = DummyClassifier(strategy="prior")
    logistic = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            (
                "classifier",
                LogisticRegression(
                    C=1.0,
                    l1_ratio=0.0,
                    solver="lbfgs",
                    max_iter=2_000,
                    random_state=RANDOM_SEED,
                ),
            ),
        ]
    )
    try:
        majority.fit(features, target)
        logistic.fit(features, target)
    except (TypeError, ValueError) as exc:
        raise BaselineError(f"Could not fit baseline models: {exc}") from exc
    return BaselineModels(majority=majority, logistic=logistic)


def validate_probability_matrix(
    probabilities: np.ndarray,
    *,
    expected_rows: int | None = None,
) -> None:
    """Enforce finite, nonnegative, normalized three-class probabilities."""
    values = np.asarray(probabilities, dtype="float64")
    if values.ndim != 2 or values.shape[1] != len(CLASS_LABELS):
        raise BaselineError(
            f"Expected an N x {len(CLASS_LABELS)} probability matrix; "
            f"received shape {values.shape}."
        )
    if expected_rows is not None and values.shape[0] != expected_rows:
        raise BaselineError(
            f"Expected {expected_rows} probability rows; received {values.shape[0]}."
        )
    if not np.isfinite(values).all():
        raise BaselineError("Predicted probabilities must all be finite.")
    if (values < 0).any():
        raise BaselineError("Predicted probabilities must be nonnegative.")
    if not np.allclose(values.sum(axis=1), 1.0, atol=1e-6, rtol=0.0):
        raise BaselineError("Each predicted probability row must sum to one.")


def predict_ordered_probabilities(model: object, frame: pd.DataFrame) -> np.ndarray:
    """Predict probabilities and map estimator classes to fixed 0/1/2 order."""
    if not hasattr(model, "classes_") or not hasattr(model, "predict_proba"):
        raise BaselineError("Baseline estimator does not expose classes_ and predict_proba.")
    classes = tuple(int(value) for value in np.asarray(model.classes_).tolist())
    if set(classes) != set(CLASS_LABELS) or len(classes) != len(CLASS_LABELS):
        raise BaselineError(f"Unexpected model class order or membership: {classes}")
    try:
        raw = np.asarray(model.predict_proba(_feature_matrix(frame)), dtype="float64")
    except (TypeError, ValueError) as exc:
        raise BaselineError(f"Could not generate baseline probabilities: {exc}") from exc
    class_positions = {class_label: index for index, class_label in enumerate(classes)}
    ordered = raw[:, [class_positions[label] for label in CLASS_LABELS]]
    validate_probability_matrix(ordered, expected_rows=len(frame))
    return ordered


def evaluate_probabilities(
    *,
    model_name: str,
    model_family: str,
    feature_set: str,
    split_name: str,
    frame: pd.DataFrame,
    probabilities: np.ndarray,
    parameters: Mapping[str, object],
) -> dict[str, object]:
    """Calculate the complete Phase 4 metric row for one model and split."""
    if split_name not in EVALUATION_SPLITS:
        raise BaselineError(
            f"Phase 4 evaluation is limited to {', '.join(EVALUATION_SPLITS)}."
        )
    validate_probability_matrix(probabilities, expected_rows=len(frame))
    target = _target_vector(frame)
    predicted = np.asarray(CLASS_LABELS, dtype="int64")[
        np.asarray(probabilities).argmax(axis=1)
    ]
    precision, recall, f1, support = precision_recall_fscore_support(
        target,
        predicted,
        labels=CLASS_LABELS,
        zero_division=0,
    )
    matrix = confusion_matrix(target, predicted, labels=CLASS_LABELS)
    row: dict[str, object] = {
        "model_name": model_name,
        "model_family": model_family,
        "feature_set": feature_set,
        "split": split_name,
        "row_count": len(frame),
        "log_loss": float(log_loss(target, probabilities, labels=CLASS_LABELS)),
        "macro_f1": float(f1.mean()),
        "accuracy": float(accuracy_score(target, predicted)),
        "best_iteration": "",
        "parameters": json.dumps(parameters, sort_keys=True, separators=(",", ":")),
    }
    for index, class_name in enumerate(CLASS_NAMES):
        row[f"precision_{class_name}"] = float(precision[index])
        row[f"recall_{class_name}"] = float(recall[index])
        row[f"f1_{class_name}"] = float(f1[index])
        row[f"support_{class_name}"] = int(support[index])
    for actual_index, actual_name in enumerate(CLASS_NAMES):
        for predicted_index, predicted_name in enumerate(CLASS_NAMES):
            row[f"confusion_{actual_name}_pred_{predicted_name}"] = int(
                matrix[actual_index, predicted_index]
            )
    if tuple(row) != MODEL_RESULT_COLUMNS:
        raise BaselineError("Metric row columns do not match the report contract.")
    return row


def evaluate_baseline_models(
    models: BaselineModels,
    splits: DatasetSplits,
    *,
    feature_set: str = "baseline",
) -> pd.DataFrame:
    """Evaluate both baselines on validation and test, never on holdout."""
    model_families = {
        "majority_baseline": "majority",
        "logistic_regression": "logistic_regression",
    }
    model_parameters: dict[str, dict[str, object]] = {
        "majority_baseline": {
            "majority_class": int(
                models.majority.classes_[np.argmax(models.majority.class_prior_)]
            ),
            "prediction_rule": "most_frequent_training_label",
            "probability_rule": "training_class_prior",
            "strategy": "prior",
            "training_class_probabilities": {
                str(int(class_label)): float(class_probability)
                for class_label, class_probability in zip(
                    models.majority.classes_, models.majority.class_prior_
                )
            },
        },
        "logistic_regression": {
            "C": 1.0,
            "imputation": "training_median",
            "l1_ratio": 0.0,
            "max_iter": 2_000,
            "random_state": RANDOM_SEED,
            "scaling": "standard",
            "solver": "lbfgs",
        },
    }
    rows = []
    split_frames = splits.by_name()
    for model_name, model in models.by_name().items():
        for split_name in EVALUATION_SPLITS:
            frame = split_frames[split_name]
            probabilities = predict_ordered_probabilities(model, frame)
            rows.append(
                evaluate_probabilities(
                    model_name=model_name,
                    model_family=model_families[model_name],
                    feature_set=feature_set,
                    split_name=split_name,
                    frame=frame,
                    probabilities=probabilities,
                    parameters=model_parameters[model_name],
                )
            )
    return pd.DataFrame(rows, columns=MODEL_RESULT_COLUMNS)


def build_preprocessing_record(
    logistic: Pipeline,
    training_frame: pd.DataFrame,
) -> dict[str, object]:
    """Serialize medians fitted exclusively from the frozen training split."""
    try:
        imputer = logistic.named_steps["imputer"]
        fitted_statistics = tuple(float(value) for value in imputer.statistics_)
    except (AttributeError, KeyError, TypeError, ValueError) as exc:
        raise BaselineError(f"Could not inspect fitted median imputer: {exc}") from exc
    if len(fitted_statistics) != len(FEATURE_COLUMNS) or not all(
        math.isfinite(value) for value in fitted_statistics
    ):
        raise BaselineError("Fitted imputation statistics are incomplete or nonfinite.")

    try:
        independently_fitted = fit_training_medians(training_frame, FEATURE_COLUMNS)
    except FeatureError as exc:
        raise BaselineError(f"Could not verify training-only medians: {exc}") from exc
    median_values = dict(zip(FEATURE_COLUMNS, fitted_statistics))
    for feature in FEATURE_COLUMNS:
        if not math.isclose(
            median_values[feature],
            independently_fitted[feature],
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise BaselineError(f"Pipeline median differs for feature {feature}.")

    training_dates = pd.to_datetime(
        training_frame["date"], format="%Y-%m-%d", errors="raise"
    )
    return {
        "feature_set": "baseline",
        "strategy": "median",
        "fitted_on_split": "train",
        "training_seasons": list(TRAIN_SEASONS),
        "training_match_count": len(training_frame),
        "training_date_min": training_dates.min().date().isoformat(),
        "training_date_max": training_dates.max().date().isoformat(),
        "feature_columns": list(FEATURE_COLUMNS),
        "median_values": median_values,
    }


def _write_json_atomic(value: Mapping[str, object], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    file_descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(file_descriptor, "w", encoding="utf-8", newline="\n") as output:
            json.dump(value, output, indent=2, ensure_ascii=False)
            output.write("\n")
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary_path, path)
    except (OSError, TypeError, ValueError) as exc:
        raise BaselineError(f"Could not atomically write {path}: {exc}") from exc
    finally:
        temporary_path.unlink(missing_ok=True)


def _write_results_atomic(results: pd.DataFrame, path: Path) -> None:
    if tuple(results.columns) != MODEL_RESULT_COLUMNS:
        raise BaselineError("Model result columns do not match the report contract.")
    path.parent.mkdir(parents=True, exist_ok=True)
    file_descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(file_descriptor, "w", encoding="utf-8", newline="") as output:
            results.to_csv(
                output,
                index=False,
                columns=MODEL_RESULT_COLUMNS,
                lineterminator="\n",
            )
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary_path, path)
    except (OSError, TypeError, ValueError) as exc:
        raise BaselineError(f"Could not atomically write model results {path}: {exc}") from exc
    finally:
        temporary_path.unlink(missing_ok=True)


def train_and_evaluate_baselines(
    feature_set: str,
    *,
    project_root: Path = PROJECT_ROOT,
) -> BaselineTrainingSummary:
    """Run Phase 4 and atomically save its reproducibility artifacts."""
    if feature_set not in SUPPORTED_FEATURE_SETS:
        raise BaselineError(
            f"Unsupported feature set {feature_set!r}; expected baseline."
        )
    project_root = project_root.resolve()
    model = read_model_dataset(
        project_root / "data" / "processed" / "model_dataset.csv"
    )
    policy = load_split_policy(project_root / "config" / "model_config.json")
    splits = split_model_dataset(model, policy)
    models = fit_baseline_models(splits.train)
    results = evaluate_baseline_models(models, splits, feature_set=feature_set)
    preprocessing = build_preprocessing_record(models.logistic, splits.train)
    manifest = build_split_manifest(splits)

    manifest_path = project_root / "data" / "processed" / "split_manifest.csv"
    results_path = project_root / "reports" / "model_results.csv"
    preprocessing_path = project_root / "models" / "preprocessing.json"
    write_split_manifest_atomic(manifest, manifest_path)
    _write_results_atomic(results, results_path)
    _write_json_atomic(preprocessing, preprocessing_path)
    return BaselineTrainingSummary(
        training_rows=len(splits.train),
        validation_rows=len(splits.validation),
        test_rows=len(splits.test),
        result_rows=len(results),
        split_manifest_path=manifest_path,
        results_path=results_path,
        preprocessing_path=preprocessing_path,
    )


def build_parser() -> argparse.ArgumentParser:
    """Build the Phase 4 baseline-training command parser."""
    parser = argparse.ArgumentParser(
        description="Train majority and logistic EPL baselines on frozen time splits."
    )
    parser.add_argument(
        "--feature-set",
        required=True,
        choices=SUPPORTED_FEATURE_SETS,
        help="feature contract to train (Phase 4 supports baseline only)",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the baseline CLI without evaluating the final holdout."""
    arguments = build_parser().parse_args(argv)
    try:
        summary = train_and_evaluate_baselines(arguments.feature_set)
    except (BaselineError, SplitError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    print(f"training_rows={summary.training_rows}")
    print(f"validation_rows={summary.validation_rows}")
    print(f"test_rows={summary.test_rows}")
    print(f"result_rows={summary.result_rows}")
    print(f"evaluated_splits={','.join(EVALUATION_SPLITS)}")
    print("holdout_evaluated=False")
    print(f"split_manifest={summary.split_manifest_path.relative_to(PROJECT_ROOT).as_posix()}")
    print(f"results={summary.results_path.relative_to(PROJECT_ROOT).as_posix()}")
    print(f"preprocessing={summary.preprocessing_path.relative_to(PROJECT_ROOT).as_posix()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
