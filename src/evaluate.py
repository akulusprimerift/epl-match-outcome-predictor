"""Load and evaluate saved Phase 5 XGBoost artifacts without refitting."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Mapping, Sequence

import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from xgboost.core import XGBoostError

from src.build_features import FeatureError, apply_training_medians
from src.clean_data import PROJECT_ROOT
from src.constants import CLASS_LABELS, FEATURE_COLUMNS
from src.split_data import SplitError, load_split_policy, read_model_dataset, split_model_dataset
from src.train_baselines import (
    BaselineError,
    evaluate_probabilities,
    validate_probability_matrix,
)


SUPPORTED_MODELS = ("model_a",)
SUPPORTED_SPLITS = ("validation", "test")


class EvaluationError(RuntimeError):
    """Raised when a saved model or its metadata cannot be evaluated safely."""


def load_json_object(path: Path) -> dict[str, object]:
    """Read a required JSON object with an actionable error."""
    try:
        with path.open("r", encoding="utf-8") as input_file:
            value = json.load(input_file)
    except FileNotFoundError as exc:
        raise EvaluationError(f"Required model artifact not found: {path}") from exc
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise EvaluationError(f"Could not read model artifact {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise EvaluationError(f"Model artifact {path} must contain a JSON object.")
    return value


def validate_model_metadata(
    metadata: Mapping[str, object],
    *,
    expected_model_name: str,
) -> None:
    """Validate the Phase 5 metadata needed for safe feature reconstruction."""
    required = {
        "model_name",
        "feature_set",
        "feature_columns",
        "classes",
        "best_iteration",
        "parameters",
        "holdout_evaluated",
    }
    missing = required.difference(metadata)
    if missing:
        raise EvaluationError(f"Model metadata is missing fields: {sorted(missing)}")
    if metadata["model_name"] != expected_model_name:
        raise EvaluationError(
            f"Metadata model name {metadata['model_name']!r} does not match "
            f"{expected_model_name!r}."
        )
    if metadata["feature_set"] != "baseline":
        raise EvaluationError("Phase 5 Model A must use the baseline feature set.")
    if not isinstance(metadata["feature_columns"], list) or tuple(
        metadata["feature_columns"]
    ) != FEATURE_COLUMNS:
        raise EvaluationError("Saved feature order differs from FEATURE_COLUMNS.")
    if not isinstance(metadata["classes"], list) or tuple(metadata["classes"]) != CLASS_LABELS:
        raise EvaluationError("Saved class mapping differs from 0/1/2.")
    best_iteration = metadata["best_iteration"]
    if not isinstance(best_iteration, int) or best_iteration < 0:
        raise EvaluationError("Saved best_iteration must be a nonnegative integer.")
    if metadata["holdout_evaluated"] is not False:
        raise EvaluationError("Phase 5 metadata must state that holdout is unopened.")
    if not isinstance(metadata["parameters"], dict):
        raise EvaluationError("Saved model parameters must be a JSON object.")


def load_preprocessing(
    path: Path,
    *,
    feature_columns: Sequence[str] = FEATURE_COLUMNS,
) -> dict[str, float]:
    """Load training-only median values in exact feature order."""
    record = load_json_object(path)
    if record.get("strategy") != "median" or record.get("fitted_on_split") != "train":
        raise EvaluationError("Preprocessing must contain training-only median imputation.")
    if tuple(record.get("feature_columns", ())) != tuple(feature_columns):
        raise EvaluationError("Preprocessing feature order differs from model metadata.")
    raw_medians = record.get("median_values")
    if not isinstance(raw_medians, dict) or set(raw_medians) != set(feature_columns):
        raise EvaluationError("Preprocessing medians do not cover the complete feature set.")
    medians: dict[str, float] = {}
    for feature in feature_columns:
        try:
            value = float(raw_medians[feature])
        except (TypeError, ValueError) as exc:
            raise EvaluationError(f"Median for {feature} is not numeric.") from exc
        if not np.isfinite(value):
            raise EvaluationError(f"Median for {feature} is not finite.")
        medians[feature] = value
    return medians


def load_xgboost_model(path: Path) -> XGBClassifier:
    """Load a native XGBoost JSON artifact and validate its class mapping."""
    if not path.is_file():
        raise EvaluationError(f"Saved XGBoost model not found: {path}")
    model = XGBClassifier()
    try:
        model.load_model(path)
    except (OSError, TypeError, ValueError, XGBoostError) as exc:
        raise EvaluationError(f"Could not load XGBoost model {path}: {exc}") from exc
    classes = tuple(int(value) for value in np.asarray(model.classes_).tolist())
    if classes != CLASS_LABELS:
        raise EvaluationError(f"Unexpected XGBoost class order: {classes}")
    return model


def predict_xgboost_probabilities(
    model: XGBClassifier,
    frame: pd.DataFrame,
    *,
    feature_columns: Sequence[str],
    medians: Mapping[str, float],
    best_iteration: int,
) -> np.ndarray:
    """Impute and return probabilities mapped through model.classes_."""
    if tuple(feature_columns) != FEATURE_COLUMNS:
        raise EvaluationError("Prediction feature order differs from FEATURE_COLUMNS.")
    try:
        transformed = apply_training_medians(frame, medians, feature_columns)
    except FeatureError as exc:
        raise EvaluationError(f"Could not apply saved preprocessing: {exc}") from exc
    features = transformed.loc[:, feature_columns]
    classes = tuple(int(value) for value in np.asarray(model.classes_).tolist())
    if set(classes) != set(CLASS_LABELS) or len(classes) != len(CLASS_LABELS):
        raise EvaluationError(f"Unexpected XGBoost classes: {classes}")
    try:
        raw = np.asarray(
            model.predict_proba(
                features,
                validate_features=True,
                iteration_range=(0, best_iteration + 1),
            ),
            dtype="float64",
        )
    except (TypeError, ValueError, XGBoostError) as exc:
        raise EvaluationError(f"Could not generate XGBoost probabilities: {exc}") from exc
    positions = {class_label: index for index, class_label in enumerate(classes)}
    ordered = raw[:, [positions[label] for label in CLASS_LABELS]]
    try:
        validate_probability_matrix(ordered, expected_rows=len(frame))
    except BaselineError as exc:
        raise EvaluationError(f"Invalid XGBoost probabilities: {exc}") from exc
    ordered = ordered / ordered.sum(axis=1, keepdims=True)
    try:
        validate_probability_matrix(ordered, expected_rows=len(frame))
    except BaselineError as exc:
        raise EvaluationError(f"Could not normalize XGBoost probabilities: {exc}") from exc
    return ordered


def evaluate_saved_model(
    model_name: str,
    split_name: str,
    *,
    project_root: Path = PROJECT_ROOT,
) -> dict[str, object]:
    """Evaluate one saved model on validation or test without refitting."""
    if model_name not in SUPPORTED_MODELS:
        raise EvaluationError(
            f"Unsupported model {model_name!r}; Phase 5 supports model_a only."
        )
    if split_name not in SUPPORTED_SPLITS:
        raise EvaluationError(
            "Phase 5 evaluation supports validation or test only; holdout is unopened."
        )
    project_root = project_root.resolve()
    metadata = load_json_object(project_root / "models" / "model_metadata.json")
    validate_model_metadata(metadata, expected_model_name=model_name)
    feature_columns = tuple(str(value) for value in metadata["feature_columns"])
    medians = load_preprocessing(
        project_root / "models" / "preprocessing.json",
        feature_columns=feature_columns,
    )
    model = load_xgboost_model(project_root / "models" / f"{model_name}_xgb.json")
    dataset = read_model_dataset(
        project_root / "data" / "processed" / "model_dataset.csv"
    )
    policy = load_split_policy(project_root / "config" / "model_config.json")
    splits = split_model_dataset(dataset, policy)
    frame = splits.by_name()[split_name]
    probabilities = predict_xgboost_probabilities(
        model,
        frame,
        feature_columns=feature_columns,
        medians=medians,
        best_iteration=int(metadata["best_iteration"]),
    )
    try:
        result = evaluate_probabilities(
            model_name=model_name,
            model_family="xgboost",
            feature_set=str(metadata["feature_set"]),
            split_name=split_name,
            frame=frame,
            probabilities=probabilities,
            parameters=metadata["parameters"],
        )
        result["best_iteration"] = int(metadata["best_iteration"])
        return result
    except BaselineError as exc:
        raise EvaluationError(f"Could not evaluate saved model: {exc}") from exc


def build_parser() -> argparse.ArgumentParser:
    """Build the saved-model evaluation CLI parser."""
    parser = argparse.ArgumentParser(
        description="Evaluate a saved EPL model without retraining or opening holdout."
    )
    parser.add_argument("--model-name", required=True, choices=SUPPORTED_MODELS)
    parser.add_argument("--split", required=True, choices=SUPPORTED_SPLITS)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run saved-model evaluation and print machine-readable metrics."""
    arguments = build_parser().parse_args(argv)
    try:
        result = evaluate_saved_model(arguments.model_name, arguments.split)
    except (EvaluationError, SplitError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    output = dict(result)
    output["holdout_evaluated"] = False
    print(json.dumps(output, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
