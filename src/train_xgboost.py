"""Tune, train, evaluate, and save the Phase 5 Model A XGBoost classifier."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from typing import Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
from xgboost import XGBClassifier
from xgboost.core import XGBoostError

from src.build_features import (
    FeatureError,
    apply_training_medians,
    fit_training_medians,
)
from src.clean_data import PROJECT_ROOT
from src.constants import (
    CLASS_LABELS,
    CLASS_NAMES,
    FEATURE_COLUMNS,
    MODEL_RESULT_COLUMNS,
    RANDOM_SEED,
    TUNING_RESULT_COLUMNS,
)
from src.evaluate import EvaluationError, predict_xgboost_probabilities
from src.matched_experiment import (
    MatchedExperimentError,
    feature_columns_for_set,
    prepare_matched_experiment,
)
from src.split_data import SplitError, load_split_policy, read_model_dataset, split_model_dataset
from src.train_baselines import (
    BaselineError,
    evaluate_probabilities,
    write_json_atomic,
    write_model_results_atomic,
)


SUPPORTED_MODEL_NAMES = ("model_a", "model_a_matched", "model_b")
SUPPORTED_FEATURE_SETS = ("baseline", "baseline_matched", "possession")
MODEL_FEATURE_SET = {
    "model_a": "baseline",
    "model_a_matched": "baseline_matched",
    "model_b": "possession",
}
OBJECTIVE = "multi:softprob"
NUM_CLASS = 3
EVAL_METRIC = "mlogloss"
APPROVED_GRID = {
    "max_depth": frozenset({2, 3, 4}),
    "learning_rate": frozenset({0.02, 0.03, 0.05}),
    "min_child_weight": frozenset({1, 3, 5}),
    "subsample": frozenset({0.75, 0.90}),
    "colsample_bytree": frozenset({0.75, 0.90}),
    "reg_lambda": frozenset({1.0, 2.0, 5.0}),
}
STARTING_PARAMETERS = {
    "max_depth": 3,
    "learning_rate": 0.03,
    "min_child_weight": 3,
    "subsample": 0.80,
    "colsample_bytree": 0.80,
    "reg_lambda": 2.0,
}
CANDIDATE_PARAMETER_NAMES = tuple(STARTING_PARAMETERS)


class XGBoostTrainingError(RuntimeError):
    """Raised when Model A tuning, training, or artifact generation fails."""


@dataclass(frozen=True)
class SearchConfig:
    """Validated, bounded XGBoost search definition."""

    n_estimators: int
    early_stopping_rounds: int
    tie_tolerance: float
    candidates: tuple[dict[str, object], ...]


@dataclass(frozen=True)
class CandidateOutcome:
    """One fitted candidate and its validation-only selection metrics."""

    candidate_id: str
    parameters: dict[str, object]
    model: XGBClassifier
    validation_log_loss: float
    validation_macro_f1: float
    validation_accuracy: float
    best_iteration: int


@dataclass(frozen=True)
class XGBoostTrainingSummary:
    """Selected configuration, metrics, and output paths."""

    selected_candidate_id: str
    attempted_candidates: int
    best_iteration: int
    validation_log_loss: float
    test_log_loss: float
    majority_test_log_loss: float | None
    logistic_test_log_loss: float | None
    model_path: Path
    metadata_path: Path
    tuning_results_path: Path
    model_results_path: Path


def _read_json_object(path: Path) -> dict[str, object]:
    try:
        with path.open("r", encoding="utf-8") as input_file:
            value = json.load(input_file)
    except FileNotFoundError as exc:
        raise XGBoostTrainingError(f"Required configuration not found: {path}") from exc
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise XGBoostTrainingError(f"Could not read configuration {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise XGBoostTrainingError(f"Configuration {path} must contain a JSON object.")
    return value


def _normalize_candidate(raw: object, position: int) -> dict[str, object]:
    if not isinstance(raw, dict):
        raise XGBoostTrainingError(f"Search candidate {position} must be a JSON object.")
    expected_fields = {"candidate_id", *CANDIDATE_PARAMETER_NAMES}
    if set(raw) != expected_fields:
        raise XGBoostTrainingError(
            f"Search candidate {position} fields differ from the approved contract."
        )
    candidate_id = raw["candidate_id"]
    if not isinstance(candidate_id, str) or not candidate_id.strip():
        raise XGBoostTrainingError(f"Search candidate {position} has an invalid ID.")
    try:
        candidate = {
            "candidate_id": candidate_id.strip(),
            "max_depth": int(raw["max_depth"]),
            "learning_rate": float(raw["learning_rate"]),
            "min_child_weight": int(raw["min_child_weight"]),
            "subsample": float(raw["subsample"]),
            "colsample_bytree": float(raw["colsample_bytree"]),
            "reg_lambda": float(raw["reg_lambda"]),
        }
    except (TypeError, ValueError) as exc:
        raise XGBoostTrainingError(
            f"Search candidate {candidate_id!r} contains nonnumeric parameters."
        ) from exc
    if candidate["max_depth"] != raw["max_depth"] or candidate[
        "min_child_weight"
    ] != raw["min_child_weight"]:
        raise XGBoostTrainingError(
            f"Search candidate {candidate_id!r} requires integer depth and child weight."
        )
    return candidate


def load_search_config(path: Path) -> SearchConfig:
    """Load and enforce the approved bounded hyperparameter search."""
    config = _read_json_object(path)
    raw_search = config.get("model_a_search")
    if not isinstance(raw_search, dict):
        raise XGBoostTrainingError("model_config.json is missing model_a_search.")
    if raw_search.get("selection_metric") != "validation_log_loss":
        raise XGBoostTrainingError("Model A selection metric must be validation log loss.")
    try:
        n_estimators = int(raw_search["n_estimators"])
        early_stopping_rounds = int(raw_search["early_stopping_rounds"])
        tie_tolerance = float(raw_search["tie_tolerance"])
    except (KeyError, TypeError, ValueError) as exc:
        raise XGBoostTrainingError("Model A search controls are missing or invalid.") from exc
    if n_estimators != 500 or early_stopping_rounds != 40:
        raise XGBoostTrainingError(
            "Model A must use 500 estimators and 40 early-stopping rounds."
        )
    if not 0 <= tie_tolerance <= 0.001:
        raise XGBoostTrainingError("Model A tie tolerance must be between 0 and 0.001.")
    raw_candidates = raw_search.get("candidates")
    if not isinstance(raw_candidates, list) or not 1 <= len(raw_candidates) <= 12:
        raise XGBoostTrainingError("Model A requires a bounded search of 1 to 12 candidates.")
    candidates = tuple(
        _normalize_candidate(raw_candidate, position)
        for position, raw_candidate in enumerate(raw_candidates, start=1)
    )
    candidate_ids = [str(candidate["candidate_id"]) for candidate in candidates]
    if len(set(candidate_ids)) != len(candidate_ids):
        raise XGBoostTrainingError("Model A candidate IDs must be unique.")

    starting_candidates = 0
    for candidate in candidates:
        parameters = {name: candidate[name] for name in CANDIDATE_PARAMETER_NAMES}
        if parameters == STARTING_PARAMETERS:
            starting_candidates += 1
            continue
        for parameter, approved_values in APPROVED_GRID.items():
            if candidate[parameter] not in approved_values:
                raise XGBoostTrainingError(
                    f"Candidate {candidate['candidate_id']!r} uses unapproved "
                    f"{parameter}={candidate[parameter]}."
                )
    if starting_candidates != 1:
        raise XGBoostTrainingError(
            "The bounded search must include the exact starting configuration once."
        )
    return SearchConfig(
        n_estimators=n_estimators,
        early_stopping_rounds=early_stopping_rounds,
        tie_tolerance=tie_tolerance,
        candidates=candidates,
    )


def _complete_parameters(
    candidate: Mapping[str, object],
    search: SearchConfig,
) -> dict[str, object]:
    return {
        "objective": OBJECTIVE,
        "num_class": NUM_CLASS,
        "n_estimators": search.n_estimators,
        "learning_rate": float(candidate["learning_rate"]),
        "max_depth": int(candidate["max_depth"]),
        "min_child_weight": int(candidate["min_child_weight"]),
        "subsample": float(candidate["subsample"]),
        "colsample_bytree": float(candidate["colsample_bytree"]),
        "reg_lambda": float(candidate["reg_lambda"]),
        "eval_metric": EVAL_METRIC,
        "early_stopping_rounds": search.early_stopping_rounds,
        "random_state": RANDOM_SEED,
        "n_jobs": 1,
        "tree_method": "hist",
    }


def _target(frame: pd.DataFrame) -> pd.Series:
    values = pd.to_numeric(frame["target"], errors="coerce")
    if values.isna().any() or set(values.astype(int).unique()) != set(CLASS_LABELS):
        raise XGBoostTrainingError("Each fitted split must contain classes 0, 1, and 2.")
    return values.astype("int64")


def fit_candidate(
    candidate: Mapping[str, object],
    search: SearchConfig,
    *,
    training_features: pd.DataFrame,
    training_target: pd.Series,
    validation_frame: pd.DataFrame,
    validation_features: pd.DataFrame,
    validation_target: pd.Series,
    medians: Mapping[str, float],
    feature_columns: Sequence[str] = FEATURE_COLUMNS,
    feature_set: str = "baseline",
) -> CandidateOutcome:
    """Fit one candidate with validation-only early stopping and metrics."""
    parameters = _complete_parameters(candidate, search)
    model = XGBClassifier(**parameters)
    try:
        model.fit(
            training_features,
            training_target,
            eval_set=[(validation_features, validation_target)],
            verbose=False,
        )
    except (TypeError, ValueError, XGBoostError) as exc:
        raise XGBoostTrainingError(
            f"Candidate {candidate['candidate_id']!r} failed to fit: {exc}"
        ) from exc
    try:
        best_iteration = int(model.best_iteration)
    except (AttributeError, TypeError, ValueError) as exc:
        raise XGBoostTrainingError(
            f"Candidate {candidate['candidate_id']!r} has no best iteration."
        ) from exc
    try:
        probabilities = predict_xgboost_probabilities(
            model,
            validation_frame,
            feature_columns=feature_columns,
            medians=medians,
            best_iteration=best_iteration,
        )
        metrics = evaluate_probabilities(
            model_name="candidate",
            model_family="xgboost",
            feature_set=feature_set,
            split_name="validation",
            frame=validation_frame,
            probabilities=probabilities,
            parameters=parameters,
        )
    except (EvaluationError, BaselineError) as exc:
        raise XGBoostTrainingError(
            f"Candidate {candidate['candidate_id']!r} validation failed: {exc}"
        ) from exc
    return CandidateOutcome(
        candidate_id=str(candidate["candidate_id"]),
        parameters=parameters,
        model=model,
        validation_log_loss=float(metrics["log_loss"]),
        validation_macro_f1=float(metrics["macro_f1"]),
        validation_accuracy=float(metrics["accuracy"]),
        best_iteration=best_iteration,
    )


def select_candidate(
    outcomes: Sequence[CandidateOutcome],
    *,
    tie_tolerance: float,
) -> CandidateOutcome:
    """Select lowest validation loss, favoring shallower candidates in a tie."""
    if not outcomes:
        raise XGBoostTrainingError("No fitted candidates are available for selection.")
    best_loss = min(outcome.validation_log_loss for outcome in outcomes)
    effectively_tied = [
        outcome
        for outcome in outcomes
        if outcome.validation_log_loss <= best_loss + tie_tolerance
    ]
    return min(
        effectively_tied,
        key=lambda outcome: (
            int(outcome.parameters["max_depth"]),
            outcome.best_iteration,
            outcome.candidate_id,
        ),
    )


def build_tuning_results(
    outcomes: Sequence[CandidateOutcome],
    selected: CandidateOutcome,
    *,
    model_name: str = "model_a",
    feature_set: str = "baseline",
) -> pd.DataFrame:
    """Create one auditable validation-only row per attempted configuration."""
    rows = []
    for outcome in outcomes:
        parameters = outcome.parameters
        rows.append(
            {
                "model_name": model_name,
                "feature_set": feature_set,
                "candidate_id": outcome.candidate_id,
                "selected": outcome.candidate_id == selected.candidate_id,
                "validation_log_loss": outcome.validation_log_loss,
                "validation_macro_f1": outcome.validation_macro_f1,
                "validation_accuracy": outcome.validation_accuracy,
                "best_iteration": outcome.best_iteration,
                "objective": parameters["objective"],
                "num_class": parameters["num_class"],
                "n_estimators": parameters["n_estimators"],
                "learning_rate": parameters["learning_rate"],
                "max_depth": parameters["max_depth"],
                "min_child_weight": parameters["min_child_weight"],
                "subsample": parameters["subsample"],
                "colsample_bytree": parameters["colsample_bytree"],
                "reg_lambda": parameters["reg_lambda"],
                "eval_metric": parameters["eval_metric"],
                "early_stopping_rounds": parameters["early_stopping_rounds"],
                "random_state": parameters["random_state"],
            }
        )
    return pd.DataFrame(rows, columns=TUNING_RESULT_COLUMNS)


def _write_tuning_results_atomic(results: pd.DataFrame, path: Path) -> None:
    if tuple(results.columns) != TUNING_RESULT_COLUMNS:
        raise XGBoostTrainingError("Tuning result columns do not match the contract.")
    path.parent.mkdir(parents=True, exist_ok=True)
    file_descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(file_descriptor, "w", encoding="utf-8", newline="") as output:
            results.to_csv(
                output,
                index=False,
                columns=TUNING_RESULT_COLUMNS,
                lineterminator="\n",
            )
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary_path, path)
    except (OSError, TypeError, ValueError) as exc:
        raise XGBoostTrainingError(f"Could not write tuning results {path}: {exc}") from exc
    finally:
        temporary_path.unlink(missing_ok=True)


def _read_existing_tuning_results(path: Path) -> pd.DataFrame:
    """Read validation-only tuning history for all completed model variants."""
    try:
        results = pd.read_csv(path)
    except FileNotFoundError:
        return pd.DataFrame(columns=TUNING_RESULT_COLUMNS)
    except (
        OSError,
        UnicodeDecodeError,
        pd.errors.EmptyDataError,
        pd.errors.ParserError,
    ) as exc:
        raise XGBoostTrainingError(
            f"Could not read tuning report {path}: {exc}"
        ) from exc
    if tuple(results.columns) != TUNING_RESULT_COLUMNS:
        raise XGBoostTrainingError(
            "Tuning report columns differ from the Phase 7 contract."
        )
    if any("test" in column or "holdout" in column for column in results.columns):
        raise XGBoostTrainingError(
            "Tuning report must contain validation-only selection metrics."
        )
    return results


def _read_existing_model_results(path: Path) -> pd.DataFrame:
    try:
        results = pd.read_csv(path)
    except FileNotFoundError as exc:
        raise XGBoostTrainingError(
            f"Baseline report not found: {path}. Run python -m src.train_baselines first."
        ) from exc
    except (
        OSError,
        UnicodeDecodeError,
        pd.errors.EmptyDataError,
        pd.errors.ParserError,
    ) as exc:
        raise XGBoostTrainingError(f"Could not read baseline report {path}: {exc}") from exc
    if tuple(results.columns) != MODEL_RESULT_COLUMNS:
        raise XGBoostTrainingError("Baseline report columns differ from the fixed contract.")
    required = {
        ("majority_baseline", "validation"),
        ("majority_baseline", "test"),
        ("logistic_regression", "validation"),
        ("logistic_regression", "test"),
    }
    observed = set(zip(results["model_name"], results["split"]))
    if not required.issubset(observed):
        raise XGBoostTrainingError("Baseline report is missing required comparison rows.")
    if "holdout" in set(results["split"]):
        raise XGBoostTrainingError("Holdout metrics already exist; Phase 5 must not proceed.")
    return results


def _comparison_metric(
    results: pd.DataFrame,
    model_name: str,
    split_name: str,
    metric: str,
) -> float:
    rows = results.loc[
        results["model_name"].eq(model_name) & results["split"].eq(split_name),
        metric,
    ]
    if len(rows) != 1:
        raise XGBoostTrainingError(
            f"Expected one {model_name}/{split_name} row in model_results.csv."
        )
    return float(rows.iloc[0])


def _save_xgboost_model_atomic(model: XGBClassifier, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    file_descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.stem}.", suffix=".json"
    )
    os.close(file_descriptor)
    temporary_path = Path(temporary_name)
    try:
        model.save_model(temporary_path)
        with temporary_path.open("r+b") as model_file:
            os.fsync(model_file.fileno())
        os.replace(temporary_path, path)
    except (OSError, TypeError, ValueError, XGBoostError) as exc:
        raise XGBoostTrainingError(f"Could not save XGBoost model {path}: {exc}") from exc
    finally:
        temporary_path.unlink(missing_ok=True)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as input_file:
            for block in iter(lambda: input_file.read(1024 * 1024), b""):
                digest.update(block)
    except OSError as exc:
        raise XGBoostTrainingError(f"Could not hash required file {path}: {exc}") from exc
    return digest.hexdigest()


def _git_state(project_root: Path) -> tuple[str, bool]:
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=project_root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=project_root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError) as exc:
        raise XGBoostTrainingError(f"Could not record Git training state: {exc}") from exc
    if len(commit) != 40:
        raise XGBoostTrainingError("Git did not return a complete commit hash.")
    return commit, bool(status.strip())


def _save_figure_atomic(figure: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    file_descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.stem}.", suffix=".png"
    )
    os.close(file_descriptor)
    temporary_path = Path(temporary_name)
    try:
        figure.savefig(temporary_path, format="png", dpi=160, bbox_inches="tight")
        with temporary_path.open("r+b") as image_file:
            os.fsync(image_file.fileno())
        os.replace(temporary_path, path)
    except (OSError, TypeError, ValueError) as exc:
        raise XGBoostTrainingError(f"Could not save report image {path}: {exc}") from exc
    finally:
        plt.close(figure)
        temporary_path.unlink(missing_ok=True)


def _plot_class_distribution(
    training: pd.DataFrame,
    validation: pd.DataFrame,
    test: pd.DataFrame,
    path: Path,
) -> None:
    frames = {"Train": training, "Validation": validation, "Test": test}
    positions = np.arange(len(CLASS_LABELS), dtype="float64")
    width = 0.24
    figure, axis = plt.subplots(figsize=(8, 5))
    colors = ("#1f77b4", "#ff7f0e", "#2ca02c")
    for index, ((split_name, frame), color) in enumerate(zip(frames.items(), colors)):
        proportions = (
            frame["target"].value_counts(normalize=True).reindex(CLASS_LABELS, fill_value=0)
        )
        axis.bar(
            positions + (index - 1) * width,
            proportions.to_numpy(),
            width,
            label=split_name,
            color=color,
        )
    axis.set_xticks(positions, ("Away win", "Draw", "Home win"))
    axis.set_ylabel("Share of fixtures")
    axis.set_title("Outcome class distribution by opened split")
    axis.set_ylim(0, 0.55)
    axis.legend()
    axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    _save_figure_atomic(figure, path)


def _plot_confusion_matrix(
    actual: pd.Series,
    probabilities: np.ndarray,
    path: Path,
    *,
    model_name: str = "model_a",
) -> None:
    predicted = np.asarray(CLASS_LABELS)[probabilities.argmax(axis=1)]
    matrix = confusion_matrix(actual, predicted, labels=CLASS_LABELS)
    figure, axis = plt.subplots(figsize=(6, 5))
    image = axis.imshow(matrix, cmap="Blues")
    labels = ("Away win", "Draw", "Home win")
    axis.set_xticks(np.arange(3), labels)
    axis.set_yticks(np.arange(3), labels)
    axis.set_xlabel("Predicted outcome")
    axis.set_ylabel("Actual outcome")
    axis.set_title("Model A confusion matrix — 2024/25 test")
    display_name = model_name.replace("_", " ").title()
    axis.set_title(f"{display_name} confusion matrix — 2024/25 test")
    threshold = matrix.max() / 2 if matrix.size else 0
    for row in range(3):
        for column in range(3):
            axis.text(
                column,
                row,
                str(matrix[row, column]),
                ha="center",
                va="center",
                color="white" if matrix[row, column] > threshold else "black",
            )
    figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    figure.tight_layout()
    _save_figure_atomic(figure, path)


def _plot_feature_importance(
    model: XGBClassifier,
    path: Path,
    *,
    feature_columns: Sequence[str] = FEATURE_COLUMNS,
    model_name: str = "model_a",
) -> None:
    booster = model.get_booster()
    gains = booster.get_score(importance_type="gain")
    importance = pd.Series(
        {feature: float(gains.get(feature, 0.0)) for feature in feature_columns},
        dtype="float64",
    ).sort_values(ascending=False)
    top = importance.head(15).sort_values()
    figure, axis = plt.subplots(figsize=(9, 6))
    axis.barh(top.index, top.values, color="#1f77b4")
    axis.set_xlabel("Average gain")
    display_name = model_name.replace("_", " ").title()
    axis.set_title(f"{display_name} feature importance (top 15)")
    axis.grid(axis="x", alpha=0.25)
    figure.tight_layout()
    _save_figure_atomic(figure, path)


def _build_preprocessing_record(
    training: pd.DataFrame,
    medians: Mapping[str, float],
    *,
    model_name: str = "model_a",
    feature_set: str = "baseline",
    feature_columns: Sequence[str] = FEATURE_COLUMNS,
) -> dict[str, object]:
    dates = pd.to_datetime(training["date"], format="%Y-%m-%d", errors="raise")
    return {
        "feature_set": feature_set,
        "strategy": "median",
        "fitted_on_split": "train",
        "used_by": [model_name],
        "training_seasons": list(
            dict.fromkeys(training["season"].astype(str).tolist())
        ),
        "training_match_count": len(training),
        "training_date_min": dates.min().date().isoformat(),
        "training_date_max": dates.max().date().isoformat(),
        "feature_columns": list(feature_columns),
        "median_values": {
            feature: float(medians[feature]) for feature in feature_columns
        },
    }


def train_xgboost_model(
    model_name: str,
    feature_set: str,
    *,
    project_root: Path = PROJECT_ROOT,
) -> XGBoostTrainingSummary:
    """Train one approved XGBoost variant without evaluating the holdout."""
    if model_name not in SUPPORTED_MODEL_NAMES:
        raise XGBoostTrainingError(
            f"Unsupported model name {model_name!r}."
        )
    if feature_set not in SUPPORTED_FEATURE_SETS:
        raise XGBoostTrainingError(
            f"Unsupported feature set {feature_set!r}."
        )
    expected_feature_set = MODEL_FEATURE_SET[model_name]
    if feature_set != expected_feature_set:
        raise XGBoostTrainingError(
            f"{model_name} requires --feature-set {expected_feature_set}."
        )
    project_root = project_root.resolve()
    config_path = project_root / "config" / "model_config.json"
    from src.freeze_model import FreezeError, assert_unfrozen
    try:
        assert_unfrozen(project_root)
    except FreezeError as exc:
        raise XGBoostTrainingError(str(exc)) from exc
    search = load_search_config(config_path)
    if model_name == "model_a":
        policy = load_split_policy(config_path)
        dataset = read_model_dataset(
            project_root / "data" / "processed" / "model_dataset.csv"
        )
        splits = split_model_dataset(dataset, policy)
        feature_columns = FEATURE_COLUMNS
        matched_manifest_path: Path | None = None
        model_b_period_start: str | None = None
    else:
        try:
            artifacts = prepare_matched_experiment(
                reset_freeze=model_name == "model_a_matched",
                project_root=project_root,
            )
        except MatchedExperimentError as exc:
            raise XGBoostTrainingError(str(exc)) from exc
        splits = artifacts.splits
        feature_columns = feature_columns_for_set(feature_set)
        matched_manifest_path = artifacts.manifest_path
        model_b_period_start = artifacts.model_b_period_start
        if model_name == "model_b":
            matched_metadata = _read_json_object(
                project_root / "models" / "model_a_matched_metadata.json"
            )
            expected_manifest_hash = _sha256(artifacts.manifest_path)
            if (
                matched_metadata.get("model_name") != "model_a_matched"
                or matched_metadata.get("feature_set") != "baseline_matched"
                or matched_metadata.get("matched_split_manifest_sha256")
                != expected_manifest_hash
                or matched_metadata.get("holdout_evaluated") is not False
            ):
                raise XGBoostTrainingError(
                    "Model A-Matched metadata does not match the frozen cohort; "
                    "retrain model_a_matched before model_b."
                )
            if not (
                project_root / "models" / "model_a_matched_xgb.json"
            ).is_file():
                raise XGBoostTrainingError(
                    "Model A-Matched artifact is missing; train it before model_b."
                )

    try:
        medians = fit_training_medians(splits.train, feature_columns)
    except FeatureError as exc:
        raise XGBoostTrainingError(f"Could not fit training-only medians: {exc}") from exc
    try:
        training_features = apply_training_medians(
            splits.train, medians, feature_columns
        ).loc[:, feature_columns]
        validation_features = apply_training_medians(
            splits.validation, medians, feature_columns
        ).loc[:, feature_columns]
    except FeatureError as exc:
        raise XGBoostTrainingError(f"Could not preprocess training splits: {exc}") from exc
    training_target = _target(splits.train)
    validation_target = _target(splits.validation)

    outcomes = []
    for candidate in search.candidates:
        outcomes.append(
            fit_candidate(
                candidate,
                search,
                training_features=training_features,
                training_target=training_target,
                validation_frame=splits.validation,
                validation_features=validation_features,
                validation_target=validation_target,
                medians=medians,
                feature_columns=feature_columns,
                feature_set=feature_set,
            )
        )
    selected = select_candidate(outcomes, tie_tolerance=search.tie_tolerance)
    tuning_results = build_tuning_results(
        outcomes,
        selected,
        model_name=model_name,
        feature_set=feature_set,
    )

    try:
        validation_probabilities = predict_xgboost_probabilities(
            selected.model,
            splits.validation,
            feature_columns=feature_columns,
            medians=medians,
            best_iteration=selected.best_iteration,
        )
        test_probabilities = predict_xgboost_probabilities(
            selected.model,
            splits.test,
            feature_columns=feature_columns,
            medians=medians,
            best_iteration=selected.best_iteration,
        )
        validation_result = evaluate_probabilities(
            model_name=model_name,
            model_family="xgboost",
            feature_set=feature_set,
            split_name="validation",
            frame=splits.validation,
            probabilities=validation_probabilities,
            parameters=selected.parameters,
        )
        test_result = evaluate_probabilities(
            model_name=model_name,
            model_family="xgboost",
            feature_set=feature_set,
            split_name="test",
            frame=splits.test,
            probabilities=test_probabilities,
            parameters=selected.parameters,
        )
    except (EvaluationError, BaselineError) as exc:
        raise XGBoostTrainingError(
            f"Selected {model_name} evaluation failed: {exc}"
        ) from exc
    validation_result["best_iteration"] = selected.best_iteration
    test_result["best_iteration"] = selected.best_iteration

    results_path = project_root / "reports" / "model_results.csv"
    existing_results = _read_existing_model_results(results_path)
    invalidated_models = {model_name}
    if model_name == "model_a_matched":
        invalidated_models.add("model_b")
    baseline_results = existing_results.loc[
        ~existing_results["model_name"].isin(invalidated_models)
    ].copy()
    combined_results = pd.concat(
        [
            baseline_results,
            pd.DataFrame([validation_result, test_result], columns=MODEL_RESULT_COLUMNS),
        ],
        ignore_index=True,
    ).loc[:, MODEL_RESULT_COLUMNS]
    if "holdout" in set(combined_results["split"]):
        raise XGBoostTrainingError("Holdout results are prohibited during Phase 7.")

    majority_test_loss: float | None = None
    logistic_test_loss: float | None = None
    if model_name == "model_a":
        majority_test_loss = _comparison_metric(
            combined_results, "majority_baseline", "test", "log_loss"
        )
        logistic_test_loss = _comparison_metric(
            combined_results, "logistic_regression", "test", "log_loss"
        )
        if float(test_result["log_loss"]) >= majority_test_loss:
            raise XGBoostTrainingError(
                "Selected Model A does not beat the majority test log loss."
            )

    model_path = project_root / "models" / f"{model_name}_xgb.json"
    if model_name == "model_a":
        metadata_path = project_root / "models" / "model_metadata.json"
        preprocessing_path = project_root / "models" / "preprocessing.json"
    else:
        metadata_path = project_root / "models" / f"{model_name}_metadata.json"
        preprocessing_path = (
            project_root / "models" / f"{model_name}_preprocessing.json"
        )
    tuning_path = project_root / "reports" / "tuning_results.csv"
    _save_xgboost_model_atomic(selected.model, model_path)
    preprocessing = _build_preprocessing_record(
        splits.train,
        medians,
        model_name=model_name,
        feature_set=feature_set,
        feature_columns=feature_columns,
    )
    write_json_atomic(preprocessing, preprocessing_path)

    git_commit, working_tree_dirty = _git_state(project_root)
    training_dates = pd.to_datetime(splits.train["date"])
    metadata = {
        "model_name": model_name,
        "feature_set": feature_set,
        "trained_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "feature_columns": list(feature_columns),
        "classes": list(CLASS_LABELS),
        "best_iteration": selected.best_iteration,
        "selected_candidate_id": selected.candidate_id,
        "parameters": selected.parameters,
        "training_match_count": len(splits.train),
        "validation_match_count": len(splits.validation),
        "test_match_count": len(splits.test),
        "holdout_match_count": len(splits.holdout),
        "training_date_min": training_dates.min().date().isoformat(),
        "training_date_max": training_dates.max().date().isoformat(),
        "source_manifest_sha256": _sha256(project_root / "data" / "raw" / "manifest.json"),
        "split_manifest_sha256": _sha256(
            matched_manifest_path
            if matched_manifest_path is not None
            else project_root / "data" / "processed" / "split_manifest.csv"
        ),
        "model_sha256": _sha256(model_path),
        "preprocessing_sha256": _sha256(preprocessing_path),
        "git_commit": git_commit,
        "working_tree_dirty_at_training": working_tree_dirty,
        "validation_metrics": {
            "log_loss": float(validation_result["log_loss"]),
            "macro_f1": float(validation_result["macro_f1"]),
            "accuracy": float(validation_result["accuracy"]),
        },
        "test_metrics": {
            "log_loss": float(test_result["log_loss"]),
            "macro_f1": float(test_result["macro_f1"]),
            "accuracy": float(test_result["accuracy"]),
        },
        "holdout_evaluated": False,
    }
    if matched_manifest_path is not None:
        metadata["matched_split_manifest_sha256"] = _sha256(
            matched_manifest_path
        )
        metadata["matched_dataset_sha256"] = _sha256(
            project_root / "data" / "processed" / "matched_model_dataset.csv"
        )
        metadata["model_b_period_start"] = model_b_period_start
    write_json_atomic(metadata, metadata_path)
    existing_tuning = _read_existing_tuning_results(tuning_path)
    invalidated_tuning_models = {model_name}
    if model_name == "model_a_matched":
        invalidated_tuning_models.add("model_b")
    combined_tuning = pd.concat(
        [
            existing_tuning.loc[
                ~existing_tuning["model_name"].isin(
                    invalidated_tuning_models
                )
            ],
            tuning_results,
        ],
        ignore_index=True,
    ).loc[:, TUNING_RESULT_COLUMNS]
    _write_tuning_results_atomic(combined_tuning, tuning_path)
    write_model_results_atomic(combined_results, results_path)
    if model_name == "model_a":
        _plot_class_distribution(
            splits.train,
            splits.validation,
            splits.test,
            project_root / "reports" / "class_distribution.png",
        )
    _plot_confusion_matrix(
        _target(splits.test),
        test_probabilities,
        project_root / "reports" / f"confusion_matrix_{model_name}.png",
        model_name=model_name,
    )
    _plot_feature_importance(
        selected.model,
        project_root / "reports" / f"feature_importance_{model_name}.png",
        feature_columns=feature_columns,
        model_name=model_name,
    )
    return XGBoostTrainingSummary(
        selected_candidate_id=selected.candidate_id,
        attempted_candidates=len(outcomes),
        best_iteration=selected.best_iteration,
        validation_log_loss=float(validation_result["log_loss"]),
        test_log_loss=float(test_result["log_loss"]),
        majority_test_log_loss=majority_test_loss,
        logistic_test_log_loss=logistic_test_loss,
        model_path=model_path,
        metadata_path=metadata_path,
        tuning_results_path=tuning_path,
        model_results_path=results_path,
    )


def train_model_a(
    model_name: str,
    feature_set: str,
    *,
    project_root: Path = PROJECT_ROOT,
) -> XGBoostTrainingSummary:
    """Backward-compatible entry point for the completed Phase 5 workflow."""
    if model_name != "model_a" or feature_set != "baseline":
        raise XGBoostTrainingError(
            "train_model_a requires model_a with the baseline feature set."
        )
    return train_xgboost_model(
        model_name, feature_set, project_root=project_root
    )


def build_parser() -> argparse.ArgumentParser:
    """Build the Phase 5/7 XGBoost training CLI parser."""
    parser = argparse.ArgumentParser(
        description="Tune and train an approved EPL XGBoost model variant."
    )
    parser.add_argument("--model-name", required=True, choices=SUPPORTED_MODEL_NAMES)
    parser.add_argument("--feature-set", required=True, choices=SUPPORTED_FEATURE_SETS)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Train an approved variant while keeping the final holdout unopened."""
    arguments = build_parser().parse_args(argv)
    try:
        summary = train_xgboost_model(
            arguments.model_name, arguments.feature_set
        )
    except (XGBoostTrainingError, SplitError, MatchedExperimentError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(f"attempted_candidates={summary.attempted_candidates}")
    print(f"selected_candidate={summary.selected_candidate_id}")
    print(f"best_iteration={summary.best_iteration}")
    print(f"validation_log_loss={summary.validation_log_loss:.12f}")
    print(f"test_log_loss={summary.test_log_loss:.12f}")
    if summary.majority_test_log_loss is not None:
        print(
            f"majority_test_log_loss={summary.majority_test_log_loss:.12f}"
        )
    if summary.logistic_test_log_loss is not None:
        print(
            f"logistic_test_log_loss={summary.logistic_test_log_loss:.12f}"
        )
    print("evaluated_splits=validation,test")
    print("holdout_evaluated=False")
    print(f"model={summary.model_path.relative_to(PROJECT_ROOT).as_posix()}")
    print(f"metadata={summary.metadata_path.relative_to(PROJECT_ROOT).as_posix()}")
    print(f"tuning_results={summary.tuning_results_path.relative_to(PROJECT_ROOT).as_posix()}")
    print(f"model_results={summary.model_results_path.relative_to(PROJECT_ROOT).as_posix()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
