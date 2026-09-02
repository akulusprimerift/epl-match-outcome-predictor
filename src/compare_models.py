"""Compare Phase 7 matched models without selecting a production candidate."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import os
from pathlib import Path
import sys
import tempfile
from typing import Mapping, Sequence

import pandas as pd

from src.clean_data import PROJECT_ROOT
from src.constants import (
    FEATURE_COLUMNS,
    MODEL_RESULT_COLUMNS,
    POSSESSION_FEATURE_ADDITIONS,
    POSSESSION_FEATURE_COLUMNS,
)
from src.evaluate import EvaluationError, evaluate_saved_model, load_json_object


PHASE_7_MODELS = ("model_a_matched", "model_b")
COMPARISON_REPORT_PATH = PROJECT_ROOT / "reports" / "possession_experiment.md"


class ModelComparisonError(RuntimeError):
    """Raised when a fair matched-model comparison cannot be proven."""


@dataclass(frozen=True)
class ModelComparisonSummary:
    """Rule outcomes and the atomically written Phase 7 report."""

    model_b_lower_log_loss: bool
    macro_f1_guardrail: bool
    integrity_and_probabilities: bool
    model_b_accepted: bool
    report_path: Path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as input_file:
            for chunk in iter(lambda: input_file.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as exc:
        raise ModelComparisonError(
            f"Could not hash required artifact {path}: {exc}"
        ) from exc
    return digest.hexdigest()


def _load_results(path: Path) -> pd.DataFrame:
    try:
        results = pd.read_csv(path)
    except FileNotFoundError as exc:
        raise ModelComparisonError(
            f"Model results not found: {path}. Train both Phase 7 models first."
        ) from exc
    except (
        OSError,
        UnicodeDecodeError,
        pd.errors.EmptyDataError,
        pd.errors.ParserError,
    ) as exc:
        raise ModelComparisonError(
            f"Could not read model results {path}: {exc}"
        ) from exc
    if tuple(results.columns) != MODEL_RESULT_COLUMNS:
        raise ModelComparisonError(
            "Model result columns differ from the fixed contract."
        )
    if "holdout" in set(results["split"].astype(str)):
        raise ModelComparisonError(
            "Holdout metrics are prohibited during Phase 7."
        )
    return results


def _metric_row(
    results: pd.DataFrame, model_name: str, split_name: str
) -> pd.Series:
    rows = results.loc[
        results["model_name"].eq(model_name)
        & results["split"].eq(split_name)
    ]
    if len(rows) != 1:
        raise ModelComparisonError(
            f"Expected one {model_name}/{split_name} metrics row; "
            f"found {len(rows)}."
        )
    return rows.iloc[0]


def _validate_metadata_pair(
    left: Mapping[str, object],
    right: Mapping[str, object],
    *,
    manifest_sha256: str,
    dataset_sha256: str,
) -> None:
    expected = {
        "model_a_matched": ("baseline_matched", FEATURE_COLUMNS),
        "model_b": ("possession", POSSESSION_FEATURE_COLUMNS),
    }
    for metadata in (left, right):
        model_name = metadata.get("model_name")
        if model_name not in expected:
            raise ModelComparisonError(
                f"Unexpected matched-model metadata name: {model_name!r}."
            )
        feature_set, feature_columns = expected[str(model_name)]
        if metadata.get("feature_set") != feature_set or tuple(
            metadata.get("feature_columns", ())
        ) != feature_columns:
            raise ModelComparisonError(
                f"{model_name} metadata violates its feature contract."
            )
        if metadata.get("holdout_evaluated") is not False:
            raise ModelComparisonError(
                f"{model_name} metadata does not keep the holdout unopened."
            )
        if metadata.get("matched_split_manifest_sha256") != manifest_sha256:
            raise ModelComparisonError(
                f"{model_name} was not trained on the current frozen match-ID manifest."
            )
        if metadata.get("matched_dataset_sha256") != dataset_sha256:
            raise ModelComparisonError(
                f"{model_name} was not trained on the current matched dataset."
            )

    for count_field in (
        "training_match_count",
        "validation_match_count",
        "test_match_count",
        "holdout_match_count",
    ):
        left_count = left.get(count_field)
        right_count = right.get(count_field)
        if (
            not isinstance(left_count, int)
            or left_count <= 0
            or not isinstance(right_count, int)
            or right_count <= 0
            or left_count != right_count
        ):
            raise ModelComparisonError(
                f"Matched model metadata differs on {count_field}."
            )
    feature_difference = set(right["feature_columns"]) - set(
        left["feature_columns"]
    )
    if feature_difference != set(POSSESSION_FEATURE_ADDITIONS):
        raise ModelComparisonError(
            "Only the three approved possession features may differ between models."
        )
    if set(left["feature_columns"]) - set(right["feature_columns"]):
        raise ModelComparisonError(
            "Model B is missing one or more matched baseline features."
        )


def _format_metric(value: object) -> str:
    return f"{float(value):.6f}"


def _build_report(
    rows: Mapping[tuple[str, str], pd.Series],
    *,
    lower_log_loss: bool,
    macro_guardrail: bool,
    integrity: bool,
    accepted: bool,
    max_macro_f1_drop: float,
) -> str:
    lines = [
        "# Phase 7 Matched Possession Experiment",
        "",
        "Model A-Matched and Model B use one frozen possession-complete fixture "
        "cohort. This comparison measures predictive association on that cohort; "
        "it does not establish that possession causes match outcomes.",
        "",
        "| Model | Split | Rows | Log loss | Macro F1 | Accuracy |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for model_name in PHASE_7_MODELS:
        for split_name in ("validation", "test"):
            row = rows[(model_name, split_name)]
            lines.append(
                "| "
                + " | ".join(
                    (
                        model_name,
                        split_name,
                        str(int(row["row_count"])),
                        _format_metric(row["log_loss"]),
                        _format_metric(row["macro_f1"]),
                        _format_metric(row["accuracy"]),
                    )
                )
                + " |"
            )
    lines.extend(
        [
            "",
            "## Model B acceptance rule",
            "",
            f"- {'PASS' if lower_log_loss else 'FAIL'} — Test log loss is lower "
            "than Model A-Matched.",
            f"- {'PASS' if macro_guardrail else 'FAIL'} — Test macro F1 is no "
            f"more than {max_macro_f1_drop:.2f} below Model A-Matched.",
            f"- {'PASS' if integrity else 'FAIL'} — Frozen-row, feature-contract, "
            "class-order, and probability-normalization checks pass.",
            "",
            f"**Phase 7 result: {'Model B passes' if accepted else 'Model B does not pass'} "
            "the declared incremental-value rule.**",
            "",
            "This is an experiment result only. Production-candidate selection "
            "belongs to Phase 8, and the 2025/26 holdout remains unopened.",
            "",
            "Per-class precision, recall, F1, support, and confusion-matrix counts "
            "are stored in `reports/model_results.csv`; model-specific confusion-"
            "matrix images are stored alongside this report.",
            "",
        ]
    )
    return "\n".join(lines)


def _write_text_atomic(text: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    file_descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(
            file_descriptor, "w", encoding="utf-8", newline="\n"
        ) as output:
            output.write(text)
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary_path, path)
    except OSError as exc:
        raise ModelComparisonError(
            f"Could not atomically write comparison report {path}: {exc}"
        ) from exc
    finally:
        temporary_path.unlink(missing_ok=True)


def apply_model_b_acceptance_rule(
    *,
    model_a_matched_log_loss: float,
    model_b_log_loss: float,
    model_a_matched_macro_f1: float,
    model_b_macro_f1: float,
    max_macro_f1_drop: float,
    integrity_and_probabilities: bool,
) -> tuple[bool, bool, bool]:
    """Return both metric checks and their combined declared outcome."""
    if not 0.0 <= max_macro_f1_drop <= 1.0:
        raise ModelComparisonError(
            "max_macro_f1_drop must be between 0 and 1."
        )
    lower_log_loss = model_b_log_loss < model_a_matched_log_loss
    macro_guardrail = model_b_macro_f1 >= (
        model_a_matched_macro_f1 - max_macro_f1_drop
    )
    accepted = (
        lower_log_loss and macro_guardrail and integrity_and_probabilities
    )
    return lower_log_loss, macro_guardrail, accepted


def compare_matched_models(
    model_names: Sequence[str],
    *,
    project_root: Path = PROJECT_ROOT,
) -> ModelComparisonSummary:
    """Validate and compare the two Phase 7 models on identical opened rows."""
    if tuple(model_names) != PHASE_7_MODELS:
        raise ModelComparisonError(
            "Phase 7 comparison requires --models model_a_matched model_b "
            "in that order."
        )
    project_root = project_root.resolve()
    results = _load_results(project_root / "reports" / "model_results.csv")
    rows = {
        (model_name, split_name): _metric_row(
            results, model_name, split_name
        )
        for model_name in PHASE_7_MODELS
        for split_name in ("validation", "test")
    }
    manifest_path = (
        project_root / "data" / "processed" / "matched_split_manifest.csv"
    )
    dataset_path = (
        project_root / "data" / "processed" / "matched_model_dataset.csv"
    )
    left = load_json_object(
        project_root / "models" / "model_a_matched_metadata.json"
    )
    right = load_json_object(
        project_root / "models" / "model_b_metadata.json"
    )
    _validate_metadata_pair(
        left,
        right,
        manifest_sha256=_sha256(manifest_path),
        dataset_sha256=_sha256(dataset_path),
    )

    for model_name in PHASE_7_MODELS:
        for split_name in ("validation", "test"):
            evaluated = evaluate_saved_model(
                model_name, split_name, project_root=project_root
            )
            recorded = rows[(model_name, split_name)]
            for metric in ("log_loss", "macro_f1", "accuracy"):
                if abs(
                    float(evaluated[metric]) - float(recorded[metric])
                ) > 1e-12:
                    raise ModelComparisonError(
                        f"Saved {model_name}/{split_name} {metric} differs "
                        "from its report."
                    )

    config = load_json_object(
        project_root / "config" / "model_config.json"
    )
    try:
        max_macro_f1_drop = float(config["max_macro_f1_drop"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ModelComparisonError(
            "model_config.json has an invalid max_macro_f1_drop."
        ) from exc
    matched_test = rows[("model_a_matched", "test")]
    model_b_test = rows[("model_b", "test")]
    integrity = True
    lower_log_loss, macro_guardrail, accepted = (
        apply_model_b_acceptance_rule(
            model_a_matched_log_loss=float(matched_test["log_loss"]),
            model_b_log_loss=float(model_b_test["log_loss"]),
            model_a_matched_macro_f1=float(matched_test["macro_f1"]),
            model_b_macro_f1=float(model_b_test["macro_f1"]),
            max_macro_f1_drop=max_macro_f1_drop,
            integrity_and_probabilities=integrity,
        )
    )
    report_path = project_root / "reports" / "possession_experiment.md"
    _write_text_atomic(
        _build_report(
            rows,
            lower_log_loss=lower_log_loss,
            macro_guardrail=macro_guardrail,
            integrity=integrity,
            accepted=accepted,
            max_macro_f1_drop=max_macro_f1_drop,
        ),
        report_path,
    )
    return ModelComparisonSummary(
        model_b_lower_log_loss=lower_log_loss,
        macro_f1_guardrail=macro_guardrail,
        integrity_and_probabilities=integrity,
        model_b_accepted=accepted,
        report_path=report_path,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compare the two coverage-matched Phase 7 models."
    )
    parser.add_argument(
        "--models",
        nargs=2,
        required=True,
        choices=PHASE_7_MODELS,
        metavar=("MODEL_A_MATCHED", "MODEL_B"),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    try:
        summary = compare_matched_models(arguments.models)
    except (ModelComparisonError, EvaluationError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(
        f"model_b_lower_test_log_loss={summary.model_b_lower_log_loss}"
    )
    print(f"macro_f1_guardrail={summary.macro_f1_guardrail}")
    print(
        "integrity_and_probabilities="
        f"{summary.integrity_and_probabilities}"
    )
    print(f"model_b_accepted={summary.model_b_accepted}")
    print("holdout_evaluated=False")
    print(
        f"report={summary.report_path.relative_to(PROJECT_ROOT).as_posix()}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
