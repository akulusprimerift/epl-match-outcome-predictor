"""Create or verify the Phase 8 record without evaluating the final holdout."""

from __future__ import annotations

import argparse
from copy import deepcopy
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CANDIDATES = ("model_a", "model_a_matched", "model_b")


class FreezeError(RuntimeError):
    """A candidate or an existing freeze cannot be verified."""


def read_json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise FreezeError(f"Expected a JSON object: {path}")
    return value


def digest(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def file_hash(path: Path, *, normalize_text: bool = False) -> str:
    data = path.read_bytes()
    if normalize_text:
        data = data.replace(b"\r\n", b"\n")
    return hashlib.sha256(data).hexdigest()


def record_hash(config: dict) -> str:
    record = deepcopy(config)
    record.pop("freeze_record_sha256", None)
    return digest(record)


def assert_unfrozen(root: Path) -> None:
    config = read_json(root / "config/model_config.json")
    if config.get("frozen_at_utc") or config.get("selected_model"):
        raise FreezeError("Phase 8 is frozen; retraining requires an explicit new freeze decision.")


def test_results(root: Path) -> dict:
    with (root / "reports/model_results.csv").open(encoding="utf-8", newline="") as source:
        rows = list(csv.DictReader(source))
    if any(row["split"] not in {"test", "validation"} for row in rows):
        raise FreezeError("Only validation/test metrics may exist before the holdout is approved.")
    if any((root / "reports").glob("*holdout*")):
        raise FreezeError("A holdout report already exists; do not create a pre-holdout freeze.")
    result = {}
    for name in CANDIDATES:
        matching = [row for row in rows if row["model_name"] == name and row["split"] == "test"]
        if len(matching) != 1:
            raise FreezeError(f"Expected exactly one test result for {name}.")
        row = matching[0]
        values = {key: float(row[key]) for key in ("log_loss", "macro_f1", "accuracy")}
        if not all(math.isfinite(value) for value in values.values()):
            raise FreezeError(f"Nonfinite test metrics for {name}.")
        if values["log_loss"] < 0 or not all(0 <= values[key] <= 1 for key in ("macro_f1", "accuracy")):
            raise FreezeError(f"Out-of-range test metrics for {name}.")
        result[name] = {**values, "row_count": int(row["row_count"])}
    return result


def choose_candidate(results: dict, max_drop: float) -> str:
    """Apply the primary metric, macro-F1 guardrail and Model B experiment gate."""
    if not math.isfinite(max_drop) or not 0 <= max_drop <= 1:
        raise FreezeError("Invalid macro-F1 guardrail.")
    best_f1 = max(row["macro_f1"] for row in results.values())
    eligible = [name for name in CANDIDATES if results[name]["macro_f1"] >= best_f1 - max_drop]
    a, b = results["model_a_matched"], results["model_b"]
    if not (b["log_loss"] < a["log_loss"] and b["macro_f1"] >= a["macro_f1"] - max_drop):
        eligible = [name for name in eligible if name != "model_b"]
    if not eligible:
        raise FreezeError("No candidate passes the declared guardrails.")
    return min(eligible, key=lambda name: (results[name]["log_loss"], CANDIDATES.index(name)))


def artifact_names(name: str) -> tuple[str, str, str]:
    return (
        f"models/{name}_xgb.json",
        "models/model_metadata.json" if name == "model_a" else f"models/{name}_metadata.json",
        "models/preprocessing.json" if name == "model_a" else f"models/{name}_preprocessing.json",
    )


def validate_candidates(root: Path, results: dict) -> None:
    """Reproduce test metrics and training medians; never predict holdout rows."""
    import numpy as np
    import pandas as pd
    from src.build_features import fit_training_medians
    from src.compare_models import _validate_metadata_pair
    from src.evaluate import evaluate_saved_model, load_preprocessing, load_xgboost_model, validate_model_metadata
    from src.matched_experiment import read_matched_model_dataset, split_matched_dataset
    from src.split_data import load_split_policy, read_model_dataset, split_model_dataset

    policy = load_split_policy(root / "config/model_config.json")
    baseline = split_model_dataset(read_model_dataset(root / "data/processed/model_dataset.csv"), policy)
    matched = split_matched_dataset(read_matched_model_dataset(root / "data/processed/matched_model_dataset.csv"), policy)
    # Freeze exactly the fixture IDs used in every split, not just equal row counts.
    for filename, splits in (("split_manifest.csv", baseline), ("matched_split_manifest.csv", matched)):
        manifest = pd.read_csv(root / "data/processed" / filename, dtype=str)
        if not manifest["match_id"].is_unique:
            raise FreezeError(f"Duplicate frozen match IDs in {filename}.")
        expected = {(str(row.match_id), str(row.season), str(row.date), split)
                    for split, frame in splits.by_name().items()
                    for row in frame.itertuples()}
        actual = set(manifest.loc[:, ["match_id", "season", "date", "split"]].itertuples(index=False, name=None))
        if actual != expected:
            raise FreezeError(f"Split manifest differs from model rows: {filename}.")
    if set(baseline.test["match_id"]) != set(matched.test["match_id"]):
        raise FreezeError("Candidate test fixtures differ; a direct comparison is invalid.")
    metadata = {}
    for name in CANDIDATES:
        model_path, meta_path, prep_path = artifact_names(name)
        meta = read_json(root / meta_path)
        metadata[name] = meta
        validate_model_metadata(meta, expected_model_name=name)
        for field, path in (("model_sha256", model_path), ("preprocessing_sha256", prep_path)):
            if meta[field] != file_hash(root / path):
                raise FreezeError(f"Saved {name} {field} does not match its artifact.")
        model = load_xgboost_model(root / model_path)
        if model.best_iteration != meta["best_iteration"] or model.get_booster().feature_names != meta["feature_columns"]:
            raise FreezeError(f"Saved {name} feature order or best iteration differs from metadata.")
        train = baseline.train if name == "model_a" else matched.train
        columns = meta["feature_columns"]
        saved = load_preprocessing(root / prep_path, feature_columns=columns)
        rebuilt = fit_training_medians(train, columns)
        if any(not np.isclose(saved[col], rebuilt[col], rtol=0, atol=1e-12) for col in columns):
            raise FreezeError(f"{name} preprocessing is not the training-only median.")
        evaluated = evaluate_saved_model(name, "test", project_root=root)
        for key in ("log_loss", "macro_f1", "accuracy", "row_count"):
            if not math.isclose(float(evaluated[key]), results[name][key], rel_tol=0, abs_tol=1e-12):
                raise FreezeError(f"{name} test {key} differs from its saved report.")
    _validate_metadata_pair(
        metadata["model_a_matched"], metadata["model_b"],
        manifest_sha256=file_hash(root / "data/processed/matched_split_manifest.csv"),
        dataset_sha256=file_hash(root / "data/processed/matched_model_dataset.csv"),
    )


def prediction_schema() -> dict:
    string = {"type": "string"}
    outcomes = ["away_win", "draw", "home_win"]
    properties = {
        "home_team": string, "away_team": string,
        "match_date": {"type": "string", "format": "date"},
        "model_name": string, "model_version": string,
        "probabilities": {"type": "object", "additionalProperties": False,
            "required": outcomes,
            "properties": {key: {"type": "number", "minimum": 0, "maximum": 1} for key in outcomes}},
        "predicted_outcome": {"type": "string", "enum": outcomes},
        "feature_as_of": string,
        "warnings": {"type": "array", "items": string},
    }
    return {"type": "object", "additionalProperties": False,
            "required": list(properties), "properties": properties}


def implementation_hashes(root: Path) -> dict:
    paths = [*root.glob("src/*.py"), *root.glob("tests/*.py"),
             root / "config/team_name_map.csv", root / "config/seasons.json",
             root / "requirements.txt", root / "requirements.lock.txt"]
    return {path.relative_to(root).as_posix(): file_hash(path, normalize_text=True)
            for path in sorted(paths)}


def verify_freeze(root: Path = PROJECT_ROOT) -> dict:
    """Read-only verification; no inference, model fit, or report writes."""
    config = read_json(root / "config/model_config.json")
    frozen = config.get("frozen_candidate")
    if not isinstance(frozen, dict) or not config.get("frozen_at_utc"):
        raise FreezeError("No Phase 8 freeze is present.")
    if config.get("freeze_record_sha256") != record_hash(config):
        raise FreezeError("Frozen configuration checksum mismatch.")
    if config["selected_model"] != frozen["model_name"]:
        raise FreezeError("Frozen candidate name differs from selection.")
    if choose_candidate(test_results(root), config["max_macro_f1_drop"]) != config["selected_model"]:
        raise FreezeError("Frozen selection differs from the test decision.")
    actual = implementation_hashes(root)
    if actual != frozen["implementation_files_sha256"] or digest(actual) != frozen["implementation_sha256"]:
        raise FreezeError("Frozen implementation checksum mismatch.")
    for path, checksum in frozen["artifacts_sha256"].items():
        resolved = (root / path).resolve()
        if not resolved.is_relative_to(root.resolve()):
            raise FreezeError("Frozen artifact path escapes the project.")
        if file_hash(resolved) != checksum:
            raise FreezeError(f"Frozen artifact checksum mismatch: {path}")
    return config


def freeze_candidate(root: Path = PROJECT_ROOT) -> dict:
    """Audit and record the selected existing model; refuse to replace a freeze."""
    from src.compare_models import _write_text_atomic
    from src.train_baselines import write_json_atomic
    from src.download_data import load_manifest

    config_path = root / "config/model_config.json"
    config = read_json(config_path)
    if config.get("frozen_at_utc"):
        return verify_freeze(root)
    results = test_results(root)
    validate_candidates(root, results)
    selected = choose_candidate(results, config["max_macro_f1_drop"])
    model_path, meta_path, prep_path = artifact_names(selected)
    meta, prep = read_json(root / meta_path), read_json(root / prep_path)
    raw_records = load_manifest(root / "data/raw/manifest.json")
    for record in raw_records:
        if file_hash(root / record["local_path"]) != record["sha256"]:
            raise FreezeError(f"Raw source checksum mismatch: {record['local_path']}")
    timestamp = datetime.now(timezone.utc).isoformat()
    parent = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, check=True, capture_output=True, text=True).stdout.strip()
    report = ["# Phase 8 Model Selection Freeze", "", f"Selected candidate: **{selected}**.", "",
              f"Recorded at {timestamp}, before final holdout evaluation.", "",
              "| Candidate | Test fixtures | Log loss | Macro F1 | Accuracy |",
              "|---|---:|---:|---:|---:|"]
    for name in CANDIDATES:
        row = results[name]
        report.append(f"| {name} | {row['row_count']} | {row['log_loss']:.12f} | {row['macro_f1']:.12f} | {row['accuracy']:.12f} |")
    report.extend(["", "## Decision and limitations", "",
        f"Section 10.6 selects {selected} using the lowest test log loss among candidates passing integrity and macro-F1 guardrails. "
        "The macro-F1 check uses the configured maximum 0.02 drop relative to the best candidate F1; "
        "Model B also must pass the Section 10.5 comparison against Model A-Matched. "
        "Model B has both the lowest log loss and highest macro F1 of all three candidates, so the choice does not depend on a borderline guardrail interpretation.", "",
        "All three candidates were compared on the identical 380 fixtures from 2024/25. "
        "Model A used 4,940 training fixtures; the matched candidates each used 1,900. "
        "Model B adds only previous-season possession for each team and their difference. "
        "Promoted-team missing values use training-only medians. The improvement measures predictive association, not causation.", "",
        "The improvement is modest and comes from one test season; it does not establish statistical significance or future performance. "
        "Logistic regression remains a useful benchmark (test log loss 1.033743, accuracy 0.489474), "
        "but the Section 10.6 candidate set is Model A, Model A-Matched, and Model B.", "",
        "## Frozen record", "",
        f"The existing model is retained without refitting: `{model_path}`. "
        f"Its zero-based best iteration is {meta['best_iteration']}; inference uses iterations [0, {meta['best_iteration'] + 1}).",
        "`config/model_config.json` records the selected feature order, full preprocessing values, parameters, labels, "
        "prediction JSON schema, probability-sum tolerance, test decision, source/artifact checksums, and implementation checksum. "
        "The original top-level baseline feature list remains for earlier phase compatibility; the candidate's authoritative list is `frozen_candidate.feature_columns`.", "",
        "Implementation hashes normalize CRLF to LF; artifact and raw-data hashes use exact bytes. "
        "The configuration checksum covers the entire config except its own `freeze_record_sha256` field. "
        f"`git_commit` identifies the pre-freeze parent {parent}; the freeze commit is the commit containing this record, "
        "avoiding a self-referential commit hash. Model JSON files retain the existing local/ignored policy; keep the saved artifacts to verify this exact freeze.", "",
        "Verification: `python -m src.freeze_model --verify`. Full acceptance: `python -m unittest discover -s tests -v`.", "",
        "The 2025/26 holdout remains unevaluated. Phase 9 requires explicit user approval after this freeze is committed. "
        "The prediction schema is frozen here; the upcoming-fixture command is still Phase 10 work.", ""])
    report_path = root / "reports/model_selection.md"
    _write_text_atomic("\n".join(report), report_path)
    files = [path for name in CANDIDATES for path in artifact_names(name)]
    files += ["data/raw/manifest.json", "reports/model_results.csv", "reports/tuning_results.csv",
              "reports/model_selection.md", "reports/possession_experiment.md"]
    files += [path.relative_to(root).as_posix() for path in sorted((root / "data/processed").glob("*.csv"))]
    files += [record["local_path"] for record in raw_records]
    implementation = implementation_hashes(root)
    config.update(selected_model=selected, frozen_at_utc=timestamp, git_commit=parent)
    config["frozen_candidate"] = {
        "model_name": selected, "feature_set": meta["feature_set"],
        "feature_columns": meta["feature_columns"], "parameters": meta["parameters"],
        "classes": meta["classes"], "target_mapping": config["target_mapping"],
        "best_iteration": meta["best_iteration"], "iteration_range": [0, meta["best_iteration"] + 1],
        "model_path": model_path, "metadata_path": meta_path, "preprocessing_path": prep_path,
        "preprocessing": prep, "prediction_schema": prediction_schema(),
        "probability_sum_tolerance": 1e-6,
        "probability_class_mapping": {"0": "away_win", "1": "draw", "2": "home_win"},
        "selection_test_results": results, "holdout_evaluated": False,
        "implementation_files_sha256": implementation, "implementation_sha256": digest(implementation),
        "artifacts_sha256": {path: file_hash(root / path) for path in sorted(set(files))},
    }
    config["freeze_record_sha256"] = record_hash(config)
    write_json_atomic(config, config_path)
    return verify_freeze(root)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify", action="store_true", help="verify the existing freeze without writes or inference")
    args = parser.parse_args(argv)
    try:
        config = verify_freeze() if args.verify else freeze_candidate()
    except (RuntimeError, OSError, ValueError, KeyError, subprocess.CalledProcessError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(f"selected_model={config['selected_model']}")
    print(f"freeze_record_sha256={config['freeze_record_sha256']}")
    print("holdout_evaluated=False")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
