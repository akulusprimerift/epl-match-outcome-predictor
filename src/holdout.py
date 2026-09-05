"""One-time, read-only-model Phase 9 evaluation with a committed audit trail."""

from __future__ import annotations

import ast
from datetime import datetime, timezone
from io import BytesIO
import json
import os
from pathlib import Path
import re
import subprocess
import tempfile

from src.freeze_model import (
    PROJECT_ROOT, FreezeError, digest, file_hash, read_json, verify_freeze,
)

PROTOCOL = "config/phase9_protocol.json"
START = "reports/final_holdout_started.json"
RECEIPT = "reports/final_holdout_receipt.json"
RESULTS = "reports/final_holdout_results.csv"
PREDICTIONS = "reports/final_holdout_predictions.json"
FIGURE = "reports/confusion_matrix_final_holdout.png"
REPORT = "reports/final_holdout.md"
OUTPUTS = (RESULTS, PREDICTIONS, FIGURE, REPORT)
ADAPTERS = {
    "src/evaluate.py": {"main", "build_parser"},
    "src/freeze_model.py": {"main", "test_results", "verify_freeze", "freeze_candidate"},
}
ADDITIONS = {"src/holdout.py", "tests/test_holdout.py"}


def git_text(root: Path, *arguments: str) -> str:
    try:
        return subprocess.run(
            ["git", *arguments], cwd=root, check=True, capture_output=True,
            text=True, encoding="utf-8",
        ).stdout
    except subprocess.CalledProcessError as exc:
        raise FreezeError(f"Could not verify the committed evaluation record: {exc.stderr.strip()}") from exc


def protected_ast(source: str, excluded: set[str]) -> str:
    """Compare all original inference logic, imports and constants, ignoring CLI adapters."""
    tree = ast.parse(source)
    tree.body = [node for node in tree.body if not (
        isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name in excluded
    )]
    return ast.dump(tree, include_attributes=False)


def verify_evaluation_extension(root: Path, config: dict, actual: dict) -> None:
    """Retain the original freeze; allow only the separately hashed Phase 9 extension."""
    protocol = read_json(root / PROTOCOL)
    frozen = config["frozen_candidate"]["implementation_files_sha256"]
    commit = protocol.get("freeze_commit", "")
    if not re.fullmatch(r"[0-9a-f]{40}", commit):
        raise FreezeError("Invalid Phase 8 freeze commit.")
    original_config = json.loads(git_text(root, "show", f"{commit}:config/model_config.json"))
    if config != original_config or protocol.get("freeze_record_sha256") != config["freeze_record_sha256"]:
        raise FreezeError("Phase 9 configuration differs from the committed Phase 8 freeze.")
    if actual != protocol.get("implementation_files_sha256"):
        raise FreezeError("Phase 9 implementation checksum mismatch.")
    if set(actual) != set(frozen) | ADDITIONS:
        raise FreezeError("Unexpected Phase 9 implementation files.")
    for path, checksum in frozen.items():
        if path in ADAPTERS:
            original = git_text(root, "show", f"{commit}:{path}")
            if protected_ast(original, ADAPTERS[path]) != protected_ast(
                (root / path).read_text(encoding="utf-8"), ADAPTERS[path]
            ):
                raise FreezeError(f"Protected frozen implementation changed: {path}")
        elif path != "tests/test_freeze_model.py" and actual[path] != checksum:
            raise FreezeError(f"Frozen implementation checksum mismatch: {path}")


def verify_started(root: Path, config: dict) -> dict:
    started = read_json(root / START)
    if started.get("freeze_record_sha256") != config["freeze_record_sha256"]:
        raise FreezeError("Holdout start record references a different freeze.")
    if started.get("protocol_sha256") != file_hash(root / PROTOCOL, normalize_text=True):
        raise FreezeError("Evaluation protocol changed after the holdout was opened.")
    commit = started.get("evaluation_commit", "")
    if not re.fullmatch(r"[0-9a-f]{40}", commit):
        raise FreezeError("Invalid pre-holdout evaluation commit.")
    committed = json.loads(git_text(root, "show", f"{commit}:{PROTOCOL}"))
    if committed != read_json(root / PROTOCOL):
        raise FreezeError("Evaluation protocol differs from its pre-holdout commit.")
    return started


def verify_holdout_receipt(root: Path, config: dict) -> dict | None:
    """Read-only verification also works after the holdout is legitimately opened."""
    if not (root / START).exists():
        if (root / RECEIPT).exists() or any((root / path).exists() for path in OUTPUTS):
            raise FreezeError("Holdout output exists without its one-time start record.")
        return None
    verify_started(root, config)
    if not (root / RECEIPT).exists():
        return None  # Interrupted runs stay locked; they are never silently rerun.
    receipt = read_json(root / RECEIPT)
    if receipt.get("status") != "complete" or receipt.get("holdout_evaluated") is not True:
        raise FreezeError("Invalid final holdout receipt.")
    if receipt.get("start_sha256") != file_hash(root / START):
        raise FreezeError("Holdout start record changed after evaluation.")
    if set(receipt.get("outputs_sha256", {})) != set(OUTPUTS):
        raise FreezeError("Final holdout output inventory is incomplete.")
    for path, checksum in receipt["outputs_sha256"].items():
        if file_hash(root / path) != checksum:
            raise FreezeError(f"Final holdout output checksum mismatch: {path}")
    return receipt


def preflight_commit(root: Path, config: dict) -> str:
    """Require the evaluation-only extension to be committed before opening holdout."""
    if git_text(root, "status", "--porcelain").strip():
        raise FreezeError("Commit the reviewed Phase 9 evaluator and protocol before opening holdout.")
    protocol = read_json(root / PROTOCOL)
    commit = git_text(root, "rev-parse", "HEAD").strip()
    git_text(root, "merge-base", "--is-ancestor", protocol["freeze_commit"], commit)
    if json.loads(git_text(root, "show", f"{commit}:{PROTOCOL}")) != protocol:
        raise FreezeError("The evaluation protocol is not committed.")
    if json.loads(git_text(root, "show", f"{commit}:config/model_config.json")) != config:
        raise FreezeError("Committed model configuration differs from the freeze.")
    # Hash every committed implementation file as well as the current working copy.
    import hashlib
    for path, checksum in protocol["implementation_files_sha256"].items():
        content = git_text(root, "show", f"{commit}:{path}").replace("\r\n", "\n")
        if hashlib.sha256(content.encode()).hexdigest() != checksum:
            raise FreezeError(f"Uncommitted evaluation implementation: {path}")
    return commit


def load_frozen_inputs(root: Path, config: dict):
    """Load existing artifacts and validate holdout membership; never fit anything."""
    import pandas as pd
    from src.evaluate import load_preprocessing, load_xgboost_model, validate_model_metadata
    from src.matched_experiment import read_matched_model_dataset, split_matched_dataset
    from src.split_data import load_split_policy, read_model_dataset, split_model_dataset

    frozen = config["frozen_candidate"]
    metadata = read_json(root / frozen["metadata_path"])
    validate_model_metadata(metadata, expected_model_name=config["selected_model"])
    for field in ("model_name", "feature_set", "feature_columns", "parameters", "classes", "best_iteration"):
        if metadata[field] != frozen[field]:
            raise FreezeError(f"Saved model differs from frozen {field}.")
    if read_json(root / frozen["preprocessing_path"]) != frozen["preprocessing"]:
        raise FreezeError("Saved preprocessing differs from its frozen record.")
    columns = frozen["feature_columns"]
    medians = load_preprocessing(root / frozen["preprocessing_path"], feature_columns=columns)
    model = load_xgboost_model(root / frozen["model_path"])
    if model.best_iteration != frozen["best_iteration"] or model.get_booster().feature_names != columns:
        raise FreezeError("Model feature order or best iteration differs from the freeze.")
    if frozen["iteration_range"] != [0, model.best_iteration + 1]:
        raise FreezeError("Invalid frozen prediction iteration range.")
    policy = load_split_policy(root / "config/model_config.json")
    if config["selected_model"] == "model_a":
        splits = split_model_dataset(read_model_dataset(root / "data/processed/model_dataset.csv"), policy)
        manifest_path = "data/processed/split_manifest.csv"
    else:
        splits = split_matched_dataset(read_matched_model_dataset(root / "data/processed/matched_model_dataset.csv"), policy)
        manifest_path = "data/processed/matched_split_manifest.csv"
    frame = splits.holdout
    manifest = pd.read_csv(root / manifest_path, dtype=str)
    expected = manifest.loc[manifest["split"].eq("holdout"), ["match_id", "season", "date"]]
    actual = frame.loc[:, ["match_id", "season", "date"]].astype(str)
    if (len(frame) != 380 or set(frame["season"]) != {"2526"}
            or not frame["match_id"].is_unique or not expected["match_id"].is_unique
            or set(actual.itertuples(index=False, name=None)) != set(expected.itertuples(index=False, name=None))):
        raise FreezeError("Final holdout must match the frozen 380 EPL fixtures from 2025/26.")
    return model, frame, medians


def holdout_metrics(frame, probabilities, frozen: dict) -> dict:
    """Same fixed metric definitions and column contract, explicitly labeled holdout."""
    import numpy as np
    from sklearn.metrics import accuracy_score, confusion_matrix, log_loss, precision_recall_fscore_support
    from src.constants import CLASS_LABELS, CLASS_NAMES, MODEL_RESULT_COLUMNS
    from src.train_baselines import validate_probability_matrix

    validate_probability_matrix(probabilities, expected_rows=len(frame))
    target = frame["target"].to_numpy()
    if len(target) == 0 or not set(target).issubset(set(CLASS_LABELS)):
        raise FreezeError("Invalid final holdout target labels.")
    predicted = np.asarray(CLASS_LABELS)[np.asarray(probabilities).argmax(axis=1)]
    precision, recall, f1, support = precision_recall_fscore_support(target, predicted, labels=CLASS_LABELS, zero_division=0)
    matrix = confusion_matrix(target, predicted, labels=CLASS_LABELS)
    result = dict(
        model_name=frozen["model_name"], model_family="xgboost", feature_set=frozen["feature_set"],
        split="holdout", row_count=len(frame), log_loss=float(log_loss(target, probabilities, labels=CLASS_LABELS)),
        macro_f1=float(f1.mean()), accuracy=float(accuracy_score(target, predicted)),
        best_iteration=frozen["best_iteration"], parameters=json.dumps(frozen["parameters"], sort_keys=True, separators=(",", ":")),
    )
    for index, name in enumerate(CLASS_NAMES):
        for metric, values in (("precision", precision), ("recall", recall), ("f1", f1), ("support", support)):
            result[f"{metric}_{name}"] = int(values[index]) if metric == "support" else float(values[index])
    for i, actual_name in enumerate(CLASS_NAMES):
        for j, predicted_name in enumerate(CLASS_NAMES):
            result[f"confusion_{actual_name}_pred_{predicted_name}"] = int(matrix[i, j])
    if tuple(result) != MODEL_RESULT_COLUMNS:
        raise FreezeError("Final holdout metrics violate the existing report column contract.")
    return result


def write_bytes_atomic(path: Path, data: bytes) -> None:
    descriptor, name = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "wb") as output:
            output.write(data)
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def save_reports(root: Path, config: dict, frame, probabilities, result: dict, started: dict) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    from src.constants import CLASS_NAMES, MODEL_RESULT_COLUMNS
    from src.train_baselines import write_json_atomic, write_model_results_atomic

    write_model_results_atomic(pd.DataFrame([result], columns=MODEL_RESULT_COLUMNS), root / RESULTS)
    write_json_atomic({
        "model_name": result["model_name"], "split": "holdout", "season": "2526",
        "probability_columns": list(CLASS_NAMES), "freeze_record_sha256": config["freeze_record_sha256"],
        "predictions": [{"match_id": str(row.match_id), "target": int(row.target),
                         "probabilities": [float(value) for value in probability]}
                        for row, probability in zip(frame.itertuples(), probabilities)],
    }, root / PREDICTIONS)
    matrix = np.array([[result[f"confusion_{a}_pred_{p}"] for p in CLASS_NAMES] for a in CLASS_NAMES])
    figure, axis = plt.subplots(figsize=(7, 6))
    rendered = axis.imshow(matrix, cmap="Blues", vmin=0)
    for (i, j), count in np.ndenumerate(matrix):
        axis.text(j, i, str(count), ha="center", va="center", color="white" if count > matrix.max() / 2 else "black")
    labels = ["Away win", "Draw", "Home win"]
    axis.set(xticks=range(3), yticks=range(3), xticklabels=labels, yticklabels=labels,
             xlabel="Predicted outcome", ylabel="Actual outcome",
             title=f"{result['model_name']} — final 2025/26 holdout\n380 EPL fixtures; no refitting")
    figure.colorbar(rendered, ax=axis, label="Fixtures")
    figure.tight_layout()
    buffer = BytesIO()
    figure.savefig(buffer, format="png", dpi=160)
    plt.close(figure)
    write_bytes_atomic(root / FIGURE, buffer.getvalue())

    test = config["frozen_candidate"]["selection_test_results"][result["model_name"]]
    lines = ["# Phase 9 — Final 2025/26 Holdout", "",
        f"Frozen candidate: **{result['model_name']}**, evaluated once on **{result['row_count']} EPL fixtures**.", "",
        "| Metric | 2024/25 test | 2025/26 final holdout | Holdout minus test |",
        "|---|---:|---:|---:|"]
    for key in ("log_loss", "macro_f1", "accuracy"):
        lines.append(f"| {key} | {test[key]:.12f} | {result[key]:.12f} | {result[key] - test[key]:+.12f} |")
    delta = result["log_loss"] - test["log_loss"]
    lines += ["", f"Primary-metric log loss {'worsened' if delta > 0 else 'improved'} by {abs(delta):.6f} "
              f"({abs(delta) / test['log_loss'] * 100:.2f}% relative to test). Lower log loss is better; higher F1 and accuracy are better.", "",
        "## Interpretation and limitations", "",
        "This is a sequential pre-match backtest: rolling features use only earlier completed fixtures, "
        "including earlier matches in the holdout season. It is not a simultaneous preseason forecast. "
        "Possession is lagged from 2024/25, never taken from the final 2025/26 season averages. "
        "Missing previous-season possession for promoted clubs uses the saved training-only medians.", "",
        "Only the previously selected model was evaluated. The result does not measure a holdout possession "
        "uplift against other candidates and does not establish statistical significance or future performance. "
        "Class-level metrics and the confusion matrix expose weaknesses that aggregate accuracy can hide. "
        "No model selection, parameter adjustment, feature change, median fitting, or retraining followed this evaluation. "
        "Any future modeling change needs a new future holdout season.", "",
        "## Integrity and acceptance", "",
        f"- Original Phase 8 freeze: `{read_json(root / PROTOCOL)['freeze_commit']}`.",
        f"- Committed pre-holdout evaluator: `{started['evaluation_commit']}`.",
        f"- Unchanged freeze record: `{config['freeze_record_sha256']}`.",
        f"- Evaluation started at `{started['started_at_utc']}`.",
        f"- Saved best iteration: {result['best_iteration']}; no fit or early-stopping calls.",
        "- Frozen artifact/source checksums and exact holdout fixture membership verified before inference.",
        "- Three finite, nonnegative class probabilities per fixture; sums checked within 0.000001.",
        "- Original report column contract and target encoding (away=0, draw=1, home=2) preserved.",
        "- Model configuration and all original training/feature/metric implementations unchanged. "
        "The separate Phase 9 protocol permits only evaluation/verification adapters and evaluation tests.",
        "- A one-time start record prevents repeat inference. Repeating the CLI verifies and returns saved results.",
        "- The completion receipt hashes all final outputs. The original metadata's holdout_evaluated=false "
        "remains an immutable statement of its pre-holdout state; the completion receipt records the current state.", "",
        "![Final holdout confusion matrix](confusion_matrix_final_holdout.png)", "",
        "Metrics: `final_holdout_results.csv`. Per-fixture probabilities: `final_holdout_predictions.json`. "
        "Audit: `final_holdout_started.json` and `final_holdout_receipt.json`.", "",
        "Phase 10 upcoming-fixture prediction is not implemented as part of this phase.", ""]
    write_bytes_atomic(root / REPORT, "\n".join(lines).encode("utf-8"))


def evaluate_final_holdout(root: Path = PROJECT_ROOT) -> dict:
    """Open only the frozen selected candidate; subsequent calls return verified reports."""
    from src.evaluate import predict_xgboost_probabilities
    from src.train_baselines import write_json_atomic

    root = root.resolve()
    config = verify_freeze(root)
    receipt = verify_holdout_receipt(root, config)
    if receipt is not None:
        # Read the hashed CSV, not unverified metric values embedded in a receipt.
        import pandas as pd
        row = pd.read_csv(root / RESULTS).iloc[0].to_dict()
        return {**row, "holdout_evaluated": True, "reused_saved_results": True}
    if (root / START).exists():
        raise FreezeError("An earlier holdout attempt is incomplete. Preserve its files; do not rerun inference or tune.")
    commit = preflight_commit(root, config)
    model, frame, medians = load_frozen_inputs(root, config)
    frozen = config["frozen_candidate"]
    started = {
        "started_at_utc": datetime.now(timezone.utc).isoformat(), "evaluation_commit": commit,
        "freeze_record_sha256": config["freeze_record_sha256"],
        "protocol_sha256": file_hash(root / PROTOCOL, normalize_text=True),
        "model_name": config["selected_model"], "season": "2526", "row_count": len(frame),
    }
    # Exclusive creation is both the one-time gate and a concurrent-run lock. A partial
    # marker after interruption deliberately fails closed instead of permitting a rerun.
    try:
        with (root / START).open("x", encoding="utf-8", newline="\n") as output:
            json.dump(started, output, indent=2)
            output.write("\n")
            output.flush()
            os.fsync(output.fileno())
    except FileExistsError as exc:
        raise FreezeError("The final holdout has already been started; concurrent inference is forbidden.") from exc
    probabilities = predict_xgboost_probabilities(
        model, frame, feature_columns=frozen["feature_columns"], medians=medians,
        best_iteration=frozen["best_iteration"],
    )
    result = holdout_metrics(frame, probabilities, frozen)
    save_reports(root, config, frame, probabilities, result, started)
    verify_freeze(root)  # Detect concurrent changes before recording completion.
    write_json_atomic({
        "status": "complete", "holdout_evaluated": True,
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "start_sha256": file_hash(root / START),
        "outputs_sha256": {path: file_hash(root / path) for path in OUTPUTS},
    }, root / RECEIPT)
    verify_holdout_receipt(root, config)
    return {**result, "holdout_evaluated": True, "reused_saved_results": False}
