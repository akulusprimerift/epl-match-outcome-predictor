"""Check Phase 11 documentation against local contracts, reports and prediction output."""

import csv
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def validate() -> None:
    from src.constants import POSSESSION_FEATURE_COLUMNS
    from src.freeze_model import verify_freeze
    from src.predict import predict_fixture

    config = verify_freeze(ROOT)
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    documents = [ROOT / "README.md", *sorted((ROOT / "docs").glob("*.md"))]
    for path in documents:
        text = path.read_text(encoding="utf-8")
        if re.search(r"(?i)(?<![a-z])[a-z]:[\\/]|/(?:home|Users|mnt)/", text):
            raise ValueError(f"Machine-specific absolute path in {path.name}")
        if re.search(r"(?i)(?:sk-[a-z0-9]{20,}|ghp_[a-z0-9]{20,}|(?:api_key|authorization)\s*[=:]\s*['\"][^'\"]{12,})", text):
            raise ValueError(f"Possible credential in {path.name}; inspect before sharing")
        for link in re.findall(r"\]\(([^)]+)\)", text):
            target = link.split("#", 1)[0]
            if not target or target.startswith(("https://", "http://")):
                continue
            if not (path.parent / target).is_file():
                raise ValueError(f"Broken documentation link: {path.name}: {link}")
    with (ROOT / "reports/model_results.csv").open(newline="", encoding="utf-8") as source:
        rows = list(csv.DictReader(source))
    for row in rows:
        if row["split"] == "test":
            expected = "| " + row["model_name"] + " | " + " | ".join(
                f"{float(row[key]):.6f}" for key in ("log_loss", "macro_f1", "accuracy")
            ) + " |"
            if expected not in readme:
                raise ValueError(f"README test metrics differ for {row['model_name']}")
    with (ROOT / "reports/final_holdout_results.csv").open(newline="", encoding="utf-8") as source:
        final = next(csv.DictReader(source))
    test = config["frozen_candidate"]["selection_test_results"]["model_b"]
    for title, key in (("Log loss", "log_loss"), ("Macro F1", "macro_f1"), ("Accuracy", "accuracy")):
        values = [test[key], float(final[key])]
        cells = [f"{value * 100:.2f}%" if key == "accuracy" else f"{value:.6f}" for value in values]
        if f"| {title} | {' | '.join(cells)} |" not in readme:
            raise ValueError(f"README holdout metrics differ for {key}")
    dictionary = (ROOT / "docs/DATA_DICTIONARY.md").read_text(encoding="utf-8")
    if "\n".join(POSSESSION_FEATURE_COLUMNS) not in dictionary:
        raise ValueError("Documented feature order differs from the frozen contract")
    sample = json.loads((ROOT / "docs/sample_prediction.json").read_text(encoding="utf-8"))
    actual = predict_fixture(sample["home_team"], sample["away_team"], sample["match_date"], root=ROOT)
    if sample != actual:
        raise ValueError("Sample JSON differs from the actual frozen prediction")
    print("Documentation checks passed: local links, path/credential scan, report metrics, feature order, exact sample and freeze.")


if __name__ == "__main__":
    try:
        validate()
    except (RuntimeError, OSError, ValueError, KeyError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(1)
