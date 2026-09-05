"""Predict a user-supplied upcoming EPL fixture using the frozen local snapshot."""

from __future__ import annotations

import argparse
from datetime import date
import json
from pathlib import Path
import re
import sys

from src.freeze_model import PROJECT_ROOT, FreezeError, read_json, verify_freeze

PROTOCOL = "config/phase10_protocol.json"
STALE_AFTER_DAYS = 14  # Warning only; never changes a feature or a probability.


class PredictionError(RuntimeError):
    """A requested fixture cannot safely use the frozen prediction pipeline."""


def verify_prediction_extension(root: Path, config: dict, actual: dict, phase9: dict) -> dict:
    """Verify additive prediction work without rewriting the pre-holdout protocol."""
    from src.holdout import git_text, protected_ast

    protocol = read_json(root / PROTOCOL)
    commit = protocol.get("phase9_complete_commit", "")
    if not re.fullmatch(r"[0-9a-f]{40}", commit):
        raise FreezeError("Invalid Phase 9 completion commit.")
    if protocol.get("freeze_record_sha256") != config["freeze_record_sha256"]:
        raise FreezeError("Prediction protocol references a different frozen model.")
    original = json.loads(git_text(root, "show", f"{commit}:config/phase9_protocol.json"))
    receipt = json.loads(git_text(root, "show", f"{commit}:reports/final_holdout_receipt.json"))
    if original != phase9 or receipt.get("status") != "complete":
        raise FreezeError("Prediction extension requires the completed original Phase 9 record.")
    expected = phase9["implementation_files_sha256"]
    if actual != protocol.get("implementation_files_sha256"):
        raise FreezeError("Phase 10 implementation checksum mismatch.")
    if set(actual) != set(expected) | {"tests/test_predict.py"}:
        raise FreezeError("Unexpected Phase 10 implementation files.")
    permitted = {"src/predict.py", "src/holdout.py", "tests/test_freeze_model.py"}
    for path, checksum in expected.items():
        if path not in permitted and actual[path] != checksum:
            raise FreezeError(f"Frozen implementation checksum mismatch: {path}")
    # The only change to the holdout module is routing verification to this extension.
    old = git_text(root, "show", f"{commit}:src/holdout.py")
    current = (root / "src/holdout.py").read_text(encoding="utf-8")
    if protected_ast(old, {"verify_evaluation_extension"}) != protected_ast(current, {"verify_evaluation_extension"}):
        raise FreezeError("The completed holdout implementation was changed.")
    return dict(expected)


def parse_prediction_date(value: str) -> date:
    if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", value):
        raise PredictionError("Prediction date must be YYYY-MM-DD.")
    try:
        return date.fromisoformat(value)
    except ValueError as exc:
        raise PredictionError(f"Invalid prediction date: {value!r}.") from exc


def canonical_teams(root: Path) -> dict[str, str]:
    from src.clean_data import load_team_name_map
    mapping = load_team_name_map(root / "config/team_name_map.csv")
    # Validate using the established mapping loader; accept canonical names, not aliases.
    return {identity.name: identity.slug for identity in mapping.values()}


def build_upcoming_features(canonical, possession, *, home: str, away: str,
                            home_slug: str, away_slug: str, match_date: date,
                            possession_coverage_threshold: float = 0.95):
    """Apply the frozen window definitions to strictly prior completed fixtures.

    The requested row has no result. Taking the last five strictly prior rows
    is equivalent to shift(1).rolling(5, min_periods=3) at that next row.
    """
    import pandas as pd
    from src.build_history import build_team_history_frame
    from src.collect_possession import season_code_from_start_year
    from src.constants import POSSESSION_FEATURE_COLUMNS, ROLLING_MIN_PERIODS, ROLLING_WINDOW
    from src.matched_experiment import _validate_team_season_possession, build_lagged_possession_features

    prior = canonical.loc[canonical["date"].astype(str).lt(match_date.isoformat())].copy()
    if prior.empty:
        raise PredictionError("No completed EPL history exists before the requested date.")
    history = build_team_history_frame(prior)
    values, warnings = {}, []
    for role, name, slug, is_home in (("home", home, home_slug, True), ("away", away, away_slug, False)):
        rows = history.loc[history["team_slug"].eq(slug)].sort_values(["date", "match_id"], kind="mergesort")
        venue = rows.loc[rows["is_home"].eq(is_home)]
        tail, venue_tail = rows.tail(ROLLING_WINDOW), venue.tail(ROLLING_WINDOW)
        for feature, column, operation in (
            ("goals_for_avg_5", "goals_for", "mean"), ("goals_against_avg_5", "goals_against", "mean"),
            ("shots_avg_5", "shots", "mean"), ("form_points_5", "points", "sum"), ("overall_ppg_5", "points", "mean"),
        ):
            series = tail[column]
            values[f"{role}_{feature}"] = float(getattr(series, operation)()) if series.count() >= ROLLING_MIN_PERIODS else float("nan")
        values[f"{role}_venue_ppg_5"] = float(venue_tail["points"].mean()) if len(venue_tail) >= ROLLING_MIN_PERIODS else float("nan")
        values[f"{role}_history_matches"] = len(rows)
        values[f"{role}_venue_history_matches"] = len(venue)
        if len(rows) < ROLLING_MIN_PERIODS or len(venue) < ROLLING_MIN_PERIODS:
            warnings.append(f"Cold start for {name}: {len(rows)} prior EPL matches and {len(venue)} in this venue role; missing features use frozen training medians.")
        if not rows.empty:
            last = date.fromisoformat(str(rows["date"].max()))
            gap = (match_date - last).days
            if gap > STALE_AFTER_DAYS:
                warnings.append(f"Stale history for {name}: latest stored EPL match is {last.isoformat()} ({gap} days before the requested fixture).")
    for edge, left, right in (
        ("goals_scored_edge", "home_goals_for_avg_5", "away_goals_for_avg_5"),
        ("defensive_edge", "away_goals_against_avg_5", "home_goals_against_avg_5"),
        ("shots_edge", "home_shots_avg_5", "away_shots_avg_5"),
        ("form_edge", "home_form_points_5", "away_form_points_5"),
        ("venue_edge", "home_venue_ppg_5", "away_venue_ppg_5"),
        ("history_edge", "home_history_matches", "away_history_matches"),
    ):
        values[edge] = values[left] - values[right]

    # Date-only upcoming fixtures use July 1 as the season boundary. This is not
    # a historical season classifier (notably the delayed 2019/20 restart).
    start_year = match_date.year if match_date.month >= 7 else match_date.year - 1
    season = season_code_from_start_year(start_year)
    source = season_code_from_start_year(start_year - 1)
    table = _validate_team_season_possession(possession)
    source_rows = table.loc[table["source_season"].eq(source)]
    source_matches = prior.loc[prior["season"].astype(str).eq(source)]
    source_teams = set(source_matches["home_team_slug"]) | set(source_matches["away_team_slug"])
    if len(source_matches) != 380 or len(source_teams) != 20 or source_rows.empty:
        raise PredictionError(f"Completed preceding EPL season {source} and its possession table are required; this snapshot cannot support target season {season}.")
    if set(source_rows["team_slug"]) != source_teams or not pd.to_numeric(source_rows["matches_recorded"], errors="coerce").eq(38).all():
        raise PredictionError(f"Possession source season {source} is incomplete or differs from its EPL teams.")
    if source_rows["average_possession_pct"].notna().mean() < possession_coverage_threshold:
        raise PredictionError(f"Possession coverage for {source} is below the frozen threshold.")
    fixture = pd.DataFrame([dict(match_id="upcoming", season=season, home_team_slug=home_slug, away_team_slug=away_slug)])
    keys = pd.concat([prior.loc[:, list(fixture.columns)], fixture], ignore_index=True)
    joined = build_lagged_possession_features(keys, table)
    row = joined.loc[joined["match_id"].eq("upcoming")].iloc[0]
    for column in ("home_previous_season_possession", "away_previous_season_possession", "possession_edge"):
        values[column] = float(row[column]) if pd.notna(row[column]) else float("nan")
    for role, name in (("home", home), ("away", away)):
        if pd.isna(values[f"{role}_previous_season_possession"]):
            warnings.append(f"Missing previous-season EPL possession for {name} ({source}); using frozen training medians, never another league or an older season.")
    missing = [column for column in POSSESSION_FEATURE_COLUMNS if pd.isna(values[column])]
    if missing:
        warnings.append("Frozen training-median imputation applied to: " + ", ".join(missing) + ".")
    return pd.DataFrame([values], columns=POSSESSION_FEATURE_COLUMNS), str(prior["date"].max()), warnings


def predict_fixture(home: str, away: str, match_date: str, *, root: Path = PROJECT_ROOT) -> dict:
    import numpy as np
    import pandas as pd
    from src.build_history import read_canonical_matches
    from src.clean_data import validate_canonical_table
    from src.constants import CLASS_NAMES, POSSESSION_FEATURE_COLUMNS
    from src.evaluate import load_preprocessing, load_xgboost_model, predict_xgboost_probabilities, validate_model_metadata
    from src.holdout import verify_holdout_receipt

    requested = parse_prediction_date(match_date)
    teams = canonical_teams(root)
    for name in (home, away):
        if name not in teams:
            raise PredictionError(f"Unknown canonical EPL team {name!r}. Use an exact name from config/team_name_map.csv.")
    if home == away:
        raise PredictionError("Home and away teams must be different.")
    config = verify_freeze(root)
    if verify_holdout_receipt(root, config) is None:
        raise PredictionError("Complete Phase 9 holdout reporting before using the prediction command.")
    frozen = config["frozen_candidate"]
    if tuple(frozen["feature_columns"]) != POSSESSION_FEATURE_COLUMNS:
        raise PredictionError("This prediction command requires the frozen Model B possession feature order.")
    canonical = read_canonical_matches(root / "data/processed/canonical_matches.csv")
    validate_canonical_table(canonical)
    latest = str(canonical["date"].max())
    if match_date <= latest:
        raise PredictionError(f"Prediction date must be after the latest stored match date ({latest}); historical and same-date backtests are not supported.")
    possession = pd.read_csv(root / "data/processed/team_season_possession.csv", dtype={"source_season": str, "target_season": str})
    frame, as_of, warnings = build_upcoming_features(
        canonical, possession, home=home, away=away, home_slug=teams[home], away_slug=teams[away], match_date=requested,
        possession_coverage_threshold=config["possession_coverage_threshold"],
    )
    metadata = read_json(root / frozen["metadata_path"])
    validate_model_metadata(metadata, expected_model_name=config["selected_model"])
    for field in ("model_name", "feature_columns", "parameters", "best_iteration", "classes", "feature_set"):
        if metadata[field] != frozen[field]:
            raise PredictionError(f"Saved metadata differs from frozen {field}.")
    if read_json(root / frozen["preprocessing_path"]) != frozen["preprocessing"]:
        raise PredictionError("Preprocessing differs from the frozen training-only values.")
    medians = load_preprocessing(root / frozen["preprocessing_path"], feature_columns=frozen["feature_columns"])
    model = load_xgboost_model(root / frozen["model_path"])
    if model.best_iteration != frozen["best_iteration"] or model.get_booster().feature_names != frozen["feature_columns"]:
        raise PredictionError("Saved model feature order or iteration differs from its freeze.")
    if frozen["iteration_range"] != [0, model.best_iteration + 1]:
        raise PredictionError("Frozen prediction iteration range is invalid.")
    probabilities = predict_xgboost_probabilities(model, frame, feature_columns=frozen["feature_columns"],
                                               medians=medians, best_iteration=frozen["best_iteration"])[0]
    warnings.extend([
        "User-supplied fixture: the local snapshot does not verify its schedule or current EPL membership.",
        "feature_as_of is the latest stored completed match date; an exact completion timestamp is unavailable.",
    ])
    return dict(home_team=home, away_team=away, match_date=match_date, model_name=config["selected_model"],
                model_version=config["freeze_record_sha256"],
                probabilities={name: float(value) for name, value in zip(CLASS_NAMES, probabilities)},
                predicted_outcome=CLASS_NAMES[int(np.argmax(probabilities))], feature_as_of=as_of, warnings=warnings)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--home", required=True, help="exact canonical home team name")
    parser.add_argument("--away", required=True, help="exact canonical away team name")
    parser.add_argument("--date", required=True, help="YYYY-MM-DD, after the latest stored EPL match")
    args = parser.parse_args(argv)
    try:
        result = predict_fixture(args.home, args.away, args.date)
        print(json.dumps(result, sort_keys=True, allow_nan=False))
    except (RuntimeError, OSError, ValueError, KeyError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
