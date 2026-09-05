"""Fixture-specific model attribution and auditable historical match statistics."""

from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
from xgboost import DMatrix

from src.build_features import apply_training_medians
from src.build_history import build_team_history_frame, read_canonical_matches
from src.constants import CLASS_NAMES, POSSESSION_FEATURE_COLUMNS
from src.download_data import load_manifest
from src.evaluate import load_xgboost_model
from src.freeze_model import PROJECT_ROOT, verify_freeze
from src.predict import build_upcoming_features, canonical_teams, predict_fixture

GROUPS = {
    "Scoring form": ("home_goals_for_avg_5", "away_goals_for_avg_5", "goals_scored_edge"),
    "Defensive form": ("home_goals_against_avg_5", "away_goals_against_avg_5", "defensive_edge"),
    "Shot volume": ("home_shots_avg_5", "away_shots_avg_5", "shots_edge"),
    "Recent points": ("home_form_points_5", "away_form_points_5", "home_overall_ppg_5", "away_overall_ppg_5", "form_edge"),
    "Home/away form": ("home_venue_ppg_5", "away_venue_ppg_5", "venue_edge"),
    "Available EPL history": ("home_history_matches", "away_history_matches", "home_venue_history_matches", "away_venue_history_matches", "history_edge"),
    "Previous-season possession": ("home_previous_season_possession", "away_previous_season_possession", "possession_edge"),
}


def finite_or_none(value):
    return float(value) if pd.notna(value) and np.isfinite(value) else None


def match_records(rows, sources: dict) -> list[dict]:
    records = []
    for row in rows.itertuples(index=False):
        records.append({
            "match_id": str(row.match_id), "date": str(row.date), "opponent": str(row.opponent),
            "venue": "Home" if row.is_home else "Away", "goals_for": int(row.goals_for),
            "goals_against": int(row.goals_against), "shots": finite_or_none(row.shots),
            "points": int(row.points), "result": "W" if row.points == 3 else "D" if row.points == 1 else "L",
            "source_url": sources[f"data/raw/football_data/E0_{row.season}.csv"],
        })
    return records


def team_evidence(history, possession, *, name: str, slug: str, role: str, source_season: str, sources: dict) -> dict:
    rows = history.loc[history.team_slug.eq(slug)].sort_values(["date", "match_id"], kind="mergesort")
    last = rows.tail(5)
    venue = rows.loc[rows.is_home.eq(role == "home")].tail(5)
    possession_row = possession.loc[possession.source_season.eq(source_season) & possession.team_slug.eq(slug)]
    record = possession_row.iloc[0] if len(possession_row) else None
    return {
        "name": name, "role": role, "history_count": len(rows), "recent_count": len(last), "venue_count": len(venue),
        "recent_points": int(last.points.sum()) if len(last) else None,
        "goals_for_average": finite_or_none(last.goals_for.mean()),
        "goals_against_average": finite_or_none(last.goals_against.mean()),
        "shots_average": finite_or_none(last.shots.mean()), "shots_observations": int(last.shots.count()),
        "venue_points_per_game": finite_or_none(venue.points.mean()),
        "possession": finite_or_none(record.average_possession_pct) if record is not None else None,
        "possession_source_season": source_season,
        "possession_matches": int(record.matches_recorded) if record is not None else None,
        "possession_source_url": str(record.source_url) if record is not None else None,
        "recent_matches": match_records(last.iloc[::-1], sources),
        "venue_matches": match_records(venue.iloc[::-1], sources),
    }


def model_attribution(model, feature_frame, probabilities: dict, best_iteration: int) -> dict:
    """Exact native TreeSHAP in raw-score space, for the same saved tree range."""
    columns = list(POSSESSION_FEATURE_COLUMNS)
    covered = [name for names in GROUPS.values() for name in names]
    if len(covered) != len(columns) or set(covered) != set(columns):
        raise RuntimeError("Attribution groups do not cover the exact frozen features once.")
    matrix = DMatrix(feature_frame.loc[:, columns], feature_names=columns)
    contributions = model.get_booster().predict(
        matrix, pred_contribs=True, approx_contribs=False, strict_shape=True,
        iteration_range=(0, best_iteration + 1), validate_features=True,
    )
    if contributions.shape != (1, 3, len(columns) + 1) or not np.isfinite(contributions).all():
        raise RuntimeError("The model did not return valid per-class feature contributions.")
    classes = [int(value) for value in model.classes_]
    ordered = np.array([contributions[0, classes.index(label), :] for label in (0, 1, 2)], dtype=float)
    margins = ordered.sum(axis=1)
    recovered = np.exp(margins - margins.max())
    recovered /= recovered.sum()
    expected = np.array([probabilities[name] for name in CLASS_NAMES])
    if not np.allclose(recovered, expected, atol=1e-6, rtol=0):
        raise RuntimeError("Explanation does not reconstruct the frozen prediction; refusing a misleading explanation.")
    ranked = sorted(range(3), key=lambda index: (-expected[index], index))
    leading = ranked[0]
    # For a predicted win, explain that club's win against the opponent's win;
    # for a draw, compare with the strongest competing win outcome.
    runner = 2 - leading if leading != 1 else ranked[1]
    differences = ordered[leading] - ordered[runner]
    groups = [{"group": title, "contribution": float(sum(differences[columns.index(column)] for column in names)),
               "features": list(names)} for title, names in GROUPS.items()]
    groups.sort(key=lambda value: -abs(value["contribution"]))
    return {
        "leading_outcome": CLASS_NAMES[leading], "comparison_outcome": CLASS_NAMES[runner],
        "groups": groups, "baseline_gap": float(differences[-1]), "total_score_gap": float(differences.sum()),
        "method": "Exact XGBoost TreeSHAP, grouped contributions comparing the selected win against the other team's win, or a leading draw against the strongest win outcome.",
        "units": "Raw model-score difference (log-odds), not probability percentage points.",
        "limitations": "This explains the fitted model, not what causes a team to win. Correlated features and the learned baseline also matter; a better raw statistic need not increase this model's score.",
    }


def comparison_statistics(home: dict, away: dict, features, imputed) -> list[dict]:
    definitions = (
        ("Points from recent EPL matches", "recent_points", "form_points_5", "points", "Recent points"),
        ("Goals scored per match", "goals_for_average", "goals_for_avg_5", "goals", "Scoring form"),
        ("Goals conceded per match", "goals_against_average", "goals_against_avg_5", "goals", "Defensive form"),
        ("Shots per match", "shots_average", "shots_avg_5", "shots", "Shot volume"),
        ("Points per match at this venue role", "venue_points_per_game", "venue_ppg_5", "points", "Home/away form"),
        ("Previous-season possession", "possession", "previous_season_possession", "%", "Previous-season possession"),
        ("Prior EPL matches available", "history_count", "history_matches", "matches", "Available EPL history"),
    )
    return [{"label": label, "unit": unit, "group": group,
             "home": home[key], "away": away[key],
             "home_model_input": float(imputed[f"home_{feature}"]),
             "away_model_input": float(imputed[f"away_{feature}"]),
             "home_imputed": bool(pd.isna(features[f"home_{feature}"])),
             "away_imputed": bool(pd.isna(features[f"away_{feature}"]))}
            for label, key, feature, unit, group in definitions]


def explain_match(home: str, away: str, match_date: str, *, root: Path = PROJECT_ROOT) -> dict:
    # Keep the public Phase 10 schema and prediction path intact.
    prediction = predict_fixture(home, away, match_date, root=root)
    config = verify_freeze(root)
    frozen = config["frozen_candidate"]
    teams = canonical_teams(root)
    canonical = read_canonical_matches(root / "data/processed/canonical_matches.csv")
    possession = pd.read_csv(root / "data/processed/team_season_possession.csv", dtype={"source_season": str, "target_season": str})
    day = date.fromisoformat(match_date)
    frame, _, _ = build_upcoming_features(canonical, possession, home=home, away=away,
        home_slug=teams[home], away_slug=teams[away], match_date=day,
        possession_coverage_threshold=config["possession_coverage_threshold"])
    imputed = apply_training_medians(frame, frozen["preprocessing"]["median_values"], POSSESSION_FEATURE_COLUMNS)
    model = load_xgboost_model(root / frozen["model_path"])
    attribution = model_attribution(model, imputed, prediction["probabilities"], frozen["best_iteration"])
    prior = canonical.loc[canonical.date.lt(match_date)]
    history = build_team_history_frame(prior)
    records = load_manifest(root / "data/raw/manifest.json")
    sources = {record["local_path"]: record["source_url"] for record in records}
    start_year = day.year if day.month >= 7 else day.year - 1
    source_season = f"{(start_year - 1) % 100:02d}{start_year % 100:02d}"
    home_stats = team_evidence(history, possession, name=home, slug=teams[home], role="home", source_season=source_season, sources=sources)
    away_stats = team_evidence(history, possession, name=away, slug=teams[away], role="away", source_season=source_season, sources=sources)
    statistics = comparison_statistics(home_stats, away_stats, frame.iloc[0], imputed.iloc[0])
    labels = {"home_win": f"{home} win", "draw": "a draw", "away_win": f"{away} win"}
    leader = prediction["predicted_outcome"]
    chance = prediction["probabilities"][leader]
    summary = f"The model ranks {labels[leader]} first at {chance:.1%}. "
    summary += ("It is still less likely than the other two outcomes combined. " if chance < .5 else "This is an estimate, not a guaranteed result. ")
    summary += f"The model influences below compare it with {labels[attribution['comparison_outcome']]}. All three outcomes remain possible."
    for group in attribution["groups"]:
        evidence = next(row for row in statistics if row["group"] == group["group"])
        group["statistics"] = evidence
        group["direction"] = "supports" if group["contribution"] > 0 else "opposes" if group["contribution"] < 0 else "neutral"
    verify_freeze(root)  # Fail closed if frozen inputs changed during the request.
    return {"prediction": prediction, "summary": summary, "attribution": attribution,
            "statistics": statistics, "teams": {"home": home_stats, "away": away_stats},
            "evidence_note": "Rolling statistics use the recent and venue matches listed below. Possession comes from the linked previous-season summary; history counts cover all prior stored EPL matches, not just the five shown. Observations are never fabricated from model medians; model-input replacements are identified separately.",
            "model_caveat": "In the final 2025/26 holdout, this model achieved 46.32% accuracy and correctly identified only 2 of 104 draws. It has not been calibrated or retuned using that holdout."}
