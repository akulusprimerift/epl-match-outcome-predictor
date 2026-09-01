"""Build leakage-safe pre-match rolling features from team history."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import date, datetime
import math
from pathlib import Path
import sys
from typing import Mapping, Sequence

import pandas as pd

from src.build_history import (
    HistoryError,
    build_team_history_frame,
    read_canonical_matches,
    read_team_history,
    validate_team_history,
    write_csv_atomic,
)
from src.clean_data import CleaningError, PROJECT_ROOT, validate_canonical_table
from src.constants import (
    AWAY_FEATURE_COLUMNS,
    CURRENT_MATCH_STAT_COLUMNS,
    FEATURE_COLUMNS,
    HOME_FEATURE_COLUMNS,
    MODEL_DATASET_COLUMNS,
    ROLLING_MIN_PERIODS,
    ROLLING_WINDOW,
    TARGET_MAPPING,
    TEAM_ROLLING_FEATURE_COLUMNS,
)


SUPPORTED_FEATURE_SETS = ("baseline",)


class FeatureError(RuntimeError):
    """Raised when feature construction or preprocessing violates its contract."""


@dataclass(frozen=True)
class FeatureBuildSummary:
    """Counts emitted after a successful feature build."""

    canonical_matches: int
    output_rows: int
    filtered_rows: int
    cold_start_rows: int
    output_path: Path


def _broadcast_first_row_by_group(
    frame: pd.DataFrame,
    group_columns: Sequence[str],
    value_columns: Sequence[str],
) -> None:
    """Give contemporaneous rows the first row's strictly prior feature values."""
    first_rows = frame.loc[
        ~frame.duplicated(list(group_columns), keep="first"),
        [*group_columns, *value_columns],
    ].set_index(list(group_columns))
    group_keys = pd.MultiIndex.from_frame(frame.loc[:, list(group_columns)])
    for column in value_columns:
        frame[column] = first_rows[column].reindex(group_keys).to_numpy()


def compute_team_rolling_features(
    history: pd.DataFrame,
    *,
    window: int = ROLLING_WINDOW,
    min_periods: int = ROLLING_MIN_PERIODS,
) -> pd.DataFrame:
    """Calculate strict pre-match team features using shift before every rolling call."""
    if window <= 0:
        raise FeatureError("Rolling window must be positive.")
    if min_periods <= 0 or min_periods > window:
        raise FeatureError("Rolling min_periods must be between 1 and the window size.")
    validate_team_history(history)

    working = history.copy()
    try:
        pd.to_datetime(working["date"], format="%Y-%m-%d", errors="raise")
    except (TypeError, ValueError) as exc:
        raise FeatureError(f"Team history contains invalid dates: {exc}") from exc
    working = working.sort_values(
        ["team_slug", "date", "match_id"], kind="mergesort"
    ).reset_index(drop=True)

    overall_group = working.groupby("team_slug", sort=False, group_keys=False)
    working["goals_for_avg_5"] = overall_group["goals_for"].transform(
        lambda values: values.shift(1).rolling(window, min_periods=min_periods).mean()
    )
    working["goals_against_avg_5"] = overall_group["goals_against"].transform(
        lambda values: values.shift(1).rolling(window, min_periods=min_periods).mean()
    )
    working["shots_avg_5"] = overall_group["shots"].transform(
        lambda values: values.shift(1).rolling(window, min_periods=min_periods).mean()
    )
    working["form_points_5"] = overall_group["points"].transform(
        lambda values: values.shift(1).rolling(window, min_periods=min_periods).sum()
    )
    working["overall_ppg_5"] = overall_group["points"].transform(
        lambda values: values.shift(1).rolling(window, min_periods=min_periods).mean()
    )
    working["history_matches"] = overall_group.cumcount().astype("int64")
    _broadcast_first_row_by_group(
        working,
        ("team_slug", "date"),
        (
            "goals_for_avg_5",
            "goals_against_avg_5",
            "shots_avg_5",
            "form_points_5",
            "overall_ppg_5",
            "history_matches",
        ),
    )

    venue_group = working.groupby(
        ["team_slug", "is_home"], sort=False, group_keys=False
    )
    working["venue_ppg_5"] = venue_group["points"].transform(
        lambda values: values.shift(1).rolling(window, min_periods=min_periods).mean()
    )
    working["venue_history_matches"] = venue_group.cumcount().astype("int64")
    _broadcast_first_row_by_group(
        working,
        ("team_slug", "is_home", "date"),
        ("venue_ppg_5", "venue_history_matches"),
    )

    return working.loc[
        :,
        [
            "team_match_id",
            "match_id",
            "date",
            "team_slug",
            "is_home",
            *TEAM_ROLLING_FEATURE_COLUMNS,
        ],
    ]


def previous_match_ids(
    history: pd.DataFrame,
    team_slug: str,
    current_match_date: date | str,
    *,
    window: int = ROLLING_WINDOW,
    is_home: bool | None = None,
) -> tuple[str, ...]:
    """Return exact prior source IDs for a strict-date rolling window audit."""
    if window <= 0:
        raise FeatureError("Source window size must be positive.")
    if isinstance(current_match_date, date):
        date_text = current_match_date.isoformat()
    else:
        date_text = str(current_match_date)
    try:
        current_date = datetime.strptime(date_text, "%Y-%m-%d").date()
    except ValueError as exc:
        raise FeatureError(
            f"Source-window date must be YYYY-MM-DD, received {date_text!r}."
        ) from exc

    history_dates = pd.to_datetime(
        history["date"], format="%Y-%m-%d", errors="raise"
    ).dt.date
    mask = history["team_slug"].eq(team_slug) & history_dates.lt(current_date)
    if is_home is not None:
        mask &= history["is_home"].astype("bool").eq(is_home)
    prior = history.loc[mask, ["date", "match_id"]].sort_values(
        ["date", "match_id"], kind="mergesort"
    )
    return tuple(prior.tail(window)["match_id"].astype(str))


def _prefixed_team_features(
    team_features: pd.DataFrame,
    *,
    is_home: bool,
    prefix: str,
) -> pd.DataFrame:
    selected = team_features.loc[
        team_features["is_home"].astype("bool").eq(is_home),
        ["match_id", *TEAM_ROLLING_FEATURE_COLUMNS],
    ].copy()
    if not selected["match_id"].is_unique:
        raise FeatureError(f"Multiple {prefix} feature rows exist for one fixture.")
    return selected.rename(
        columns={column: f"{prefix}_{column}" for column in TEAM_ROLLING_FEATURE_COLUMNS}
    )


def build_model_dataset_frame(
    canonical: pd.DataFrame,
    history: pd.DataFrame | None = None,
    *,
    feature_set: str = "baseline",
) -> pd.DataFrame:
    """Join home/away strict pre-match features to one labeled fixture row."""
    if feature_set not in SUPPORTED_FEATURE_SETS:
        raise FeatureError(
            f"Unsupported feature set {feature_set!r}; expected one of "
            f"{', '.join(SUPPORTED_FEATURE_SETS)}."
        )
    validate_canonical_table(canonical)
    if history is None:
        history = build_team_history_frame(canonical)
    else:
        validate_team_history(history, canonical)

    team_features = compute_team_rolling_features(history)
    home_features = _prefixed_team_features(
        team_features, is_home=True, prefix="home"
    )
    away_features = _prefixed_team_features(
        team_features, is_home=False, prefix="away"
    )

    model = canonical.loc[
        :, ["match_id", "season", "date", "home_team", "away_team"]
    ].copy()
    model["target"] = canonical["result_code"].map(TARGET_MAPPING)
    if model["target"].isna().any():
        invalid = sorted(set(canonical.loc[model["target"].isna(), "result_code"]))
        raise FeatureError(f"Cannot map result codes to targets: {invalid}")
    model["target"] = model["target"].astype("int64")
    try:
        model = model.merge(
            home_features, on="match_id", how="left", validate="one_to_one"
        ).merge(away_features, on="match_id", how="left", validate="one_to_one")
    except pd.errors.MergeError as exc:
        raise FeatureError(f"Could not join one feature row per fixture: {exc}") from exc

    # Rolling values are legitimately NaN during cold starts, but counts are
    # always present when both team-history rows joined successfully.
    count_columns = [
        "home_history_matches",
        "home_venue_history_matches",
        "away_history_matches",
        "away_venue_history_matches",
    ]
    if model[count_columns].isna().any(axis=None):
        missing_ids = model.loc[
            model[count_columns].isna().any(axis=1), "match_id"
        ].tolist()[:5]
        raise FeatureError(f"Missing home or away feature joins: {missing_ids}")

    model["goals_scored_edge"] = (
        model["home_goals_for_avg_5"] - model["away_goals_for_avg_5"]
    )
    model["defensive_edge"] = (
        model["away_goals_against_avg_5"] - model["home_goals_against_avg_5"]
    )
    model["shots_edge"] = model["home_shots_avg_5"] - model["away_shots_avg_5"]
    model["form_edge"] = model["home_form_points_5"] - model["away_form_points_5"]
    model["venue_edge"] = model["home_venue_ppg_5"] - model["away_venue_ppg_5"]
    model["history_edge"] = (
        model["home_history_matches"] - model["away_history_matches"]
    )

    model = model.loc[:, MODEL_DATASET_COLUMNS].sort_values(
        ["date", "match_id"], kind="mergesort"
    ).reset_index(drop=True)
    validate_model_dataset(model, canonical)
    return model


def _series_values_match(actual: pd.Series, expected: pd.Series) -> bool:
    actual_numeric = pd.to_numeric(actual, errors="coerce")
    expected_numeric = pd.to_numeric(expected, errors="coerce")
    if (actual.notna() & actual_numeric.isna()).any():
        return False
    close = actual_numeric.sub(expected_numeric).abs().le(1e-12).fillna(False)
    both_missing = actual_numeric.isna() & expected_numeric.isna()
    return bool((close | both_missing).all())


def validate_model_dataset(
    model: pd.DataFrame,
    canonical: pd.DataFrame | None = None,
) -> None:
    """Validate feature, label, edge direction, and one-row fixture invariants."""
    if tuple(model.columns) != MODEL_DATASET_COLUMNS:
        raise FeatureError("Model dataset columns do not match the required order.")
    if model.empty:
        raise FeatureError("Model dataset contains no fixtures.")
    if model["match_id"].isna().any() or not model["match_id"].is_unique:
        raise FeatureError("Model dataset must contain one unique row per fixture.")
    if not model["target"].isin(TARGET_MAPPING.values()).all():
        raise FeatureError("Model targets must use only the fixed 0/1/2 encoding.")

    forbidden = set(FEATURE_COLUMNS).intersection(CURRENT_MATCH_STAT_COLUMNS)
    if forbidden:
        raise FeatureError(f"Current-match statistics entered FEATURE_COLUMNS: {forbidden}")
    if "home_advantage" in FEATURE_COLUMNS:
        raise FeatureError("A constant home_advantage feature is prohibited.")
    if any("possession" in column for column in FEATURE_COLUMNS):
        raise FeatureError("Baseline FEATURE_COLUMNS must exclude possession.")

    for column in (
        "home_history_matches",
        "away_history_matches",
        "home_venue_history_matches",
        "away_venue_history_matches",
    ):
        values = pd.to_numeric(model[column], errors="coerce")
        if values.isna().any() or values.lt(0).any() or values.mod(1).ne(0).any():
            raise FeatureError(f"History-count feature {column} is invalid.")

    edge_expectations = {
        "goals_scored_edge": model["home_goals_for_avg_5"]
        - model["away_goals_for_avg_5"],
        "defensive_edge": model["away_goals_against_avg_5"]
        - model["home_goals_against_avg_5"],
        "shots_edge": model["home_shots_avg_5"] - model["away_shots_avg_5"],
        "form_edge": model["home_form_points_5"] - model["away_form_points_5"],
        "venue_edge": model["home_venue_ppg_5"] - model["away_venue_ppg_5"],
        "history_edge": model["home_history_matches"]
        - model["away_history_matches"],
    }
    for edge_column, expected in edge_expectations.items():
        if not _series_values_match(model[edge_column], expected):
            raise FeatureError(f"Directional edge {edge_column} has an invalid sign.")

    dates = pd.to_datetime(model["date"], format="%Y-%m-%d", errors="raise")
    if not dates.is_monotonic_increasing:
        raise FeatureError("Model dataset must be chronological.")
    if canonical is not None:
        if len(model) != len(canonical):
            raise FeatureError(
                "Phase 3 retains every canonical fixture; output row count differs."
            )
        if set(model["match_id"]) != set(canonical["match_id"]):
            raise FeatureError("Model and canonical fixture IDs do not agree.")


def fit_training_medians(
    training_frame: pd.DataFrame,
    feature_columns: Sequence[str] = FEATURE_COLUMNS,
) -> dict[str, float]:
    """Fit cold-start medians from an explicitly supplied training frame only."""
    if training_frame.empty:
        raise FeatureError("Cannot fit imputation medians on an empty training frame.")
    missing_columns = set(feature_columns).difference(training_frame.columns)
    if missing_columns:
        raise FeatureError(
            f"Training frame is missing feature columns: {sorted(missing_columns)}"
        )

    medians: dict[str, float] = {}
    for column in feature_columns:
        values = pd.to_numeric(training_frame[column], errors="coerce")
        invalid = training_frame[column].notna() & values.isna()
        if invalid.any():
            raise FeatureError(f"Training feature {column} contains nonnumeric values.")
        median = values.median(skipna=True)
        if pd.isna(median) or not math.isfinite(float(median)):
            raise FeatureError(f"Training feature {column} has no finite median.")
        medians[column] = float(median)
    return medians


def apply_training_medians(
    frame: pd.DataFrame,
    medians: Mapping[str, float],
    feature_columns: Sequence[str] = FEATURE_COLUMNS,
) -> pd.DataFrame:
    """Apply already-fitted training medians without learning from this frame."""
    missing_columns = set(feature_columns).difference(frame.columns)
    if missing_columns:
        raise FeatureError(f"Frame is missing feature columns: {sorted(missing_columns)}")
    missing_medians = set(feature_columns).difference(medians)
    if missing_medians:
        raise FeatureError(f"Missing fitted medians: {sorted(missing_medians)}")

    transformed = frame.copy()
    for column in feature_columns:
        median = float(medians[column])
        if not math.isfinite(median):
            raise FeatureError(f"Imputation median for {column} is not finite.")
        values = pd.to_numeric(transformed[column], errors="coerce")
        invalid = transformed[column].notna() & values.isna()
        if invalid.any():
            raise FeatureError(f"Feature {column} contains nonnumeric values.")
        transformed[column] = values.fillna(median)
    return transformed


def build_and_save_features(
    feature_set: str,
    project_root: Path = PROJECT_ROOT,
) -> FeatureBuildSummary:
    """Build and atomically save the requested Phase 3 feature set."""
    project_root = project_root.resolve()
    canonical = read_canonical_matches(
        project_root / "data" / "processed" / "canonical_matches.csv"
    )
    history = read_team_history(
        project_root / "data" / "processed" / "team_match_history.csv"
    )
    model = build_model_dataset_frame(
        canonical,
        history,
        feature_set=feature_set,
    )
    output_path = project_root / "data" / "processed" / "model_dataset.csv"
    write_csv_atomic(model, output_path, MODEL_DATASET_COLUMNS)
    cold_start_rows = int(
        (
            model["home_history_matches"].lt(ROLLING_MIN_PERIODS)
            | model["away_history_matches"].lt(ROLLING_MIN_PERIODS)
        ).sum()
    )
    return FeatureBuildSummary(
        canonical_matches=len(canonical),
        output_rows=len(model),
        filtered_rows=len(canonical) - len(model),
        cold_start_rows=cold_start_rows,
        output_path=output_path,
    )


def build_parser() -> argparse.ArgumentParser:
    """Build the Phase 3 feature command-line parser."""
    parser = argparse.ArgumentParser(
        description="Build leakage-safe pre-match EPL rolling features."
    )
    parser.add_argument(
        "--feature-set",
        required=True,
        choices=SUPPORTED_FEATURE_SETS,
        help="feature contract to build (Phase 3 supports baseline only)",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the leakage-safe feature CLI."""
    parser = build_parser()
    arguments = parser.parse_args(argv)
    try:
        summary = build_and_save_features(arguments.feature_set)
    except (FeatureError, HistoryError, CleaningError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    print(f"canonical_matches={summary.canonical_matches}")
    print(f"output_rows={summary.output_rows}")
    print(f"filtered_rows={summary.filtered_rows}")
    print(f"cold_start_rows={summary.cold_start_rows}")
    print(f"output={summary.output_path.relative_to(PROJECT_ROOT).as_posix()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
