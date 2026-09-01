"""Build the two-row-per-fixture team match history table."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import os
from pathlib import Path
import sys
import tempfile
from typing import Sequence

import pandas as pd

from src.clean_data import (
    CANONICAL_MATCH_COLUMNS,
    CleaningError,
    PROJECT_ROOT,
    validate_canonical_table,
)
from src.constants import TEAM_HISTORY_COLUMNS


CANONICAL_MATCHES_PATH = PROJECT_ROOT / "data" / "processed" / "canonical_matches.csv"
TEAM_MATCH_HISTORY_PATH = (
    PROJECT_ROOT / "data" / "processed" / "team_match_history.csv"
)


class HistoryError(RuntimeError):
    """Raised when team-history input or output violates its contract."""


@dataclass(frozen=True)
class HistoryBuildSummary:
    """Counts emitted after a successful team-history build."""

    canonical_matches: int
    history_rows: int
    output_path: Path


def points_from_score(goals_for: int, goals_against: int) -> int:
    """Return EPL points from one team's view of a completed fixture."""
    if goals_for > goals_against:
        return 3
    if goals_for == goals_against:
        return 1
    return 0


def read_canonical_matches(path: Path = CANONICAL_MATCHES_PATH) -> pd.DataFrame:
    """Read the Phase 2 canonical table with stable identity dtypes."""
    try:
        frame = pd.read_csv(
            path,
            dtype={
                "match_id": "string",
                "season": "string",
                "date": "string",
                "kickoff_time": "string",
                "home_team": "string",
                "away_team": "string",
                "home_team_slug": "string",
                "away_team_slug": "string",
                "result_code": "string",
                "football_data_source_file": "string",
            },
            low_memory=False,
        )
    except FileNotFoundError as exc:
        raise HistoryError(
            f"Canonical match table not found: {path}. Run python -m src.clean_data first."
        ) from exc
    except (
        OSError,
        UnicodeDecodeError,
        pd.errors.EmptyDataError,
        pd.errors.ParserError,
    ) as exc:
        raise HistoryError(f"Could not read canonical match table {path}: {exc}") from exc

    if tuple(frame.columns) != CANONICAL_MATCH_COLUMNS:
        raise HistoryError(
            f"Canonical match table {path} does not match the required column order."
        )
    return frame


def _points_series(goals_for: pd.Series, goals_against: pd.Series) -> pd.Series:
    values = [
        points_from_score(int(scored), int(conceded))
        for scored, conceded in zip(goals_for, goals_against)
    ]
    return pd.Series(values, index=goals_for.index, dtype="int64")


def build_team_history_frame(canonical: pd.DataFrame) -> pd.DataFrame:
    """Expand each canonical fixture into home- and away-team perspectives."""
    validate_canonical_table(canonical)
    canonical = canonical.reset_index(drop=True)

    home = pd.DataFrame(
        {
            "team_match_id": canonical["match_id"].astype("string") + "|home",
            "match_id": canonical["match_id"].astype("string"),
            "season": canonical["season"].astype("string"),
            "date": canonical["date"].astype("string"),
            "team": canonical["home_team"].astype("string"),
            "team_slug": canonical["home_team_slug"].astype("string"),
            "opponent": canonical["away_team"].astype("string"),
            "opponent_slug": canonical["away_team_slug"].astype("string"),
            "is_home": pd.Series([True] * len(canonical), dtype="bool"),
            "goals_for": pd.to_numeric(canonical["home_goals"]).astype("int64"),
            "goals_against": pd.to_numeric(canonical["away_goals"]).astype("int64"),
            "shots": pd.to_numeric(canonical["home_shots"]).astype("Int64"),
            "possession": pd.to_numeric(
                canonical["home_possession"], errors="coerce"
            ).astype("Float64"),
        }
    )
    home["points"] = _points_series(home["goals_for"], home["goals_against"])

    away = pd.DataFrame(
        {
            "team_match_id": canonical["match_id"].astype("string") + "|away",
            "match_id": canonical["match_id"].astype("string"),
            "season": canonical["season"].astype("string"),
            "date": canonical["date"].astype("string"),
            "team": canonical["away_team"].astype("string"),
            "team_slug": canonical["away_team_slug"].astype("string"),
            "opponent": canonical["home_team"].astype("string"),
            "opponent_slug": canonical["home_team_slug"].astype("string"),
            "is_home": pd.Series([False] * len(canonical), dtype="bool"),
            "goals_for": pd.to_numeric(canonical["away_goals"]).astype("int64"),
            "goals_against": pd.to_numeric(canonical["home_goals"]).astype("int64"),
            "shots": pd.to_numeric(canonical["away_shots"]).astype("Int64"),
            "possession": pd.to_numeric(
                canonical["away_possession"], errors="coerce"
            ).astype("Float64"),
        }
    )
    away["points"] = _points_series(away["goals_for"], away["goals_against"])

    history = pd.concat([home, away], ignore_index=True)
    history = history.loc[:, TEAM_HISTORY_COLUMNS].sort_values(
        ["date", "match_id", "is_home"],
        ascending=[True, True, False],
        kind="mergesort",
    ).reset_index(drop=True)
    validate_team_history(history, canonical)
    return history


def validate_team_history(
    history: pd.DataFrame,
    canonical: pd.DataFrame | None = None,
) -> None:
    """Validate the complete team-history schema and two-row fixture invariant."""
    if tuple(history.columns) != TEAM_HISTORY_COLUMNS:
        raise HistoryError("Team history columns do not match the required order.")
    if history.empty:
        raise HistoryError("Team history contains no rows.")
    if history["team_match_id"].isna().any() or not history["team_match_id"].is_unique:
        raise HistoryError("team_match_id values must be nonempty and unique.")

    fixture_counts = history.groupby("match_id", sort=False).size()
    if not fixture_counts.eq(2).all():
        invalid_ids = fixture_counts[fixture_counts.ne(2)].index.tolist()[:5]
        raise HistoryError(
            f"Every fixture must produce exactly two history rows; invalid: {invalid_ids}"
        )

    venue_counts = history.groupby("match_id", sort=False)["is_home"].agg(
        lambda values: set(bool(value) for value in values)
    )
    if not venue_counts.map(lambda values: values == {False, True}).all():
        raise HistoryError("Every fixture must have one home and one away history row.")
    team_counts = history.groupby("match_id", sort=False)["team_slug"].nunique()
    if not team_counts.eq(2).all():
        raise HistoryError("Every fixture must contain two distinct team histories.")

    expected_points = [
        points_from_score(int(scored), int(conceded))
        for scored, conceded in zip(history["goals_for"], history["goals_against"])
    ]
    if not history["points"].astype("int64").equals(
        pd.Series(expected_points, index=history.index, dtype="int64")
    ):
        raise HistoryError("Team-history points disagree with goals for and against.")
    if (pd.to_numeric(history["goals_for"]) < 0).any() or (
        pd.to_numeric(history["goals_against"]) < 0
    ).any():
        raise HistoryError("Team-history goals must be nonnegative.")
    if not history["points"].isin([0, 1, 3]).all():
        raise HistoryError("Team-history points must be 0, 1, or 3.")

    parsed_dates = pd.to_datetime(history["date"], format="%Y-%m-%d", errors="raise")
    if not parsed_dates.is_monotonic_increasing:
        raise HistoryError("Team history must be chronological.")

    if canonical is not None:
        if len(history) != 2 * len(canonical):
            raise HistoryError(
                "Team history row count must be exactly twice canonical match count."
            )
        if set(history["match_id"]) != set(canonical["match_id"]):
            raise HistoryError("Team history and canonical match IDs do not agree.")


def read_team_history(path: Path = TEAM_MATCH_HISTORY_PATH) -> pd.DataFrame:
    """Read the persisted team history with explicit nullable dtypes."""
    try:
        frame = pd.read_csv(
            path,
            dtype={
                "team_match_id": "string",
                "match_id": "string",
                "season": "string",
                "date": "string",
                "team": "string",
                "team_slug": "string",
                "opponent": "string",
                "opponent_slug": "string",
                "is_home": "string",
            },
            low_memory=False,
        )
    except FileNotFoundError as exc:
        raise HistoryError(
            f"Team history not found: {path}. Run python -m src.build_history first."
        ) from exc
    except (
        OSError,
        UnicodeDecodeError,
        pd.errors.EmptyDataError,
        pd.errors.ParserError,
    ) as exc:
        raise HistoryError(f"Could not read team history {path}: {exc}") from exc

    if tuple(frame.columns) != TEAM_HISTORY_COLUMNS:
        raise HistoryError(f"Team history {path} has an unexpected column order.")
    venue_mapping = {"True": True, "False": False}
    invalid_venues = set(frame["is_home"].dropna()) - set(venue_mapping)
    if invalid_venues or frame["is_home"].isna().any():
        raise HistoryError(f"Team history has invalid is_home values: {invalid_venues}")
    frame["is_home"] = frame["is_home"].map(venue_mapping).astype("bool")
    for column in ("goals_for", "goals_against", "points"):
        frame[column] = pd.to_numeric(frame[column], errors="raise").astype("int64")
    frame["shots"] = pd.to_numeric(frame["shots"], errors="raise").astype("Int64")
    frame["possession"] = pd.to_numeric(
        frame["possession"], errors="raise"
    ).astype("Float64")
    validate_team_history(frame)
    return frame


def write_csv_atomic(
    frame: pd.DataFrame,
    output_path: Path,
    columns: Sequence[str],
) -> None:
    """Write a CSV through a same-directory temporary file and atomic rename."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    file_descriptor, temporary_name = tempfile.mkstemp(
        dir=output_path.parent,
        prefix=f".{output_path.name}.",
        suffix=".tmp",
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(file_descriptor, "w", encoding="utf-8", newline="") as output:
            frame.to_csv(
                output,
                index=False,
                columns=columns,
                lineterminator="\n",
            )
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary_path, output_path)
    except (OSError, TypeError, ValueError) as exc:
        raise HistoryError(f"Could not atomically write {output_path}: {exc}") from exc
    finally:
        temporary_path.unlink(missing_ok=True)


def build_and_save_team_history(project_root: Path = PROJECT_ROOT) -> HistoryBuildSummary:
    """Read canonical matches, build histories, and save the validated output."""
    project_root = project_root.resolve()
    canonical_path = project_root / "data" / "processed" / "canonical_matches.csv"
    output_path = project_root / "data" / "processed" / "team_match_history.csv"
    canonical = read_canonical_matches(canonical_path)
    history = build_team_history_frame(canonical)
    write_csv_atomic(history, output_path, TEAM_HISTORY_COLUMNS)
    return HistoryBuildSummary(
        canonical_matches=len(canonical),
        history_rows=len(history),
        output_path=output_path,
    )


def build_parser() -> argparse.ArgumentParser:
    """Build the Phase 3 history command-line parser."""
    return argparse.ArgumentParser(
        description="Build two chronological team-history rows per EPL fixture."
    )


def main(argv: Sequence[str] | None = None) -> int:
    """Run the team-history CLI."""
    parser = build_parser()
    parser.parse_args(argv)
    try:
        summary = build_and_save_team_history()
    except (HistoryError, CleaningError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    print(f"canonical_matches={summary.canonical_matches}")
    print(f"history_rows={summary.history_rows}")
    print(f"output={summary.output_path.relative_to(PROJECT_ROOT).as_posix()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
