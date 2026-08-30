"""Build the validated canonical EPL match table from immutable raw CSVs."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import date, datetime
import os
from pathlib import Path
import re
import sys
import tempfile
from typing import Mapping, Sequence

import pandas as pd

from src.download_data import (
    IngestionError,
    LEAGUE_CODE,
    PROJECT_ROOT,
    REQUIRED_COLUMNS,
    Season,
    load_seasons,
    verify_football_data_manifest,
)


TEAM_NAME_MAP_PATH = PROJECT_ROOT / "config" / "team_name_map.csv"
FOOTBALL_DATA_PROVIDER = "football_data"
TEAM_MAP_COLUMNS = (
    "provider",
    "provider_team_name",
    "canonical_team_name",
    "canonical_team_slug",
)
CANONICAL_MATCH_COLUMNS = (
    "match_id",
    "season",
    "date",
    "kickoff_time",
    "home_team",
    "away_team",
    "home_team_slug",
    "away_team_slug",
    "home_goals",
    "away_goals",
    "result_code",
    "home_shots",
    "away_shots",
    "home_possession",
    "away_possession",
    "football_data_source_file",
    "api_fixture_id",
)
SOURCE_DATE_FORMATS = ("%d/%m/%y", "%d/%m/%Y")
SOURCE_TIME_FORMAT = "%H:%M"
VALID_RESULT_CODES = frozenset({"A", "D", "H"})


class CleaningError(RuntimeError):
    """Base class for actionable canonical-data cleaning failures."""


class RawDataError(CleaningError):
    """Raised when required raw data cannot be read or violates its contract."""


class DateValidationError(CleaningError):
    """Raised when a source match date is missing or malformed."""


class KickoffTimeValidationError(CleaningError):
    """Raised when a provided source kickoff time is malformed."""


class TeamMappingError(CleaningError):
    """Raised when the team map is malformed or a provider name is unknown."""


class NumericValidationError(CleaningError):
    """Raised when a goal or shot value is invalid."""


class ResultValidationError(CleaningError):
    """Raised when a result code is invalid or disagrees with final goals."""


class DuplicateMatchError(CleaningError):
    """Raised when one match ID has conflicting canonical records."""


class CanonicalIntegrityError(CleaningError):
    """Raised when the completed canonical table violates its output contract."""


@dataclass(frozen=True)
class TeamIdentity:
    """The canonical identity assigned to a provider team name."""

    name: str
    slug: str


@dataclass(frozen=True)
class SeasonCleaningResult:
    """One cleaned season plus source-quality counts."""

    frame: pd.DataFrame
    input_rows: int
    missing_shots: int


@dataclass(frozen=True)
class CleaningSummary:
    """Auditable counts emitted after a successful canonical build."""

    input_rows: int
    output_rows: int
    duplicate_rows: int
    missing_shots: int
    unresolved_teams: int
    output_path: Path


def parse_match_date(value: object) -> date:
    """Parse an explicit day-first Football-Data date format."""
    if pd.isna(value):
        raise DateValidationError("Match date is missing.")
    text = str(value).strip()
    if not text:
        raise DateValidationError("Match date is empty.")
    for date_format in SOURCE_DATE_FORMATS:
        try:
            return datetime.strptime(text, date_format).date()
        except ValueError:
            continue
    supported = ", ".join(SOURCE_DATE_FORMATS)
    raise DateValidationError(
        f"Invalid match date {text!r}; expected an explicit day-first format: {supported}."
    )


def parse_optional_kickoff_time(value: object) -> str | None:
    """Normalize a provided kickoff time without fabricating missing values."""
    if pd.isna(value):
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        parsed = datetime.strptime(text, SOURCE_TIME_FORMAT)
    except ValueError as exc:
        raise KickoffTimeValidationError(
            f"Invalid kickoff time {text!r}; expected HH:MM."
        ) from exc
    return parsed.strftime(SOURCE_TIME_FORMAT)


def load_team_name_map(
    path: Path = TEAM_NAME_MAP_PATH,
) -> dict[tuple[str, str], TeamIdentity]:
    """Load exact provider-to-canonical team mappings with conflict checks."""
    try:
        frame = pd.read_csv(path, dtype="string", keep_default_na=False)
    except FileNotFoundError as exc:
        raise TeamMappingError(f"Team-name mapping file not found: {path}") from exc
    except (
        OSError,
        UnicodeDecodeError,
        pd.errors.EmptyDataError,
        pd.errors.ParserError,
    ) as exc:
        raise TeamMappingError(f"Could not read team-name mapping {path}: {exc}") from exc

    missing_columns = set(TEAM_MAP_COLUMNS).difference(frame.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise TeamMappingError(f"Team-name mapping {path} is missing columns: {missing}")

    mappings: dict[tuple[str, str], TeamIdentity] = {}
    canonical_slugs: dict[str, str] = {}
    slug_names: dict[str, str] = {}
    for row_number, row in enumerate(frame.itertuples(index=False), start=2):
        values = {
            column: str(getattr(row, column)).strip() for column in TEAM_MAP_COLUMNS
        }
        empty_fields = [field for field, value in values.items() if not value]
        if empty_fields:
            fields = ", ".join(empty_fields)
            raise TeamMappingError(
                f"Team-name mapping row {row_number} has empty fields: {fields}"
            )

        slug = values["canonical_team_slug"]
        if re.fullmatch(r"[a-z0-9]+(?:-[a-z0-9]+)*", slug) is None:
            raise TeamMappingError(
                f"Team-name mapping row {row_number} has invalid slug {slug!r}."
            )

        key = (values["provider"], values["provider_team_name"])
        if key in mappings:
            raise TeamMappingError(
                f"Duplicate provider team mapping for {key[0]}:{key[1]}."
            )

        canonical_name = values["canonical_team_name"]
        previous_slug = canonical_slugs.get(canonical_name)
        if previous_slug is not None and previous_slug != slug:
            raise TeamMappingError(
                f"Canonical team {canonical_name!r} maps to multiple slugs."
            )
        previous_name = slug_names.get(slug)
        if previous_name is not None and previous_name != canonical_name:
            raise TeamMappingError(f"Canonical slug {slug!r} maps to multiple teams.")

        identity = TeamIdentity(name=canonical_name, slug=slug)
        mappings[key] = identity
        canonical_slugs[canonical_name] = slug
        slug_names[slug] = canonical_name

    if not mappings:
        raise TeamMappingError(f"Team-name mapping {path} contains no rows.")
    return mappings


def resolve_team_name(
    provider: str,
    provider_team_name: str,
    mappings: Mapping[tuple[str, str], TeamIdentity],
) -> TeamIdentity:
    """Resolve one exact provider team name or fail without fuzzy matching."""
    normalized_provider_name = provider_team_name.strip()
    key = (provider, normalized_provider_name)
    try:
        return mappings[key]
    except KeyError as exc:
        raise TeamMappingError(
            f"Unmapped team name for provider {provider!r}: "
            f"{normalized_provider_name!r}. Add it to config/team_name_map.csv."
        ) from exc


def parse_nonnegative_integer_series(
    values: pd.Series,
    column_name: str,
    *,
    allow_missing: bool,
) -> pd.Series:
    """Convert numeric source values to integers while preserving allowed missingness."""
    text = values.astype("string").str.strip()
    provided = values.notna() & text.ne("").fillna(False)
    numeric = pd.to_numeric(text.where(provided), errors="coerce")

    invalid = provided & numeric.isna()
    if invalid.any():
        bad_value = text.loc[invalid].iloc[0]
        raise NumericValidationError(
            f"Invalid numeric value in {column_name}: {bad_value!r}."
        )
    if not allow_missing and numeric.isna().any():
        raise NumericValidationError(f"Required numeric column {column_name} is missing.")

    fractional = numeric.notna() & numeric.mod(1).ne(0)
    if fractional.any():
        bad_value = numeric.loc[fractional].iloc[0]
        raise NumericValidationError(
            f"Non-integer value in {column_name}: {bad_value!r}."
        )
    negative = numeric.notna() & numeric.lt(0)
    if negative.any():
        bad_value = numeric.loc[negative].iloc[0]
        raise NumericValidationError(
            f"Negative value in {column_name}: {bad_value!r}."
        )

    if allow_missing:
        return numeric.astype("Int64")
    return numeric.astype("int64")


def expected_result_code(home_goals: int, away_goals: int) -> str:
    """Return H, D, or A from final goals."""
    if home_goals > away_goals:
        return "H"
    if home_goals < away_goals:
        return "A"
    return "D"


def validate_result_code(home_goals: int, away_goals: int, result_code: object) -> str:
    """Validate a source result code against the final score."""
    if pd.isna(result_code):
        raise ResultValidationError("Full-time result code is missing.")
    normalized = str(result_code).strip()
    if normalized not in VALID_RESULT_CODES:
        raise ResultValidationError(
            f"Invalid full-time result code {normalized!r}; expected A, D, or H."
        )
    expected = expected_result_code(home_goals, away_goals)
    if normalized != expected:
        raise ResultValidationError(
            f"Result code {normalized!r} disagrees with score "
            f"{home_goals}-{away_goals}; expected {expected!r}."
        )
    return normalized


def generate_match_id(
    season_code: str,
    match_date: date | str,
    home_team_slug: str,
    away_team_slug: str,
) -> str:
    """Build the deterministic canonical match identifier."""
    date_text = match_date.isoformat() if isinstance(match_date, date) else str(match_date)
    try:
        normalized_date = datetime.strptime(date_text, "%Y-%m-%d").date().isoformat()
    except ValueError as exc:
        raise DateValidationError(
            f"Canonical match ID requires an ISO date, received {date_text!r}."
        ) from exc
    return f"{season_code}|{normalized_date}|{home_team_slug}|{away_team_slug}"


def _required_text_series(frame: pd.DataFrame, column_name: str) -> pd.Series:
    values = frame[column_name].astype("string").str.strip()
    invalid = values.isna() | values.eq("").fillna(True)
    if invalid.any():
        raise RawDataError(f"Required source column {column_name} contains an empty value.")
    return values


def clean_season_frame(
    raw_frame: pd.DataFrame,
    season: Season,
    source_file: str,
    mappings: Mapping[tuple[str, str], TeamIdentity],
) -> SeasonCleaningResult:
    """Transform one raw EPL season into canonical columns without mutating input."""
    frame = raw_frame.dropna(how="all").reset_index(drop=True).copy()
    if frame.empty:
        raise RawDataError(f"Raw source {source_file} contains no match rows.")

    missing_columns = REQUIRED_COLUMNS.difference(frame.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise RawDataError(f"Raw source {source_file} is missing columns: {missing}")

    divisions = _required_text_series(frame, "Div")
    invalid_divisions = sorted(set(divisions[divisions.ne(LEAGUE_CODE)].tolist()))
    if invalid_divisions:
        raise RawDataError(
            f"Raw source {source_file} contains non-EPL divisions: "
            f"{', '.join(invalid_divisions)}"
        )

    parsed_dates = [parse_match_date(value) for value in frame["Date"]]
    if "Time" in frame.columns:
        kickoff_times = [
            parse_optional_kickoff_time(value) for value in frame["Time"]
        ]
    else:
        kickoff_times = [None] * len(frame)

    home_provider_names = _required_text_series(frame, "HomeTeam")
    away_provider_names = _required_text_series(frame, "AwayTeam")
    provider_names = set(home_provider_names.tolist()) | set(
        away_provider_names.tolist()
    )
    unknown_names = sorted(
        name
        for name in provider_names
        if (FOOTBALL_DATA_PROVIDER, name) not in mappings
    )
    if unknown_names:
        raise TeamMappingError(
            "Unmapped football_data team names: "
            f"{', '.join(unknown_names)}. Add them to config/team_name_map.csv."
        )

    home_identities = [
        resolve_team_name(FOOTBALL_DATA_PROVIDER, name, mappings)
        for name in home_provider_names
    ]
    away_identities = [
        resolve_team_name(FOOTBALL_DATA_PROVIDER, name, mappings)
        for name in away_provider_names
    ]
    home_goals = parse_nonnegative_integer_series(
        frame["FTHG"], "FTHG", allow_missing=False
    )
    away_goals = parse_nonnegative_integer_series(
        frame["FTAG"], "FTAG", allow_missing=False
    )
    home_shots = parse_nonnegative_integer_series(
        frame["HS"], "HS", allow_missing=True
    )
    away_shots = parse_nonnegative_integer_series(
        frame["AS"], "AS", allow_missing=True
    )
    result_codes = [
        validate_result_code(int(home), int(away), result)
        for home, away, result in zip(home_goals, away_goals, frame["FTR"])
    ]

    home_slugs = [identity.slug for identity in home_identities]
    away_slugs = [identity.slug for identity in away_identities]
    match_ids = [
        generate_match_id(season.code, match_date, home_slug, away_slug)
        for match_date, home_slug, away_slug in zip(
            parsed_dates, home_slugs, away_slugs
        )
    ]
    row_count = len(frame)
    canonical = pd.DataFrame(
        {
            "match_id": pd.Series(match_ids, dtype="string"),
            "season": pd.Series([season.code] * row_count, dtype="string"),
            "date": pd.Series(
                [match_date.isoformat() for match_date in parsed_dates], dtype="string"
            ),
            "kickoff_time": pd.Series(kickoff_times, dtype="string"),
            "home_team": pd.Series(
                [identity.name for identity in home_identities], dtype="string"
            ),
            "away_team": pd.Series(
                [identity.name for identity in away_identities], dtype="string"
            ),
            "home_team_slug": pd.Series(home_slugs, dtype="string"),
            "away_team_slug": pd.Series(away_slugs, dtype="string"),
            "home_goals": home_goals,
            "away_goals": away_goals,
            "result_code": pd.Series(result_codes, dtype="string"),
            "home_shots": home_shots,
            "away_shots": away_shots,
            "home_possession": pd.Series([pd.NA] * row_count, dtype="Float64"),
            "away_possession": pd.Series([pd.NA] * row_count, dtype="Float64"),
            "football_data_source_file": pd.Series(
                [source_file] * row_count, dtype="string"
            ),
            "api_fixture_id": pd.Series([pd.NA] * row_count, dtype="Int64"),
        },
        columns=CANONICAL_MATCH_COLUMNS,
    )
    missing_shots = int(home_shots.isna().sum() + away_shots.isna().sum())
    return SeasonCleaningResult(
        frame=canonical,
        input_rows=row_count,
        missing_shots=missing_shots,
    )


def deduplicate_canonical_matches(frame: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Remove exact duplicate fixtures and reject conflicting duplicate records."""
    duplicate_mask = frame["match_id"].duplicated(keep=False)
    if not duplicate_mask.any():
        return frame.reset_index(drop=True), 0

    drop_indices: list[int] = []
    comparison_columns = [
        column
        for column in CANONICAL_MATCH_COLUMNS
        if column != "football_data_source_file"
    ]
    duplicate_rows = frame.loc[duplicate_mask]
    for match_id, group in duplicate_rows.groupby("match_id", sort=True):
        unique_payloads = group[comparison_columns].drop_duplicates()
        if len(unique_payloads) != 1:
            sources = ", ".join(
                sorted(set(group["football_data_source_file"].astype(str)))
            )
            raise DuplicateMatchError(
                f"Conflicting duplicate records for match_id {match_id}: {sources}"
            )
        ordered_group = group.sort_values(
            "football_data_source_file", kind="mergesort"
        )
        drop_indices.extend(ordered_group.index[1:].tolist())

    deduplicated = frame.drop(index=drop_indices).reset_index(drop=True)
    return deduplicated, len(drop_indices)


def validate_canonical_table(frame: pd.DataFrame) -> None:
    """Assert the canonical table's complete Phase 2 integrity contract."""
    if tuple(frame.columns) != CANONICAL_MATCH_COLUMNS:
        raise CanonicalIntegrityError(
            "Canonical table columns do not match the required explicit order."
        )
    if frame.empty:
        raise CanonicalIntegrityError("Canonical table contains no matches.")
    match_ids = frame["match_id"].astype("string").str.strip()
    if (
        match_ids.isna().any()
        or match_ids.eq("").fillna(True).any()
        or not match_ids.is_unique
    ):
        raise CanonicalIntegrityError("Canonical match_id values must be nonempty and unique.")

    for column_name in (
        "season",
        "home_team",
        "away_team",
        "home_team_slug",
        "away_team_slug",
        "result_code",
        "football_data_source_file",
    ):
        values = frame[column_name].astype("string").str.strip()
        if values.isna().any() or values.eq("").fillna(True).any():
            raise CanonicalIntegrityError(
                f"Canonical column {column_name} must be nonempty."
            )

    try:
        dates = pd.to_datetime(frame["date"], format="%Y-%m-%d", errors="raise")
    except (TypeError, ValueError) as exc:
        raise CanonicalIntegrityError(f"Canonical dates are invalid: {exc}") from exc
    if not dates.is_monotonic_increasing:
        raise CanonicalIntegrityError("Canonical matches are not chronological.")

    home_goals_values = parse_nonnegative_integer_series(
        frame["home_goals"], "home_goals", allow_missing=False
    )
    away_goals_values = parse_nonnegative_integer_series(
        frame["away_goals"], "away_goals", allow_missing=False
    )
    parse_nonnegative_integer_series(
        frame["home_shots"], "home_shots", allow_missing=True
    )
    parse_nonnegative_integer_series(
        frame["away_shots"], "away_shots", allow_missing=True
    )

    for position, row in enumerate(frame.itertuples(index=False)):
        home_goals = int(home_goals_values.iloc[position])
        away_goals = int(away_goals_values.iloc[position])
        validate_result_code(home_goals, away_goals, row.result_code)

        season_code = str(row.season)
        expected_id = generate_match_id(
            season_code,
            str(row.date),
            str(row.home_team_slug),
            str(row.away_team_slug),
        )
        if row.match_id != expected_id:
            raise CanonicalIntegrityError(
                f"Non-deterministic match_id for {row.match_id}; expected {expected_id}."
            )
        expected_source = f"data/raw/football_data/E0_{season_code}.csv"
        if row.football_data_source_file != expected_source:
            raise CanonicalIntegrityError(
                f"Invalid source provenance for {row.match_id}: "
                f"{row.football_data_source_file}"
            )
        if row.home_team_slug == row.away_team_slug:
            raise CanonicalIntegrityError(
                f"Fixture {row.match_id} has the same home and away team."
            )
        parse_optional_kickoff_time(row.kickoff_time)


def write_canonical_matches_atomic(frame: pd.DataFrame, output_path: Path) -> None:
    """Write canonical CSV data through a temporary file and atomic rename."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    file_descriptor, temporary_name = tempfile.mkstemp(
        dir=output_path.parent,
        prefix=f".{output_path.name}.",
        suffix=".tmp",
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(file_descriptor, "w", encoding="utf-8", newline="") as output:
            frame.to_csv(output, index=False, columns=CANONICAL_MATCH_COLUMNS, lineterminator="\n")
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary_path, output_path)
    except (OSError, TypeError, ValueError) as exc:
        raise CleaningError(
            f"Could not atomically write canonical table {output_path}: {exc}"
        ) from exc
    finally:
        temporary_path.unlink(missing_ok=True)


def build_canonical_matches(project_root: Path = PROJECT_ROOT) -> CleaningSummary:
    """Build, validate, and atomically save all configured EPL seasons."""
    project_root = project_root.resolve()
    seasons = load_seasons(project_root / "config" / "seasons.json")
    verified_manifest_files = verify_football_data_manifest(project_root)
    if verified_manifest_files != len(seasons):
        raise RawDataError(
            f"Expected {len(seasons)} verified Football-Data files, "
            f"found {verified_manifest_files}."
        )
    mappings = load_team_name_map(project_root / "config" / "team_name_map.csv")

    cleaned_seasons: list[pd.DataFrame] = []
    input_rows = 0
    missing_shots = 0
    for season in seasons:
        relative_source = f"data/raw/football_data/E0_{season.code}.csv"
        raw_path = project_root / relative_source
        try:
            raw_frame = pd.read_csv(raw_path, low_memory=False)
        except FileNotFoundError as exc:
            raise RawDataError(f"Required raw source is missing: {raw_path}") from exc
        except (
            OSError,
            UnicodeDecodeError,
            pd.errors.EmptyDataError,
            pd.errors.ParserError,
        ) as exc:
            raise RawDataError(f"Could not read raw source {raw_path}: {exc}") from exc

        cleaned = clean_season_frame(
            raw_frame,
            season,
            relative_source,
            mappings,
        )
        cleaned_seasons.append(cleaned.frame)
        input_rows += cleaned.input_rows
        missing_shots += cleaned.missing_shots

    canonical = pd.concat(cleaned_seasons, ignore_index=True)
    canonical, duplicate_rows = deduplicate_canonical_matches(canonical)
    canonical = canonical.sort_values(
        ["date", "kickoff_time", "home_team_slug", "away_team_slug", "match_id"],
        kind="mergesort",
        na_position="last",
    ).reset_index(drop=True)
    if input_rows - duplicate_rows != len(canonical):
        raise CanonicalIntegrityError(
            "Canonical output row count is not explained by input and duplicate counts."
        )
    validate_canonical_table(canonical)

    output_path = project_root / "data" / "processed" / "canonical_matches.csv"
    write_canonical_matches_atomic(canonical, output_path)
    return CleaningSummary(
        input_rows=input_rows,
        output_rows=len(canonical),
        duplicate_rows=duplicate_rows,
        missing_shots=missing_shots,
        unresolved_teams=0,
        output_path=output_path,
    )


def build_parser() -> argparse.ArgumentParser:
    """Build the Phase 2 command-line parser."""
    return argparse.ArgumentParser(
        description="Build the validated canonical EPL match table."
    )


def main(argv: Sequence[str] | None = None) -> int:
    """Run the canonical data-cleaning CLI."""
    parser = build_parser()
    parser.parse_args(argv)
    try:
        summary = build_canonical_matches()
    except (CleaningError, IngestionError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    print(f"input_rows={summary.input_rows}")
    print(f"output_rows={summary.output_rows}")
    print(f"duplicate_rows={summary.duplicate_rows}")
    print(f"missing_shots={summary.missing_shots}")
    print(f"unresolved_teams={summary.unresolved_teams}")
    print(f"output={summary.output_path.relative_to(PROJECT_ROOT).as_posix()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
