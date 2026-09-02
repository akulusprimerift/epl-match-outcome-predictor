"""Build the validated canonical EPL match table from immutable raw CSVs."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import date, datetime
import json
import os
from pathlib import Path
import re
import sys
import tempfile
from typing import Mapping, Sequence

import pandas as pd

from src.collect_possession import (
    API_TEAM_PROVIDER,
    ApiPossessionFixture,
    PossessionCollectionError,
    load_cached_possession_fixtures,
    season_year_to_code,
)
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
POSSESSION_COVERAGE_COLUMNS = (
    "scope",
    "season",
    "team",
    "completed_fixtures",
    "possession_complete_fixtures",
    "coverage",
    "threshold",
    "meets_threshold",
    "model_b_period_start",
)
POSSESSION_JOIN_REPORT_COLUMNS = (
    "api_fixture_id",
    "season",
    "date",
    "home_team",
    "away_team",
    "canonical_match_ids",
    "reason",
)


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
    possession_included: bool
    api_fixtures_loaded: int
    possession_matches_joined: int
    unmatched_api_fixtures: int
    ambiguous_api_fixtures: int
    model_b_period_start: str | None


@dataclass(frozen=True)
class PossessionJoinResult:
    """An enriched canonical frame plus explicit join-audit tables."""

    frame: pd.DataFrame
    matched_count: int
    unmatched: pd.DataFrame
    ambiguous: pd.DataFrame


@dataclass(frozen=True)
class PossessionCoverageResult:
    """Coverage rows and the first season satisfying the configured rule."""

    frame: pd.DataFrame
    model_b_period_start: str | None


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


def parse_optional_percentage_series(
    values: pd.Series, column_name: str
) -> pd.Series:
    """Validate 0--100 percentages without replacing missing values."""
    text = values.astype("string").str.strip()
    provided = values.notna() & text.ne("").fillna(False)
    numeric = pd.to_numeric(text.where(provided), errors="coerce")
    invalid = provided & numeric.isna()
    if invalid.any():
        bad_value = text.loc[invalid].iloc[0]
        raise NumericValidationError(
            f"Invalid numeric value in {column_name}: {bad_value!r}."
        )
    out_of_range = numeric.notna() & (numeric.lt(0.0) | numeric.gt(100.0))
    if out_of_range.any():
        bad_value = numeric.loc[out_of_range].iloc[0]
        raise NumericValidationError(
            f"Percentage in {column_name} is outside 0--100: {bad_value!r}."
        )
    return numeric.astype("Float64")


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
    home_possession = parse_optional_percentage_series(
        frame["home_possession"], "home_possession"
    )
    away_possession = parse_optional_percentage_series(
        frame["away_possession"], "away_possession"
    )
    api_fixture_ids = parse_nonnegative_integer_series(
        frame["api_fixture_id"], "api_fixture_id", allow_missing=True
    )
    if api_fixture_ids.eq(0).fillna(False).any():
        raise CanonicalIntegrityError("Canonical api_fixture_id values must be positive.")
    if api_fixture_ids.dropna().duplicated().any():
        raise CanonicalIntegrityError(
            "Canonical nonmissing api_fixture_id values must be unique."
        )
    possession_without_fixture = (
        home_possession.notna() | away_possession.notna()
    ) & api_fixture_ids.isna()
    if possession_without_fixture.any():
        raise CanonicalIntegrityError(
            "Canonical possession values require API fixture provenance."
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


def join_possession_fixtures(
    canonical: pd.DataFrame,
    fixtures: Sequence[ApiPossessionFixture],
    mappings: Mapping[tuple[str, str], TeamIdentity],
    *,
    project_root: Path = PROJECT_ROOT,
) -> PossessionJoinResult:
    """Join API fixtures to canonical matches with exact, one-to-one keys."""
    enriched = canonical.copy()
    fixture_ids = [fixture.fixture_id for fixture in fixtures]
    if len(fixture_ids) != len(set(fixture_ids)):
        raise CanonicalIntegrityError(
            "Cached API possession fixtures contain duplicate fixture IDs."
        )

    provider_names = {
        fixture.home_team_name for fixture in fixtures
    } | {fixture.away_team_name for fixture in fixtures}
    unknown_names = sorted(
        name
        for name in provider_names
        if (API_TEAM_PROVIDER, name) not in mappings
    )
    if unknown_names:
        raise TeamMappingError(
            "Unmapped api_football team names: "
            f"{', '.join(unknown_names)}. Add them to config/team_name_map.csv."
        )

    canonical_by_key: dict[tuple[str, str, str, str], list[int]] = {}
    for index, row in enriched.iterrows():
        key = (
            str(row["season"]),
            str(row["date"]),
            str(row["home_team_slug"]),
            str(row["away_team_slug"]),
        )
        canonical_by_key.setdefault(key, []).append(int(index))

    normalized: list[
        tuple[
            ApiPossessionFixture,
            str,
            TeamIdentity,
            TeamIdentity,
            tuple[str, str, str, str],
        ]
    ] = []
    for fixture in fixtures:
        season_code = season_year_to_code(
            fixture.season_year, project_root=project_root
        )
        home_identity = resolve_team_name(
            API_TEAM_PROVIDER, fixture.home_team_name, mappings
        )
        away_identity = resolve_team_name(
            API_TEAM_PROVIDER, fixture.away_team_name, mappings
        )
        key = (
            season_code,
            fixture.date,
            home_identity.slug,
            away_identity.slug,
        )
        normalized.append(
            (fixture, season_code, home_identity, away_identity, key)
        )

    api_by_key: dict[
        tuple[str, str, str, str],
        list[
            tuple[
                ApiPossessionFixture,
                str,
                TeamIdentity,
                TeamIdentity,
                tuple[str, str, str, str],
            ]
        ],
    ] = {}
    for item in normalized:
        api_by_key.setdefault(item[4], []).append(item)

    unmatched_rows: list[dict[str, object]] = []
    ambiguous_rows: list[dict[str, object]] = []
    claimed_canonical_indices: set[int] = set()
    matched_count = 0

    def audit_row(
        item: tuple[
            ApiPossessionFixture,
            str,
            TeamIdentity,
            TeamIdentity,
            tuple[str, str, str, str],
        ],
        *,
        candidate_indices: Sequence[int],
        reason: str,
    ) -> dict[str, object]:
        fixture, season_code, home_identity, away_identity, _ = item
        match_ids = ";".join(
            sorted(
                str(enriched.loc[index, "match_id"])
                for index in candidate_indices
            )
        )
        return {
            "api_fixture_id": fixture.fixture_id,
            "season": season_code,
            "date": fixture.date,
            "home_team": home_identity.name,
            "away_team": away_identity.name,
            "canonical_match_ids": match_ids,
            "reason": reason,
        }

    for key in sorted(api_by_key):
        items = sorted(api_by_key[key], key=lambda item: item[0].fixture_id)
        candidate_indices = canonical_by_key.get(key, [])
        if len(items) > 1:
            for item in items:
                ambiguous_rows.append(
                    audit_row(
                        item,
                        candidate_indices=candidate_indices,
                        reason="duplicate_api_join_key",
                    )
                )
            continue

        item = items[0]
        fixture = item[0]
        if not candidate_indices:
            unmatched_rows.append(
                audit_row(
                    item,
                    candidate_indices=(),
                    reason="no_canonical_match",
                )
            )
            continue
        if len(candidate_indices) > 1:
            ambiguous_rows.append(
                audit_row(
                    item,
                    candidate_indices=candidate_indices,
                    reason="multiple_canonical_matches",
                )
            )
            continue

        canonical_index = candidate_indices[0]
        if canonical_index in claimed_canonical_indices:
            ambiguous_rows.append(
                audit_row(
                    item,
                    candidate_indices=candidate_indices,
                    reason="canonical_match_already_claimed",
                )
            )
            continue
        enriched.at[canonical_index, "home_possession"] = (
            pd.NA
            if fixture.home_possession is None
            else fixture.home_possession
        )
        enriched.at[canonical_index, "away_possession"] = (
            pd.NA
            if fixture.away_possession is None
            else fixture.away_possession
        )
        enriched.at[canonical_index, "api_fixture_id"] = fixture.fixture_id
        claimed_canonical_indices.add(canonical_index)
        matched_count += 1

    enriched["home_possession"] = pd.to_numeric(
        enriched["home_possession"], errors="raise"
    ).astype("Float64")
    enriched["away_possession"] = pd.to_numeric(
        enriched["away_possession"], errors="raise"
    ).astype("Float64")
    enriched["api_fixture_id"] = pd.to_numeric(
        enriched["api_fixture_id"], errors="raise"
    ).astype("Int64")
    unmatched = pd.DataFrame(
        unmatched_rows, columns=POSSESSION_JOIN_REPORT_COLUMNS
    )
    ambiguous = pd.DataFrame(
        ambiguous_rows, columns=POSSESSION_JOIN_REPORT_COLUMNS
    )
    return PossessionJoinResult(
        frame=enriched,
        matched_count=matched_count,
        unmatched=unmatched,
        ambiguous=ambiguous,
    )


def build_possession_coverage(
    canonical: pd.DataFrame, threshold: float
) -> PossessionCoverageResult:
    """Report complete-pair possession coverage by season and by team."""
    if not 0.0 <= threshold <= 1.0:
        raise CleaningError(
            f"Possession coverage threshold must be between 0 and 1: {threshold!r}"
        )
    complete = canonical["home_possession"].notna() & canonical[
        "away_possession"
    ].notna()
    season_order = list(
        dict.fromkeys(canonical["season"].astype("string").tolist())
    )
    season_rows: list[dict[str, object]] = []
    team_rows: list[dict[str, object]] = []
    eligible_seasons: list[str] = []

    def coverage_row(
        *,
        scope: str,
        season: str,
        team: str,
        mask: pd.Series,
    ) -> dict[str, object]:
        total = int(mask.sum())
        complete_count = int((mask & complete).sum())
        coverage = complete_count / total if total else 0.0
        meets_threshold = coverage >= threshold
        return {
            "scope": scope,
            "season": season,
            "team": team,
            "completed_fixtures": total,
            "possession_complete_fixtures": complete_count,
            "coverage": round(coverage, 6),
            "threshold": threshold,
            "meets_threshold": meets_threshold,
            "model_b_period_start": "",
        }

    seasons = canonical["season"].astype("string")
    for season in season_order:
        season_mask = seasons.eq(season).fillna(False)
        row = coverage_row(
            scope="season", season=season, team="", mask=season_mask
        )
        season_rows.append(row)
        if bool(row["meets_threshold"]):
            eligible_seasons.append(season)

        season_frame = canonical.loc[season_mask]
        teams = sorted(
            set(season_frame["home_team"].astype(str))
            | set(season_frame["away_team"].astype(str))
        )
        for team in teams:
            team_mask = season_mask & (
                canonical["home_team"].eq(team)
                | canonical["away_team"].eq(team)
            )
            team_rows.append(
                coverage_row(
                    scope="team", season=season, team=team, mask=team_mask
                )
            )

    period_start = eligible_seasons[0] if eligible_seasons else None
    for row in [*season_rows, *team_rows]:
        row["model_b_period_start"] = period_start or ""
    coverage_frame = pd.DataFrame(
        [*season_rows, *team_rows], columns=POSSESSION_COVERAGE_COLUMNS
    )
    return PossessionCoverageResult(
        frame=coverage_frame,
        model_b_period_start=period_start,
    )


def load_possession_coverage_threshold(project_root: Path = PROJECT_ROOT) -> float:
    """Load the fixed Phase 6 coverage threshold from model configuration."""
    path = project_root / "config" / "model_config.json"
    try:
        configured = json.loads(path.read_text(encoding="utf-8"))
        threshold = float(configured["possession_coverage_threshold"])
    except FileNotFoundError as exc:
        raise CleaningError(f"Model configuration not found: {path}") from exc
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
        raise CleaningError(
            f"Could not load possession coverage threshold from {path}: {exc}"
        ) from exc
    if not 0.0 <= threshold <= 1.0:
        raise CleaningError(
            f"Possession coverage threshold must be between 0 and 1: {threshold!r}"
        )
    return threshold


def write_report_atomic(frame: pd.DataFrame, output_path: Path) -> None:
    """Write one report CSV through a temporary file and atomic rename."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    file_descriptor, temporary_name = tempfile.mkstemp(
        dir=output_path.parent,
        prefix=f".{output_path.name}.",
        suffix=".tmp",
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(
            file_descriptor, "w", encoding="utf-8", newline=""
        ) as output:
            frame.to_csv(output, index=False, lineterminator="\n")
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary_path, output_path)
    except (OSError, TypeError, ValueError) as exc:
        raise CleaningError(
            f"Could not atomically write report {output_path}: {exc}"
        ) from exc
    finally:
        temporary_path.unlink(missing_ok=True)


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


def build_canonical_matches(
    project_root: Path = PROJECT_ROOT, *, include_possession: bool = False
) -> CleaningSummary:
    """Build, optionally enrich, validate, and save all configured EPL seasons."""
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

    api_fixtures_loaded = 0
    possession_matches_joined = 0
    unmatched_api_fixtures = 0
    ambiguous_api_fixtures = 0
    model_b_period_start: str | None = None
    if include_possession:
        api_fixtures = load_cached_possession_fixtures(project_root)
        api_fixtures_loaded = len(api_fixtures)
        join_result = join_possession_fixtures(
            canonical,
            api_fixtures,
            mappings,
            project_root=project_root,
        )
        canonical = join_result.frame
        possession_matches_joined = join_result.matched_count
        unmatched_api_fixtures = len(join_result.unmatched)
        ambiguous_api_fixtures = len(join_result.ambiguous)
        coverage = build_possession_coverage(
            canonical, load_possession_coverage_threshold(project_root)
        )
        model_b_period_start = coverage.model_b_period_start
        validate_canonical_table(canonical)

        reports_directory = project_root / "reports"
        write_report_atomic(
            join_result.unmatched,
            reports_directory / "possession_unmatched.csv",
        )
        write_report_atomic(
            join_result.ambiguous,
            reports_directory / "possession_ambiguous.csv",
        )
        write_report_atomic(
            coverage.frame,
            reports_directory / "possession_coverage.csv",
        )

    output_path = project_root / "data" / "processed" / "canonical_matches.csv"
    write_canonical_matches_atomic(canonical, output_path)
    return CleaningSummary(
        input_rows=input_rows,
        output_rows=len(canonical),
        duplicate_rows=duplicate_rows,
        missing_shots=missing_shots,
        unresolved_teams=0,
        output_path=output_path,
        possession_included=include_possession,
        api_fixtures_loaded=api_fixtures_loaded,
        possession_matches_joined=possession_matches_joined,
        unmatched_api_fixtures=unmatched_api_fixtures,
        ambiguous_api_fixtures=ambiguous_api_fixtures,
        model_b_period_start=model_b_period_start,
    )


def build_parser() -> argparse.ArgumentParser:
    """Build the canonical-data command-line parser."""
    parser = argparse.ArgumentParser(
        description="Build the validated canonical EPL match table."
    )
    parser.add_argument(
        "--include-possession",
        action="store_true",
        help="join manifested API-Football possession caches and write coverage reports",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the canonical data-cleaning CLI."""
    parser = build_parser()
    arguments = parser.parse_args(argv)
    try:
        summary = build_canonical_matches(
            include_possession=arguments.include_possession
        )
    except (CleaningError, IngestionError, PossessionCollectionError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    print(f"input_rows={summary.input_rows}")
    print(f"output_rows={summary.output_rows}")
    print(f"duplicate_rows={summary.duplicate_rows}")
    print(f"missing_shots={summary.missing_shots}")
    print(f"unresolved_teams={summary.unresolved_teams}")
    if summary.possession_included:
        print(f"api_fixtures_loaded={summary.api_fixtures_loaded}")
        print(
            f"possession_matches_joined={summary.possession_matches_joined}"
        )
        print(
            f"unmatched_api_fixtures={summary.unmatched_api_fixtures}"
        )
        print(
            f"ambiguous_api_fixtures={summary.ambiguous_api_fixtures}"
        )
        print(
            "model_b_period_start="
            f"{summary.model_b_period_start or 'none'}"
        )
        print("coverage_report=reports/possession_coverage.csv")
    print(f"output={summary.output_path.relative_to(PROJECT_ROOT).as_posix()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
