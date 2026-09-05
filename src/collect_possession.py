"""Collect leakage-safe EPL team-season possession averages from SofaScore."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from datetime import datetime, timezone
from http.client import HTTPException
import json
import math
import os
from pathlib import Path
import re
import sys
import tempfile
import time
from typing import Any, Callable, Mapping, Sequence
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from src.download_data import (
    PROJECT_ROOT,
    ManifestError,
    load_manifest,
    load_seasons,
    sha256_file,
    write_manifest_atomic,
)


TOURNAMENT_ID = 17
SOFASCORE_SOURCE = "sofascore"
SOFASCORE_TEAM_PROVIDER = "sofascore"
SOFASCORE_BASE_URLS = (
    "https://www.sofascore.com/api/v1",
    "https://api.sofascore.com/api/v1",
)
REQUEST_TIMEOUT_SECONDS = 30.0
DEFAULT_REQUEST_DELAY_SECONDS = 0.5
DEFAULT_ATTEMPTS = 5
DEFAULT_FIRST_SEASON = 2017
DEFAULT_LAST_SEASON = 2025
USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 Chrome/128.0 Safari/537.36"
)

SEASONS_ENDPOINT = "seasons"
STANDINGS_ENDPOINT = "standings/total"
TEAM_STATISTICS_ENDPOINT = "statistics/overall"
AGGREGATE_EXPORT_ENDPOINT = "team-season-average-possession-export"
AGGREGATE_EXPORT_FILENAME = (
    "sofascore_epl_average_possession_2017-18_to_2025-26.csv"
)
AGGREGATE_EXPORT_COLUMNS = (
    "season",
    "season_id",
    "team",
    "team_id",
    "average_ball_possession",
    "matches",
    "source_url",
)

TEAM_SEASON_POSSESSION_COLUMNS = (
    "source_season",
    "target_season",
    "source_season_start_year",
    "sofascore_season_id",
    "team",
    "team_slug",
    "sofascore_team_name",
    "sofascore_team_id",
    "average_possession_pct",
    "matches_recorded",
    "source_url",
)
POSSESSION_COVERAGE_COLUMNS = (
    "scope",
    "source_season",
    "target_season",
    "team",
    "expected_teams",
    "available_team_averages",
    "coverage",
    "threshold",
    "meets_threshold",
    "model_b_period_start",
)


class PossessionCollectionError(RuntimeError):
    """Base class for actionable SofaScore collection failures."""


class SofaScoreConfigurationError(PossessionCollectionError):
    """Raised when collector configuration is invalid."""


class SofaScoreRequestError(PossessionCollectionError):
    """Raised when a SofaScore request cannot be completed."""


class SofaScoreResponseError(PossessionCollectionError):
    """Raised when a SofaScore response violates the expected contract."""


class SofaScoreCacheError(PossessionCollectionError):
    """Raised when an immutable SofaScore cache fails provenance validation."""


class TeamMappingError(PossessionCollectionError):
    """Raised when a SofaScore team name lacks an explicit canonical mapping."""


class RequestBudgetReached(PossessionCollectionError):
    """Raised internally before a request would exceed the run budget."""


@dataclass(frozen=True)
class SofaScoreSeason:
    """One validated EPL season from the SofaScore season directory."""

    start_year: int
    season_code: str
    season_id: int
    name: str


@dataclass(frozen=True)
class SofaScoreTeam:
    """One team listed in an EPL season table."""

    team_id: int
    name: str


@dataclass(frozen=True)
class TeamSeasonPossession:
    """One possibly-missing team-season average with source provenance."""

    source_season: str
    target_season: str
    source_season_start_year: int
    sofascore_season_id: int
    team: str
    team_slug: str
    sofascore_team_name: str
    sofascore_team_id: int
    average_possession_pct: float | None
    matches_recorded: int | None
    source_url: str


@dataclass(frozen=True)
class CollectionSummary:
    """Auditable counts from one bounded, resumable collection run."""

    requested_seasons: tuple[int, ...]
    seasons_found: int
    teams_found: int
    statistics_cached: int
    statistics_downloaded: int
    failed_statistics: int
    requests_made: int
    stopped_before_budget: bool
    output_rows: int
    complete_rows: int
    model_b_period_start: str | None
    output_path: Path
    coverage_path: Path


@dataclass
class _RequestBudget:
    maximum: int
    used: int = 0

    def take(self) -> None:
        if self.used >= self.maximum:
            raise RequestBudgetReached(
                "Request budget reached; rerun the same command to resume from cache."
            )
        self.used += 1


def seasons_path() -> str:
    return f"/unique-tournament/{TOURNAMENT_ID}/seasons"


def standings_path(season_id: int) -> str:
    return (
        f"/unique-tournament/{TOURNAMENT_ID}/season/{season_id}/"
        "standings/total"
    )


def team_statistics_path(team_id: int, season_id: int) -> str:
    return (
        f"/team/{team_id}/unique-tournament/{TOURNAMENT_ID}/"
        f"season/{season_id}/statistics/overall"
    )


def primary_url(path: str) -> str:
    return f"{SOFASCORE_BASE_URLS[0]}{path}"


def season_code_from_start_year(start_year: int) -> str:
    if isinstance(start_year, bool) or start_year < 1992 or start_year > 2098:
        raise SofaScoreConfigurationError(
            f"Invalid EPL season start year {start_year!r}."
        )
    return f"{start_year % 100:02d}{(start_year + 1) % 100:02d}"


def parse_season_start_year(season: Mapping[str, object]) -> int | None:
    """Parse values such as 2017/2018, 17/18, or 2017-18."""
    text = f"{season.get('name', '')} {season.get('year', '')}"
    four_digit = re.search(r"\b(20\d{2})\s*[/\-]", text)
    if four_digit:
        return int(four_digit.group(1))
    two_digit = re.search(r"\b(\d{2})\s*[/\-]\s*\d{2}\b", text)
    if two_digit:
        year = int(two_digit.group(1))
        return 2000 + year if year < 90 else 1900 + year
    return None


def parse_average_possession(value: object) -> float | None:
    """Parse a numeric percentage while preserving missing values."""
    if value is None:
        return None
    if isinstance(value, bool):
        raise SofaScoreResponseError(f"Invalid average possession {value!r}.")
    candidate = value
    if isinstance(value, str):
        candidate = value.strip().removesuffix("%").strip()
        if not candidate:
            return None
    try:
        parsed = float(candidate)
    except (TypeError, ValueError) as exc:
        raise SofaScoreResponseError(
            f"Invalid average possession {value!r}."
        ) from exc
    if not math.isfinite(parsed) or parsed < 0.0 or parsed > 100.0:
        raise SofaScoreResponseError(
            f"Average possession {value!r} must be between 0 and 100."
        )
    return parsed


def _positive_integer(value: object, *, field_name: str) -> int:
    if isinstance(value, bool):
        raise SofaScoreResponseError(f"Invalid {field_name} {value!r}.")
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise SofaScoreResponseError(
            f"Invalid {field_name} {value!r}."
        ) from exc
    if parsed <= 0 or str(parsed) != str(value).strip():
        raise SofaScoreResponseError(f"Invalid {field_name} {value!r}.")
    return parsed


def _optional_nonnegative_integer(value: object, *, field_name: str) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool):
        raise SofaScoreResponseError(f"Invalid {field_name} {value!r}.")
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise SofaScoreResponseError(
            f"Invalid {field_name} {value!r}."
        ) from exc
    if parsed < 0 or str(parsed) != str(value).strip():
        raise SofaScoreResponseError(f"Invalid {field_name} {value!r}.")
    return parsed


def _payload_object(payload: object, *, endpoint: str) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise SofaScoreResponseError(
            f"SofaScore {endpoint} response must be a JSON object."
        )
    return payload


def parse_seasons(payload: object) -> list[SofaScoreSeason]:
    data = _payload_object(payload, endpoint=SEASONS_ENDPOINT)
    items = data.get("seasons")
    if not isinstance(items, list):
        raise SofaScoreResponseError(
            "SofaScore seasons response requires a seasons list."
        )
    seasons: list[SofaScoreSeason] = []
    seen_years: set[int] = set()
    seen_ids: set[int] = set()
    for position, item in enumerate(items):
        if not isinstance(item, dict):
            raise SofaScoreResponseError(
                f"SofaScore season item {position} must be an object."
            )
        start_year = parse_season_start_year(item)
        if start_year is None:
            continue
        season_id = _positive_integer(item.get("id"), field_name="season id")
        name = item.get("name")
        if not isinstance(name, str) or not name.strip():
            raise SofaScoreResponseError(
                f"SofaScore season {season_id} has an invalid name."
            )
        if start_year in seen_years or season_id in seen_ids:
            raise SofaScoreResponseError(
                f"SofaScore seasons response duplicates {start_year} or {season_id}."
            )
        seasons.append(
            SofaScoreSeason(
                start_year=start_year,
                season_code=season_code_from_start_year(start_year),
                season_id=season_id,
                name=name.strip(),
            )
        )
        seen_years.add(start_year)
        seen_ids.add(season_id)
    return sorted(seasons, key=lambda item: item.start_year)


def parse_standing_teams(payload: object) -> list[SofaScoreTeam]:
    data = _payload_object(payload, endpoint=STANDINGS_ENDPOINT)
    standings = data.get("standings")
    if not isinstance(standings, list):
        raise SofaScoreResponseError(
            "SofaScore standings response requires a standings list."
        )
    teams: dict[int, str] = {}
    for standing in standings:
        if not isinstance(standing, dict):
            continue
        rows = standing.get("rows")
        if not isinstance(rows, list):
            continue
        for row in rows:
            team = row.get("team") if isinstance(row, dict) else None
            if not isinstance(team, dict):
                continue
            team_id = _positive_integer(team.get("id"), field_name="team id")
            name = team.get("name")
            if not isinstance(name, str) or not name.strip():
                raise SofaScoreResponseError(
                    f"SofaScore team {team_id} has an invalid name."
                )
            normalized = name.strip()
            if team_id in teams and teams[team_id] != normalized:
                raise SofaScoreResponseError(
                    f"SofaScore team id {team_id} has conflicting names."
                )
            teams[team_id] = normalized
    if not teams:
        raise SofaScoreResponseError("SofaScore standings contain no teams.")
    return [SofaScoreTeam(team_id, teams[team_id]) for team_id in sorted(teams)]


def parse_team_statistics(payload: object) -> tuple[float | None, int | None]:
    data = _payload_object(payload, endpoint=TEAM_STATISTICS_ENDPOINT)
    statistics = data.get("statistics")
    if not isinstance(statistics, dict):
        raise SofaScoreResponseError(
            "SofaScore team statistics response requires a statistics object."
        )
    return (
        parse_average_possession(statistics.get("averageBallPossession")),
        _optional_nonnegative_integer(
            statistics.get("matches"), field_name="matches"
        ),
    )


def _format_utc(moment: datetime) -> str:
    if moment.tzinfo is None:
        raise SofaScoreConfigurationError(
            "The collector clock must return a timezone-aware value."
        )
    return (
        moment.astimezone(timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )


def _relative_path(project_root: Path, destination: Path) -> str:
    try:
        return destination.resolve().relative_to(project_root.resolve()).as_posix()
    except ValueError as exc:
        raise SofaScoreCacheError(
            f"SofaScore cache path escapes the project root: {destination}"
        ) from exc


def _manifest_record_for_path(
    records: Sequence[Mapping[str, Any]], local_path: str
) -> Mapping[str, Any] | None:
    matches = [record for record in records if record.get("local_path") == local_path]
    if len(matches) > 1:
        raise SofaScoreCacheError(
            f"Manifest contains duplicate records for {local_path}."
        )
    return matches[0] if matches else None


def _load_cached_payload(
    destination: Path,
    *,
    project_root: Path,
    endpoint: str,
    season_code: str,
    season_id: int | None,
    team_id: int | None,
) -> tuple[dict[str, Any], Mapping[str, Any]] | None:
    records = load_manifest(project_root / "data" / "raw" / "manifest.json")
    local_path = _relative_path(project_root, destination)
    record = _manifest_record_for_path(records, local_path)
    if not destination.exists():
        if record is not None:
            raise SofaScoreCacheError(
                f"Manifested SofaScore cache is missing: {destination}"
            )
        return None
    if record is None:
        raise SofaScoreCacheError(
            f"SofaScore cache exists without manifest provenance: {destination}"
        )
    expected = {
        "source": SOFASCORE_SOURCE,
        "season": season_code,
        "endpoint": endpoint,
        "sofascore_season_id": season_id,
        "sofascore_team_id": team_id,
    }
    mismatches = [key for key, value in expected.items() if record.get(key) != value]
    if mismatches:
        raise SofaScoreCacheError(
            f"SofaScore manifest mismatch for {local_path}: {', '.join(mismatches)}."
        )
    source_url = record.get("source_url")
    if not isinstance(source_url, str) or not source_url.startswith(SOFASCORE_BASE_URLS):
        raise SofaScoreCacheError(
            f"SofaScore manifest has an invalid source URL for {local_path}."
        )
    if sha256_file(destination) != record["sha256"]:
        raise SofaScoreCacheError(
            f"Checksum mismatch for immutable SofaScore cache {destination}."
        )
    try:
        payload = json.loads(destination.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise SofaScoreCacheError(
            f"Could not read cached SofaScore response {destination}: {exc}"
        ) from exc
    return _payload_object(payload, endpoint=endpoint), record


def _fetch_json(
    path: str,
    *,
    budget: _RequestBudget,
    opener: Callable[..., Any],
    timeout: float,
    attempts: int,
    sleep_fn: Callable[[float], None],
) -> tuple[bytes, dict[str, Any], str]:
    last_error: BaseException | None = None
    for base_url in SOFASCORE_BASE_URLS:
        source_url = f"{base_url}{path}"
        for attempt in range(attempts):
            budget.take()
            request = Request(
                source_url,
                headers={"Accept": "application/json", "User-Agent": USER_AGENT},
            )
            try:
                with opener(request, timeout=timeout) as response:
                    status = getattr(response, "status", None)
                    if status is not None and not 200 <= int(status) < 300:
                        raise SofaScoreRequestError(
                            f"HTTP {status} from SofaScore for {source_url}."
                        )
                    raw_bytes = response.read()
                payload = json.loads(raw_bytes.decode("utf-8"))
                return raw_bytes, _payload_object(payload, endpoint=path), source_url
            except HTTPError as exc:
                last_error = exc
                if exc.code in (403, 404):
                    break
                if exc.code == 429:
                    retry_after = exc.headers.get("Retry-After") if exc.headers else None
                    try:
                        wait = float(retry_after) if retry_after is not None else 2**attempt
                    except ValueError:
                        wait = float(2**attempt)
                    sleep_fn(max(wait, 2.0))
                    continue
                sleep_fn(float(2**attempt))
            except (
                URLError,
                TimeoutError,
                HTTPException,
                OSError,
                UnicodeDecodeError,
                json.JSONDecodeError,
                SofaScoreRequestError,
                SofaScoreResponseError,
            ) as exc:
                last_error = exc
                sleep_fn(float(2**attempt))
        # A 403/404 or exhausted retries moves to the documented fallback host.
    raise SofaScoreRequestError(f"Could not retrieve {path}: {last_error}")


def _write_cache(
    raw_bytes: bytes,
    destination: Path,
    *,
    project_root: Path,
    endpoint: str,
    season_code: str,
    season_id: int | None,
    team_id: int | None,
    team_name: str | None,
    source_url: str,
    row_count: int,
    clock: Callable[[], datetime],
) -> None:
    if destination.exists():
        raise SofaScoreCacheError(
            f"Refusing to overwrite immutable SofaScore cache {destination}."
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=destination.parent, prefix=f".{destination.name}.", suffix=".tmp"
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as output:
            output.write(raw_bytes)
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary_path, destination)
    except OSError as exc:
        raise SofaScoreCacheError(
            f"Could not atomically cache SofaScore response {destination}: {exc}"
        ) from exc
    finally:
        temporary_path.unlink(missing_ok=True)

    try:
        manifest_path = project_root / "data" / "raw" / "manifest.json"
        records = load_manifest(manifest_path)
        local_path = _relative_path(project_root, destination)
        if _manifest_record_for_path(records, local_path) is not None:
            raise SofaScoreCacheError(
                f"Manifest record appeared during collection for {local_path}."
            )
        record = {
            "source": SOFASCORE_SOURCE,
            "season": season_code,
            "endpoint": endpoint,
            "sofascore_season_id": season_id,
            "sofascore_team_id": team_id,
            "sofascore_team_name": team_name,
            "source_url": source_url,
            "local_path": local_path,
            "retrieved_at_utc": _format_utc(clock()),
            "sha256": sha256_file(destination),
            "row_count": row_count,
        }
        write_manifest_atomic([*records, record], manifest_path)
    except Exception:
        destination.unlink(missing_ok=True)
        raise


def _load_or_fetch(
    path: str,
    destination: Path,
    *,
    project_root: Path,
    endpoint: str,
    season_code: str,
    season_id: int | None,
    team_id: int | None,
    team_name: str | None,
    parser: Callable[[object], Sequence[object] | tuple[object, object]],
    budget: _RequestBudget,
    opener: Callable[..., Any],
    timeout: float,
    attempts: int,
    sleep_fn: Callable[[float], None],
    request_delay: float,
) -> tuple[dict[str, Any], bool]:
    cached = _load_cached_payload(
        destination,
        project_root=project_root,
        endpoint=endpoint,
        season_code=season_code,
        season_id=season_id,
        team_id=team_id,
    )
    if cached is not None:
        payload, record = cached
        parsed = parser(payload)
        parsed_row_count = 1 if endpoint == TEAM_STATISTICS_ENDPOINT else len(parsed)
        if parsed_row_count != int(record["row_count"]):
            raise SofaScoreCacheError(
                f"Row-count mismatch for SofaScore cache {destination}."
            )
        return payload, True

    raw_bytes, payload, source_url = _fetch_json(
        path,
        budget=budget,
        opener=opener,
        timeout=timeout,
        attempts=attempts,
        sleep_fn=sleep_fn,
    )
    parsed = parser(payload)
    parsed_row_count = 1 if endpoint == TEAM_STATISTICS_ENDPOINT else len(parsed)
    _write_cache(
        raw_bytes,
        destination,
        project_root=project_root,
        endpoint=endpoint,
        season_code=season_code,
        season_id=season_id,
        team_id=team_id,
        team_name=team_name,
        source_url=source_url,
        row_count=parsed_row_count,
        clock=lambda: datetime.now(timezone.utc),
    )
    if request_delay:
        sleep_fn(request_delay)
    return payload, False


def _team_map(project_root: Path) -> dict[str, tuple[str, str]]:
    path = project_root / "config" / "team_name_map.csv"
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as source:
            rows = list(csv.DictReader(source))
    except (OSError, UnicodeDecodeError, csv.Error) as exc:
        raise TeamMappingError(f"Could not read team mapping {path}: {exc}") from exc
    mappings: dict[str, tuple[str, str]] = {}
    for row in rows:
        if row.get("provider") != SOFASCORE_TEAM_PROVIDER:
            continue
        provider_name = str(row.get("provider_team_name", "")).strip()
        canonical_name = str(row.get("canonical_team_name", "")).strip()
        slug = str(row.get("canonical_team_slug", "")).strip()
        if not provider_name or not canonical_name or not slug:
            raise TeamMappingError("SofaScore team mapping contains an empty field.")
        identity = (canonical_name, slug)
        if provider_name in mappings and mappings[provider_name] != identity:
            raise TeamMappingError(
                f"Conflicting SofaScore mapping for {provider_name!r}."
            )
        mappings[provider_name] = identity
    if not mappings:
        raise TeamMappingError("No SofaScore team mappings are configured.")
    return mappings


def _write_csv_atomic(
    rows: Sequence[Mapping[str, object]], columns: Sequence[str], destination: Path
) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=destination.parent, prefix=f".{destination.name}.", suffix=".tmp"
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as output:
            writer = csv.DictWriter(output, fieldnames=list(columns), lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary_path, destination)
    except (OSError, csv.Error, ValueError) as exc:
        raise PossessionCollectionError(
            f"Could not atomically write {destination}: {exc}"
        ) from exc
    finally:
        temporary_path.unlink(missing_ok=True)


def _coverage_threshold(project_root: Path) -> float:
    path = project_root / "config" / "model_config.json"
    try:
        configured = json.loads(path.read_text(encoding="utf-8"))
        threshold = float(configured["possession_coverage_threshold"])
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, KeyError, ValueError) as exc:
        raise SofaScoreConfigurationError(
            f"Could not load possession coverage threshold from {path}: {exc}"
        ) from exc
    if not 0.0 <= threshold <= 1.0:
        raise SofaScoreConfigurationError(
            "possession_coverage_threshold must be between zero and one."
        )
    return threshold


def _aggregate_export_path(project_root: Path) -> Path:
    return (
        project_root
        / "data"
        / "raw"
        / "sofascore"
        / AGGREGATE_EXPORT_FILENAME
    )


def _load_aggregate_export(
    project_root: Path,
) -> list[TeamSeasonPossession] | None:
    """Load a manifested SofaScore web export when direct API access is blocked."""
    path = _aggregate_export_path(project_root)
    if not path.exists():
        return None

    records = load_manifest(project_root / "data" / "raw" / "manifest.json")
    local_path = _relative_path(project_root, path)
    record = _manifest_record_for_path(records, local_path)
    if record is None:
        raise SofaScoreCacheError(
            f"SofaScore aggregate export exists without manifest provenance: {path}"
        )
    if (
        record.get("source") != SOFASCORE_SOURCE
        or record.get("endpoint") != AGGREGATE_EXPORT_ENDPOINT
    ):
        raise SofaScoreCacheError(
            f"SofaScore aggregate export manifest mismatch for {local_path}."
        )
    if sha256_file(path) != record["sha256"]:
        raise SofaScoreCacheError(
            f"Checksum mismatch for immutable SofaScore aggregate export {path}."
        )

    mappings = _team_map(project_root)
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as source:
            reader = csv.DictReader(source)
            if tuple(reader.fieldnames or ()) != AGGREGATE_EXPORT_COLUMNS:
                raise SofaScoreCacheError(
                    f"Unexpected columns in SofaScore aggregate export {path}."
                )
            raw_rows = list(reader)
    except (OSError, UnicodeDecodeError, csv.Error) as exc:
        raise SofaScoreCacheError(
            f"Could not read SofaScore aggregate export {path}: {exc}"
        ) from exc

    if len(raw_rows) != int(record["row_count"]):
        raise SofaScoreCacheError(
            f"Row-count mismatch for SofaScore aggregate export {path}."
        )

    output_rows: list[TeamSeasonPossession] = []
    seen_keys: set[tuple[str, int]] = set()
    seen_slugs: dict[str, set[str]] = {}
    season_ids: dict[str, int] = {}
    for line_number, raw in enumerate(raw_rows, start=2):
        label = str(raw.get("season", "")).strip()
        match = re.fullmatch(r"(20\d{2})/(\d{2})", label)
        if match is None or int(match.group(2)) != (int(match.group(1)) + 1) % 100:
            raise SofaScoreCacheError(
                f"Invalid season {label!r} in {path} on line {line_number}."
            )
        start_year = int(match.group(1))
        source_season = season_code_from_start_year(start_year)
        target_season = season_code_from_start_year(start_year + 1)
        season_id = _positive_integer(
            raw.get("season_id"), field_name="season id"
        )
        team_id = _positive_integer(raw.get("team_id"), field_name="team id")
        team_name = str(raw.get("team", "")).strip()
        identity = mappings.get(team_name)
        if identity is None:
            raise TeamMappingError(
                f"Unmapped SofaScore team name {team_name!r}; add it to "
                "config/team_name_map.csv."
            )
        possession = parse_average_possession(raw.get("average_ball_possession"))
        matches = _optional_nonnegative_integer(
            raw.get("matches"), field_name="matches"
        )
        if possession is None or matches is None:
            raise SofaScoreCacheError(
                f"Missing possession or match count in {path} on line {line_number}."
            )
        source_url = str(raw.get("source_url", "")).strip()
        expected_url = primary_url(team_statistics_path(team_id, season_id))
        if source_url != expected_url:
            raise SofaScoreCacheError(
                f"Unexpected SofaScore URL in {path} on line {line_number}."
            )
        if source_season in season_ids and season_ids[source_season] != season_id:
            raise SofaScoreCacheError(
                f"Conflicting season ids for {source_season} in {path}."
            )
        season_ids[source_season] = season_id
        key = (source_season, team_id)
        if key in seen_keys:
            raise SofaScoreCacheError(
                f"Duplicate team-season {key!r} in {path}."
            )
        seen_keys.add(key)
        canonical_name, slug = identity
        season_slugs = seen_slugs.setdefault(source_season, set())
        if slug in season_slugs:
            raise TeamMappingError(
                f"Multiple SofaScore teams map to {slug!r} in season "
                f"{source_season}."
            )
        season_slugs.add(slug)
        output_rows.append(
            TeamSeasonPossession(
                source_season=source_season,
                target_season=target_season,
                source_season_start_year=start_year,
                sofascore_season_id=season_id,
                team=canonical_name,
                team_slug=slug,
                sofascore_team_name=team_name,
                sofascore_team_id=team_id,
                average_possession_pct=possession,
                matches_recorded=matches,
                source_url=source_url,
            )
        )
    return sorted(output_rows, key=lambda row: (row.source_season, row.team_slug))


def _write_processed_outputs(
    output_rows: Sequence[TeamSeasonPossession],
    project_root: Path,
) -> tuple[list[TeamSeasonPossession], str | None, Path, Path]:
    """Write the canonical table and its season/team coverage audit."""
    rows = sorted(output_rows, key=lambda row: (row.source_season, row.team_slug))
    threshold = _coverage_threshold(project_root)
    coverage_rows: list[dict[str, object]] = []
    qualifying_targets: list[str] = []
    for source_season in sorted({row.source_season for row in rows}):
        season_rows = [row for row in rows if row.source_season == source_season]
        target_season = season_rows[0].target_season
        complete_count = sum(
            row.average_possession_pct is not None for row in season_rows
        )
        coverage = complete_count / len(season_rows)
        meets = coverage >= threshold
        if meets:
            qualifying_targets.append(target_season)
        coverage_rows.append(
            {
                "scope": "season",
                "source_season": source_season,
                "target_season": target_season,
                "team": "",
                "expected_teams": len(season_rows),
                "available_team_averages": complete_count,
                "coverage": coverage,
                "threshold": threshold,
                "meets_threshold": meets,
                "model_b_period_start": "",
            }
        )
        for row in season_rows:
            available = int(row.average_possession_pct is not None)
            coverage_rows.append(
                {
                    "scope": "team",
                    "source_season": source_season,
                    "target_season": target_season,
                    "team": row.team,
                    "expected_teams": 1,
                    "available_team_averages": available,
                    "coverage": float(available),
                    "threshold": threshold,
                    "meets_threshold": bool(available),
                    "model_b_period_start": "",
                }
            )

    model_b_period_start = min(qualifying_targets) if qualifying_targets else None
    for row in coverage_rows:
        if row["scope"] == "season":
            row["model_b_period_start"] = model_b_period_start or ""

    serialized: list[dict[str, object]] = []
    for row in rows:
        values = row.__dict__.copy()
        values["average_possession_pct"] = (
            "" if row.average_possession_pct is None else row.average_possession_pct
        )
        values["matches_recorded"] = (
            "" if row.matches_recorded is None else row.matches_recorded
        )
        serialized.append(values)

    output_path = project_root / "data" / "processed" / "team_season_possession.csv"
    coverage_path = project_root / "reports" / "possession_coverage.csv"
    _write_csv_atomic(serialized, TEAM_SEASON_POSSESSION_COLUMNS, output_path)
    _write_csv_atomic(coverage_rows, POSSESSION_COVERAGE_COLUMNS, coverage_path)
    return rows, model_b_period_start, output_path, coverage_path


def build_processed_possession(
    project_root: Path = PROJECT_ROOT,
) -> tuple[list[TeamSeasonPossession], str | None, Path, Path]:
    """Rebuild the team-season table and coverage report only from raw caches."""
    project_root = project_root.resolve()
    aggregate_rows = _load_aggregate_export(project_root)
    if aggregate_rows is not None:
        return _write_processed_outputs(aggregate_rows, project_root)

    manifest_path = project_root / "data" / "raw" / "manifest.json"
    records = load_manifest(manifest_path)
    mappings = _team_map(project_root)
    standings_records = sorted(
        (
            record
            for record in records
            if record.get("source") == SOFASCORE_SOURCE
            and record.get("endpoint") == STANDINGS_ENDPOINT
        ),
        key=lambda record: str(record.get("season")),
    )
    output_rows: list[TeamSeasonPossession] = []
    coverage_rows: list[dict[str, object]] = []
    threshold = _coverage_threshold(project_root)
    qualifying_targets: list[str] = []

    for standing_record in standings_records:
        source_season = str(standing_record["season"])
        season_id = _positive_integer(
            standing_record.get("sofascore_season_id"), field_name="season id"
        )
        if not re.fullmatch(r"\d{4}", source_season):
            raise SofaScoreCacheError(
                f"Invalid canonical season code in {standing_record['local_path']}."
            )
        start_year = 2000 + int(source_season[:2])
        if start_year > 2089:
            start_year -= 100
        target_season = season_code_from_start_year(start_year + 1)
        standing_path = project_root / str(standing_record["local_path"])
        cached = _load_cached_payload(
            standing_path,
            project_root=project_root,
            endpoint=STANDINGS_ENDPOINT,
            season_code=source_season,
            season_id=season_id,
            team_id=None,
        )
        if cached is None:  # pragma: no cover - manifested missing handled above
            raise SofaScoreCacheError(f"Missing standings cache {standing_path}.")
        teams = parse_standing_teams(cached[0])
        complete_count = 0
        seen_team_slugs: set[str] = set()
        for team in sorted(teams, key=lambda item: item.name):
            identity = mappings.get(team.name)
            if identity is None:
                raise TeamMappingError(
                    f"Unmapped SofaScore team name {team.name!r}; add it to "
                    "config/team_name_map.csv."
                )
            matching = [
                record
                for record in records
                if record.get("source") == SOFASCORE_SOURCE
                and record.get("endpoint") == TEAM_STATISTICS_ENDPOINT
                and record.get("sofascore_season_id") == season_id
                and record.get("sofascore_team_id") == team.team_id
            ]
            if len(matching) > 1:
                raise SofaScoreCacheError(
                    f"Duplicate team-statistics caches for season {season_id}, "
                    f"team {team.team_id}."
                )
            possession: float | None = None
            matches: int | None = None
            source_url = primary_url(team_statistics_path(team.team_id, season_id))
            if matching:
                record = matching[0]
                stats_path = project_root / str(record["local_path"])
                stats_cached = _load_cached_payload(
                    stats_path,
                    project_root=project_root,
                    endpoint=TEAM_STATISTICS_ENDPOINT,
                    season_code=source_season,
                    season_id=season_id,
                    team_id=team.team_id,
                )
                if stats_cached is None:  # pragma: no cover
                    raise SofaScoreCacheError(f"Missing statistics cache {stats_path}.")
                possession, matches = parse_team_statistics(stats_cached[0])
                source_url = str(record["source_url"])
            if possession is not None:
                complete_count += 1
            canonical_name, slug = identity
            if slug in seen_team_slugs:
                raise TeamMappingError(
                    f"Multiple SofaScore teams map to {slug!r} in season "
                    f"{source_season}."
                )
            seen_team_slugs.add(slug)
            output_rows.append(
                TeamSeasonPossession(
                    source_season=source_season,
                    target_season=target_season,
                    source_season_start_year=start_year,
                    sofascore_season_id=season_id,
                    team=canonical_name,
                    team_slug=slug,
                    sofascore_team_name=team.name,
                    sofascore_team_id=team.team_id,
                    average_possession_pct=possession,
                    matches_recorded=matches,
                    source_url=source_url,
                )
            )

        coverage = complete_count / len(teams)
        meets = coverage >= threshold
        if meets:
            qualifying_targets.append(target_season)
        coverage_rows.append(
            {
                "scope": "season",
                "source_season": source_season,
                "target_season": target_season,
                "team": "",
                "expected_teams": len(teams),
                "available_team_averages": complete_count,
                "coverage": coverage,
                "threshold": threshold,
                "meets_threshold": meets,
                "model_b_period_start": "",
            }
        )
        for row in output_rows[-len(teams) :]:
            available = int(row.average_possession_pct is not None)
            coverage_rows.append(
                {
                    "scope": "team",
                    "source_season": source_season,
                    "target_season": target_season,
                    "team": row.team,
                    "expected_teams": 1,
                    "available_team_averages": available,
                    "coverage": float(available),
                    "threshold": threshold,
                    "meets_threshold": bool(available),
                    "model_b_period_start": "",
                }
            )

    model_b_period_start = min(qualifying_targets) if qualifying_targets else None
    for row in coverage_rows:
        if row["scope"] == "season":
            row["model_b_period_start"] = model_b_period_start or ""

    output_rows.sort(key=lambda row: (row.source_season, row.team_slug))
    serialized = []
    for row in output_rows:
        values = row.__dict__.copy()
        values["average_possession_pct"] = (
            "" if row.average_possession_pct is None else row.average_possession_pct
        )
        values["matches_recorded"] = (
            "" if row.matches_recorded is None else row.matches_recorded
        )
        serialized.append(values)

    output_path = project_root / "data" / "processed" / "team_season_possession.csv"
    coverage_path = project_root / "reports" / "possession_coverage.csv"
    _write_csv_atomic(serialized, TEAM_SEASON_POSSESSION_COLUMNS, output_path)
    _write_csv_atomic(coverage_rows, POSSESSION_COVERAGE_COLUMNS, coverage_path)
    return output_rows, model_b_period_start, output_path, coverage_path


def collect_possession_averages(
    start_years: Sequence[int],
    max_requests: int,
    *,
    project_root: Path = PROJECT_ROOT,
    opener: Callable[..., Any] = urlopen,
    timeout: float = REQUEST_TIMEOUT_SECONDS,
    attempts: int = DEFAULT_ATTEMPTS,
    request_delay: float = DEFAULT_REQUEST_DELAY_SECONDS,
    sleep_fn: Callable[[float], None] = time.sleep,
) -> CollectionSummary:
    """Collect requested seasons within a strict budget and resume from cache."""
    project_root = project_root.resolve()
    requested = tuple(sorted(set(start_years)))
    if not requested:
        raise SofaScoreConfigurationError("At least one season is required.")
    if isinstance(max_requests, bool) or max_requests <= 0:
        raise SofaScoreConfigurationError("--max-requests must be positive.")
    if timeout <= 0 or attempts <= 0 or request_delay < 0:
        raise SofaScoreConfigurationError(
            "Timeout and attempts must be positive; request delay cannot be negative."
        )
    configured_years = {
        int(season.label[:4])
        for season in load_seasons(project_root / "config" / "seasons.json")
    }
    unknown = sorted(set(requested).difference(configured_years))
    if unknown:
        raise SofaScoreConfigurationError(
            "Requested seasons are outside config/seasons.json: "
            + ", ".join(str(year) for year in unknown)
        )

    if _aggregate_export_path(project_root).exists():
        rows, period_start, output_path, coverage_path = build_processed_possession(
            project_root
        )
        requested_codes = {
            season_code_from_start_year(start_year) for start_year in requested
        }
        selected_rows = [
            row for row in rows if row.source_season in requested_codes
        ]
        found_codes = {row.source_season for row in selected_rows}
        missing_codes = sorted(requested_codes.difference(found_codes))
        if missing_codes:
            raise SofaScoreCacheError(
                "SofaScore aggregate export does not contain requested seasons: "
                + ", ".join(missing_codes)
            )
        return CollectionSummary(
            requested_seasons=requested,
            seasons_found=len(found_codes),
            teams_found=len(selected_rows),
            statistics_cached=len(selected_rows),
            statistics_downloaded=0,
            failed_statistics=0,
            requests_made=0,
            stopped_before_budget=False,
            output_rows=len(rows),
            complete_rows=sum(
                row.average_possession_pct is not None for row in rows
            ),
            model_b_period_start=period_start,
            output_path=output_path,
            coverage_path=coverage_path,
        )

    raw_directory = project_root / "data" / "raw" / "sofascore"
    budget = _RequestBudget(max_requests)
    seasons_cache = raw_directory / "seasons.json"
    seasons_payload, _ = _load_or_fetch(
        seasons_path(),
        seasons_cache,
        project_root=project_root,
        endpoint=SEASONS_ENDPOINT,
        season_code="all",
        season_id=None,
        team_id=None,
        team_name=None,
        parser=parse_seasons,
        budget=budget,
        opener=opener,
        timeout=timeout,
        attempts=attempts,
        sleep_fn=sleep_fn,
        request_delay=request_delay,
    )
    seasons_by_year = {season.start_year: season for season in parse_seasons(seasons_payload)}
    missing = [year for year in requested if year not in seasons_by_year]
    if missing:
        raise SofaScoreResponseError(
            "SofaScore season directory does not contain: "
            + ", ".join(str(year) for year in missing)
        )

    teams_found = 0
    cached_statistics = 0
    downloaded_statistics = 0
    failed_statistics = 0
    seasons_found = 0
    stopped = False
    for start_year in requested:
        season = seasons_by_year[start_year]
        standings_cache = raw_directory / f"standings_{season.season_id}.json"
        try:
            standings_payload, _ = _load_or_fetch(
                standings_path(season.season_id),
                standings_cache,
                project_root=project_root,
                endpoint=STANDINGS_ENDPOINT,
                season_code=season.season_code,
                season_id=season.season_id,
                team_id=None,
                team_name=None,
                parser=parse_standing_teams,
                budget=budget,
                opener=opener,
                timeout=timeout,
                attempts=attempts,
                sleep_fn=sleep_fn,
                request_delay=request_delay,
            )
        except RequestBudgetReached:
            stopped = True
            break
        teams = parse_standing_teams(standings_payload)
        seasons_found += 1
        teams_found += len(teams)
        for team in teams:
            stats_cache = raw_directory / (
                f"team_{team.team_id}_season_{season.season_id}_statistics_overall.json"
            )
            try:
                _, cached = _load_or_fetch(
                    team_statistics_path(team.team_id, season.season_id),
                    stats_cache,
                    project_root=project_root,
                    endpoint=TEAM_STATISTICS_ENDPOINT,
                    season_code=season.season_code,
                    season_id=season.season_id,
                    team_id=team.team_id,
                    team_name=team.name,
                    parser=parse_team_statistics,
                    budget=budget,
                    opener=opener,
                    timeout=timeout,
                    attempts=attempts,
                    sleep_fn=sleep_fn,
                    request_delay=request_delay,
                )
            except RequestBudgetReached:
                stopped = True
                break
            except SofaScoreRequestError:
                failed_statistics += 1
                continue
            if cached:
                cached_statistics += 1
            else:
                downloaded_statistics += 1
        if stopped:
            break

    rows, period_start, output_path, coverage_path = build_processed_possession(
        project_root
    )
    complete_rows = sum(row.average_possession_pct is not None for row in rows)
    return CollectionSummary(
        requested_seasons=requested,
        seasons_found=seasons_found,
        teams_found=teams_found,
        statistics_cached=cached_statistics,
        statistics_downloaded=downloaded_statistics,
        failed_statistics=failed_statistics,
        requests_made=budget.used,
        stopped_before_budget=stopped,
        output_rows=len(rows),
        complete_rows=complete_rows,
        model_b_period_start=period_start,
        output_path=output_path,
        coverage_path=coverage_path,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Collect immutable SofaScore EPL team-season possession averages. "
            "A target season must use only the preceding source season."
        )
    )
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument(
        "--season", type=int, metavar="YEAR", help="one EPL season start year"
    )
    selection.add_argument(
        "--all",
        action="store_true",
        help="collect the configured default range 2017/18 through 2025/26",
    )
    parser.add_argument("--first-season", type=int, default=DEFAULT_FIRST_SEASON)
    parser.add_argument("--last-season", type=int, default=DEFAULT_LAST_SEASON)
    parser.add_argument("--max-requests", type=int, default=250, metavar="N")
    parser.add_argument(
        "--request-delay",
        type=float,
        default=DEFAULT_REQUEST_DELAY_SECONDS,
        metavar="SECONDS",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    if arguments.all:
        if arguments.first_season > arguments.last_season:
            print("error: --first-season must not exceed --last-season", file=sys.stderr)
            return 1
        years = range(arguments.first_season, arguments.last_season + 1)
    else:
        years = (arguments.season,)
    try:
        summary = collect_possession_averages(
            years,
            arguments.max_requests,
            request_delay=arguments.request_delay,
        )
    except (PossessionCollectionError, ManifestError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    print("source=sofascore")
    print("requested_seasons=" + ",".join(str(year) for year in summary.requested_seasons))
    print(f"seasons_found={summary.seasons_found}")
    print(f"teams_found={summary.teams_found}")
    print(f"statistics_cached={summary.statistics_cached}")
    print(f"statistics_downloaded={summary.statistics_downloaded}")
    print(f"failed_statistics={summary.failed_statistics}")
    print(f"requests_made={summary.requests_made}")
    print(f"stopped_before_budget={str(summary.stopped_before_budget).lower()}")
    print(f"complete_rows={summary.complete_rows}/{summary.output_rows}")
    print(f"model_b_period_start={summary.model_b_period_start or 'none'}")
    print(f"output={summary.output_path.relative_to(PROJECT_ROOT).as_posix()}")
    print(f"coverage={summary.coverage_path.relative_to(PROJECT_ROOT).as_posix()}")
    if summary.stopped_before_budget:
        print("resume=rerun the same command; verified caches will be skipped")
    return 1 if summary.failed_statistics else 0


if __name__ == "__main__":
    sys.exit(main())
