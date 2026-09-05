"""Legacy API-Football fixture-possession support retained for old caches."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
from http.client import HTTPException
import json
import math
import os
from pathlib import Path
import sys
import tempfile
from typing import Any, Callable, Mapping, Sequence
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from src.download_data import (
    PROJECT_ROOT,
    ManifestError,
    ManifestIntegrityError,
    load_manifest,
    load_seasons,
    sha256_file,
    write_manifest_atomic,
)


API_BASE_URL = "https://v3.football.api-sports.io"
API_SOURCE = "api-football"
API_TEAM_PROVIDER = "api_football"
API_KEY_ENVIRONMENT_VARIABLE = "API_FOOTBALL_KEY"
EPL_LEAGUE_ID = 39
REQUEST_TIMEOUT_SECONDS = 30.0
USER_AGENT = "epl-match-outcome-predictor/1.0"
FIXTURES_ENDPOINT = "fixtures"
STATISTICS_ENDPOINT = "fixtures/statistics"


class PossessionCollectionError(RuntimeError):
    """Base class for expected, actionable possession-collection failures."""


class ApiConfigurationError(PossessionCollectionError):
    """Raised when collection configuration or credentials are invalid."""


class ApiRequestError(PossessionCollectionError):
    """Raised when an API-Football request cannot be completed."""


class ApiQuotaError(ApiRequestError):
    """Raised when API-Football reports that its request quota is exhausted."""


class ApiResponseError(PossessionCollectionError):
    """Raised when a response violates the expected API-Football contract."""


class ApiCacheError(PossessionCollectionError):
    """Raised when an immutable cached response fails provenance validation."""


class PossessionValueError(PossessionCollectionError):
    """Raised when an API possession value is malformed or outside 0--100."""


@dataclass(frozen=True)
class ApiFixture:
    """Validated EPL fixture metadata from a cached fixtures response."""

    fixture_id: int
    season_year: int
    date: str
    home_team_id: int
    home_team_name: str
    away_team_id: int
    away_team_name: str


@dataclass(frozen=True)
class ApiPossessionFixture(ApiFixture):
    """One cached API fixture plus its possibly-missing possession values."""

    home_possession: float | None
    away_possession: float | None


@dataclass(frozen=True)
class CollectionSummary:
    """Auditable counts from one bounded, resumable collection run."""

    season_year: int
    fixture_count: int
    statistics_cached: int
    statistics_downloaded: int
    requests_made: int
    stopped_before_quota: bool


def fixtures_url(season_year: int) -> str:
    """Build the fixed completed-EPL-fixtures endpoint for one season."""
    query = urlencode(
        {"league": EPL_LEAGUE_ID, "season": season_year, "status": "FT"}
    )
    return f"{API_BASE_URL}/fixtures?{query}"


def statistics_url(fixture_id: int) -> str:
    """Build the fixture-statistics endpoint without embedding credentials."""
    return f"{API_BASE_URL}/fixtures/statistics?{urlencode({'fixture': fixture_id})}"


def season_year_to_code(
    season_year: int, *, project_root: Path = PROJECT_ROOT
) -> str:
    """Resolve an API start year to an explicitly configured canonical season."""
    seasons = load_seasons(project_root / "config" / "seasons.json")
    for season in seasons:
        if season.label.startswith(f"{season_year}/"):
            return season.code
    configured = ", ".join(season.label[:4] for season in seasons)
    raise ApiConfigurationError(
        f"API season {season_year!r} is not configured. "
        f"Configured start years: {configured}"
    )


def parse_possession(value: object) -> float | None:
    """Parse an API possession percentage while preserving true missingness."""
    if value is None:
        return None
    if isinstance(value, bool):
        raise PossessionValueError(f"Invalid possession value {value!r}.")

    candidate: object = value
    if isinstance(value, str):
        text = value.strip()
        if not text.endswith("%"):
            raise PossessionValueError(
                f"Invalid possession value {value!r}; "
                "expected a percentage such as '56%'."
            )
        candidate = text[:-1].strip()
        if not candidate:
            raise PossessionValueError(f"Invalid possession value {value!r}.")

    try:
        parsed = float(candidate)
    except (TypeError, ValueError) as exc:
        raise PossessionValueError(f"Invalid possession value {value!r}.") from exc
    if not math.isfinite(parsed) or parsed < 0.0 or parsed > 100.0:
        raise PossessionValueError(
            f"Possession value {value!r} must be between 0 and 100."
        )
    return parsed


def _format_utc(moment: datetime) -> str:
    if moment.tzinfo is None:
        raise ApiConfigurationError(
            "The collector clock must return a timezone-aware value."
        )
    normalized = moment.astimezone(timezone.utc).replace(microsecond=0)
    return normalized.isoformat().replace("+00:00", "Z")


def _response_items(payload: object, *, endpoint: str) -> list[Any]:
    if not isinstance(payload, dict):
        raise ApiResponseError(
            f"API-Football {endpoint} response must be a JSON object."
        )
    errors = payload.get("errors")
    if errors not in (None, {}, [], ""):
        error_text = json.dumps(errors, ensure_ascii=True).lower()
        if (
            "limit" in error_text
            or "quota" in error_text
            or "request" in error_text
        ):
            raise ApiQuotaError(
                "API-Football reported exhausted request quota; "
                "resume after quota is available."
            )
        raise ApiResponseError(
            f"API-Football rejected the {endpoint} request; "
            "verify the key and subscription."
        )
    response = payload.get("response")
    if not isinstance(response, list):
        raise ApiResponseError(
            f"API-Football {endpoint} response requires a JSON response list."
        )
    return response


def _read_http_response(
    source_url: str,
    api_key: str | None,
    *,
    opener: Callable[..., Any],
    timeout: float,
) -> tuple[bytes, dict[str, Any], int | None]:
    if api_key is None or not api_key.strip():
        raise ApiConfigurationError(
            f"{API_KEY_ENVIRONMENT_VARIABLE} is missing or empty. "
            "Set it locally before requesting uncached API data."
        )
    request = Request(
        source_url,
        headers={
            "User-Agent": USER_AGENT,
            "x-apisports-key": api_key.strip(),
        },
    )
    try:
        with opener(request, timeout=timeout) as response:
            status = getattr(response, "status", None)
            if status is not None and not 200 <= int(status) < 300:
                raise ApiRequestError(
                    f"HTTP {status} from API-Football for {source_url}."
                )
            raw_bytes = response.read()
            headers = getattr(response, "headers", {})
            remaining_text = None
            if hasattr(headers, "get"):
                remaining_text = headers.get("x-ratelimit-requests-remaining")
                if remaining_text is None:
                    remaining_text = headers.get(
                        "X-RateLimit-Requests-Remaining"
                    )
    except HTTPError as exc:
        if exc.code == 429:
            raise ApiQuotaError(
                "API-Football returned HTTP 429; "
                "resume after quota is available."
            ) from exc
        raise ApiRequestError(
            f"HTTP {exc.code} from API-Football for {source_url}: {exc.reason}"
        ) from exc
    except URLError as exc:
        raise ApiRequestError(f"Could not reach API-Football: {exc.reason}") from exc
    except TimeoutError as exc:
        raise ApiRequestError(
            f"Timed out requesting API-Football endpoint {source_url}."
        ) from exc
    except HTTPException as exc:
        raise ApiRequestError(f"API-Football HTTP response failed: {exc}") from exc
    except OSError as exc:
        raise ApiRequestError(f"Could not read API-Football response: {exc}") from exc

    try:
        payload = json.loads(raw_bytes.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ApiResponseError(
            f"API-Football returned invalid JSON for {source_url}."
        ) from exc
    if not isinstance(payload, dict):
        raise ApiResponseError(
            f"API-Football returned a non-object JSON response for {source_url}."
        )

    remaining: int | None = None
    if remaining_text is not None:
        try:
            remaining = int(str(remaining_text))
        except ValueError as exc:
            raise ApiResponseError(
                "API-Football returned an invalid remaining-quota header."
            ) from exc
        if remaining < 0:
            raise ApiResponseError(
                "API-Football returned a negative remaining-quota header."
            )
    return raw_bytes, payload, remaining


def _relative_cache_path(project_root: Path, destination: Path) -> str:
    try:
        return (
            destination.resolve()
            .relative_to(project_root.resolve())
            .as_posix()
        )
    except ValueError as exc:
        raise ApiCacheError(
            f"API cache path escapes the project root: {destination}"
        ) from exc


def _manifest_record_for_path(
    records: Sequence[Mapping[str, Any]], local_path: str
) -> Mapping[str, Any] | None:
    matches = [
        record for record in records if record.get("local_path") == local_path
    ]
    if len(matches) > 1:
        raise ApiCacheError(
            f"Manifest contains duplicate records for {local_path}."
        )
    return matches[0] if matches else None


def _verify_api_record(
    record: Mapping[str, Any],
    *,
    season_year: int,
    endpoint: str,
    fixture_id: int | None,
    source_url: str,
    local_path: str,
) -> None:
    expected = {
        "source": API_SOURCE,
        "season": str(season_year),
        "endpoint": endpoint,
        "fixture_id": fixture_id,
        "source_url": source_url,
        "local_path": local_path,
    }
    mismatches = [
        key for key, value in expected.items() if record.get(key) != value
    ]
    if mismatches:
        raise ApiCacheError(
            f"API cache manifest mismatch for {local_path}: "
            f"{', '.join(mismatches)}."
        )


def _load_cached_payload(
    destination: Path,
    *,
    project_root: Path,
    records: Sequence[Mapping[str, Any]],
    season_year: int,
    endpoint: str,
    fixture_id: int | None,
    source_url: str,
) -> dict[str, Any] | None:
    local_path = _relative_cache_path(project_root, destination)
    record = _manifest_record_for_path(records, local_path)
    if not destination.exists():
        if record is not None:
            raise ApiCacheError(
                f"Manifested API cache file is missing: {destination}"
            )
        return None
    if record is None:
        raise ApiCacheError(
            f"API cache {destination} exists without a manifest record; "
            "refusing to infer provenance."
        )
    _verify_api_record(
        record,
        season_year=season_year,
        endpoint=endpoint,
        fixture_id=fixture_id,
        source_url=source_url,
        local_path=local_path,
    )
    try:
        checksum = sha256_file(destination)
    except ManifestIntegrityError as exc:
        raise ApiCacheError(str(exc)) from exc
    if checksum != record["sha256"]:
        raise ApiCacheError(
            f"Checksum mismatch for immutable API cache {destination}."
        )
    try:
        payload = json.loads(destination.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ApiCacheError(
            f"Could not read cached API response {destination}: {exc}"
        ) from exc
    response = _response_items(payload, endpoint=endpoint)
    if len(response) != record["row_count"]:
        raise ApiCacheError(
            f"Row-count mismatch for immutable API cache {destination}."
        )
    return payload


def _write_api_cache(
    raw_bytes: bytes,
    payload: Mapping[str, Any],
    destination: Path,
    *,
    project_root: Path,
    season_year: int,
    endpoint: str,
    fixture_id: int | None,
    source_url: str,
    clock: Callable[[], datetime],
) -> None:
    """Atomically preserve exact response bytes, then record their provenance."""
    if destination.exists():
        raise ApiCacheError(
            f"Refusing to overwrite immutable API cache {destination}."
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    file_descriptor, temporary_name = tempfile.mkstemp(
        dir=destination.parent,
        prefix=f".{destination.name}.",
        suffix=".tmp",
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(file_descriptor, "wb") as output:
            output.write(raw_bytes)
            output.flush()
            os.fsync(output.fileno())
        if destination.exists():
            raise ApiCacheError(
                f"API cache appeared during collection: {destination}"
            )
        os.replace(temporary_path, destination)
    except OSError as exc:
        raise ApiCacheError(
            f"Could not atomically cache API response {destination}: {exc}"
        ) from exc
    finally:
        temporary_path.unlink(missing_ok=True)

    try:
        manifest_path = project_root / "data" / "raw" / "manifest.json"
        records = load_manifest(manifest_path)
        local_path = _relative_cache_path(project_root, destination)
        if _manifest_record_for_path(records, local_path) is not None:
            raise ApiCacheError(
                f"Manifest record appeared during collection for {local_path}."
            )
        response = _response_items(payload, endpoint=endpoint)
        record = {
            "source": API_SOURCE,
            "season": str(season_year),
            "endpoint": endpoint,
            "fixture_id": fixture_id,
            "source_url": source_url,
            "local_path": local_path,
            "retrieved_at_utc": _format_utc(clock()),
            "sha256": sha256_file(destination),
            "row_count": len(response),
        }
        write_manifest_atomic([*records, record], manifest_path)
    except Exception:
        try:
            destination.unlink(missing_ok=True)
        except OSError as cleanup_error:
            raise ApiCacheError(
                "API manifest update failed and the unmanifested response "
                f"could not be removed: {destination}: {cleanup_error}"
            ) from cleanup_error
        raise


def _positive_integer(value: object, *, field_name: str) -> int:
    if isinstance(value, bool):
        raise ApiResponseError(f"API fixture has invalid {field_name}.")
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise ApiResponseError(
            f"API fixture has invalid {field_name}."
        ) from exc
    if parsed <= 0 or str(parsed) != str(value).strip():
        raise ApiResponseError(f"API fixture has invalid {field_name}.")
    return parsed


def _required_name(value: object, *, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ApiResponseError(f"API fixture has invalid {field_name}.")
    return value.strip()


def parse_fixture_list(
    payload: object, *, season_year: int
) -> list[ApiFixture]:
    """Validate a completed EPL fixture listing and return stable metadata."""
    items = _response_items(payload, endpoint=FIXTURES_ENDPOINT)
    fixtures: list[ApiFixture] = []
    seen_ids: set[int] = set()
    for position, item in enumerate(items):
        if not isinstance(item, dict):
            raise ApiResponseError(
                f"Fixture response item {position} must be an object."
            )
        fixture = item.get("fixture")
        league = item.get("league")
        teams = item.get("teams")
        if (
            not isinstance(fixture, dict)
            or not isinstance(league, dict)
            or not isinstance(teams, dict)
        ):
            raise ApiResponseError(
                f"Fixture response item {position} is missing fixture, "
                "league, or teams metadata."
            )
        if league.get("id") != EPL_LEAGUE_ID or league.get("season") != season_year:
            raise ApiResponseError(
                f"Fixture response item {position} is not EPL league "
                f"{EPL_LEAGUE_ID}, season {season_year}."
            )
        status = fixture.get("status")
        if not isinstance(status, dict) or status.get("short") != "FT":
            raise ApiResponseError(
                f"Fixture response item {position} is not completed with status FT."
            )
        fixture_id = _positive_integer(
            fixture.get("id"), field_name="fixture id"
        )
        if fixture_id in seen_ids:
            raise ApiResponseError(
                f"Fixture response contains duplicate fixture id {fixture_id}."
            )
        date_text = fixture.get("date")
        if not isinstance(date_text, str):
            raise ApiResponseError(f"Fixture {fixture_id} has an invalid date.")
        try:
            parsed_datetime = datetime.fromisoformat(
                date_text.replace("Z", "+00:00")
            )
        except ValueError as exc:
            raise ApiResponseError(
                f"Fixture {fixture_id} has an invalid ISO date."
            ) from exc
        if parsed_datetime.tzinfo is None:
            raise ApiResponseError(
                f"Fixture {fixture_id} date must include a timezone."
            )

        home = teams.get("home")
        away = teams.get("away")
        if not isinstance(home, dict) or not isinstance(away, dict):
            raise ApiResponseError(
                f"Fixture {fixture_id} has invalid home/away teams."
            )
        home_id = _positive_integer(
            home.get("id"), field_name="home team id"
        )
        away_id = _positive_integer(
            away.get("id"), field_name="away team id"
        )
        if home_id == away_id:
            raise ApiResponseError(
                f"Fixture {fixture_id} has the same home and away team id."
            )
        fixtures.append(
            ApiFixture(
                fixture_id=fixture_id,
                season_year=season_year,
                date=parsed_datetime.date().isoformat(),
                home_team_id=home_id,
                home_team_name=_required_name(
                    home.get("name"), field_name="home team name"
                ),
                away_team_id=away_id,
                away_team_name=_required_name(
                    away.get("name"), field_name="away team name"
                ),
            )
        )
        seen_ids.add(fixture_id)
    return sorted(fixtures, key=lambda item: item.fixture_id)


def _team_possession(
    statistics: object, *, fixture_id: int, team_id: int
) -> float | None:
    if not isinstance(statistics, list):
        raise ApiResponseError(
            f"Fixture {fixture_id} team {team_id} statistics must be a list."
        )
    matches = [
        item
        for item in statistics
        if isinstance(item, dict) and item.get("type") == "Ball Possession"
    ]
    if len(matches) > 1:
        raise ApiResponseError(
            f"Fixture {fixture_id} team {team_id} has duplicate "
            "Ball Possession statistics."
        )
    if not matches:
        return None
    return parse_possession(matches[0].get("value"))


def parse_fixture_statistics(
    payload: object, fixture: ApiFixture
) -> ApiPossessionFixture:
    """Extract home/away possession by stable API team ID, regardless of order."""
    items = _response_items(payload, endpoint=STATISTICS_ENDPOINT)
    by_team: dict[int, Mapping[str, Any]] = {}
    expected_ids = {fixture.home_team_id, fixture.away_team_id}
    for position, item in enumerate(items):
        if not isinstance(item, dict) or not isinstance(item.get("team"), dict):
            raise ApiResponseError(
                f"Fixture {fixture.fixture_id} statistics item {position} "
                "lacks team metadata."
            )
        team_id = _positive_integer(
            item["team"].get("id"), field_name="statistics team id"
        )
        if team_id not in expected_ids:
            raise ApiResponseError(
                f"Fixture {fixture.fixture_id} statistics contain "
                f"unexpected team id {team_id}."
            )
        if team_id in by_team:
            raise ApiResponseError(
                f"Fixture {fixture.fixture_id} statistics duplicate team id {team_id}."
            )
        by_team[team_id] = item

    def possession_for(team_id: int) -> float | None:
        item = by_team.get(team_id)
        if item is None:
            return None
        return _team_possession(
            item.get("statistics"),
            fixture_id=fixture.fixture_id,
            team_id=team_id,
        )

    return ApiPossessionFixture(
        **fixture.__dict__,
        home_possession=possession_for(fixture.home_team_id),
        away_possession=possession_for(fixture.away_team_id),
    )


def collect_season(
    season_year: int,
    max_requests: int,
    *,
    project_root: Path = PROJECT_ROOT,
    api_key: str | None = None,
    opener: Callable[..., Any] = urlopen,
    timeout: float = REQUEST_TIMEOUT_SECONDS,
    clock: Callable[[], datetime] = lambda: datetime.now(timezone.utc),
) -> CollectionSummary:
    """Collect one EPL season within a strict request budget and resume from cache."""
    project_root = project_root.resolve()
    season_year_to_code(season_year, project_root=project_root)
    if isinstance(max_requests, bool) or max_requests <= 0:
        raise ApiConfigurationError(
            "--max-requests must be a positive integer."
        )
    if timeout <= 0:
        raise ApiConfigurationError("API request timeout must be positive.")

    raw_directory = project_root / "data" / "raw" / "api_football"
    manifest_path = project_root / "data" / "raw" / "manifest.json"
    request_count = 0
    server_quota_remaining: int | None = None
    listing_path = raw_directory / f"fixtures_{season_year}.json"
    listing_url = fixtures_url(season_year)
    records = load_manifest(manifest_path)
    listing_payload = _load_cached_payload(
        listing_path,
        project_root=project_root,
        records=records,
        season_year=season_year,
        endpoint=FIXTURES_ENDPOINT,
        fixture_id=None,
        source_url=listing_url,
    )
    if listing_payload is None:
        raw_bytes, listing_payload, server_quota_remaining = _read_http_response(
            listing_url,
            api_key,
            opener=opener,
            timeout=timeout,
        )
        request_count += 1
        _response_items(listing_payload, endpoint=FIXTURES_ENDPOINT)
        _write_api_cache(
            raw_bytes,
            listing_payload,
            listing_path,
            project_root=project_root,
            season_year=season_year,
            endpoint=FIXTURES_ENDPOINT,
            fixture_id=None,
            source_url=listing_url,
            clock=clock,
        )

    fixtures = parse_fixture_list(listing_payload, season_year=season_year)
    cached_statistics = 0
    downloaded_statistics = 0
    stopped = False
    for fixture in fixtures:
        stats_path = (
            raw_directory / f"fixture_{fixture.fixture_id}_statistics.json"
        )
        stats_url = statistics_url(fixture.fixture_id)
        records = load_manifest(manifest_path)
        stats_payload = _load_cached_payload(
            stats_path,
            project_root=project_root,
            records=records,
            season_year=season_year,
            endpoint=STATISTICS_ENDPOINT,
            fixture_id=fixture.fixture_id,
            source_url=stats_url,
        )
        if stats_payload is not None:
            parse_fixture_statistics(stats_payload, fixture)
            cached_statistics += 1
            continue
        if request_count >= max_requests or server_quota_remaining == 0:
            stopped = True
            break
        raw_bytes, stats_payload, server_quota_remaining = _read_http_response(
            stats_url,
            api_key,
            opener=opener,
            timeout=timeout,
        )
        request_count += 1
        _response_items(stats_payload, endpoint=STATISTICS_ENDPOINT)
        _write_api_cache(
            raw_bytes,
            stats_payload,
            stats_path,
            project_root=project_root,
            season_year=season_year,
            endpoint=STATISTICS_ENDPOINT,
            fixture_id=fixture.fixture_id,
            source_url=stats_url,
            clock=clock,
        )
        parse_fixture_statistics(stats_payload, fixture)
        downloaded_statistics += 1

    return CollectionSummary(
        season_year=season_year,
        fixture_count=len(fixtures),
        statistics_cached=cached_statistics,
        statistics_downloaded=downloaded_statistics,
        requests_made=request_count,
        stopped_before_quota=stopped,
    )


def load_cached_possession_fixtures(
    project_root: Path = PROJECT_ROOT,
) -> list[ApiPossessionFixture]:
    """Load every fully cached, manifested API fixture-statistics response."""
    project_root = project_root.resolve()
    manifest_path = project_root / "data" / "raw" / "manifest.json"
    records = load_manifest(manifest_path)
    listing_records = sorted(
        (
            record
            for record in records
            if record.get("source") == API_SOURCE
            and record.get("endpoint") == FIXTURES_ENDPOINT
        ),
        key=lambda item: str(item.get("season")),
    )
    loaded: list[ApiPossessionFixture] = []
    consumed_statistics_paths: set[str] = set()
    for record in listing_records:
        try:
            season_year = int(record["season"])
        except (TypeError, ValueError) as exc:
            raise ApiCacheError(
                "API fixture-list manifest record has an invalid season."
            ) from exc
        season_year_to_code(season_year, project_root=project_root)
        listing_path = project_root / str(record["local_path"])
        payload = _load_cached_payload(
            listing_path,
            project_root=project_root,
            records=records,
            season_year=season_year,
            endpoint=FIXTURES_ENDPOINT,
            fixture_id=None,
            source_url=fixtures_url(season_year),
        )
        if payload is None:  # pragma: no cover - manifested missing handled above
            raise ApiCacheError(
                f"Manifested fixture listing is missing: {listing_path}"
            )
        for fixture in parse_fixture_list(payload, season_year=season_year):
            stats_path = (
                project_root
                / "data"
                / "raw"
                / "api_football"
                / f"fixture_{fixture.fixture_id}_statistics.json"
            )
            stats_local_path = _relative_cache_path(project_root, stats_path)
            stats_record = _manifest_record_for_path(records, stats_local_path)
            if stats_record is None and not stats_path.exists():
                continue
            stats_payload = _load_cached_payload(
                stats_path,
                project_root=project_root,
                records=records,
                season_year=season_year,
                endpoint=STATISTICS_ENDPOINT,
                fixture_id=fixture.fixture_id,
                source_url=statistics_url(fixture.fixture_id),
            )
            if stats_payload is None:  # pragma: no cover - state handled above
                continue
            consumed_statistics_paths.add(stats_local_path)
            loaded.append(parse_fixture_statistics(stats_payload, fixture))

    orphaned = sorted(
        str(record["local_path"])
        for record in records
        if record.get("source") == API_SOURCE
        and record.get("endpoint") == STATISTICS_ENDPOINT
        and str(record["local_path"]) not in consumed_statistics_paths
    )
    if orphaned:
        raise ApiCacheError(
            "Statistics caches have no matching configured EPL fixture listing: "
            + ", ".join(orphaned)
        )
    return sorted(
        loaded, key=lambda item: (item.season_year, item.fixture_id)
    )


def build_parser() -> argparse.ArgumentParser:
    """Build the Phase 6 possession-collector CLI."""
    parser = argparse.ArgumentParser(
        description="Collect immutable API-Football EPL possession responses."
    )
    parser.add_argument(
        "--season",
        type=int,
        required=True,
        metavar="YEAR",
        help="configured EPL season start year, such as 2025",
    )
    parser.add_argument(
        "--max-requests",
        type=int,
        required=True,
        metavar="N",
        help="strict maximum number of API requests for this run",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the bounded API-Football collection CLI without printing the key."""
    arguments = build_parser().parse_args(argv)
    try:
        summary = collect_season(
            arguments.season,
            arguments.max_requests,
            api_key=os.environ.get(API_KEY_ENVIRONMENT_VARIABLE),
        )
    except (PossessionCollectionError, ManifestError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    print(f"season={summary.season_year}")
    print(f"fixtures={summary.fixture_count}")
    print(f"statistics_cached={summary.statistics_cached}")
    print(f"statistics_downloaded={summary.statistics_downloaded}")
    print(f"requests_made={summary.requests_made}")
    print(
        "stopped_before_quota="
        f"{str(summary.stopped_before_quota).lower()}"
    )
    if summary.stopped_before_quota:
        print(
            "resume=rerun the same command; "
            "verified cached responses will be skipped"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
