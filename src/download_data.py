"""Download and preserve season-level Football-Data EPL CSV files."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
from http.client import HTTPException
import json
import os
from pathlib import Path
import re
import sys
import tempfile
from typing import Any, Callable, Sequence
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SEASONS_CONFIG_PATH = PROJECT_ROOT / "config" / "seasons.json"
FOOTBALL_DATA_DIRECTORY = PROJECT_ROOT / "data" / "raw" / "football_data"
MANIFEST_PATH = PROJECT_ROOT / "data" / "raw" / "manifest.json"
FOOTBALL_DATA_BASE_URL = "https://www.football-data.co.uk/mmz4281"
FOOTBALL_DATA_SOURCE = "football-data"
LEAGUE_CODE = "E0"
DOWNLOAD_TIMEOUT_SECONDS = 30.0
USER_AGENT = "epl-match-outcome-predictor/1.0"
REQUIRED_COLUMNS = frozenset(
    {
        "Div",
        "Date",
        "HomeTeam",
        "AwayTeam",
        "FTHG",
        "FTAG",
        "FTR",
        "HS",
        "AS",
    }
)
MANIFEST_REQUIRED_FIELDS = frozenset(
    {
        "source",
        "season",
        "source_url",
        "local_path",
        "retrieved_at_utc",
        "sha256",
        "row_count",
    }
)


class IngestionError(RuntimeError):
    """Base class for expected, actionable ingestion failures."""


class ConfigurationError(IngestionError):
    """Raised when season configuration is missing or invalid."""


class DownloadError(IngestionError):
    """Raised when a source response cannot be downloaded."""


class CsvValidationError(IngestionError):
    """Raised when a downloaded CSV violates the source contract."""


class EmptyDownloadError(CsvValidationError):
    """Raised when a downloaded CSV contains no match rows."""


class MissingColumnsError(CsvValidationError):
    """Raised when a downloaded CSV omits required columns."""


class InvalidDivisionError(CsvValidationError):
    """Raised when a downloaded CSV contains a non-EPL row."""


class ManifestError(IngestionError):
    """Raised when the raw-data manifest is missing required structure."""


class ManifestIntegrityError(ManifestError):
    """Raised when a local raw file disagrees with its manifest record."""


@dataclass(frozen=True)
class Season:
    """A configured EPL season label and Football-Data season code."""

    label: str
    code: str


@dataclass(frozen=True)
class DownloadResult:
    """The outcome and validated metadata for one season download."""

    season: Season
    path: Path
    row_count: int
    sha256: str
    status: str


def build_season_url(season_code: str) -> str:
    """Return the configured Football-Data EPL URL for a season code."""
    if re.fullmatch(r"\d{4}", season_code) is None:
        raise ConfigurationError(
            f"Invalid season code {season_code!r}; expected exactly four digits."
        )
    return f"{FOOTBALL_DATA_BASE_URL}/{season_code}/E0.csv"


def load_seasons(path: Path = SEASONS_CONFIG_PATH) -> list[Season]:
    """Load and validate the ordered season configuration."""
    try:
        configured = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ConfigurationError(f"Season configuration not found: {path}") from exc
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ConfigurationError(
            f"Could not read season configuration {path}: {exc}"
        ) from exc

    if not isinstance(configured, list) or not configured:
        raise ConfigurationError(
            f"Season configuration {path} must contain a nonempty JSON list."
        )

    seasons: list[Season] = []
    seen_codes: set[str] = set()
    seen_labels: set[str] = set()
    for index, item in enumerate(configured):
        if not isinstance(item, dict):
            raise ConfigurationError(
                f"Season entry {index} in {path} must be a JSON object."
            )
        label = item.get("label")
        code = item.get("code")
        if not isinstance(label, str) or not isinstance(code, str):
            raise ConfigurationError(
                f"Season entry {index} in {path} requires string label and code fields."
            )
        build_season_url(code)
        if code in seen_codes or label in seen_labels:
            raise ConfigurationError(
                f"Duplicate season label or code in {path}: {label!r}, {code!r}."
            )
        seasons.append(Season(label=label, code=code))
        seen_codes.add(code)
        seen_labels.add(label)

    return seasons


def sha256_file(path: Path) -> str:
    """Calculate a file's SHA-256 checksum without loading it all into memory."""
    digest = hashlib.sha256()
    try:
        with path.open("rb") as source:
            for chunk in iter(lambda: source.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as exc:
        raise ManifestIntegrityError(f"Could not hash raw file {path}: {exc}") from exc
    return digest.hexdigest()


def validate_csv(path: Path) -> int:
    """Validate one source CSV and return its number of nonempty match rows."""
    try:
        if path.stat().st_size == 0:
            raise EmptyDownloadError(f"Downloaded CSV is empty: {path}")
    except FileNotFoundError as exc:
        raise CsvValidationError(f"Downloaded CSV does not exist: {path}") from exc
    except OSError as exc:
        raise CsvValidationError(f"Could not inspect downloaded CSV {path}: {exc}") from exc

    try:
        with path.open("r", encoding="utf-8-sig", newline="") as source:
            reader = csv.DictReader(source)
            if not reader.fieldnames:
                raise EmptyDownloadError(
                    f"Downloaded CSV is empty or has no header: {path}"
                )

            missing_columns = REQUIRED_COLUMNS.difference(reader.fieldnames)
            if missing_columns:
                missing = ", ".join(sorted(missing_columns))
                raise MissingColumnsError(
                    f"Downloaded CSV {path} is missing required columns: {missing}"
                )

            row_count = 0
            for line_number, row in enumerate(reader, start=2):
                string_values = (
                    value for value in row.values() if isinstance(value, str)
                )
                if not any(value.strip() for value in string_values):
                    continue
                row_count += 1
                division = (row.get("Div") or "").strip()
                if division != LEAGUE_CODE:
                    raise InvalidDivisionError(
                        f"Downloaded CSV {path} has division {division!r} "
                        f"on line {line_number}; expected only {LEAGUE_CODE}."
                    )
    except UnicodeDecodeError as exc:
        raise CsvValidationError(
            f"Downloaded CSV {path} is not valid UTF-8: {exc}"
        ) from exc
    except csv.Error as exc:
        raise CsvValidationError(f"Could not parse downloaded CSV {path}: {exc}") from exc
    except OSError as exc:
        raise CsvValidationError(f"Could not read downloaded CSV {path}: {exc}") from exc

    if row_count == 0:
        raise EmptyDownloadError(f"Downloaded CSV contains no match rows: {path}")
    return row_count


def load_manifest(path: Path = MANIFEST_PATH) -> list[dict[str, Any]]:
    """Load and validate the data manifest, returning an empty list if absent."""
    if not path.exists():
        return []

    try:
        records = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ManifestError(f"Could not read data manifest {path}: {exc}") from exc

    if not isinstance(records, list):
        raise ManifestError(f"Data manifest {path} must contain a JSON list.")

    seen_local_paths: set[str] = set()
    for index, record in enumerate(records):
        if not isinstance(record, dict):
            raise ManifestError(f"Manifest record {index} must be a JSON object.")
        missing_fields = MANIFEST_REQUIRED_FIELDS.difference(record)
        if missing_fields:
            missing = ", ".join(sorted(missing_fields))
            raise ManifestError(f"Manifest record {index} is missing fields: {missing}")
        if not isinstance(record["local_path"], str):
            raise ManifestError(f"Manifest record {index} has an invalid local_path.")
        if record["local_path"] in seen_local_paths:
            raise ManifestError(
                f"Data manifest contains duplicate local path: {record['local_path']}"
            )
        seen_local_paths.add(record["local_path"])
        for field in (
            "source",
            "season",
            "source_url",
            "retrieved_at_utc",
        ):
            if not isinstance(record[field], str) or not record[field]:
                raise ManifestError(
                    f"Manifest record {index} has an invalid {field}."
                )
        try:
            retrieved_at = datetime.fromisoformat(
                record["retrieved_at_utc"].replace("Z", "+00:00")
            )
        except ValueError as exc:
            raise ManifestError(
                f"Manifest record {index} has an invalid retrieved_at_utc."
            ) from exc
        if retrieved_at.tzinfo is None:
            raise ManifestError(
                f"Manifest record {index} retrieved_at_utc must include a timezone."
            )
        if (
            not isinstance(record["sha256"], str)
            or re.fullmatch(r"[0-9a-f]{64}", record["sha256"]) is None
        ):
            raise ManifestError(f"Manifest record {index} has an invalid SHA-256.")
        if not isinstance(record["row_count"], int) or record["row_count"] < 0:
            raise ManifestError(f"Manifest record {index} has an invalid row_count.")

    return records


def _manifest_sort_key(record: dict[str, Any]) -> tuple[str, str, str, str, str]:
    """Return a stable ordering key that also supports later API records."""
    return (
        str(record.get("source", "")),
        str(record.get("season", "")),
        str(record.get("endpoint", "")),
        str(record.get("fixture_id", "")),
        str(record.get("local_path", "")),
    )


def write_manifest_atomic(records: list[dict[str, Any]], path: Path) -> None:
    """Write sorted manifest records through a temporary file and atomic rename."""
    path.parent.mkdir(parents=True, exist_ok=True)
    file_descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(file_descriptor, "w", encoding="utf-8", newline="\n") as output:
            json.dump(
                sorted(records, key=_manifest_sort_key),
                output,
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
            output.write("\n")
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary_path, path)
    except (OSError, TypeError, ValueError) as exc:
        raise ManifestError(f"Could not write data manifest {path}: {exc}") from exc
    finally:
        temporary_path.unlink(missing_ok=True)


def _football_data_record(
    records: list[dict[str, Any]], season_code: str
) -> dict[str, Any] | None:
    matching = [
        record
        for record in records
        if record.get("source") == FOOTBALL_DATA_SOURCE
        and record.get("season") == season_code
    ]
    if len(matching) > 1:
        raise ManifestError(
            f"Data manifest has multiple Football-Data records for season {season_code}."
        )
    return matching[0] if matching else None


def _upsert_football_data_record(
    records: list[dict[str, Any]], new_record: dict[str, Any]
) -> list[dict[str, Any]]:
    retained = [
        record
        for record in records
        if not (
            record.get("source") == FOOTBALL_DATA_SOURCE
            and record.get("season") == new_record["season"]
        )
    ]
    retained.append(new_record)
    return retained


def _format_utc(moment: datetime) -> str:
    if moment.tzinfo is None:
        raise ConfigurationError("The downloader clock must return a timezone-aware value.")
    normalized = moment.astimezone(timezone.utc).replace(microsecond=0)
    return normalized.isoformat().replace("+00:00", "Z")


def _download_to_temporary_file(
    source_url: str,
    directory: Path,
    opener: Callable[..., Any],
    timeout: float,
) -> Path:
    """Stream one URL to a temporary file in the destination directory."""
    directory.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    request = Request(source_url, headers={"User-Agent": USER_AGENT})

    def discard_temporary_file() -> None:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)

    try:
        with opener(request, timeout=timeout) as response:
            status = getattr(response, "status", None)
            if status is not None and not 200 <= int(status) < 300:
                raise DownloadError(
                    f"HTTP {status} while downloading {source_url}."
                )

            with tempfile.NamedTemporaryFile(
                mode="wb",
                delete=False,
                dir=directory,
                prefix=".E0_",
                suffix=".csv.tmp",
            ) as output:
                temporary_path = Path(output.name)
                while True:
                    chunk = response.read(1024 * 1024)
                    if not chunk:
                        break
                    output.write(chunk)
                output.flush()
                os.fsync(output.fileno())
    except HTTPError as exc:
        discard_temporary_file()
        raise DownloadError(
            f"HTTP {exc.code} while downloading {source_url}: {exc.reason}"
        ) from exc
    except URLError as exc:
        discard_temporary_file()
        raise DownloadError(f"Could not download {source_url}: {exc.reason}") from exc
    except TimeoutError as exc:
        discard_temporary_file()
        raise DownloadError(f"Timed out while downloading {source_url}.") from exc
    except HTTPException as exc:
        discard_temporary_file()
        raise DownloadError(
            f"HTTP response failed while downloading {source_url}: {exc}"
        ) from exc
    except OSError as exc:
        discard_temporary_file()
        raise DownloadError(f"Could not save download from {source_url}: {exc}") from exc
    except Exception:
        discard_temporary_file()
        raise

    if temporary_path is None:
        raise DownloadError(f"Download from {source_url} produced no temporary file.")
    return temporary_path


def _manifest_record(
    season: Season,
    source_url: str,
    local_path: str,
    retrieved_at_utc: str,
    checksum: str,
    row_count: int,
) -> dict[str, Any]:
    return {
        "source": FOOTBALL_DATA_SOURCE,
        "season": season.code,
        "source_url": source_url,
        "local_path": local_path,
        "retrieved_at_utc": retrieved_at_utc,
        "sha256": checksum,
        "row_count": row_count,
    }


def download_season(
    season: Season,
    *,
    project_root: Path = PROJECT_ROOT,
    opener: Callable[..., Any] = urlopen,
    timeout: float = DOWNLOAD_TIMEOUT_SECONDS,
    clock: Callable[[], datetime] = lambda: datetime.now(timezone.utc),
) -> DownloadResult:
    """Download one configured season or validate and reuse its immutable cache."""
    project_root = project_root.resolve()
    raw_directory = project_root / "data" / "raw" / "football_data"
    manifest_path = project_root / "data" / "raw" / "manifest.json"
    destination = raw_directory / f"E0_{season.code}.csv"
    source_url = build_season_url(season.code)
    local_path = destination.relative_to(project_root).as_posix()
    records = load_manifest(manifest_path)
    existing_record = _football_data_record(records, season.code)

    if existing_record is not None:
        if existing_record["source_url"] != source_url:
            raise ManifestIntegrityError(
                f"Manifest URL mismatch for season {season.code}; refusing to replace raw data."
            )
        if existing_record["local_path"] != local_path:
            raise ManifestIntegrityError(
                f"Manifest path mismatch for season {season.code}; refusing to replace raw data."
            )

    if destination.exists():
        if existing_record is None:
            raise ManifestIntegrityError(
                f"Raw file {destination} exists without a manifest record; "
                "refusing to overwrite or infer its provenance."
            )
        row_count = validate_csv(destination)
        checksum = sha256_file(destination)
        if checksum != existing_record["sha256"]:
            raise ManifestIntegrityError(
                f"Checksum mismatch for immutable raw file {destination}."
            )
        if row_count != existing_record["row_count"]:
            raise ManifestIntegrityError(
                f"Row-count mismatch for immutable raw file {destination}."
            )
        return DownloadResult(
            season=season,
            path=destination,
            row_count=row_count,
            sha256=checksum,
            status="cached",
        )

    temporary_path = _download_to_temporary_file(
        source_url, raw_directory, opener, timeout
    )
    try:
        row_count = validate_csv(temporary_path)
        checksum = sha256_file(temporary_path)
        if destination.exists():
            raise ManifestIntegrityError(
                f"Raw file {destination} appeared during download; refusing to replace it."
            )
        try:
            os.replace(temporary_path, destination)
        except OSError as exc:
            raise DownloadError(
                f"Could not atomically place raw file {destination}: {exc}"
            ) from exc
        downloaded_record = _manifest_record(
            season,
            source_url,
            local_path,
            _format_utc(clock()),
            checksum,
            row_count,
        )
        write_manifest_atomic(
            _upsert_football_data_record(records, downloaded_record), manifest_path
        )
    finally:
        temporary_path.unlink(missing_ok=True)

    return DownloadResult(
        season=season,
        path=destination,
        row_count=row_count,
        sha256=checksum,
        status="downloaded",
    )


def verify_football_data_manifest(project_root: Path = PROJECT_ROOT) -> int:
    """Verify every Football-Data manifest checksum, row count, and source URL."""
    project_root = project_root.resolve()
    manifest_path = project_root / "data" / "raw" / "manifest.json"
    records = load_manifest(manifest_path)
    verified_count = 0
    for record in records:
        if record.get("source") != FOOTBALL_DATA_SOURCE:
            continue

        season_code = str(record["season"])
        if record["source_url"] != build_season_url(season_code):
            raise ManifestIntegrityError(
                f"Manifest URL does not match season {season_code}."
            )
        expected_local_path = (
            f"data/raw/football_data/E0_{season_code}.csv"
        )
        if record["local_path"] != expected_local_path:
            raise ManifestIntegrityError(
                f"Manifest path does not match season {season_code}."
            )
        raw_path = (project_root / str(record["local_path"])).resolve()
        try:
            raw_path.relative_to(project_root)
        except ValueError as exc:
            raise ManifestIntegrityError(
                f"Manifest path escapes the project root: {record['local_path']}"
            ) from exc
        if not raw_path.is_file():
            raise ManifestIntegrityError(f"Manifest raw file is missing: {raw_path}")
        checksum = sha256_file(raw_path)
        if checksum != record["sha256"]:
            raise ManifestIntegrityError(f"Manifest checksum mismatch: {raw_path}")
        row_count = validate_csv(raw_path)
        if row_count != record["row_count"]:
            raise ManifestIntegrityError(f"Manifest row-count mismatch: {raw_path}")
        verified_count += 1

    return verified_count


def build_parser() -> argparse.ArgumentParser:
    """Build the Phase 1 command-line parser."""
    parser = argparse.ArgumentParser(
        description="Download immutable Football-Data EPL season CSVs."
    )
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument(
        "--all",
        dest="download_all",
        action="store_true",
        help="download every season in config/seasons.json",
    )
    selection.add_argument(
        "--season",
        metavar="CODE",
        help="download one configured four-digit season code, such as 2425",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the Football-Data downloader CLI."""
    parser = build_parser()
    arguments = parser.parse_args(argv)
    try:
        seasons = load_seasons()
        if arguments.download_all:
            selected_seasons = seasons
        else:
            selected_seasons = [
                season for season in seasons if season.code == arguments.season
            ]
            if not selected_seasons:
                configured_codes = ", ".join(season.code for season in seasons)
                raise ConfigurationError(
                    f"Unknown season code {arguments.season!r}. "
                    f"Configured codes: {configured_codes}"
                )

        for season in selected_seasons:
            result = download_season(season)
            print(
                f"{season.code} {result.status}: {result.row_count} rows, "
                f"sha256={result.sha256}"
            )
    except IngestionError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
