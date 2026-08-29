"""Tests for Phase 1 Football-Data ingestion."""

from datetime import datetime, timezone
import hashlib
from io import BytesIO
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from urllib.error import HTTPError

import pandas as pd

from src.download_data import (
    CsvValidationError,
    DownloadError,
    EmptyDownloadError,
    InvalidDivisionError,
    ManifestIntegrityError,
    MissingColumnsError,
    Season,
    build_season_url,
    download_season,
    load_manifest,
    load_seasons,
    sha256_file,
    validate_csv,
    verify_football_data_manifest,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
FIXTURES = PROJECT_ROOT / "tests" / "fixtures"
EXPECTED_SEASON_CODES = [
    "1011",
    "1112",
    "1213",
    "1314",
    "1415",
    "1516",
    "1617",
    "1718",
    "1819",
    "1920",
    "2021",
    "2122",
    "2223",
    "2324",
    "2425",
    "2526",
]


class FootballDataValidationTests(unittest.TestCase):
    """Verify URL generation, configuration, checksums, and CSV contracts."""

    def test_season_url_generation(self) -> None:
        self.assertEqual(
            build_season_url("2425"),
            "https://www.football-data.co.uk/mmz4281/2425/E0.csv",
        )

    def test_configured_seasons_cover_required_range(self) -> None:
        self.assertEqual(
            [season.code for season in load_seasons()], EXPECTED_SEASON_CODES
        )

    def test_sha256_matches_independent_calculation(self) -> None:
        fixture = FIXTURES / "valid_e0.csv"
        expected = hashlib.sha256(fixture.read_bytes()).hexdigest()
        self.assertEqual(sha256_file(fixture), expected)

    def test_valid_csv_row_count(self) -> None:
        self.assertEqual(validate_csv(FIXTURES / "valid_e0.csv"), 2)

    def test_missing_column_failure(self) -> None:
        with self.assertRaisesRegex(MissingColumnsError, "AS"):
            validate_csv(FIXTURES / "missing_columns.csv")

    def test_empty_download_failure_uses_local_fixture(self) -> None:
        with self.assertRaises(EmptyDownloadError):
            validate_csv(FIXTURES / "empty.csv")

    def test_non_epl_row_failure(self) -> None:
        with self.assertRaisesRegex(InvalidDivisionError, "E1"):
            validate_csv(FIXTURES / "non_epl.csv")

    def test_decoding_failure_is_actionable(self) -> None:
        invalid_utf8 = (
            b"Div,Date,HomeTeam,AwayTeam,FTHG,FTAG,FTR,HS,AS\n"
            b"E0,16/08/24,Home,\xff,1,0,H,10,8\n"
        )
        with TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "invalid_encoding.csv"
            path.write_bytes(invalid_utf8)
            with self.assertRaisesRegex(CsvValidationError, "valid UTF-8"):
                validate_csv(path)


class FootballDataDownloadTests(unittest.TestCase):
    """Verify atomic validation, manifest integrity, and cache idempotency."""

    def setUp(self) -> None:
        self.season = Season(label="2024/25", code="2425")
        self.payload = (FIXTURES / "valid_e0.csv").read_bytes()
        self.fixed_time = datetime(2026, 8, 29, 1, 2, 3, tzinfo=timezone.utc)

    def _opener(self, calls: list[str]):
        payload = self.payload

        def open_response(request, *, timeout):
            calls.append(request.full_url)
            self.assertGreater(timeout, 0)
            return BytesIO(payload)

        return open_response

    def test_idempotent_rerun_does_not_request_or_modify_raw_file(self) -> None:
        calls: list[str] = []
        with TemporaryDirectory() as temporary_directory:
            project_root = Path(temporary_directory)
            first = download_season(
                self.season,
                project_root=project_root,
                opener=self._opener(calls),
                clock=lambda: self.fixed_time,
            )
            raw_bytes = first.path.read_bytes()
            manifest_bytes = (
                project_root / "data" / "raw" / "manifest.json"
            ).read_bytes()

            second = download_season(
                self.season,
                project_root=project_root,
                opener=self._opener(calls),
                clock=lambda: self.fixed_time,
            )

            self.assertEqual(first.status, "downloaded")
            self.assertEqual(second.status, "cached")
            self.assertEqual(len(calls), 1)
            self.assertEqual(second.path.read_bytes(), raw_bytes)
            self.assertEqual(
                (project_root / "data" / "raw" / "manifest.json").read_bytes(),
                manifest_bytes,
            )
            self.assertEqual(len(load_manifest(project_root / "data/raw/manifest.json")), 1)
            self.assertEqual(verify_football_data_manifest(project_root), 1)

    def test_manifest_checksum_verification_detects_mutation(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            project_root = Path(temporary_directory)
            result = download_season(
                self.season,
                project_root=project_root,
                opener=self._opener([]),
                clock=lambda: self.fixed_time,
            )
            result.path.write_bytes(result.path.read_bytes() + b"\n")

            with self.assertRaisesRegex(ManifestIntegrityError, "checksum"):
                verify_football_data_manifest(project_root)

    def test_invalid_download_never_becomes_a_raw_file(self) -> None:
        invalid_payload = (FIXTURES / "missing_columns.csv").read_bytes()

        def invalid_opener(request, *, timeout):
            return BytesIO(invalid_payload)

        with TemporaryDirectory() as temporary_directory:
            project_root = Path(temporary_directory)
            with self.assertRaises(MissingColumnsError):
                download_season(
                    self.season,
                    project_root=project_root,
                    opener=invalid_opener,
                    clock=lambda: self.fixed_time,
                )

            expected_raw_file = (
                project_root / "data" / "raw" / "football_data" / "E0_2425.csv"
            )
            self.assertFalse(expected_raw_file.exists())
            self.assertFalse(
                (project_root / "data" / "raw" / "manifest.json").exists()
            )
            self.assertEqual(
                list(expected_raw_file.parent.glob("*.tmp")),
                [],
            )

    def test_http_failure_is_actionable_and_leaves_no_raw_file(self) -> None:
        def failing_opener(request, *, timeout):
            raise HTTPError(request.full_url, 503, "Unavailable", {}, None)

        with TemporaryDirectory() as temporary_directory:
            project_root = Path(temporary_directory)
            with self.assertRaisesRegex(DownloadError, "HTTP 503"):
                download_season(
                    self.season,
                    project_root=project_root,
                    opener=failing_opener,
                    clock=lambda: self.fixed_time,
                )

            raw_directory = project_root / "data" / "raw" / "football_data"
            self.assertEqual(list(raw_directory.glob("E0_*.csv")), [])
            self.assertEqual(list(raw_directory.glob("*.tmp")), [])

    def test_unmanifested_raw_file_is_never_overwritten(self) -> None:
        def unexpected_opener(request, *, timeout):
            self.fail("The downloader must not request an unmanifested existing file.")

        with TemporaryDirectory() as temporary_directory:
            project_root = Path(temporary_directory)
            raw_path = (
                project_root / "data" / "raw" / "football_data" / "E0_2425.csv"
            )
            raw_path.parent.mkdir(parents=True)
            raw_path.write_bytes(self.payload)

            with self.assertRaisesRegex(
                ManifestIntegrityError, "without a manifest record"
            ):
                download_season(
                    self.season,
                    project_root=project_root,
                    opener=unexpected_opener,
                    clock=lambda: self.fixed_time,
                )
            self.assertEqual(raw_path.read_bytes(), self.payload)

    def test_manifest_is_deterministically_sorted(self) -> None:
        calls: list[str] = []
        with TemporaryDirectory() as temporary_directory:
            project_root = Path(temporary_directory)
            later = Season(label="2024/25", code="2425")
            earlier = Season(label="2023/24", code="2324")
            for season in (later, earlier):
                download_season(
                    season,
                    project_root=project_root,
                    opener=self._opener(calls),
                    clock=lambda: self.fixed_time,
                )

            manifest = json.loads(
                (project_root / "data" / "raw" / "manifest.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(
                [record["season"] for record in manifest], ["2324", "2425"]
            )


class FootballDataRepositoryIntegrityTests(unittest.TestCase):
    """Validate every configured raw season stored in the repository."""

    def test_all_configured_seasons_match_manifest_and_source_contract(self) -> None:
        seasons = load_seasons()
        manifest = load_manifest(PROJECT_ROOT / "data" / "raw" / "manifest.json")
        football_data_records = {
            record["season"]: record
            for record in manifest
            if record["source"] == "football-data"
        }
        configured_codes = {season.code for season in seasons}

        self.assertEqual(set(football_data_records), configured_codes)
        self.assertEqual(verify_football_data_manifest(PROJECT_ROOT), len(seasons))

        raw_directory = PROJECT_ROOT / "data" / "raw" / "football_data"
        expected_paths = {
            raw_directory / f"E0_{season.code}.csv" for season in seasons
        }
        self.assertEqual(set(raw_directory.glob("E0_*.csv")), expected_paths)

        for season in seasons:
            with self.subTest(season=season.code):
                raw_path = raw_directory / f"E0_{season.code}.csv"
                frame = pd.read_csv(raw_path)
                match_rows = frame.dropna(how="all")
                self.assertFalse(match_rows.empty)
                self.assertTrue(
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
                    }.issubset(frame.columns)
                )
                self.assertTrue(match_rows["Div"].notna().all())
                self.assertEqual(set(match_rows["Div"].unique()), {"E0"})
                self.assertEqual(
                    len(match_rows), football_data_records[season.code]["row_count"]
                )


if __name__ == "__main__":
    unittest.main()
