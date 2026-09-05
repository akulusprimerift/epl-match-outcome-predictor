"""Regression tests for legacy API-Football cache and join compatibility."""

from __future__ import annotations

from datetime import datetime, timezone
from io import BytesIO
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from urllib.parse import parse_qs, urlparse

import pandas as pd

from src.clean_data import (
    CANONICAL_MATCH_COLUMNS,
    POSSESSION_COVERAGE_COLUMNS,
    POSSESSION_JOIN_REPORT_COLUMNS,
    TeamMappingError,
    build_possession_coverage,
    generate_match_id,
    join_possession_fixtures,
    load_team_name_map,
    validate_canonical_table,
)
from src.api_football_legacy import (
    API_KEY_ENVIRONMENT_VARIABLE,
    ApiConfigurationError,
    ApiFixture,
    ApiPossessionFixture,
    PossessionValueError,
    collect_season,
    load_cached_possession_fixtures,
    parse_fixture_statistics,
    parse_possession,
)
from src.download_data import load_manifest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
FIXED_TIME = datetime(2026, 9, 1, 12, 0, tzinfo=timezone.utc)


def api_fixture_item(
    fixture_id: int,
    date: str,
    home_id: int,
    home_name: str,
    away_id: int,
    away_name: str,
) -> dict[str, object]:
    """Build one realistic completed EPL API fixture response item."""
    return {
        "fixture": {
            "id": fixture_id,
            "date": f"{date}T14:00:00+00:00",
            "status": {"short": "FT"},
        },
        "league": {"id": 39, "season": 2024},
        "teams": {
            "home": {"id": home_id, "name": home_name},
            "away": {"id": away_id, "name": away_name},
        },
    }


def statistics_payload(
    home_id: int,
    home_value: object,
    away_id: int,
    away_value: object,
) -> dict[str, object]:
    """Build reversed-order statistics to prove team-ID based extraction."""
    return {
        "response": [
            {
                "team": {"id": away_id},
                "statistics": [
                    {"type": "Ball Possession", "value": away_value}
                ],
            },
            {
                "team": {"id": home_id},
                "statistics": [
                    {"type": "Ball Possession", "value": home_value}
                ],
            },
        ],
        "errors": {},
    }


class FakeResponse(BytesIO):
    """Minimal urllib-compatible response for deterministic collection tests."""

    def __init__(self, payload: bytes, *, remaining: int = 100) -> None:
        super().__init__(payload)
        self.status = 200
        self.headers = {
            "x-ratelimit-requests-remaining": str(remaining),
        }


class PossessionParsingTests(unittest.TestCase):
    """Verify percentages and missing values retain their exact meaning."""

    def test_percentage_and_null_parsing(self) -> None:
        self.assertEqual(parse_possession("56%"), 56.0)
        self.assertEqual(parse_possession(" 56.5% "), 56.5)
        self.assertEqual(parse_possession(42), 42.0)
        self.assertIsNone(parse_possession(None))

    def test_malformed_or_out_of_range_percentage_is_rejected(self) -> None:
        for value in ("56", "unknown%", -1, 101, True):
            with self.subTest(value=value):
                with self.assertRaises(PossessionValueError):
                    parse_possession(value)

    def test_home_and_away_are_not_forced_to_sum_to_100(self) -> None:
        fixture = ApiFixture(
            fixture_id=1001,
            season_year=2024,
            date="2024-08-17",
            home_team_id=42,
            home_team_name="Arsenal",
            away_team_id=49,
            away_team_name="Chelsea",
        )
        parsed = parse_fixture_statistics(
            statistics_payload(42, "56%", 49, "41%"), fixture
        )
        self.assertEqual(parsed.home_possession, 56.0)
        self.assertEqual(parsed.away_possession, 41.0)


class ResumableCollectorTests(unittest.TestCase):
    """Verify bounded requests, immutable caches, resume, and secret handling."""

    def setUp(self) -> None:
        self.fixture_listing = {
            "response": [
                api_fixture_item(
                    1001,
                    "2024-08-17",
                    42,
                    "Arsenal",
                    49,
                    "Chelsea",
                ),
                api_fixture_item(
                    1002,
                    "2024-08-18",
                    49,
                    "Chelsea",
                    42,
                    "Arsenal",
                ),
            ],
            "errors": {},
        }
        self.statistics = {
            1001: statistics_payload(42, "56%", 49, "41%"),
            1002: statistics_payload(49, None, 42, "49%"),
        }

    @staticmethod
    def _prepare_project(project_root: Path) -> None:
        config_directory = project_root / "config"
        config_directory.mkdir(parents=True)
        (config_directory / "seasons.json").write_text(
            '[{"label": "2024/25", "code": "2425"}]\n',
            encoding="utf-8",
        )

    def _opener(self, calls: list[tuple[str, str | None]]):
        listing = self.fixture_listing
        statistics = self.statistics

        def open_response(request, *, timeout):
            self.assertGreater(timeout, 0)
            key = request.get_header("X-apisports-key")
            calls.append((request.full_url, key))
            parsed = urlparse(request.full_url)
            query = parse_qs(parsed.query)
            if parsed.path.endswith("/fixtures/statistics"):
                fixture_id = int(query["fixture"][0])
                payload = statistics[fixture_id]
            else:
                self.assertEqual(query["league"], ["39"])
                self.assertEqual(query["season"], ["2024"])
                self.assertEqual(query["status"], ["FT"])
                payload = listing
            raw_bytes = json.dumps(
                payload, separators=(",", ":"), sort_keys=False
            ).encode("utf-8")
            return FakeResponse(raw_bytes)

        return open_response

    def test_quota_stop_resume_cache_skip_and_manifest_provenance(self) -> None:
        calls: list[tuple[str, str | None]] = []
        secret = "unit-test-secret-never-persist"
        with TemporaryDirectory() as temporary_directory:
            project_root = Path(temporary_directory)
            self._prepare_project(project_root)
            opener = self._opener(calls)

            first = collect_season(
                2024,
                2,
                project_root=project_root,
                api_key=secret,
                opener=opener,
                clock=lambda: FIXED_TIME,
            )
            self.assertEqual(first.requests_made, 2)
            self.assertEqual(first.statistics_downloaded, 1)
            self.assertTrue(first.stopped_before_quota)

            listing_path = (
                project_root / "data/raw/api_football/fixtures_2024.json"
            )
            listing_bytes = listing_path.read_bytes()
            second = collect_season(
                2024,
                2,
                project_root=project_root,
                api_key=secret,
                opener=opener,
                clock=lambda: FIXED_TIME,
            )
            self.assertEqual(second.requests_made, 1)
            self.assertEqual(second.statistics_cached, 1)
            self.assertEqual(second.statistics_downloaded, 1)
            self.assertFalse(second.stopped_before_quota)
            self.assertEqual(listing_path.read_bytes(), listing_bytes)

            calls_before_cached_run = len(calls)
            third = collect_season(
                2024,
                1,
                project_root=project_root,
                api_key=None,
                opener=opener,
                clock=lambda: FIXED_TIME,
            )
            self.assertEqual(third.requests_made, 0)
            self.assertEqual(third.statistics_cached, 2)
            self.assertEqual(len(calls), calls_before_cached_run)

            records = load_manifest(
                project_root / "data/raw/manifest.json"
            )
            self.assertEqual(len(records), 3)
            self.assertTrue(
                all(
                    record["source"] == "api-football"
                    and "endpoint" in record
                    and "fixture_id" in record
                    for record in records
                )
            )
            persisted = (
                project_root / "data/raw/manifest.json"
            ).read_text(encoding="utf-8")
            for path in (project_root / "data/raw/api_football").glob("*.json"):
                persisted += path.read_text(encoding="utf-8")
            self.assertNotIn(secret, persisted)
            self.assertTrue(all(call_key == secret for _, call_key in calls))

            fixtures = load_cached_possession_fixtures(project_root)
            self.assertEqual(len(fixtures), 2)
            self.assertEqual(fixtures[0].home_possession, 56.0)
            self.assertEqual(fixtures[0].away_possession, 41.0)
            self.assertIsNone(fixtures[1].home_possession)
            self.assertEqual(fixtures[1].away_possession, 49.0)

    def test_missing_key_fails_before_any_network_request(self) -> None:
        calls: list[tuple[str, str | None]] = []
        with TemporaryDirectory() as temporary_directory:
            project_root = Path(temporary_directory)
            self._prepare_project(project_root)
            with self.assertRaisesRegex(
                ApiConfigurationError, API_KEY_ENVIRONMENT_VARIABLE
            ):
                collect_season(
                    2024,
                    1,
                    project_root=project_root,
                    api_key=None,
                    opener=self._opener(calls),
                )
        self.assertEqual(calls, [])

    def test_server_remaining_header_stops_before_another_request(self) -> None:
        calls: list[str] = []
        raw_listing = json.dumps(self.fixture_listing).encode("utf-8")

        def quota_opener(request, *, timeout):
            calls.append(request.full_url)
            return FakeResponse(raw_listing, remaining=0)

        with TemporaryDirectory() as temporary_directory:
            project_root = Path(temporary_directory)
            self._prepare_project(project_root)
            summary = collect_season(
                2024,
                90,
                project_root=project_root,
                api_key="test-key",
                opener=quota_opener,
                clock=lambda: FIXED_TIME,
            )
        self.assertEqual(summary.requests_made, 1)
        self.assertEqual(summary.statistics_downloaded, 0)
        self.assertTrue(summary.stopped_before_quota)
        self.assertEqual(len(calls), 1)


def canonical_row(
    season: str,
    date: str,
    home_name: str,
    home_slug: str,
    away_name: str,
    away_slug: str,
) -> dict[str, object]:
    """Build a valid canonical row for exact-join tests."""
    return {
        "match_id": generate_match_id(season, date, home_slug, away_slug),
        "season": season,
        "date": date,
        "kickoff_time": "15:00",
        "home_team": home_name,
        "away_team": away_name,
        "home_team_slug": home_slug,
        "away_team_slug": away_slug,
        "home_goals": 1,
        "away_goals": 0,
        "result_code": "H",
        "home_shots": 10,
        "away_shots": 8,
        "home_possession": pd.NA,
        "away_possession": pd.NA,
        "football_data_source_file": f"data/raw/football_data/E0_{season}.csv",
        "api_fixture_id": pd.NA,
    }


def possession_fixture(
    fixture_id: int,
    date: str,
    home_id: int,
    home_name: str,
    away_id: int,
    away_name: str,
    home_possession: float | None,
    away_possession: float | None,
) -> ApiPossessionFixture:
    return ApiPossessionFixture(
        fixture_id=fixture_id,
        season_year=2024,
        date=date,
        home_team_id=home_id,
        home_team_name=home_name,
        away_team_id=away_id,
        away_team_name=away_name,
        home_possession=home_possession,
        away_possession=away_possession,
    )


class PossessionJoinTests(unittest.TestCase):
    """Verify exact one-to-one joins and explicit exception reports."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.mappings = load_team_name_map()

    def setUp(self) -> None:
        self.canonical = pd.DataFrame(
            [
                canonical_row(
                    "2425",
                    "2024-08-17",
                    "Arsenal",
                    "arsenal",
                    "Chelsea",
                    "chelsea",
                ),
                canonical_row(
                    "2425",
                    "2024-08-18",
                    "Chelsea",
                    "chelsea",
                    "Arsenal",
                    "arsenal",
                ),
                canonical_row(
                    "2425",
                    "2024-08-19",
                    "Liverpool",
                    "liverpool",
                    "Everton",
                    "everton",
                ),
            ],
            columns=CANONICAL_MATCH_COLUMNS,
        )

    def test_unique_join_and_explicit_unmatched_and_ambiguous_rows(self) -> None:
        fixtures = [
            possession_fixture(
                1001,
                "2024-08-17",
                42,
                "Arsenal",
                49,
                "Chelsea",
                56.0,
                44.0,
            ),
            possession_fixture(
                1002,
                "2024-08-20",
                42,
                "Arsenal",
                45,
                "Everton",
                51.0,
                49.0,
            ),
            possession_fixture(
                1003,
                "2024-08-18",
                49,
                "Chelsea",
                42,
                "Arsenal",
                48.0,
                52.0,
            ),
            possession_fixture(
                1004,
                "2024-08-18",
                49,
                "Chelsea",
                42,
                "Arsenal",
                47.0,
                53.0,
            ),
        ]
        result = join_possession_fixtures(
            self.canonical, fixtures, self.mappings
        )
        self.assertEqual(result.matched_count, 1)
        self.assertEqual(len(result.unmatched), 1)
        self.assertEqual(len(result.ambiguous), 2)
        self.assertEqual(
            tuple(result.unmatched.columns), POSSESSION_JOIN_REPORT_COLUMNS
        )
        self.assertEqual(
            tuple(result.ambiguous.columns), POSSESSION_JOIN_REPORT_COLUMNS
        )
        self.assertEqual(
            set(result.ambiguous["reason"]), {"duplicate_api_join_key"}
        )
        self.assertEqual(result.frame.loc[0, "api_fixture_id"], 1001)
        self.assertEqual(result.frame.loc[0, "home_possession"], 56.0)
        self.assertEqual(result.frame.loc[0, "away_possession"], 44.0)
        self.assertTrue(result.frame.loc[1, "home_possession"] is pd.NA)
        self.assertEqual(result.frame["api_fixture_id"].dropna().nunique(), 1)
        validate_canonical_table(result.frame)

    def test_unknown_api_team_requires_an_explicit_mapping(self) -> None:
        unknown = possession_fixture(
            1005,
            "2024-08-17",
            999,
            "Mystery FC",
            49,
            "Chelsea",
            50.0,
            50.0,
        )
        with self.assertRaisesRegex(TeamMappingError, "Mystery FC"):
            join_possession_fixtures(
                self.canonical, [unknown], self.mappings
            )

    def test_missing_possession_remains_missing_after_a_valid_join(self) -> None:
        fixture = possession_fixture(
            1006,
            "2024-08-17",
            42,
            "Arsenal",
            49,
            "Chelsea",
            None,
            44.0,
        )
        result = join_possession_fixtures(
            self.canonical, [fixture], self.mappings
        )
        self.assertTrue(pd.isna(result.frame.loc[0, "home_possession"]))
        self.assertEqual(result.frame["home_possession"].fillna(-1).iloc[0], -1)
        self.assertEqual(result.frame.loc[0, "away_possession"], 44.0)


class PossessionCoverageTests(unittest.TestCase):
    """Verify the report declares the first season meeting the 95% rule."""

    def test_coverage_is_reported_by_season_and_team(self) -> None:
        frame = pd.DataFrame(
            [
                canonical_row(
                    "2324",
                    "2023-08-12",
                    "Arsenal",
                    "arsenal",
                    "Chelsea",
                    "chelsea",
                ),
                canonical_row(
                    "2425",
                    "2024-08-17",
                    "Arsenal",
                    "arsenal",
                    "Chelsea",
                    "chelsea",
                ),
            ],
            columns=CANONICAL_MATCH_COLUMNS,
        )
        frame.loc[0, ["home_possession", "away_possession"]] = [55.0, 45.0]
        coverage = build_possession_coverage(frame, 0.95)
        self.assertEqual(
            tuple(coverage.frame.columns), POSSESSION_COVERAGE_COLUMNS
        )
        self.assertEqual(coverage.model_b_period_start, "2324")
        season_rows = coverage.frame.loc[coverage.frame["scope"].eq("season")]
        self.assertEqual(season_rows["coverage"].tolist(), [1.0, 0.0])
        self.assertEqual(
            set(coverage.frame.loc[coverage.frame["scope"].eq("team"), "team"]),
            {"Arsenal", "Chelsea"},
        )
        self.assertTrue(
            coverage.frame["model_b_period_start"].eq("2324").all()
        )


if __name__ == "__main__":
    unittest.main()
