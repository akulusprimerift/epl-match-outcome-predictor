"""Phase 6 SofaScore team-season possession collection tests."""

from __future__ import annotations

from io import BytesIO
import csv
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from urllib.error import HTTPError

from src.collect_possession import (
    AGGREGATE_EXPORT_COLUMNS,
    AGGREGATE_EXPORT_ENDPOINT,
    AGGREGATE_EXPORT_FILENAME,
    POSSESSION_COVERAGE_COLUMNS,
    TEAM_SEASON_POSSESSION_COLUMNS,
    TeamMappingError,
    build_processed_possession,
    collect_possession_averages,
    parse_average_possession,
    parse_season_start_year,
    parse_seasons,
    parse_standing_teams,
    parse_team_statistics,
    season_code_from_start_year,
)


class FakeResponse(BytesIO):
    status = 200
    headers: dict[str, str] = {}

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
        return False


def json_response(value: object) -> FakeResponse:
    return FakeResponse(json.dumps(value).encode("utf-8"))


class SofaScoreParsingTests(unittest.TestCase):
    def test_season_formats_and_codes(self) -> None:
        self.assertEqual(parse_season_start_year({"year": "17/18"}), 2017)
        self.assertEqual(parse_season_start_year({"name": "Premier League 2017-18"}), 2017)
        self.assertEqual(season_code_from_start_year(2017), "1718")
        self.assertEqual(season_code_from_start_year(2025), "2526")

    def test_season_directory_and_standings_are_validated(self) -> None:
        seasons = parse_seasons(
            {"seasons": [{"id": 13380, "name": "Premier League 17/18", "year": "17/18"}]}
        )
        self.assertEqual((seasons[0].start_year, seasons[0].season_code), (2017, "1718"))
        teams = parse_standing_teams(
            {
                "standings": [
                    {
                        "rows": [
                            {"team": {"id": 1, "name": "Alpha FC"}},
                            {"team": {"id": 2, "name": "Beta FC"}},
                        ]
                    }
                ]
            }
        )
        self.assertEqual([(team.team_id, team.name) for team in teams], [(1, "Alpha FC"), (2, "Beta FC")])

    def test_average_possession_and_missingness(self) -> None:
        self.assertEqual(parse_average_possession(56.25), 56.25)
        self.assertEqual(parse_average_possession("51.5%"), 51.5)
        self.assertIsNone(parse_average_possession(None))
        self.assertEqual(
            parse_team_statistics(
                {"statistics": {"averageBallPossession": 56.25, "matches": 38}}
            ),
            (56.25, 38),
        )
        for invalid in (-1, 101, float("inf"), True, "not-a-number"):
            with self.subTest(invalid=invalid):
                with self.assertRaisesRegex(RuntimeError, "average possession|Average possession"):
                    parse_average_possession(invalid)


class SofaScoreCollectorTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.project_root = Path(self.temporary.name)
        (self.project_root / "config").mkdir(parents=True)
        (self.project_root / "data" / "raw").mkdir(parents=True)
        (self.project_root / "data" / "processed").mkdir(parents=True)
        (self.project_root / "reports").mkdir(parents=True)
        (self.project_root / "config" / "seasons.json").write_text(
            json.dumps([{"label": "2017/18", "code": "1718"}]),
            encoding="utf-8",
        )
        (self.project_root / "config" / "model_config.json").write_text(
            json.dumps({"possession_coverage_threshold": 0.95}),
            encoding="utf-8",
        )
        self._write_mapping(include_beta=True)

        self.payloads = {
            "/unique-tournament/17/seasons": {
                "seasons": [
                    {"id": 13380, "name": "Premier League 17/18", "year": "17/18"}
                ]
            },
            "/unique-tournament/17/season/13380/standings/total": {
                "standings": [
                    {
                        "rows": [
                            {"team": {"id": 1, "name": "Alpha FC"}},
                            {"team": {"id": 2, "name": "Beta FC"}},
                        ]
                    }
                ]
            },
            "/team/1/unique-tournament/17/season/13380/statistics/overall": {
                "statistics": {"averageBallPossession": 55.0, "matches": 38}
            },
            "/team/2/unique-tournament/17/season/13380/statistics/overall": {
                "statistics": {"averageBallPossession": None, "matches": 38}
            },
        }
        self.requested: list[str] = []

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def _write_mapping(self, *, include_beta: bool) -> None:
        rows = [
            ["sofascore", "Alpha FC", "Alpha", "alpha"],
        ]
        if include_beta:
            rows.append(["sofascore", "Beta FC", "Beta", "beta"])
        path = self.project_root / "config" / "team_name_map.csv"
        with path.open("w", encoding="utf-8", newline="") as output:
            writer = csv.writer(output, lineterminator="\n")
            writer.writerow(
                ["provider", "provider_team_name", "canonical_team_name", "canonical_team_slug"]
            )
            writer.writerows(rows)

    def opener(self, request, timeout=30):
        del timeout
        url = request.full_url
        self.requested.append(url)
        path = url.split("/api/v1", 1)[1]
        if path not in self.payloads:
            raise AssertionError(f"Unexpected URL: {url}")
        return json_response(self.payloads[path])

    def test_budget_stop_resume_cache_skip_and_leakage_safe_target(self) -> None:
        first = collect_possession_averages(
            [2017],
            3,
            project_root=self.project_root,
            opener=self.opener,
            attempts=1,
            request_delay=0,
            sleep_fn=lambda seconds: None,
        )
        self.assertTrue(first.stopped_before_budget)
        self.assertEqual(first.requests_made, 3)
        self.assertEqual(first.output_rows, 2)
        self.assertEqual(first.complete_rows, 1)

        self.requested.clear()
        second = collect_possession_averages(
            [2017],
            1,
            project_root=self.project_root,
            opener=self.opener,
            attempts=1,
            request_delay=0,
            sleep_fn=lambda seconds: None,
        )
        self.assertFalse(second.stopped_before_budget)
        self.assertEqual(second.requests_made, 1)
        self.assertEqual(len(self.requested), 1)
        self.assertEqual(second.complete_rows, 1)  # Beta's null is preserved.
        self.assertEqual(second.model_b_period_start, None)

        with second.output_path.open("r", encoding="utf-8", newline="") as source:
            rows = list(csv.DictReader(source))
        self.assertEqual(tuple(rows[0]), TEAM_SEASON_POSSESSION_COLUMNS)
        self.assertEqual({row["source_season"] for row in rows}, {"1718"})
        self.assertEqual({row["target_season"] for row in rows}, {"1819"})
        self.assertEqual(rows[1]["average_possession_pct"], "")

        with second.coverage_path.open("r", encoding="utf-8", newline="") as source:
            coverage = list(csv.DictReader(source))
        self.assertEqual(tuple(coverage[0]), POSSESSION_COVERAGE_COLUMNS)
        season_row = next(row for row in coverage if row["scope"] == "season")
        self.assertEqual(season_row["coverage"], "0.5")

        self.requested.clear()
        third = collect_possession_averages(
            [2017],
            1,
            project_root=self.project_root,
            opener=self.opener,
            attempts=1,
            request_delay=0,
            sleep_fn=lambda seconds: None,
        )
        self.assertEqual(third.requests_made, 0)
        self.assertEqual(self.requested, [])
        manifest = json.loads(
            (self.project_root / "data" / "raw" / "manifest.json").read_text(encoding="utf-8")
        )
        self.assertEqual(len(manifest), 4)
        self.assertTrue(all(record["source"] == "sofascore" for record in manifest))

    def test_unknown_team_mapping_fails_explicitly(self) -> None:
        collect_possession_averages(
            [2017],
            4,
            project_root=self.project_root,
            opener=self.opener,
            attempts=1,
            request_delay=0,
            sleep_fn=lambda seconds: None,
        )
        self._write_mapping(include_beta=False)
        with self.assertRaisesRegex(TeamMappingError, "Beta FC"):
            build_processed_possession(self.project_root)

    def test_manifested_aggregate_export_skips_network_and_builds_outputs(self) -> None:
        raw_dir = self.project_root / "data" / "raw" / "sofascore"
        raw_dir.mkdir(parents=True, exist_ok=True)
        export_path = raw_dir / AGGREGATE_EXPORT_FILENAME
        rows = [
            {
                "season": "2017/18",
                "season_id": "1",
                "team": "Alpha FC",
                "team_id": "10",
                "average_ball_possession": "55.5",
                "matches": "38",
                "source_url": (
                    "https://www.sofascore.com/api/v1/team/10/unique-tournament/17/"
                    "season/1/statistics/overall"
                ),
            },
            {
                "season": "2017/18",
                "season_id": "1",
                "team": "Beta FC",
                "team_id": "20",
                "average_ball_possession": "44.5",
                "matches": "38",
                "source_url": (
                    "https://www.sofascore.com/api/v1/team/20/unique-tournament/17/"
                    "season/1/statistics/overall"
                ),
            },
        ]
        with export_path.open("w", encoding="utf-8", newline="") as output:
            writer = csv.DictWriter(output, fieldnames=AGGREGATE_EXPORT_COLUMNS)
            writer.writeheader()
            writer.writerows(rows)

        manifest_path = self.project_root / "data" / "raw" / "manifest.json"
        manifest_path.write_text(
            json.dumps(
                [
                    {
                        "acquisition_method": "sofascore-web-statistics",
                        "endpoint": AGGREGATE_EXPORT_ENDPOINT,
                        "local_path": export_path.relative_to(self.project_root).as_posix(),
                        "retrieved_at_utc": "2026-09-05T06:25:55Z",
                        "row_count": 2,
                        "season": "1718",
                        "sha256": hashlib.sha256(export_path.read_bytes()).hexdigest(),
                        "source": "sofascore",
                        "source_url": (
                            "https://www.sofascore.com/api/v1/unique-tournament/17/seasons"
                        ),
                    }
                ]
            ),
            encoding="utf-8",
        )

        def fail_if_called(_request, timeout=30):
            del timeout
            raise AssertionError(
                "network should not be used when the aggregate export exists"
            )

        summary = collect_possession_averages(
            [2017],
            1,
            project_root=self.project_root,
            opener=fail_if_called,
            attempts=1,
            request_delay=0,
            sleep_fn=lambda seconds: None,
        )

        self.assertEqual(summary.requests_made, 0)
        self.assertEqual(summary.complete_rows, 2)
        self.assertEqual(summary.output_rows, 2)
        self.assertEqual(summary.model_b_period_start, "1819")

        with summary.output_path.open("r", encoding="utf-8", newline="") as source:
            processed = list(csv.DictReader(source))
        self.assertEqual(len(processed), 2)
        self.assertEqual(processed[0]["source_season"], "1718")
        self.assertEqual(processed[0]["team"], "Alpha")
        self.assertEqual(processed[0]["average_possession_pct"], "55.5")

    def test_forbidden_primary_host_falls_back_to_secondary_host(self) -> None:
        calls: list[str] = []

        def fallback_opener(request, timeout=30):
            del timeout
            url = request.full_url
            calls.append(url)
            if url.startswith("https://www.sofascore.com"):
                raise HTTPError(url, 403, "Forbidden", {}, None)
            path = url.split("/api/v1", 1)[1]
            return json_response(self.payloads[path])

        summary = collect_possession_averages(
            [2017],
            8,
            project_root=self.project_root,
            opener=fallback_opener,
            attempts=1,
            request_delay=0,
            sleep_fn=lambda seconds: None,
        )
        self.assertFalse(summary.stopped_before_budget)
        self.assertEqual(summary.requests_made, 8)
        self.assertEqual(len(calls), 8)
        self.assertTrue(any(url.startswith("https://api.sofascore.com") for url in calls))


if __name__ == "__main__":
    unittest.main()
