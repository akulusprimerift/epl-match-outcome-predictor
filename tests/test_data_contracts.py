"""Tests for Phase 2 cleaning and canonical match contracts."""

from datetime import date
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import pandas as pd

from src.clean_data import (
    CANONICAL_MATCH_COLUMNS,
    DateValidationError,
    DuplicateMatchError,
    KickoffTimeValidationError,
    NumericValidationError,
    ResultValidationError,
    TeamMappingError,
    deduplicate_canonical_matches,
    expected_result_code,
    generate_match_id,
    load_team_name_map,
    parse_match_date,
    parse_nonnegative_integer_series,
    parse_optional_kickoff_time,
    resolve_team_name,
    validate_canonical_table,
    validate_result_code,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def canonical_row(**overrides):
    """Return a minimal valid canonical row for duplicate tests."""
    row = {
        "match_id": "2425|2024-08-17|arsenal|chelsea",
        "season": "2425",
        "date": "2024-08-17",
        "kickoff_time": "15:00",
        "home_team": "Arsenal",
        "away_team": "Chelsea",
        "home_team_slug": "arsenal",
        "away_team_slug": "chelsea",
        "home_goals": 2,
        "away_goals": 1,
        "result_code": "H",
        "home_shots": 12,
        "away_shots": 8,
        "home_possession": pd.NA,
        "away_possession": pd.NA,
        "football_data_source_file": "data/raw/football_data/E0_2425.csv",
        "api_fixture_id": pd.NA,
    }
    row.update(overrides)
    return row


class DateAndNumericContractTests(unittest.TestCase):
    """Verify explicit parsing and nonnegative integer rules."""

    def test_date_parsing_across_source_formats(self) -> None:
        self.assertEqual(parse_match_date("14/08/10"), date(2010, 8, 14))
        self.assertEqual(parse_match_date("08/08/2015"), date(2015, 8, 8))

    def test_ambiguous_or_iso_source_date_is_rejected(self) -> None:
        with self.assertRaises(DateValidationError):
            parse_match_date("2015-08-08")

    def test_optional_kickoff_time_is_not_fabricated(self) -> None:
        self.assertIsNone(parse_optional_kickoff_time(None))
        self.assertIsNone(parse_optional_kickoff_time(pd.NA))
        self.assertEqual(parse_optional_kickoff_time("20:00"), "20:00")

    def test_malformed_kickoff_time_is_rejected(self) -> None:
        with self.assertRaisesRegex(KickoffTimeValidationError, "expected HH:MM"):
            parse_optional_kickoff_time("25:00")

    def test_negative_goal_is_rejected(self) -> None:
        with self.assertRaisesRegex(NumericValidationError, "Negative"):
            parse_nonnegative_integer_series(
                pd.Series([-1]), "FTHG", allow_missing=False
            )

    def test_fractional_shot_is_rejected(self) -> None:
        with self.assertRaisesRegex(NumericValidationError, "Non-integer"):
            parse_nonnegative_integer_series(
                pd.Series([10.5]), "HS", allow_missing=True
            )


class ResultContractTests(unittest.TestCase):
    """Verify full-time results agree with final goals."""

    def test_expected_result_codes(self) -> None:
        self.assertEqual(expected_result_code(2, 1), "H")
        self.assertEqual(expected_result_code(1, 1), "D")
        self.assertEqual(expected_result_code(0, 3), "A")

    def test_result_code_matches_score(self) -> None:
        self.assertEqual(validate_result_code(2, 1, "H"), "H")
        with self.assertRaisesRegex(ResultValidationError, "disagrees"):
            validate_result_code(2, 1, "A")

    def test_invalid_result_code_is_rejected(self) -> None:
        with self.assertRaisesRegex(ResultValidationError, "Invalid"):
            validate_result_code(1, 1, "X")


class TeamMappingContractTests(unittest.TestCase):
    """Verify exact team normalization and unknown-name failure."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.mappings = load_team_name_map()

    def test_provider_team_name_maps_to_canonical_identity(self) -> None:
        identity = resolve_team_name("football_data", "Man United", self.mappings)
        self.assertEqual(identity.name, "Manchester United")
        self.assertEqual(identity.slug, "manchester-united")

        api_identity = resolve_team_name(
            "api_football", "Manchester United", self.mappings
        )
        self.assertEqual(api_identity.name, "Manchester United")
        self.assertEqual(api_identity.slug, "manchester-united")

    def test_unknown_team_mapping_is_rejected(self) -> None:
        with self.assertRaisesRegex(TeamMappingError, "Unmapped team name"):
            resolve_team_name("football_data", "Unknown FC", self.mappings)

    def test_conflicting_mapping_file_is_rejected(self) -> None:
        content = (
            "provider,provider_team_name,canonical_team_name,canonical_team_slug\n"
            "football_data,Team One,Team One,shared-slug\n"
            "football_data,Team Two,Team Two,shared-slug\n"
        )
        with TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "team_name_map.csv"
            path.write_text(content, encoding="utf-8")
            with self.assertRaisesRegex(TeamMappingError, "multiple teams"):
                load_team_name_map(path)


class MatchIdentityAndDuplicateTests(unittest.TestCase):
    """Verify deterministic IDs and duplicate conflict behavior."""

    def test_match_id_is_deterministic(self) -> None:
        expected = "2425|2024-08-17|arsenal|wolverhampton-wanderers"
        self.assertEqual(
            generate_match_id(
                "2425",
                date(2024, 8, 17),
                "arsenal",
                "wolverhampton-wanderers",
            ),
            expected,
        )
        self.assertEqual(
            generate_match_id(
                "2425",
                "2024-08-17",
                "arsenal",
                "wolverhampton-wanderers",
            ),
            expected,
        )

    def test_exact_duplicate_is_counted_and_removed(self) -> None:
        frame = pd.DataFrame(
            [canonical_row(), canonical_row()], columns=CANONICAL_MATCH_COLUMNS
        )
        deduplicated, duplicate_count = deduplicate_canonical_matches(frame)
        self.assertEqual(duplicate_count, 1)
        self.assertEqual(len(deduplicated), 1)

    def test_conflicting_duplicate_is_rejected(self) -> None:
        frame = pd.DataFrame(
            [canonical_row(), canonical_row(home_goals=3)],
            columns=CANONICAL_MATCH_COLUMNS,
        )
        with self.assertRaisesRegex(DuplicateMatchError, "Conflicting"):
            deduplicate_canonical_matches(frame)

    def test_fractional_canonical_goal_is_rejected(self) -> None:
        frame = pd.DataFrame(
            [canonical_row(home_goals=1.5)], columns=CANONICAL_MATCH_COLUMNS
        )
        with self.assertRaisesRegex(NumericValidationError, "Non-integer"):
            validate_canonical_table(frame)


class CanonicalRepositoryContractTests(unittest.TestCase):
    """Validate the generated repository-level canonical match table."""

    def test_canonical_matches_satisfy_phase_two_contract(self) -> None:
        path = PROJECT_ROOT / "data" / "processed" / "canonical_matches.csv"
        self.assertTrue(path.is_file(), f"Missing canonical output: {path}")
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
        )
        validate_canonical_table(frame)
        self.assertEqual(len(frame), 6080)
        self.assertTrue(frame["match_id"].is_unique)
        self.assertEqual(frame.groupby("season").size().to_dict(), {
            "1011": 380,
            "1112": 380,
            "1213": 380,
            "1314": 380,
            "1415": 380,
            "1516": 380,
            "1617": 380,
            "1718": 380,
            "1819": 380,
            "1920": 380,
            "2021": 380,
            "2122": 380,
            "2223": 380,
            "2324": 380,
            "2425": 380,
            "2526": 380,
        })
        for column in ("home_possession", "away_possession"):
            provided = frame[column].dropna()
            self.assertTrue(provided.between(0, 100).all())
        possession_provided = frame["home_possession"].notna() | frame[
            "away_possession"
        ].notna()
        self.assertTrue(frame.loc[possession_provided, "api_fixture_id"].notna().all())
        self.assertTrue(frame["api_fixture_id"].dropna().is_unique)

        mappings = load_team_name_map()
        canonical_identities = {
            (identity.name, identity.slug)
            for (provider, _), identity in mappings.items()
            if provider == "football_data"
        }
        observed_identities = set(
            zip(frame["home_team"], frame["home_team_slug"])
        ) | set(zip(frame["away_team"], frame["away_team_slug"]))
        self.assertTrue(observed_identities.issubset(canonical_identities))


if __name__ == "__main__":
    unittest.main()
