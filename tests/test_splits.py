"""Phase 4 frozen season split and manifest tests."""

from pathlib import Path
import unittest

import pandas as pd

from src.constants import (
    HOLDOUT_SEASONS,
    SPLIT_MANIFEST_COLUMNS,
    SPLIT_ORDER,
    TEST_SEASONS,
    TRAIN_SEASONS,
    VALIDATION_SEASONS,
)
from src.split_data import (
    SPLIT_MANIFEST_PATH,
    build_split_manifest,
    load_split_policy,
    read_model_dataset,
    split_model_dataset,
    validate_dataset_splits,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class FrozenSplitTests(unittest.TestCase):
    """Verify exact memberships, disjoint IDs, and temporal ordering."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.policy = load_split_policy()
        cls.model = read_model_dataset()
        cls.splits = split_model_dataset(cls.model, cls.policy)

    def test_season_policy_is_exact(self) -> None:
        self.assertEqual(self.policy.train, TRAIN_SEASONS)
        self.assertEqual(self.policy.validation, VALIDATION_SEASONS)
        self.assertEqual(self.policy.test, TEST_SEASONS)
        self.assertEqual(self.policy.holdout, HOLDOUT_SEASONS)
        self.assertEqual(
            self.policy.by_name(),
            {
                "train": TRAIN_SEASONS,
                "validation": VALIDATION_SEASONS,
                "test": TEST_SEASONS,
                "holdout": HOLDOUT_SEASONS,
            },
        )

    def test_split_counts_and_seasons_match_policy(self) -> None:
        expected_counts = {
            "train": 4_940,
            "validation": 380,
            "test": 380,
            "holdout": 380,
        }
        for split_name, frame in self.splits.by_name().items():
            with self.subTest(split=split_name):
                self.assertEqual(len(frame), expected_counts[split_name])
                self.assertEqual(
                    set(frame["season"].astype(str)),
                    set(self.policy.by_name()[split_name]),
                )

    def test_match_ids_are_disjoint_and_complete(self) -> None:
        split_id_sets = {
            name: set(frame["match_id"])
            for name, frame in self.splits.by_name().items()
        }
        for index, left_name in enumerate(SPLIT_ORDER):
            for right_name in SPLIT_ORDER[index + 1 :]:
                with self.subTest(left=left_name, right=right_name):
                    self.assertTrue(
                        split_id_sets[left_name].isdisjoint(split_id_sets[right_name])
                    )
        combined = set().union(*split_id_sets.values())
        self.assertEqual(combined, set(self.model["match_id"]))
        self.assertEqual(sum(map(len, split_id_sets.values())), len(self.model))

    def test_splits_are_strictly_chronological(self) -> None:
        validate_dataset_splits(self.splits, self.model, self.policy)
        frames = self.splits.by_name()
        for earlier_name, later_name in zip(SPLIT_ORDER, SPLIT_ORDER[1:]):
            with self.subTest(earlier=earlier_name, later=later_name):
                earlier_max = pd.to_datetime(frames[earlier_name]["date"]).max()
                later_min = pd.to_datetime(frames[later_name]["date"]).min()
                self.assertLess(earlier_max, later_min)

    def test_persisted_manifest_matches_rebuilt_assignments(self) -> None:
        path = PROJECT_ROOT / SPLIT_MANIFEST_PATH.relative_to(PROJECT_ROOT)
        self.assertTrue(path.is_file(), f"Missing split manifest: {path}")
        persisted = pd.read_csv(
            path,
            dtype={"match_id": "string", "season": "string", "date": "string"},
        )
        expected = build_split_manifest(self.splits)
        self.assertEqual(tuple(persisted.columns), SPLIT_MANIFEST_COLUMNS)
        pd.testing.assert_frame_equal(persisted, expected)


if __name__ == "__main__":
    unittest.main()
