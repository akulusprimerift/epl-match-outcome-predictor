"""Create and validate the frozen chronological Phase 4 dataset splits."""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
import tempfile
from typing import Mapping

import pandas as pd

from src.build_features import FeatureError, validate_model_dataset
from src.clean_data import PROJECT_ROOT
from src.constants import (
    FEATURE_COLUMNS,
    HOLDOUT_SEASONS,
    MODEL_DATASET_COLUMNS,
    RANDOM_SEED,
    ROLLING_MIN_PERIODS,
    ROLLING_WINDOW,
    SPLIT_MANIFEST_COLUMNS,
    SPLIT_ORDER,
    TARGET_MAPPING,
    TEST_SEASONS,
    TRAIN_SEASONS,
    VALIDATION_SEASONS,
)


MODEL_CONFIG_PATH = PROJECT_ROOT / "config" / "model_config.json"
MODEL_DATASET_PATH = PROJECT_ROOT / "data" / "processed" / "model_dataset.csv"
SPLIT_MANIFEST_PATH = PROJECT_ROOT / "data" / "processed" / "split_manifest.csv"


class SplitError(RuntimeError):
    """Raised when split configuration or chronology violates the fixed policy."""


@dataclass(frozen=True)
class SplitPolicy:
    """Season membership for every frozen chronological split."""

    train: tuple[str, ...]
    validation: tuple[str, ...]
    test: tuple[str, ...]
    holdout: tuple[str, ...]

    def by_name(self) -> dict[str, tuple[str, ...]]:
        """Return split seasons in their required chronological order."""
        return {
            "train": self.train,
            "validation": self.validation,
            "test": self.test,
            "holdout": self.holdout,
        }


@dataclass(frozen=True)
class DatasetSplits:
    """The four disjoint model-table partitions."""

    train: pd.DataFrame
    validation: pd.DataFrame
    test: pd.DataFrame
    holdout: pd.DataFrame

    def by_name(self) -> dict[str, pd.DataFrame]:
        """Return split frames in their required chronological order."""
        return {
            "train": self.train,
            "validation": self.validation,
            "test": self.test,
            "holdout": self.holdout,
        }


def _load_json_object(path: Path) -> Mapping[str, object]:
    try:
        with path.open("r", encoding="utf-8") as input_file:
            value = json.load(input_file)
    except FileNotFoundError as exc:
        raise SplitError(f"Model configuration not found: {path}") from exc
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise SplitError(f"Could not read model configuration {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise SplitError(f"Model configuration {path} must contain a JSON object.")
    return value


def load_split_policy(path: Path = MODEL_CONFIG_PATH) -> SplitPolicy:
    """Load the model config and enforce every fixed Phase 4 decision."""
    config = _load_json_object(path)
    required_values = {
        "random_seed": RANDOM_SEED,
        "target_mapping": TARGET_MAPPING,
        "rolling_window": ROLLING_WINDOW,
        "rolling_min_periods": ROLLING_MIN_PERIODS,
        "train_seasons": list(TRAIN_SEASONS),
        "validation_seasons": list(VALIDATION_SEASONS),
        "test_seasons": list(TEST_SEASONS),
        "holdout_seasons": list(HOLDOUT_SEASONS),
        "feature_columns": list(FEATURE_COLUMNS),
    }
    for key, expected in required_values.items():
        if config.get(key) != expected:
            raise SplitError(
                f"Model configuration field {key!r} differs from the fixed contract."
            )

    policy = SplitPolicy(
        train=tuple(str(value) for value in config["train_seasons"]),
        validation=tuple(str(value) for value in config["validation_seasons"]),
        test=tuple(str(value) for value in config["test_seasons"]),
        holdout=tuple(str(value) for value in config["holdout_seasons"]),
    )
    season_sets = [set(seasons) for seasons in policy.by_name().values()]
    for index, left in enumerate(season_sets):
        for right in season_sets[index + 1 :]:
            if left.intersection(right):
                raise SplitError("Configured season splits overlap.")
    return policy


def read_model_dataset(path: Path = MODEL_DATASET_PATH) -> pd.DataFrame:
    """Read and validate the persisted Phase 3 model table."""
    try:
        frame = pd.read_csv(
            path,
            dtype={
                "match_id": "string",
                "season": "string",
                "date": "string",
                "home_team": "string",
                "away_team": "string",
            },
            low_memory=False,
        )
    except FileNotFoundError as exc:
        raise SplitError(
            f"Model dataset not found: {path}. "
            "Run python -m src.build_features --feature-set baseline first."
        ) from exc
    except (
        OSError,
        UnicodeDecodeError,
        pd.errors.EmptyDataError,
        pd.errors.ParserError,
    ) as exc:
        raise SplitError(f"Could not read model dataset {path}: {exc}") from exc
    if tuple(frame.columns) != MODEL_DATASET_COLUMNS:
        raise SplitError(f"Model dataset {path} has an unexpected column order.")
    try:
        validate_model_dataset(frame)
    except FeatureError as exc:
        raise SplitError(f"Model dataset failed validation: {exc}") from exc
    return frame


def split_model_dataset(
    model: pd.DataFrame,
    policy: SplitPolicy,
) -> DatasetSplits:
    """Partition the model table without shuffling or random selection."""
    try:
        validate_model_dataset(model)
    except FeatureError as exc:
        raise SplitError(f"Cannot split an invalid model dataset: {exc}") from exc

    configured_seasons = set().union(
        *(set(seasons) for seasons in policy.by_name().values())
    )
    observed_seasons = set(model["season"].astype(str))
    unknown_seasons = observed_seasons.difference(configured_seasons)
    missing_seasons = configured_seasons.difference(observed_seasons)
    if unknown_seasons:
        raise SplitError(
            f"Model dataset contains unconfigured seasons: {sorted(unknown_seasons)}"
        )
    if missing_seasons:
        raise SplitError(
            f"Model dataset is missing configured seasons: {sorted(missing_seasons)}"
        )

    split_frames: dict[str, pd.DataFrame] = {}
    for split_name in SPLIT_ORDER:
        seasons = policy.by_name()[split_name]
        frame = model.loc[model["season"].astype(str).isin(seasons)].copy()
        frame = frame.sort_values(["date", "match_id"], kind="mergesort").reset_index(
            drop=True
        )
        if frame.empty:
            raise SplitError(f"The {split_name} split is empty.")
        split_frames[split_name] = frame

    splits = DatasetSplits(**split_frames)
    validate_dataset_splits(splits, model, policy)
    return splits


def validate_dataset_splits(
    splits: DatasetSplits,
    model: pd.DataFrame,
    policy: SplitPolicy,
) -> None:
    """Assert split membership, disjointness, completeness, and chronology."""
    seen_ids: set[str] = set()
    total_rows = 0
    for split_name, frame in splits.by_name().items():
        if frame.empty:
            raise SplitError(f"The {split_name} split is empty.")
        expected_seasons = set(policy.by_name()[split_name])
        actual_seasons = set(frame["season"].astype(str))
        if actual_seasons != expected_seasons:
            raise SplitError(
                f"The {split_name} split has seasons {sorted(actual_seasons)}; "
                f"expected {sorted(expected_seasons)}."
            )
        match_ids = set(frame["match_id"].astype(str))
        overlap = seen_ids.intersection(match_ids)
        if overlap:
            raise SplitError(f"match_id occurs in multiple splits: {sorted(overlap)[:5]}")
        seen_ids.update(match_ids)
        total_rows += len(frame)

        dates = pd.to_datetime(frame["date"], format="%Y-%m-%d", errors="raise")
        if not dates.is_monotonic_increasing:
            raise SplitError(f"The {split_name} split is not chronological.")

    if total_rows != len(model) or seen_ids != set(model["match_id"].astype(str)):
        raise SplitError("Frozen splits do not cover the complete model dataset exactly once.")

    frames = splits.by_name()
    for earlier_name, later_name in zip(SPLIT_ORDER, SPLIT_ORDER[1:]):
        earlier_max = pd.to_datetime(frames[earlier_name]["date"]).max()
        later_min = pd.to_datetime(frames[later_name]["date"]).min()
        if not earlier_max < later_min:
            raise SplitError(
                f"Chronology violation: {earlier_name} does not precede {later_name}."
            )


def build_split_manifest(splits: DatasetSplits) -> pd.DataFrame:
    """Build the reproducible one-row-per-match split assignment table."""
    parts = []
    for split_name, frame in splits.by_name().items():
        part = frame.loc[:, ["match_id", "season", "date"]].copy()
        part["split"] = split_name
        parts.append(part)
    manifest = pd.concat(parts, ignore_index=True).loc[:, SPLIT_MANIFEST_COLUMNS]
    if manifest["match_id"].isna().any() or not manifest["match_id"].is_unique:
        raise SplitError("Split manifest match IDs must be nonempty and unique.")
    return manifest


def write_split_manifest_atomic(
    manifest: pd.DataFrame,
    path: Path = SPLIT_MANIFEST_PATH,
) -> None:
    """Write the split manifest atomically with deterministic column ordering."""
    if tuple(manifest.columns) != SPLIT_MANIFEST_COLUMNS:
        raise SplitError("Split manifest columns do not match the required order.")
    path.parent.mkdir(parents=True, exist_ok=True)
    file_descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(file_descriptor, "w", encoding="utf-8", newline="") as output:
            manifest.to_csv(
                output,
                index=False,
                columns=SPLIT_MANIFEST_COLUMNS,
                lineterminator="\n",
            )
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary_path, path)
    except (OSError, TypeError, ValueError) as exc:
        raise SplitError(f"Could not atomically write split manifest {path}: {exc}") from exc
    finally:
        temporary_path.unlink(missing_ok=True)


def build_and_save_splits(
    model_path: Path = MODEL_DATASET_PATH,
    config_path: Path = MODEL_CONFIG_PATH,
    manifest_path: Path = SPLIT_MANIFEST_PATH,
) -> DatasetSplits:
    """Load, split, validate, and persist the frozen split membership."""
    model = read_model_dataset(model_path)
    policy = load_split_policy(config_path)
    splits = split_model_dataset(model, policy)
    write_split_manifest_atomic(build_split_manifest(splits), manifest_path)
    return splits
