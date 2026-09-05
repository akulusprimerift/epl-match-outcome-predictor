"""Build and freeze the possession-complete Phase 7 matched experiment."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Sequence

import numpy as np
import pandas as pd

from src.build_features import FeatureError, build_model_dataset_frame, validate_model_dataset
from src.build_history import (
    HistoryError,
    build_team_history_frame,
    read_canonical_matches,
    write_csv_atomic,
)
from src.clean_data import PROJECT_ROOT
from src.collect_possession import (
    PossessionCollectionError,
    TEAM_SEASON_POSSESSION_COLUMNS,
    build_processed_possession,
    season_code_from_start_year,
)
from src.constants import (
    FEATURE_COLUMNS,
    MATCHED_MODEL_DATASET_COLUMNS,
    POSSESSION_FEATURE_ADDITIONS,
    POSSESSION_FEATURE_COLUMNS,
    SPLIT_MANIFEST_COLUMNS,
    SPLIT_ORDER,
)
from src.split_data import DatasetSplits, SplitPolicy, load_split_policy
from src.download_data import ManifestError


MATCHED_DATASET_PATH = (
    PROJECT_ROOT / "data" / "processed" / "matched_model_dataset.csv"
)
MATCHED_SPLIT_MANIFEST_PATH = (
    PROJECT_ROOT / "data" / "processed" / "matched_split_manifest.csv"
)


class MatchedExperimentError(RuntimeError):
    """Raised when the possession-matched experiment violates its contract."""


@dataclass(frozen=True)
class MatchedExperimentArtifacts:
    """The frozen cohort, split frames, and auditable output paths."""

    dataset: pd.DataFrame
    splits: DatasetSplits
    manifest: pd.DataFrame
    model_b_period_start: str
    dataset_path: Path
    manifest_path: Path


def _target_season_for_source(source_season: str) -> str:
    """Return the canonical season immediately after a four-digit season code."""
    if re.fullmatch(r"\d{4}", source_season) is None:
        raise MatchedExperimentError(
            f"Invalid possession source season {source_season!r}."
        )
    start_year = 2000 + int(source_season[:2])
    if start_year > 2089:
        start_year -= 100
    expected_source = season_code_from_start_year(start_year)
    if expected_source != source_season:
        raise MatchedExperimentError(
            f"Invalid possession source season {source_season!r}."
        )
    return season_code_from_start_year(start_year + 1)


def _validate_team_season_possession(frame: pd.DataFrame) -> pd.DataFrame:
    """Validate the Phase 6 table before any fixture-level feature join."""
    if tuple(frame.columns) != TEAM_SEASON_POSSESSION_COLUMNS:
        raise MatchedExperimentError(
            "Team-season possession columns differ from the Phase 6 contract."
        )
    if frame.empty:
        raise MatchedExperimentError("Team-season possession data is empty.")

    working = frame.copy()
    for column in ("source_season", "target_season", "team", "team_slug"):
        if working[column].isna().any() or working[column].astype(str).str.strip().eq("").any():
            raise MatchedExperimentError(
                f"Team-season possession requires complete {column} values."
            )
        working[column] = working[column].astype(str)

    for source_season, target_season in working.loc[
        :, ["source_season", "target_season"]
    ].itertuples(index=False, name=None):
        if _target_season_for_source(source_season) != target_season:
            raise MatchedExperimentError(
                "Possession rows must map source season N-1 to target season N."
            )

    key_columns = ["source_season", "team_slug"]
    if working.duplicated(key_columns).any() or working.duplicated(
        ["target_season", "team_slug"]
    ).any():
        raise MatchedExperimentError(
            "Team-season possession requires one row per season and team."
        )
    source_counts = working.groupby("target_season")["source_season"].nunique()
    if source_counts.gt(1).any():
        raise MatchedExperimentError(
            "A target season maps to multiple possession source seasons."
        )

    numeric = pd.to_numeric(working["average_possession_pct"], errors="coerce")
    invalid_numeric = working["average_possession_pct"].notna() & numeric.isna()
    if invalid_numeric.any() or numeric.dropna().lt(0).any() or numeric.dropna().gt(100).any():
        raise MatchedExperimentError(
            "Team-season average possession must be numeric within 0--100 or missing."
        )
    working["average_possession_pct"] = numeric.astype("Float64")
    return working


def _canonical_team_slugs_by_season(
    canonical: pd.DataFrame,
) -> dict[str, set[str]]:
    teams: dict[str, set[str]] = {}
    for season, group in canonical.groupby(canonical["season"].astype(str), sort=False):
        teams[str(season)] = set(group["home_team_slug"].astype(str)).union(
            group["away_team_slug"].astype(str)
        )
    return teams


def build_lagged_possession_features(
    canonical: pd.DataFrame,
    team_season_possession: pd.DataFrame,
) -> pd.DataFrame:
    """Join each fixture only to both clubs' preceding completed EPL season."""
    required = {"match_id", "season", "home_team_slug", "away_team_slug"}
    missing = required.difference(canonical.columns)
    if missing:
        raise MatchedExperimentError(
            f"Canonical fixtures are missing lagged-possession keys: {sorted(missing)}"
        )
    possession = _validate_team_season_possession(team_season_possession)
    fixture_keys = canonical.loc[
        :, ["match_id", "season", "home_team_slug", "away_team_slug"]
    ].copy()
    fixture_keys["season"] = fixture_keys["season"].astype(str)

    lookup = possession.loc[
        :, ["source_season", "target_season", "team_slug", "average_possession_pct"]
    ]
    home_lookup = lookup.rename(
        columns={
            "team_slug": "home_team_slug",
            "average_possession_pct": "home_previous_season_possession",
            "source_season": "home_possession_source_season",
            "target_season": "season",
        }
    )
    away_lookup = lookup.rename(
        columns={
            "team_slug": "away_team_slug",
            "average_possession_pct": "away_previous_season_possession",
            "source_season": "away_possession_source_season",
            "target_season": "season",
        }
    )
    try:
        joined = fixture_keys.merge(
            home_lookup,
            on=["season", "home_team_slug"],
            how="left",
            validate="many_to_one",
        ).merge(
            away_lookup,
            on=["season", "away_team_slug"],
            how="left",
            validate="many_to_one",
        )
    except pd.errors.MergeError as exc:
        raise MatchedExperimentError(
            f"Could not join lagged team-season possession: {exc}"
        ) from exc

    source_by_target = (
        possession.loc[:, ["target_season", "source_season"]]
        .drop_duplicates()
        .set_index("target_season")["source_season"]
        .to_dict()
    )
    canonical_teams = _canonical_team_slugs_by_season(canonical)
    for role in ("home", "away"):
        source_column = f"{role}_possession_source_season"
        team_column = f"{role}_team_slug"
        for row in joined.loc[joined[source_column].isna()].itertuples(index=False):
            target_season = str(row.season)
            expected_source = source_by_target.get(target_season)
            if expected_source is None:
                continue
            team_slug = str(getattr(row, team_column))
            if team_slug in canonical_teams.get(str(expected_source), set()):
                raise MatchedExperimentError(
                    f"Missing preceding-season possession for established EPL team "
                    f"{team_slug!r} in target season {target_season}."
                )

    joined["possession_edge"] = (
        joined["home_previous_season_possession"]
        - joined["away_previous_season_possession"]
    )
    return joined.loc[:, ["match_id", *POSSESSION_FEATURE_ADDITIONS]]


def build_possession_complete_dataset(
    canonical: pd.DataFrame,
    history: pd.DataFrame,
    *,
    team_season_possession: pd.DataFrame,
    model_b_period_start: str,
) -> pd.DataFrame:
    """Build the one lagged-possession cohort shared by both Phase 7 models."""
    try:
        baseline = build_model_dataset_frame(
            canonical, history, feature_set="baseline"
        )
    except (FeatureError, HistoryError) as exc:
        raise MatchedExperimentError(
            f"Could not build baseline features for the matched cohort: {exc}"
        ) from exc
    possession_features = build_lagged_possession_features(
        canonical, team_season_possession
    )
    try:
        matched = baseline.merge(
            possession_features, on="match_id", how="left", validate="one_to_one"
        )
    except pd.errors.MergeError as exc:
        raise MatchedExperimentError(
            f"Could not join possession features one-to-one: {exc}"
        ) from exc

    season_order = list(dict.fromkeys(canonical["season"].astype(str).tolist()))
    if model_b_period_start not in season_order:
        raise MatchedExperimentError(
            f"Model B period start {model_b_period_start!r} is not canonical."
        )
    eligible_seasons = set(
        season_order[season_order.index(model_b_period_start) :]
    )
    complete_mask = (
        matched["season"].astype(str).isin(eligible_seasons)
    )
    matched = matched.loc[complete_mask, MATCHED_MODEL_DATASET_COLUMNS].copy()
    matched = matched.sort_values(["date", "match_id"], kind="mergesort").reset_index(
        drop=True
    )
    validate_matched_model_dataset(matched, canonical=canonical)
    return matched


def validate_matched_model_dataset(
    frame: pd.DataFrame,
    *,
    canonical: pd.DataFrame | None = None,
) -> None:
    """Validate the shared cohort and baseline-plus-possession feature contract."""
    if tuple(frame.columns) != MATCHED_MODEL_DATASET_COLUMNS:
        raise MatchedExperimentError(
            "Matched model dataset columns differ from the fixed contract."
        )
    if frame.empty:
        raise MatchedExperimentError(
            "Possession-complete model dataset is empty; collect sufficient EPL possession data."
        )
    if frame["match_id"].isna().any() or not frame["match_id"].is_unique:
        raise MatchedExperimentError(
            "Matched model dataset requires unique, nonmissing match IDs."
        )
    baseline = frame.loc[:, [*frame.columns[:6], *FEATURE_COLUMNS]].copy()
    try:
        validate_model_dataset(baseline)
    except FeatureError as exc:
        raise MatchedExperimentError(
            f"Matched baseline features failed validation: {exc}"
        ) from exc
    for column in (
        "home_previous_season_possession",
        "away_previous_season_possession",
    ):
        values = pd.to_numeric(frame[column], errors="coerce")
        invalid = frame[column].notna() & values.isna()
        if invalid.any() or values.dropna().lt(0).any() or values.dropna().gt(100).any():
            raise MatchedExperimentError(
                f"Matched feature {column} must contain 0--100 values or be missing."
            )
    expected_edge = (
        frame["home_previous_season_possession"]
        - frame["away_previous_season_possession"]
    )
    observed_edge = pd.to_numeric(frame["possession_edge"], errors="coerce")
    both_available = expected_edge.notna()
    if (
        (frame["possession_edge"].notna() & observed_edge.isna()).any()
        or observed_edge.loc[~both_available].notna().any()
        or not np.allclose(
            observed_edge.loc[both_available],
            pd.to_numeric(expected_edge.loc[both_available], errors="raise"),
            rtol=0.0,
            atol=1e-12,
        )
    ):
        raise MatchedExperimentError("Possession edge has an invalid sign.")
    if set(POSSESSION_FEATURE_COLUMNS).difference(frame.columns):
        raise MatchedExperimentError("Matched dataset is missing model features.")
    if set(POSSESSION_FEATURE_ADDITIONS) != (
        set(POSSESSION_FEATURE_COLUMNS) - set(FEATURE_COLUMNS)
    ):
        raise MatchedExperimentError(
            "Only possession-related columns may differ between matched models."
        )
    if canonical is not None:
        canonical_by_id = canonical.set_index("match_id")
        missing_ids = set(frame["match_id"]).difference(canonical_by_id.index)
        if missing_ids:
            raise MatchedExperimentError(
                f"Matched rows are absent from canonical data: {sorted(missing_ids)[:5]}"
            )
        current = canonical_by_id.loc[frame["match_id"]]
        if not current["season"].astype(str).reset_index(drop=True).equals(
            frame["season"].astype(str).reset_index(drop=True)
        ):
            raise MatchedExperimentError(
                "Matched rows do not preserve canonical fixture seasons."
            )


def feature_columns_for_set(feature_set: str) -> tuple[str, ...]:
    """Return the only approved feature order for one model variant."""
    if feature_set in {"baseline", "baseline_matched"}:
        return FEATURE_COLUMNS
    if feature_set == "possession":
        return POSSESSION_FEATURE_COLUMNS
    raise MatchedExperimentError(f"Unsupported feature set {feature_set!r}.")


def split_matched_dataset(
    frame: pd.DataFrame, policy: SplitPolicy
) -> DatasetSplits:
    """Apply frozen seasons to the cohort without requiring every season row."""
    validate_matched_model_dataset(frame)
    configured = set().union(
        *(set(seasons) for seasons in policy.by_name().values())
    )
    unknown = set(frame["season"].astype(str)).difference(configured)
    if unknown:
        raise MatchedExperimentError(
            f"Matched dataset contains unconfigured seasons: {sorted(unknown)}"
        )
    split_frames: dict[str, pd.DataFrame] = {}
    seen_ids: set[str] = set()
    for split_name in SPLIT_ORDER:
        split_frame = frame.loc[
            frame["season"].astype(str).isin(policy.by_name()[split_name])
        ].copy()
        split_frame = split_frame.sort_values(
            ["date", "match_id"], kind="mergesort"
        ).reset_index(drop=True)
        if split_frame.empty:
            raise MatchedExperimentError(
                f"Possession-complete {split_name} split is empty; "
                "coverage is insufficient for Phase 7."
            )
        match_ids = set(split_frame["match_id"].astype(str))
        if seen_ids.intersection(match_ids):
            raise MatchedExperimentError("A matched fixture occurs in multiple splits.")
        seen_ids.update(match_ids)
        split_frames[split_name] = split_frame
    if seen_ids != set(frame["match_id"].astype(str)):
        raise MatchedExperimentError(
            "Matched splits do not cover the cohort exactly once."
        )
    for earlier, later in zip(SPLIT_ORDER, SPLIT_ORDER[1:]):
        earlier_max = pd.to_datetime(split_frames[earlier]["date"]).max()
        later_min = pd.to_datetime(split_frames[later]["date"]).min()
        if not earlier_max < later_min:
            raise MatchedExperimentError(
                f"Matched split chronology violation: {earlier} vs {later}."
            )
    return DatasetSplits(**split_frames)


def build_matched_split_manifest(splits: DatasetSplits) -> pd.DataFrame:
    """Freeze one exact match-ID list for both Phase 7 models."""
    parts: list[pd.DataFrame] = []
    for split_name in SPLIT_ORDER:
        part = splits.by_name()[split_name].loc[
            :, ["match_id", "season", "date"]
        ].copy()
        part["split"] = split_name
        parts.append(part)
    manifest = pd.concat(parts, ignore_index=True).loc[
        :, SPLIT_MANIFEST_COLUMNS
    ]
    if manifest["match_id"].isna().any() or not manifest["match_id"].is_unique:
        raise MatchedExperimentError(
            "Matched split manifest IDs must be nonmissing and unique."
        )
    return manifest


def read_matched_model_dataset(path: Path = MATCHED_DATASET_PATH) -> pd.DataFrame:
    """Read and validate the frozen matched feature table."""
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
        raise MatchedExperimentError(
            f"Frozen matched dataset not found: {path}. Train model_a_matched first."
        ) from exc
    except (
        OSError,
        UnicodeDecodeError,
        pd.errors.EmptyDataError,
        pd.errors.ParserError,
    ) as exc:
        raise MatchedExperimentError(
            f"Could not read frozen matched dataset {path}: {exc}"
        ) from exc
    validate_matched_model_dataset(frame)
    return frame


def _assert_frozen_equal(
    frozen: pd.DataFrame,
    rebuilt: pd.DataFrame,
    *,
    artifact_name: str,
) -> None:
    try:
        pd.testing.assert_frame_equal(
            frozen.reset_index(drop=True),
            rebuilt.reset_index(drop=True),
            check_dtype=False,
            check_exact=False,
            rtol=0.0,
            atol=1e-12,
        )
    except AssertionError as exc:
        raise MatchedExperimentError(
            f"Current data differs from frozen {artifact_name}; rerun "
            "model_a_matched before training model_b."
        ) from exc


def prepare_matched_experiment(
    *,
    reset_freeze: bool,
    project_root: Path = PROJECT_ROOT,
) -> MatchedExperimentArtifacts:
    """Build the cohort, then freeze it or verify the existing freeze exactly."""
    project_root = project_root.resolve()
    canonical = read_canonical_matches(
        project_root / "data" / "processed" / "canonical_matches.csv"
    )
    history = build_team_history_frame(canonical)
    try:
        possession_rows, model_b_period_start, _, _ = build_processed_possession(
            project_root
        )
    except (PossessionCollectionError, ManifestError) as exc:
        raise MatchedExperimentError(
            f"Could not load validated team-season possession: {exc}"
        ) from exc
    if model_b_period_start is None:
        raise MatchedExperimentError(
            "Possession coverage is below the configured threshold in every season; "
            "collect and join sufficient EPL possession data before Phase 7 training."
        )
    possession = pd.DataFrame(
        [row.__dict__ for row in possession_rows],
        columns=TEAM_SEASON_POSSESSION_COLUMNS,
    )
    dataset = build_possession_complete_dataset(
        canonical,
        history,
        team_season_possession=possession,
        model_b_period_start=model_b_period_start,
    )
    policy = load_split_policy(project_root / "config" / "model_config.json")
    splits = split_matched_dataset(dataset, policy)
    manifest = build_matched_split_manifest(splits)
    dataset_path = (
        project_root / "data" / "processed" / "matched_model_dataset.csv"
    )
    manifest_path = (
        project_root / "data" / "processed" / "matched_split_manifest.csv"
    )

    if reset_freeze:
        write_csv_atomic(dataset, dataset_path, MATCHED_MODEL_DATASET_COLUMNS)
        write_csv_atomic(manifest, manifest_path, SPLIT_MANIFEST_COLUMNS)
    else:
        frozen_dataset = read_matched_model_dataset(dataset_path)
        try:
            frozen_manifest = pd.read_csv(
                manifest_path,
                dtype={
                    "match_id": "string",
                    "season": "string",
                    "date": "string",
                    "split": "string",
                },
            )
        except FileNotFoundError as exc:
            raise MatchedExperimentError(
                "Frozen matched split manifest is missing; train model_a_matched first."
            ) from exc
        except (
            OSError,
            UnicodeDecodeError,
            pd.errors.EmptyDataError,
            pd.errors.ParserError,
        ) as exc:
            raise MatchedExperimentError(
                f"Could not read frozen matched split manifest: {exc}"
            ) from exc
        if tuple(frozen_manifest.columns) != SPLIT_MANIFEST_COLUMNS:
            raise MatchedExperimentError(
                "Frozen matched split manifest columns differ from the contract."
            )
        _assert_frozen_equal(
            frozen_dataset, dataset, artifact_name="matched model dataset"
        )
        _assert_frozen_equal(
            frozen_manifest, manifest, artifact_name="matched split manifest"
        )

    return MatchedExperimentArtifacts(
        dataset=dataset,
        splits=splits,
        manifest=manifest,
        model_b_period_start=model_b_period_start,
        dataset_path=dataset_path,
        manifest_path=manifest_path,
    )


def assert_identical_model_match_ids(
    manifest: pd.DataFrame,
    model_a_matched_ids: Sequence[str],
    model_b_ids: Sequence[str],
) -> None:
    """Assert both variants consume the same complete frozen cohort."""
    expected = set(manifest["match_id"].astype(str))
    left = set(str(value) for value in model_a_matched_ids)
    right = set(str(value) for value in model_b_ids)
    if left != expected or right != expected or left != right:
        raise MatchedExperimentError(
            "Model A-Matched and Model B match-ID sets differ from the frozen cohort."
        )
