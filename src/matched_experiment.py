"""Build and freeze the possession-complete Phase 7 matched experiment."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from src.build_features import FeatureError, build_model_dataset_frame, validate_model_dataset
from src.build_history import (
    HistoryError,
    build_team_history_frame,
    read_canonical_matches,
    validate_team_history,
    write_csv_atomic,
)
from src.clean_data import (
    PROJECT_ROOT,
    build_possession_coverage,
    load_possession_coverage_threshold,
)
from src.constants import (
    FEATURE_COLUMNS,
    MATCHED_MODEL_DATASET_COLUMNS,
    POSSESSION_FEATURE_ADDITIONS,
    POSSESSION_FEATURE_COLUMNS,
    ROLLING_MIN_PERIODS,
    ROLLING_WINDOW,
    SPLIT_MANIFEST_COLUMNS,
    SPLIT_ORDER,
)
from src.split_data import DatasetSplits, SplitPolicy, load_split_policy


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


def _validate_rolling_controls(window: int, min_periods: int) -> None:
    if window <= 0:
        raise MatchedExperimentError("Possession rolling window must be positive.")
    if min_periods <= 0 or min_periods > window:
        raise MatchedExperimentError(
            "Possession rolling min_periods must be between 1 and the window size."
        )


def compute_possession_rolling_features(
    history: pd.DataFrame,
    *,
    window: int = ROLLING_WINDOW,
    min_periods: int = ROLLING_MIN_PERIODS,
) -> pd.DataFrame:
    """Compute means from the previous possession-complete EPL matches only."""
    _validate_rolling_controls(window, min_periods)
    try:
        validate_team_history(history)
    except HistoryError as exc:
        raise MatchedExperimentError(
            f"Cannot build possession features from invalid team history: {exc}"
        ) from exc

    working = history.copy()
    try:
        parsed_dates = pd.to_datetime(
            working["date"], format="%Y-%m-%d", errors="raise"
        )
    except (TypeError, ValueError) as exc:
        raise MatchedExperimentError(
            f"Team history contains invalid dates: {exc}"
        ) from exc
    possession = pd.to_numeric(working["possession"], errors="coerce")
    invalid = working["possession"].notna() & possession.isna()
    if invalid.any():
        raise MatchedExperimentError("Team history contains nonnumeric possession.")
    if (possession.dropna().lt(0) | possession.dropna().gt(100)).any():
        raise MatchedExperimentError("Team history possession must be within 0--100.")

    working["_parsed_date"] = parsed_dates
    working["_possession_numeric"] = possession
    working = working.sort_values(
        ["team_slug", "date", "match_id"], kind="mergesort"
    ).reset_index(drop=True)
    averages = pd.Series(pd.NA, index=working.index, dtype="Float64")

    for _, team_group in working.groupby("team_slug", sort=False):
        prior_complete: list[float] = []
        for _, same_date in team_group.groupby("_parsed_date", sort=False):
            recent = prior_complete[-window:]
            if len(recent) >= min_periods:
                averages.loc[same_date.index] = float(np.mean(recent))
            current_complete = same_date["_possession_numeric"].dropna().tolist()
            prior_complete.extend(float(value) for value in current_complete)

    result = working.loc[
        :, ["team_match_id", "match_id", "date", "team_slug", "is_home"]
    ].copy()
    result["possession_avg_5"] = averages
    return result


def previous_possession_match_ids(
    history: pd.DataFrame,
    team_slug: str,
    current_match_date: date | str,
    *,
    window: int = ROLLING_WINDOW,
) -> tuple[str, ...]:
    """Return exact strictly-prior possession-complete source fixture IDs."""
    if window <= 0:
        raise MatchedExperimentError("Possession source window must be positive.")
    date_text = (
        current_match_date.isoformat()
        if isinstance(current_match_date, date)
        else str(current_match_date)
    )
    try:
        current_date = datetime.strptime(date_text, "%Y-%m-%d").date()
    except ValueError as exc:
        raise MatchedExperimentError(
            f"Possession source-window date must be YYYY-MM-DD: {date_text!r}."
        ) from exc
    dates = pd.to_datetime(
        history["date"], format="%Y-%m-%d", errors="raise"
    ).dt.date
    possession = pd.to_numeric(history["possession"], errors="coerce")
    mask = (
        history["team_slug"].eq(team_slug)
        & dates.lt(current_date)
        & possession.notna()
    )
    prior = history.loc[mask, ["date", "match_id"]].sort_values(
        ["date", "match_id"], kind="mergesort"
    )
    return tuple(prior.tail(window)["match_id"].astype(str))


def build_possession_complete_dataset(
    canonical: pd.DataFrame,
    history: pd.DataFrame,
    *,
    model_b_period_start: str,
) -> pd.DataFrame:
    """Build one cohort shared by baseline-matched and possession models."""
    try:
        baseline = build_model_dataset_frame(
            canonical, history, feature_set="baseline"
        )
    except (FeatureError, HistoryError) as exc:
        raise MatchedExperimentError(
            f"Could not build baseline features for the matched cohort: {exc}"
        ) from exc
    possession_features = compute_possession_rolling_features(history)

    home = possession_features.loc[
        possession_features["is_home"].astype("bool"),
        ["match_id", "possession_avg_5"],
    ].rename(columns={"possession_avg_5": "home_possession_avg_5"})
    away = possession_features.loc[
        ~possession_features["is_home"].astype("bool"),
        ["match_id", "possession_avg_5"],
    ].rename(columns={"possession_avg_5": "away_possession_avg_5"})
    if not home["match_id"].is_unique or not away["match_id"].is_unique:
        raise MatchedExperimentError(
            "Possession features contain multiple home or away rows per fixture."
        )
    try:
        matched = baseline.merge(
            home, on="match_id", how="left", validate="one_to_one"
        ).merge(away, on="match_id", how="left", validate="one_to_one")
    except pd.errors.MergeError as exc:
        raise MatchedExperimentError(
            f"Could not join possession features one-to-one: {exc}"
        ) from exc
    matched["possession_edge"] = (
        matched["home_possession_avg_5"]
        - matched["away_possession_avg_5"]
    )

    season_order = list(dict.fromkeys(canonical["season"].astype(str).tolist()))
    if model_b_period_start not in season_order:
        raise MatchedExperimentError(
            f"Model B period start {model_b_period_start!r} is not canonical."
        )
    eligible_seasons = set(
        season_order[season_order.index(model_b_period_start) :]
    )
    current_complete = canonical.loc[
        canonical["home_possession"].notna()
        & canonical["away_possession"].notna(),
        "match_id",
    ].astype(str)
    complete_mask = (
        matched["season"].astype(str).isin(eligible_seasons)
        & matched["match_id"].astype(str).isin(set(current_complete))
        & matched["home_possession_avg_5"].notna()
        & matched["away_possession_avg_5"].notna()
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
    for column in ("home_possession_avg_5", "away_possession_avg_5"):
        values = pd.to_numeric(frame[column], errors="coerce")
        if values.isna().any() or values.lt(0).any() or values.gt(100).any():
            raise MatchedExperimentError(
                f"Matched feature {column} must contain complete 0--100 values."
            )
    expected_edge = (
        frame["home_possession_avg_5"] - frame["away_possession_avg_5"]
    )
    if not np.allclose(
        pd.to_numeric(frame["possession_edge"], errors="coerce"),
        pd.to_numeric(expected_edge, errors="coerce"),
        rtol=0.0,
        atol=1e-12,
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
        if current["home_possession"].isna().any() or current[
            "away_possession"
        ].isna().any():
            raise MatchedExperimentError(
                "Matched rows require complete current-fixture possession coverage."
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
    threshold = load_possession_coverage_threshold(project_root)
    coverage = build_possession_coverage(canonical, threshold)
    if coverage.model_b_period_start is None:
        raise MatchedExperimentError(
            f"Possession coverage is below {threshold:.0%} in every season; "
            "collect and join sufficient EPL possession data before Phase 7 training."
        )
    dataset = build_possession_complete_dataset(
        canonical,
        history,
        model_b_period_start=coverage.model_b_period_start,
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
        model_b_period_start=coverage.model_b_period_start,
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
