"""Negative-control feature construction for emotion experiments."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import EMOTION_FEATURE_COLUMNS
from .preprocessing import validate_required_columns


def shuffled_column_names(columns: tuple[str, ...], *, prefix: str = "shuffled_") -> tuple[str, ...]:
    return tuple(f"{prefix}{col}" for col in columns)


def shuffle_features_within_split(
    frame: pd.DataFrame,
    *,
    feature_cols: tuple[str, ...] = EMOTION_FEATURE_COLUMNS,
    split_col: str = "split",
    seed: int = 209066,
    prefix: str = "shuffled_",
) -> pd.DataFrame:
    """Shuffle feature rows within each split while preserving all identifiers.

    The output keeps row count, split membership, and all non-feature columns
    unchanged. New columns contain split-preserving shuffled feature values.
    """

    validate_required_columns(frame, [split_col, *feature_cols])
    rng = np.random.default_rng(seed)
    out = frame.copy()
    shuffled_cols = shuffled_column_names(feature_cols, prefix=prefix)
    for col in shuffled_cols:
        if col in out.columns:
            raise ValueError(f"Output column already exists: {col}")

    for split_value, index in frame.groupby(split_col).groups.items():
        index = np.asarray(list(index))
        permutation = rng.permutation(index)
        values = frame.loc[permutation, list(feature_cols)].to_numpy()
        out.loc[index, list(shuffled_cols)] = values
    return out


def replace_with_shuffled_features(
    frame: pd.DataFrame,
    *,
    feature_cols: tuple[str, ...] = EMOTION_FEATURE_COLUMNS,
    split_col: str = "split",
    seed: int = 209066,
) -> pd.DataFrame:
    """Return a copy where feature columns are permuted within split."""

    shuffled = shuffle_features_within_split(
        frame,
        feature_cols=feature_cols,
        split_col=split_col,
        seed=seed,
    )
    out = frame.copy()
    for col in feature_cols:
        out[col] = shuffled[f"shuffled_{col}"].to_numpy()
    return out


def assert_split_preserving_shuffle(
    original: pd.DataFrame,
    shuffled: pd.DataFrame,
    *,
    feature_cols: tuple[str, ...] = EMOTION_FEATURE_COLUMNS,
    split_col: str = "split",
) -> None:
    """Validate row-count, split-membership, and marginal feature preservation."""

    validate_required_columns(original, [split_col, *feature_cols])
    validate_required_columns(shuffled, [split_col, *feature_cols])
    if len(original) != len(shuffled):
        raise ValueError("Shuffled frame changed row count")
    if not original[split_col].reset_index(drop=True).equals(
        shuffled[split_col].reset_index(drop=True)
    ):
        raise ValueError("Shuffled frame changed split membership order")
    for split_value in original[split_col].drop_duplicates():
        left = original.loc[original[split_col] == split_value, list(feature_cols)]
        right = shuffled.loc[shuffled[split_col] == split_value, list(feature_cols)]
        for col in feature_cols:
            if not np.allclose(
                np.sort(left[col].to_numpy(dtype=float)),
                np.sort(right[col].to_numpy(dtype=float)),
            ):
                raise ValueError(
                    f"Shuffled frame changed marginal distribution for {col} in {split_value}"
                )
