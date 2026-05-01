"""Author-level splitting and split diagnostics."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from .config import TARGET_COLUMNS
from .preprocessing import validate_required_columns


@dataclass(frozen=True)
class SplitResult:
    authors: pd.DataFrame
    method: str
    warnings: tuple[str, ...] = ()


def make_author_frame(
    posts: pd.DataFrame,
    *,
    author_col: str = "author",
    mbti_col: str = "mbti",
    target_cols: tuple[str, ...] = TARGET_COLUMNS,
) -> pd.DataFrame:
    """Collapse post rows to one author row and reject conflicting labels."""

    validate_required_columns(posts, [author_col, mbti_col, *target_cols])
    grouped = posts.groupby(author_col).agg(
        {mbti_col: "nunique", **{col: "nunique" for col in target_cols}}
    )
    conflict_mask = (grouped > 1).any(axis=1)
    if conflict_mask.any():
        raise ValueError(
            "Author-label conflicts remain after preprocessing: "
            f"{conflict_mask.sum()} authors"
        )

    first_cols = [author_col, mbti_col, *target_cols]
    return posts[first_cols].drop_duplicates(author_col).reset_index(drop=True)


def _can_stratify(labels: pd.Series) -> bool:
    counts = labels.value_counts()
    return len(counts) > 1 and counts.min() >= 2


def _split_once(
    frame: pd.DataFrame,
    *,
    test_size: float,
    seed: int,
    stratify_col: str | None,
) -> tuple[pd.DataFrame, pd.DataFrame, str, str | None]:
    stratify = frame[stratify_col] if stratify_col and _can_stratify(frame[stratify_col]) else None
    method = f"stratified_by_{stratify_col}" if stratify is not None else "random"
    warning = None if stratify is not None else f"Fell back to random split for size={test_size}."
    left, right = train_test_split(
        frame,
        test_size=test_size,
        random_state=seed,
        shuffle=True,
        stratify=stratify,
    )
    return left.copy(), right.copy(), method, warning


def split_authors(
    author_frame: pd.DataFrame,
    *,
    train_size: float = 0.70,
    val_size: float = 0.15,
    test_size: float = 0.15,
    seed: int = 209066,
    stratify_col: str = "mbti",
) -> SplitResult:
    """Create train/validation/test splits at author level."""

    if not np.isclose(train_size + val_size + test_size, 1.0):
        raise ValueError("train_size + val_size + test_size must equal 1.0")
    if len(author_frame) < 3:
        raise ValueError("Need at least three authors to create train/val/test splits")

    author_frame = author_frame.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    warnings: list[str] = []

    temp_size = val_size + test_size
    try:
        train, temp, method1, warning1 = _split_once(
            author_frame, test_size=temp_size, seed=seed, stratify_col=stratify_col
        )
    except ValueError as exc:
        train, temp, method1, warning1 = _split_once(
            author_frame, test_size=temp_size, seed=seed, stratify_col=None
        )
        warning1 = f"Stratified first split failed ({exc}); used random split."
    if warning1:
        warnings.append(warning1)

    relative_test_size = test_size / temp_size
    try:
        val, test, method2, warning2 = _split_once(
            temp, test_size=relative_test_size, seed=seed + 1, stratify_col=stratify_col
        )
    except ValueError as exc:
        val, test, method2, warning2 = _split_once(
            temp, test_size=relative_test_size, seed=seed + 1, stratify_col=None
        )
        warning2 = f"Stratified validation/test split failed ({exc}); used random split."
    if warning2:
        warnings.append(warning2)

    train = train.assign(split="train")
    val = val.assign(split="val")
    test = test.assign(split="test")
    out = pd.concat([train, val, test], ignore_index=True)
    return SplitResult(out, method=f"{method1}; {method2}", warnings=tuple(warnings))


def attach_splits(
    posts: pd.DataFrame,
    author_splits: pd.DataFrame,
    *,
    author_col: str = "author",
) -> pd.DataFrame:
    validate_required_columns(author_splits, [author_col, "split"])
    split_map = author_splits[[author_col, "split"]].drop_duplicates(author_col)
    out = posts.merge(split_map, on=author_col, how="left", validate="many_to_one")
    if out["split"].isna().any():
        missing = out.loc[out["split"].isna(), author_col].nunique()
        raise ValueError(f"{missing} authors did not receive a split")
    return out


def split_balance_table(
    author_splits: pd.DataFrame,
    *,
    target_cols: tuple[str, ...] = TARGET_COLUMNS,
) -> pd.DataFrame:
    validate_required_columns(author_splits, ["split", *target_cols])
    rows = []
    for split_name, group in author_splits.groupby("split"):
        for target in target_cols:
            rows.append(
                {
                    "split": split_name,
                    "target": target,
                    "n_authors": int(len(group)),
                    "positive_share": float(group[target].mean()),
                    "positive_count": int(group[target].sum()),
                    "negative_count": int((1 - group[target]).sum()),
                }
            )
    return pd.DataFrame(rows)
