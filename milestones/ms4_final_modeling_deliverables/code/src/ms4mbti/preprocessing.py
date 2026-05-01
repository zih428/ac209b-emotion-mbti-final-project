"""Reddit MBTI preprocessing and leakage audits."""

from __future__ import annotations

import re
from itertools import product
from typing import Literal

import numpy as np
import pandas as pd

from .config import DIMENSION_SPECS, DIMENSIONS, TARGET_COLUMNS


VALID_MBTI_TYPES = tuple(
    "".join(parts)
    for parts in product(("E", "I"), ("N", "S"), ("F", "T"), ("J", "P"))
)

MBTI_TERM_PATTERN = re.compile(
    r"\b(?:"
    + "|".join(VALID_MBTI_TYPES)
    + r")\b|\bMBTI\b|\bMyers[-\s]?Briggs\b",
    flags=re.IGNORECASE,
)


def normalize_text(value: object) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return ""
    return re.sub(r"\s+", " ", str(value)).strip()


def standardize_reddit_schema(df: pd.DataFrame) -> pd.DataFrame:
    """Return canonical `author`, `text`, `mbti` columns plus passthrough data."""

    columns = set(df.columns)
    text_col = "body" if "body" in columns else "text" if "text" in columns else None
    mbti_col = "mbti" if "mbti" in columns else "type" if "type" in columns else None
    missing = [
        name
        for name, present in {
            "author": "author" in columns,
            "body/text": text_col is not None,
            "mbti/type": mbti_col is not None,
        }.items()
        if not present
    ]
    if missing:
        raise ValueError(f"Missing required Reddit columns: {', '.join(missing)}")

    out = df.copy()
    out["author"] = out["author"].map(normalize_text)
    out["text"] = out[text_col].map(normalize_text)
    out["mbti"] = out[mbti_col].map(lambda value: normalize_text(value).upper())
    return out


def derive_mbti_targets(df: pd.DataFrame) -> pd.DataFrame:
    """Add one binary target per MBTI dimension using configured positive labels."""

    if "mbti" not in df:
        raise ValueError("derive_mbti_targets requires an `mbti` column")
    out = df.copy()
    valid_mask = out["mbti"].isin(VALID_MBTI_TYPES)
    if not valid_mask.all():
        invalid = sorted(out.loc[~valid_mask, "mbti"].dropna().unique().tolist())
        raise ValueError(f"Invalid MBTI labels found: {invalid[:10]}")
    for key in DIMENSIONS:
        spec = DIMENSION_SPECS[key]
        out[spec.target_col] = (
            out["mbti"].str[spec.mbti_position] == spec.positive_label
        ).astype("int8")
    return out


def clean_reddit_frame(df: pd.DataFrame, *, drop_blank: bool = True) -> pd.DataFrame:
    """Standardize, remove invalid rows, and derive binary targets."""

    out = standardize_reddit_schema(df)
    if drop_blank:
        out = out.loc[(out["author"] != "") & (out["text"] != "")].copy()
    out = out.loc[out["mbti"].isin(VALID_MBTI_TYPES)].copy()
    return derive_mbti_targets(out)


def mask_mbti_terms(text: object, mask_token: str = "[TYPE_MASK]") -> str:
    """Mask explicit MBTI type terms and label phrases."""

    return MBTI_TERM_PATTERN.sub(mask_token, normalize_text(text))


def add_masked_text(
    df: pd.DataFrame,
    *,
    text_col: str = "text",
    output_col: str = "text_masked",
    mask_token: str = "[TYPE_MASK]",
) -> pd.DataFrame:
    out = df.copy()
    normalized = out[text_col].map(normalize_text)
    out[output_col] = normalized
    has_term = normalized.str.contains(MBTI_TERM_PATTERN, regex=True, na=False)
    if has_term.any():
        out.loc[has_term, output_col] = normalized.loc[has_term].str.replace(
            MBTI_TERM_PATTERN,
            mask_token,
            regex=True,
        )
    return out


def word_count(text: object) -> int:
    return len(re.findall(r"\b\w+\b", normalize_text(text)))


def add_word_count(
    df: pd.DataFrame, *, text_col: str = "text_masked", output_col: str = "word_count"
) -> pd.DataFrame:
    out = df.copy()
    out[output_col] = out[text_col].map(word_count).astype("int32")
    return out


def audit_mbti_leakage(
    df: pd.DataFrame,
    *,
    text_col: str = "text",
    author_col: str = "author",
    mbti_col: str = "mbti",
) -> dict[str, float | int]:
    """Measure explicit MBTI-label mentions before or after masking."""

    if df.empty:
        return {
            "n_posts": 0,
            "n_authors": 0,
            "posts_with_any_term": 0,
            "post_share_with_any_term": 0.0,
            "authors_with_any_term": 0,
            "author_share_with_any_term": 0.0,
            "posts_with_own_type": 0,
            "post_share_with_own_type": 0.0,
        }

    text = df[text_col].map(normalize_text)
    has_any = text.str.contains(MBTI_TERM_PATTERN, regex=True, na=False)

    has_own = pd.Series(False, index=df.index)
    for mbti_type in VALID_MBTI_TYPES:
        type_mask = df[mbti_col] == mbti_type
        if not type_mask.any():
            continue
        pattern = re.compile(rf"\b{re.escape(mbti_type)}\b", re.IGNORECASE)
        has_own.loc[type_mask] = text.loc[type_mask].str.contains(
            pattern, regex=True, na=False
        )
    authors = df[author_col]
    n_authors = authors.nunique()
    authors_with_any = authors.loc[has_any].nunique()
    return {
        "n_posts": int(len(df)),
        "n_authors": int(n_authors),
        "posts_with_any_term": int(has_any.sum()),
        "post_share_with_any_term": float(has_any.mean()),
        "authors_with_any_term": int(authors_with_any),
        "author_share_with_any_term": float(authors_with_any / n_authors)
        if n_authors
        else 0.0,
        "posts_with_own_type": int(has_own.sum()),
        "post_share_with_own_type": float(has_own.mean()),
    }


def author_label_conflicts(
    df: pd.DataFrame, *, author_col: str = "author", mbti_col: str = "mbti"
) -> pd.DataFrame:
    """Return authors that appear with more than one MBTI type."""

    grouped = (
        df.groupby(author_col)[mbti_col]
        .agg(unique_count="nunique", labels=lambda values: tuple(sorted(set(values))))
        .reset_index()
    )
    return grouped.loc[grouped["unique_count"] > 1].reset_index(drop=True)


def resolve_author_label_conflicts(
    df: pd.DataFrame,
    *,
    strategy: Literal["drop", "modal"] = "drop",
    author_col: str = "author",
    mbti_col: str = "mbti",
) -> pd.DataFrame:
    """Resolve repeated authors with conflicting MBTI labels before splitting."""

    conflicts = author_label_conflicts(df, author_col=author_col, mbti_col=mbti_col)
    if conflicts.empty:
        return df.copy()

    if strategy == "drop":
        conflicted = set(conflicts[author_col])
        return df.loc[~df[author_col].isin(conflicted)].copy()

    if strategy == "modal":
        out = df.copy()
        modal_labels = (
            out.groupby(author_col)[mbti_col]
            .agg(lambda values: sorted(values.value_counts().items(), key=lambda x: (-x[1], x[0]))[0][0])
            .to_dict()
        )
        out[mbti_col] = out[author_col].map(modal_labels)
        out = out.drop(columns=list(TARGET_COLUMNS), errors="ignore")
        return derive_mbti_targets(out)

    raise ValueError(f"Unknown conflict strategy: {strategy}")


def validate_required_columns(df: pd.DataFrame, columns: list[str] | tuple[str, ...]) -> None:
    missing = [column for column in columns if column not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
