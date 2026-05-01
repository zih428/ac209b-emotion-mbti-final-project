"""Stage 2 token truncation audit helpers."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import TARGET_COLUMNS
from .preprocessing import normalize_text, validate_required_columns


def simple_token_count(text: object) -> int:
    return len(normalize_text(text).split())


def count_tokens(text: object, tokenizer=None) -> int:
    """Count tokens with a provided tokenizer, falling back to whitespace."""

    value = normalize_text(text)
    if tokenizer is None:
        return simple_token_count(value)
    if hasattr(tokenizer, "tokenize"):
        return len(tokenizer.tokenize(value))
    if callable(tokenizer):
        tokens = tokenizer(value)
        if isinstance(tokens, dict) and "input_ids" in tokens:
            return len(tokens["input_ids"])
        return len(tokens)
    raise TypeError("tokenizer must be None, callable, or expose `.tokenize`")


def token_truncation_audit(
    df: pd.DataFrame,
    *,
    text_col: str = "text_masked",
    author_col: str = "author",
    split_col: str = "split",
    target_cols: tuple[str, ...] = TARGET_COLUMNS,
    max_length: int = 128,
    tokenizer=None,
) -> dict[str, pd.DataFrame | dict[str, float | int]]:
    """Summarize post- and author-level truncation exposure."""

    validate_required_columns(df, [text_col, author_col])
    out = df.copy()
    out["token_length"] = out[text_col].map(lambda text: count_tokens(text, tokenizer))
    out["is_truncated"] = out["token_length"] > max_length

    lengths = out["token_length"].to_numpy()
    summary = {
        "n_posts": int(len(out)),
        "max_length": int(max_length),
        "median": float(np.percentile(lengths, 50)) if len(lengths) else 0.0,
        "p90": float(np.percentile(lengths, 90)) if len(lengths) else 0.0,
        "p95": float(np.percentile(lengths, 95)) if len(lengths) else 0.0,
        "p99": float(np.percentile(lengths, 99)) if len(lengths) else 0.0,
        "share_over_max": float(out["is_truncated"].mean()) if len(out) else 0.0,
    }

    author_exposure = (
        out.groupby(author_col)["is_truncated"]
        .agg(author_has_truncation="max", author_truncated_share="mean")
        .reset_index()
    )
    author_summary = {
        "n_authors": int(len(author_exposure)),
        "share_authors_with_any_truncation": float(
            author_exposure["author_has_truncation"].mean()
        )
        if len(author_exposure)
        else 0.0,
        "mean_within_author_truncated_share": float(
            author_exposure["author_truncated_share"].mean()
        )
        if len(author_exposure)
        else 0.0,
    }

    by_split = pd.DataFrame()
    if split_col in out.columns:
        by_split = (
            out.groupby(split_col)
            .agg(
                n_posts=("token_length", "size"),
                median=("token_length", "median"),
                p95=("token_length", lambda values: float(np.percentile(values, 95))),
                share_over_max=("is_truncated", "mean"),
            )
            .reset_index()
        )

    dimension_rows = []
    present_targets = [col for col in target_cols if col in out.columns]
    for target in present_targets:
        for value, group in out.groupby(target):
            dimension_rows.append(
                {
                    "target": target,
                    "target_value": int(value),
                    "n_posts": int(len(group)),
                    "median": float(group["token_length"].median()),
                    "p95": float(np.percentile(group["token_length"], 95)),
                    "share_over_max": float(group["is_truncated"].mean()),
                }
            )
    by_dimension = pd.DataFrame(dimension_rows)

    return {
        "post_summary": summary,
        "author_summary": author_summary,
        "by_split": by_split,
        "by_dimension": by_dimension,
    }
