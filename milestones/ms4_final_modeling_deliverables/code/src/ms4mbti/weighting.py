"""Class and author-balancing weights for Stage 2 training."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import TARGET_COLUMNS
from .preprocessing import validate_required_columns


def effective_label_distribution(
    labels: pd.DataFrame,
    *,
    target_cols: tuple[str, ...] = TARGET_COLUMNS,
    sample_weight: np.ndarray | pd.Series | None = None,
) -> pd.DataFrame:
    validate_required_columns(labels, list(target_cols))
    weights = (
        np.asarray(sample_weight, dtype=float)
        if sample_weight is not None
        else np.ones(len(labels), dtype=float)
    )
    if len(weights) != len(labels):
        raise ValueError("sample_weight length must match labels")

    rows = []
    for target in target_cols:
        y = labels[target].to_numpy(dtype=float)
        pos = float(weights[y == 1].sum())
        neg = float(weights[y == 0].sum())
        total = pos + neg
        rows.append(
            {
                "target": target,
                "effective_positive": pos,
                "effective_negative": neg,
                "positive_share": pos / total if total else 0.0,
            }
        )
    return pd.DataFrame(rows)


def compute_pos_weights(
    labels: pd.DataFrame,
    *,
    target_cols: tuple[str, ...] = TARGET_COLUMNS,
    variant: str = "inverse",
    sample_weight: np.ndarray | pd.Series | None = None,
) -> pd.Series:
    """Compute BCE pos_weight values from an effective training distribution."""

    dist = effective_label_distribution(
        labels, target_cols=target_cols, sample_weight=sample_weight
    )
    values = {}
    for row in dist.itertuples(index=False):
        if row.effective_positive <= 0 or row.effective_negative <= 0:
            weight = 1.0
        else:
            weight = row.effective_negative / row.effective_positive
            if variant == "sqrt":
                weight = float(np.sqrt(weight))
            elif variant != "inverse":
                raise ValueError("variant must be `inverse` or `sqrt`")
        values[row.target] = float(weight)
    return pd.Series(values, name=f"pos_weight_{variant}")


def author_post_loss_weights(
    posts: pd.DataFrame,
    *,
    author_col: str = "author",
    normalize_mean: bool = True,
) -> pd.Series:
    """Weight each post by inverse retained posts for its author."""

    validate_required_columns(posts, [author_col])
    counts = posts[author_col].map(posts[author_col].value_counts()).astype(float)
    weights = 1.0 / counts
    if normalize_mean and len(weights):
        weights = weights / weights.mean()
    return pd.Series(weights.to_numpy(), index=posts.index, name="post_loss_weight")
