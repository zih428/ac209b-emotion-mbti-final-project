"""Author-level aggregation, threshold tuning, and metrics."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from .config import TARGET_COLUMNS
from .preprocessing import validate_required_columns


@dataclass(frozen=True)
class ThresholdResult:
    thresholds: pd.Series
    validation_scores: pd.Series


def sigmoid(logits: np.ndarray) -> np.ndarray:
    logits = np.asarray(logits, dtype=float)
    return 1.0 / (1.0 + np.exp(-logits))


def aggregate_post_scores(
    posts: pd.DataFrame,
    *,
    score_cols: tuple[str, ...],
    target_cols: tuple[str, ...] = TARGET_COLUMNS,
    author_col: str = "author",
    split_col: str = "split",
) -> pd.DataFrame:
    """Average post-level probabilities to author-level scores."""

    validate_required_columns(posts, [author_col, *score_cols, *target_cols])
    agg_spec = {score: "mean" for score in score_cols}
    agg_spec.update({target: "first" for target in target_cols})
    if split_col in posts.columns:
        agg_spec[split_col] = "first"
    return posts.groupby(author_col).agg(agg_spec).reset_index()


def _metric_for_threshold(y_true: np.ndarray, y_pred: np.ndarray, objective: str) -> float:
    if objective == "balanced_accuracy":
        return float(balanced_accuracy_score(y_true, y_pred))
    if objective == "f1":
        return float(f1_score(y_true, y_pred, zero_division=0))
    raise ValueError("objective must be `balanced_accuracy` or `f1`")


def tune_thresholds(
    validation_authors: pd.DataFrame,
    *,
    score_cols: tuple[str, ...],
    target_cols: tuple[str, ...] = TARGET_COLUMNS,
    objective: str = "balanced_accuracy",
    grid: np.ndarray | None = None,
) -> ThresholdResult:
    """Tune one threshold per dimension on validation authors."""

    validate_required_columns(validation_authors, [*score_cols, *target_cols])
    if len(score_cols) != len(target_cols):
        raise ValueError("score_cols and target_cols must have the same length")
    grid = grid if grid is not None else np.linspace(0.05, 0.95, 181)

    thresholds = {}
    scores = {}
    for score_col, target_col in zip(score_cols, target_cols, strict=True):
        y_true = validation_authors[target_col].to_numpy(dtype=int)
        y_score = validation_authors[score_col].to_numpy(dtype=float)
        if len(np.unique(y_true)) < 2:
            thresholds[target_col] = 0.5
            scores[target_col] = math.nan
            continue
        candidates = []
        for threshold in grid:
            y_pred = (y_score >= threshold).astype(int)
            metric = _metric_for_threshold(y_true, y_pred, objective)
            candidates.append((metric, -abs(threshold - 0.5), threshold))
        best_metric, _, best_threshold = max(candidates)
        thresholds[target_col] = float(best_threshold)
        scores[target_col] = float(best_metric)
    return ThresholdResult(
        thresholds=pd.Series(thresholds, name="threshold"),
        validation_scores=pd.Series(scores, name=f"validation_{objective}"),
    )


def metric_table(
    authors: pd.DataFrame,
    *,
    score_cols: tuple[str, ...],
    thresholds: pd.Series | dict[str, float],
    target_cols: tuple[str, ...] = TARGET_COLUMNS,
) -> pd.DataFrame:
    """Compute author-level metrics for thresholded and continuous scores."""

    validate_required_columns(authors, [*score_cols, *target_cols])
    threshold_lookup = dict(thresholds)
    rows = []
    for score_col, target_col in zip(score_cols, target_cols, strict=True):
        y_true = authors[target_col].to_numpy(dtype=int)
        y_score = authors[score_col].to_numpy(dtype=float)
        threshold = float(threshold_lookup.get(target_col, 0.5))
        y_pred = (y_score >= threshold).astype(int)

        if len(np.unique(y_true)) < 2:
            roc_auc = math.nan
            average_precision = math.nan
        else:
            roc_auc = float(roc_auc_score(y_true, y_score))
            average_precision = float(average_precision_score(y_true, y_score))

        rows.append(
            {
                "target": target_col,
                "n_authors": int(len(authors)),
                "threshold": threshold,
                "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
                "f1": float(f1_score(y_true, y_pred, zero_division=0)),
                "minority_precision": float(
                    precision_score(y_true, y_pred, zero_division=0)
                ),
                "minority_recall": float(recall_score(y_true, y_pred, zero_division=0)),
                "roc_auc": roc_auc,
                "average_precision": average_precision,
                "raw_accuracy": float((y_true == y_pred).mean()) if len(y_true) else math.nan,
                "positive_rate_pred": float(y_pred.mean()) if len(y_pred) else math.nan,
                "positive_rate_true": float(y_true.mean()) if len(y_true) else math.nan,
            }
        )
    return pd.DataFrame(rows)


def evaluate_with_validation_thresholds(
    author_scores: pd.DataFrame,
    *,
    score_cols: tuple[str, ...],
    target_cols: tuple[str, ...] = TARGET_COLUMNS,
    split_col: str = "split",
    objective: str = "balanced_accuracy",
) -> dict[str, pd.DataFrame | ThresholdResult]:
    """Tune on validation authors and evaluate validation/test splits."""

    validate_required_columns(author_scores, [split_col, *score_cols, *target_cols])
    validation = author_scores.loc[author_scores[split_col] == "val"].copy()
    test = author_scores.loc[author_scores[split_col] == "test"].copy()
    if validation.empty:
        raise ValueError("No validation authors available for threshold tuning")
    if test.empty:
        raise ValueError("No test authors available for final metric table")

    threshold_result = tune_thresholds(
        validation,
        score_cols=score_cols,
        target_cols=target_cols,
        objective=objective,
    )
    return {
        "thresholds": threshold_result,
        "validation_metrics": metric_table(
            validation,
            score_cols=score_cols,
            target_cols=target_cols,
            thresholds=threshold_result.thresholds,
        ),
        "test_metrics": metric_table(
            test,
            score_cols=score_cols,
            target_cols=target_cols,
            thresholds=threshold_result.thresholds,
        ),
    }


def majority_baseline_author_scores(
    train_authors: pd.DataFrame,
    all_authors: pd.DataFrame,
    *,
    target_cols: tuple[str, ...] = TARGET_COLUMNS,
) -> pd.DataFrame:
    """Create constant author-level scores from training majority classes."""

    validate_required_columns(train_authors, list(target_cols))
    out = all_authors.copy()
    for target in target_cols:
        majority_positive = float(train_authors[target].mean() >= 0.5)
        out[f"score_majority_{target}"] = majority_positive
    return out
