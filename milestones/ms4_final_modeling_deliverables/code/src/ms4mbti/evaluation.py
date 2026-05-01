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


def _threshold_lookup(thresholds: pd.Series | dict[str, float]) -> dict[str, float]:
    return dict(thresholds)


def _threshold_metric_value(
    y_true: np.ndarray,
    y_score: np.ndarray,
    *,
    threshold: float,
    metric: str,
) -> float:
    y_pred = (y_score >= threshold).astype(int)
    if metric == "balanced_accuracy":
        if len(np.unique(y_true)) < 2:
            return 0.5
        return float(balanced_accuracy_score(y_true, y_pred))
    if metric == "f1":
        return float(f1_score(y_true, y_pred, zero_division=0))
    if metric == "minority_recall":
        return float(recall_score(y_true, y_pred, zero_division=0))
    if metric == "minority_precision":
        return float(precision_score(y_true, y_pred, zero_division=0))
    raise ValueError(
        "metric must be balanced_accuracy, f1, minority_recall, or minority_precision"
    )


def paired_bootstrap_delta(
    baseline: pd.DataFrame,
    comparison: pd.DataFrame,
    *,
    baseline_score_cols: tuple[str, ...],
    comparison_score_cols: tuple[str, ...],
    baseline_thresholds: pd.Series | dict[str, float],
    comparison_thresholds: pd.Series | dict[str, float],
    comparison_name: str,
    target_cols: tuple[str, ...] = TARGET_COLUMNS,
    author_col: str = "author",
    metric: str = "balanced_accuracy",
    n_bootstrap: int = 2000,
    seed: int = 209066,
) -> pd.DataFrame:
    """Paired bootstrap CI for comparison-minus-baseline author metrics."""

    validate_required_columns(baseline, [author_col, *baseline_score_cols, *target_cols])
    validate_required_columns(comparison, [author_col, *comparison_score_cols, *target_cols])
    if len(baseline_score_cols) != len(target_cols) or len(comparison_score_cols) != len(target_cols):
        raise ValueError("score column tuples must match target_cols length")

    base_cols = [author_col, *target_cols, *baseline_score_cols]
    comp_cols = [author_col, *comparison_score_cols]
    joined = baseline[base_cols].merge(
        comparison[comp_cols],
        on=author_col,
        how="inner",
        validate="one_to_one",
        suffixes=("_baseline", "_comparison"),
    )
    if joined.empty:
        raise ValueError("No paired authors available for bootstrap delta")

    base_threshold_lookup = _threshold_lookup(baseline_thresholds)
    comp_threshold_lookup = _threshold_lookup(comparison_thresholds)
    rng = np.random.default_rng(seed)
    rows = []
    target_deltas: list[float] = []

    for target, base_score, comp_score in zip(
        target_cols, baseline_score_cols, comparison_score_cols, strict=True
    ):
        base_threshold = float(base_threshold_lookup.get(target, 0.5))
        comp_threshold = float(comp_threshold_lookup.get(target, 0.5))
        point = _threshold_metric_value(
            joined[target].to_numpy(dtype=int),
            joined[comp_score].to_numpy(dtype=float),
            threshold=comp_threshold,
            metric=metric,
        ) - _threshold_metric_value(
            joined[target].to_numpy(dtype=int),
            joined[base_score].to_numpy(dtype=float),
            threshold=base_threshold,
            metric=metric,
        )
        target_deltas.append(point)
        boot = []
        for _ in range(n_bootstrap):
            sample_idx = rng.integers(0, len(joined), size=len(joined))
            sample = joined.iloc[sample_idx]
            comp_value = _threshold_metric_value(
                sample[target].to_numpy(dtype=int),
                sample[comp_score].to_numpy(dtype=float),
                threshold=comp_threshold,
                metric=metric,
            )
            base_value = _threshold_metric_value(
                sample[target].to_numpy(dtype=int),
                sample[base_score].to_numpy(dtype=float),
                threshold=base_threshold,
                metric=metric,
            )
            boot.append(comp_value - base_value)
        rows.append(
            {
                "comparison": comparison_name,
                "metric": metric,
                "target": target,
                "point_estimate": float(point),
                "ci_lower": float(np.nanpercentile(boot, 2.5)),
                "ci_upper": float(np.nanpercentile(boot, 97.5)),
                "n_bootstrap": int(n_bootstrap),
                "n_authors": int(len(joined)),
            }
        )

    boot_mean = []
    for _ in range(n_bootstrap):
        sample_idx = rng.integers(0, len(joined), size=len(joined))
        sample = joined.iloc[sample_idx]
        deltas = []
        for target, base_score, comp_score in zip(
            target_cols, baseline_score_cols, comparison_score_cols, strict=True
        ):
            comp_value = _threshold_metric_value(
                sample[target].to_numpy(dtype=int),
                sample[comp_score].to_numpy(dtype=float),
                threshold=float(comp_threshold_lookup.get(target, 0.5)),
                metric=metric,
            )
            base_value = _threshold_metric_value(
                sample[target].to_numpy(dtype=int),
                sample[base_score].to_numpy(dtype=float),
                threshold=float(base_threshold_lookup.get(target, 0.5)),
                metric=metric,
            )
            deltas.append(comp_value - base_value)
        boot_mean.append(float(np.nanmean(deltas)))
    rows.append(
        {
            "comparison": comparison_name,
            "metric": metric,
            "target": "mean",
            "point_estimate": float(np.nanmean(target_deltas)),
            "ci_lower": float(np.nanpercentile(boot_mean, 2.5)),
            "ci_upper": float(np.nanpercentile(boot_mean, 97.5)),
            "n_bootstrap": int(n_bootstrap),
            "n_authors": int(len(joined)),
        }
    )
    return pd.DataFrame(rows)
