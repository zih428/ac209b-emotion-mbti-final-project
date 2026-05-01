"""Experiment-selection helpers that avoid test-set peeking."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class RecipeSelection:
    selected_variant: str
    summary: pd.DataFrame
    primary_metric: str
    tie_breaker: str


def model_comparison_plan() -> pd.DataFrame:
    """Return the planned MS4 model table used by the notebook skeleton."""

    rows = [
        {
            "model_id": "majority",
            "family": "baseline",
            "core": True,
            "training_unit": "author",
            "uses_text": False,
            "uses_emotion": False,
            "stage1_emotion": "none",
            "stage2_text": "none",
            "loss_or_fit": "training-split majority label",
            "aggregation": "constant author score",
            "role": "imbalance floor",
        },
        {
            "model_id": "linear_tfidf",
            "family": "linear sanity baseline",
            "core": True,
            "training_unit": "author",
            "uses_text": True,
            "uses_emotion": False,
            "stage1_emotion": "none",
            "stage2_text": "TF-IDF author document",
            "loss_or_fit": "class-balanced logistic regression",
            "aggregation": "already author-level",
            "role": "cheap lexical sanity check",
        },
        {
            "model_id": "fixed_gru_text_only",
            "family": "fixed GRU",
            "core": True,
            "training_unit": "post",
            "uses_text": True,
            "uses_emotion": False,
            "stage1_emotion": "none",
            "stage2_text": "GRU max_len=128",
            "loss_or_fit": "weighted BCE, recipe selected on validation text-only model",
            "aggregation": "mean post probability per author + validation threshold",
            "role": "critical corrected text-only control",
        },
        {
            "model_id": "fixed_gru_rnn_emotion",
            "family": "fixed GRU",
            "core": True,
            "training_unit": "post",
            "uses_text": True,
            "uses_emotion": True,
            "stage1_emotion": "MS3 RNN emotion probabilities",
            "stage2_text": "same GRU as text-only",
            "loss_or_fit": "same locked weighted BCE recipe",
            "aggregation": "same soft author aggregation",
            "role": "bridge from MS3 emotion channel",
        },
        {
            "model_id": "fixed_gru_distilbert_emotion",
            "family": "fixed GRU",
            "core": True,
            "training_unit": "post",
            "uses_text": True,
            "uses_emotion": True,
            "stage1_emotion": "DistilBERT emotion probabilities",
            "stage2_text": "same GRU as text-only",
            "loss_or_fit": "same locked weighted BCE recipe",
            "aggregation": "same soft author aggregation",
            "role": "headline emotion-informed final model",
        },
        {
            "model_id": "stage2_distilbert_text",
            "family": "future work",
            "core": False,
            "training_unit": "post",
            "uses_text": True,
            "uses_emotion": "optional",
            "stage1_emotion": "optional",
            "stage2_text": "DistilBERT MBTI encoder",
            "loss_or_fit": "not core MS4",
            "aggregation": "not core MS4",
            "role": "future work, avoids confounding emotion comparison",
        },
    ]
    return pd.DataFrame(rows)


def select_weight_recipe(
    validation_metric_tables: dict[str, pd.DataFrame],
    *,
    primary_metric: str = "balanced_accuracy",
    tie_breaker: str = "f1",
    tolerance: float = 1e-6,
) -> RecipeSelection:
    """Select one class-weight recipe using validation metrics only.

    The intended caller passes validation metrics from the fixed text-only GRU
    for `inverse` and `sqrt` variants, then applies the selected recipe
    unchanged to all fixed GRU models.
    """

    if not validation_metric_tables:
        raise ValueError("At least one validation metric table is required")

    rows = []
    for variant, table in validation_metric_tables.items():
        for metric in (primary_metric, tie_breaker):
            if metric not in table.columns:
                raise ValueError(f"Metric table for {variant!r} lacks {metric!r}")
        rows.append(
            {
                "variant": variant,
                f"mean_{primary_metric}": float(table[primary_metric].mean()),
                f"mean_{tie_breaker}": float(table[tie_breaker].mean()),
            }
        )
    summary = pd.DataFrame(rows)
    best_primary = summary[f"mean_{primary_metric}"].max()
    candidates = summary.loc[
        summary[f"mean_{primary_metric}"] >= best_primary - tolerance
    ].copy()
    candidates = candidates.sort_values(
        [f"mean_{tie_breaker}", "variant"], ascending=[False, True]
    )
    selected = str(candidates.iloc[0]["variant"])
    return RecipeSelection(
        selected_variant=selected,
        summary=summary.sort_values(f"mean_{primary_metric}", ascending=False),
        primary_metric=primary_metric,
        tie_breaker=tie_breaker,
    )
