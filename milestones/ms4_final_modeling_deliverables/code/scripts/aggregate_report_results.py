#!/usr/bin/env python3
"""Build report-ready MS4 result tables and figures."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, f1_score

CODE_DIR = Path(__file__).resolve().parents[1]
MS4_DIR = CODE_DIR.parent
SRC_DIR = CODE_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from ms4mbti.config import EMOTION_LABELS, RunConfig
from ms4mbti.data import load_emotion_dataset
from ms4mbti.evaluation import paired_bootstrap_delta
from ms4mbti.transformer_author import make_score_columns
from ms4mbti.viz import (
    PALETTE,
    plot_emotion_increment_deltas,
    plot_frozen_transformer_comparison,
    plot_post_budget_ablation,
    plot_set_attention_comparison,
    set_plot_style,
)


MODEL_SOURCES = {
    "majority_author": CODE_DIR
    / "artifacts"
    / "runs"
    / "author_baselines"
    / "metrics_majority_author.csv",
    "linear_tfidf_author": CODE_DIR
    / "artifacts"
    / "runs"
    / "author_baselines"
    / "metrics_linear_tfidf_author.csv",
    "stage2_text_gru_sqrt": CODE_DIR
    / "artifacts"
    / "runs"
    / "stage2_text_gru_full"
    / "metrics_stage2_text_gru.csv",
    "stage2_text_gru_inverse": CODE_DIR
    / "artifacts"
    / "runs"
    / "stage2_text_gru_inverse_full"
    / "metrics_stage2_text_gru.csv",
    "stage2_text_emotion_gru": CODE_DIR
    / "artifacts"
    / "runs"
    / "stage2_text_emotion_gru_full"
    / "metrics_stage2_text_gru.csv",
}

HISTORY_SOURCES = {
    "stage2_text_gru_sqrt": CODE_DIR
    / "artifacts"
    / "runs"
    / "stage2_text_gru_full"
    / "history.csv",
    "stage2_text_gru_inverse": CODE_DIR
    / "artifacts"
    / "runs"
    / "stage2_text_gru_inverse_full"
    / "history.csv",
    "stage2_text_emotion_gru": CODE_DIR
    / "artifacts"
    / "runs"
    / "stage2_text_emotion_gru_full"
    / "history.csv",
}

DISPLAY_NAMES = {
    "majority_author": "Majority",
    "linear_tfidf_author": "TF-IDF Logistic",
    "stage2_text_gru_sqrt": "GRU Text",
    "stage2_text_gru_inverse": "GRU Text Inverse Weight",
    "stage2_text_emotion_gru": "GRU Text + Emotion",
}

AUTHOR_SCORE_SOURCES = {
    "majority_author": {
        "path": CODE_DIR / "artifacts" / "runs" / "author_baselines" / "author_scores_majority_author.csv",
        "thresholds": CODE_DIR
        / "artifacts"
        / "runs"
        / "author_baselines"
        / "thresholds_majority_author.csv",
        "score_prefix": "score_majority",
    },
    "linear_tfidf_author": {
        "path": CODE_DIR
        / "artifacts"
        / "runs"
        / "author_baselines"
        / "author_scores_linear_tfidf_author.csv",
        "thresholds": CODE_DIR
        / "artifacts"
        / "runs"
        / "author_baselines"
        / "thresholds_linear_tfidf_author.csv",
        "score_prefix": "score_linear",
    },
    "stage2_text_gru_sqrt": {
        "path": CODE_DIR
        / "artifacts"
        / "runs"
        / "stage2_text_gru_full"
        / "author_scores_stage2_text_gru.csv",
        "thresholds": CODE_DIR
        / "artifacts"
        / "runs"
        / "stage2_text_gru_full"
        / "thresholds_stage2_text_gru.csv",
        "score_prefix": "score_gru",
    },
    "stage2_text_gru_inverse": {
        "path": CODE_DIR
        / "artifacts"
        / "runs"
        / "stage2_text_gru_inverse_full"
        / "author_scores_stage2_text_gru.csv",
        "thresholds": CODE_DIR
        / "artifacts"
        / "runs"
        / "stage2_text_gru_inverse_full"
        / "thresholds_stage2_text_gru.csv",
        "score_prefix": "score_gru",
    },
    "stage2_text_emotion_gru": {
        "path": CODE_DIR
        / "artifacts"
        / "runs"
        / "stage2_text_emotion_gru_full"
        / "author_scores_stage2_text_gru.csv",
        "thresholds": CODE_DIR
        / "artifacts"
        / "runs"
        / "stage2_text_emotion_gru_full"
        / "thresholds_stage2_text_gru.csv",
        "score_prefix": "score_gru",
    },
}

GRU_DIAGNOSTIC_DIR = CODE_DIR / "artifacts" / "runs" / "stage2_text_emotion_gru_full"
GRU_DIAGNOSTIC_AUTHOR_SCORES = GRU_DIAGNOSTIC_DIR / "author_scores_stage2_text_gru.csv"
GRU_DIAGNOSTIC_THRESHOLDS = GRU_DIAGNOSTIC_DIR / "thresholds_stage2_text_gru.csv"
TEXT_GRU_128_DIR = CODE_DIR / "artifacts" / "runs" / "stage2_text_gru_full"
TEXT_GRU_256_DIR = CODE_DIR / "artifacts" / "runs" / "stage2_text_gru_len256_full"
EMOTION_CACHE = CODE_DIR / "artifacts" / "cache" / "emotion_probs_full.parquet"
PREPROCESSED_POSTS = CODE_DIR / "artifacts" / "preprocessed" / "full" / "modeling_posts.parquet"
TRANSFORMER_AUTHOR_DIR = CODE_DIR / "artifacts" / "runs" / "transformer_author"
SET_ATTENTION_AUTHOR_DIR = CODE_DIR / "artifacts" / "runs" / "set_attention_author"
SET_ATTENTION_CONFUSION_MODEL_ID = "set_attention_text_p200"
SET_ATTENTION_CONFUSION_AUTHOR_SCORES = (
    SET_ATTENTION_AUTHOR_DIR / f"author_scores_{SET_ATTENTION_CONFUSION_MODEL_ID}.csv"
)
SET_ATTENTION_CONFUSION_THRESHOLDS = (
    SET_ATTENTION_AUTHOR_DIR / f"thresholds_{SET_ATTENTION_CONFUSION_MODEL_ID}.csv"
)
TARGETS = ("target_E", "target_S", "target_T", "target_J")

SUPPLEMENTAL_SET_ATTENTION_RUNS = [
    {
        "run_id": "p200_e5_seed209067",
        "analysis": "multi_seed",
        "seed": 209067,
        "epochs": 5,
        "post_budget": 200,
        "run_dir": CODE_DIR / "artifacts" / "runs" / "set_attention_author_p200_e5_seed209067",
    },
    {
        "run_id": "p200_e5_seed209068",
        "analysis": "multi_seed",
        "seed": 209068,
        "epochs": 5,
        "post_budget": 200,
        "run_dir": CODE_DIR / "artifacts" / "runs" / "set_attention_author_p200_e5_seed209068",
    },
    {
        "run_id": "p200_e10_seed209066",
        "analysis": "epoch_sensitivity",
        "seed": 209066,
        "epochs": 10,
        "post_budget": 200,
        "run_dir": CODE_DIR / "artifacts" / "runs" / "set_attention_author_p200_e10_seed209066",
    },
    {
        "run_id": "p200_e20_seed209066",
        "analysis": "epoch_sensitivity",
        "seed": 209066,
        "epochs": 20,
        "post_budget": 200,
        "run_dir": CODE_DIR / "artifacts" / "runs" / "set_attention_author_p200_e20_seed209066",
    },
]

TRANSFORMER_DISPLAY_NAMES = {
    "frozen_text_mean_std": "Frozen Text",
    "frozen_emotion_only": "Emotion Only",
    "frozen_text_shuffled_emotion": "Frozen Text + Shuffled Emotion",
    "frozen_text_real_emotion": "Frozen Text + Real Emotion",
    "frozen_text_controls": "Frozen Text + Controls",
    "frozen_text_real_emotion_controls": "Frozen Text + Real Emotion + Controls",
}

SET_ATTENTION_DISPLAY_PREFIXES = {
    "mean_pool_mlp_text": "Mean Pool MLP",
    "mean_std_pool_mlp_text": "Mean+Std Pool MLP",
    "set_attention_text": "Set Attention Text",
    "set_attention_text_shuffled_emotion": "Set Attention Text + Shuffled Emotion",
    "set_attention_text_real_emotion": "Set Attention Text + Real Emotion",
    "set_attention_text_controls": "Set Attention Text + Controls",
    "set_attention_text_real_emotion_controls": "Set Attention Text + Real Emotion + Controls",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=MS4_DIR / "report" / "results",
    )
    return parser.parse_args()


def require_files(paths: dict[str, Path]) -> None:
    missing = {name: path for name, path in paths.items() if not path.exists()}
    if missing:
        formatted = "\n".join(f"- {name}: {path}" for name, path in missing.items())
        raise FileNotFoundError(f"Missing required result artifacts:\n{formatted}")


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True, default=str), encoding="utf-8")


def load_metrics() -> pd.DataFrame:
    require_files(MODEL_SOURCES)
    frames = []
    for model_id, path in MODEL_SOURCES.items():
        frame = pd.read_csv(path)
        frame["model_id"] = model_id
        frame["model_name"] = DISPLAY_NAMES[model_id]
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def load_histories() -> pd.DataFrame:
    require_files(HISTORY_SOURCES)
    frames = []
    for model_id, path in HISTORY_SOURCES.items():
        frame = pd.read_csv(path)
        frame["model_id"] = model_id
        frame["model_name"] = DISPLAY_NAMES[model_id]
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def make_summary(metrics: pd.DataFrame) -> pd.DataFrame:
    test = metrics.loc[metrics["split"] == "test"].copy()
    columns = [
        "balanced_accuracy",
        "f1",
        "minority_recall",
        "roc_auc",
        "average_precision",
    ]
    summary = (
        test.groupby(["model_id", "model_name"], as_index=False)[columns]
        .mean()
        .sort_values("balanced_accuracy", ascending=False)
    )
    return summary


def save_model_summary_plot(summary: pd.DataFrame, output_dir: Path) -> Path:
    plot_df = summary.sort_values("balanced_accuracy", ascending=True)
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    sns.barplot(
        data=plot_df,
        y="model_name",
        x="balanced_accuracy",
        color=PALETTE["blue"],
        ax=ax,
    )
    ax.axvline(0.5, color=PALETTE["muted"], linestyle="--", linewidth=1)
    ax.set_xlim(0.45, max(0.72, float(plot_df["balanced_accuracy"].max()) + 0.03))
    ax.set_xlabel("Mean test balanced accuracy")
    ax.set_ylabel("")
    for container in ax.containers:
        ax.bar_label(container, fmt="%.3f", padding=4)
    fig.tight_layout()
    path = output_dir / "fig_model_summary_balanced_accuracy.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return path


def save_target_comparison_plot(metrics: pd.DataFrame, output_dir: Path) -> Path:
    test = metrics.loc[metrics["split"] == "test"].copy()
    keep = [
        "majority_author",
        "linear_tfidf_author",
        "stage2_text_gru_sqrt",
        "stage2_text_emotion_gru",
    ]
    test = test.loc[test["model_id"].isin(keep)].copy()
    fig, ax = plt.subplots(figsize=(10, 5.2))
    sns.barplot(
        data=test,
        x="target",
        y="balanced_accuracy",
        hue="model_name",
        ax=ax,
    )
    ax.axhline(0.5, color=PALETTE["muted"], linestyle="--", linewidth=1)
    ax.set_ylim(0.45, 0.78)
    ax.set_xlabel("")
    ax.set_ylabel("Test balanced accuracy")
    ax.legend(title="", loc="lower right")
    fig.tight_layout()
    path = output_dir / "fig_target_balanced_accuracy.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return path


def save_training_curve_plot(histories: pd.DataFrame, output_dir: Path) -> Path:
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    sns.lineplot(
        data=histories,
        x="epoch",
        y="val_loss",
        hue="model_name",
        marker="o",
        ax=ax,
    )
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Validation loss")
    ax.legend(title="")
    fig.tight_layout()
    path = output_dir / "fig_gru_validation_loss.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return path


def save_emotion_gain_plot(metrics: pd.DataFrame, output_dir: Path) -> Path:
    test = metrics.loc[
        (metrics["split"] == "test")
        & metrics["model_id"].isin(["stage2_text_gru_sqrt", "stage2_text_emotion_gru"])
    ].copy()
    wide = test.pivot(index="target", columns="model_id", values="balanced_accuracy")
    wide["emotion_gain"] = (
        wide["stage2_text_emotion_gru"] - wide["stage2_text_gru_sqrt"]
    )
    plot_df = wide.reset_index()
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    sns.barplot(data=plot_df, x="target", y="emotion_gain", color=PALETTE["teal"], ax=ax)
    ax.axhline(0, color=PALETTE["muted"], linewidth=1)
    ax.set_xlabel("")
    ax.set_ylabel("Balanced accuracy gain")
    for container in ax.containers:
        ax.bar_label(container, fmt="%+.3f", padding=4)
    fig.tight_layout()
    path = output_dir / "fig_emotion_gain_by_target.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return path


def load_gru_diagnostic_author_scores() -> tuple[pd.DataFrame, pd.DataFrame]:
    require_files(
        {
            "gru_diagnostic_author_scores": GRU_DIAGNOSTIC_AUTHOR_SCORES,
            "gru_diagnostic_thresholds": GRU_DIAGNOSTIC_THRESHOLDS,
        }
    )
    return pd.read_csv(GRU_DIAGNOSTIC_AUTHOR_SCORES), pd.read_csv(GRU_DIAGNOSTIC_THRESHOLDS)


def load_set_attention_confusion_author_scores() -> tuple[pd.DataFrame, pd.DataFrame]:
    require_files(
        {
            "set_attention_author_scores": SET_ATTENTION_CONFUSION_AUTHOR_SCORES,
            "set_attention_thresholds": SET_ATTENTION_CONFUSION_THRESHOLDS,
        }
    )
    return (
        pd.read_csv(SET_ATTENTION_CONFUSION_AUTHOR_SCORES),
        pd.read_csv(SET_ATTENTION_CONFUSION_THRESHOLDS),
    )


def load_author_score_inputs() -> dict[str, dict[str, Any]]:
    paths: dict[str, Path] = {}
    for model_id, spec in AUTHOR_SCORE_SOURCES.items():
        paths[f"{model_id}_author_scores"] = spec["path"]
        paths[f"{model_id}_thresholds"] = spec["thresholds"]
    require_files(paths)

    loaded = {}
    for model_id, spec in AUTHOR_SCORE_SOURCES.items():
        loaded[model_id] = {
            "author_scores": pd.read_csv(spec["path"]),
            "thresholds": pd.read_csv(spec["thresholds"]),
            "score_prefix": spec["score_prefix"],
        }
    return loaded


def score_columns(prefix: str) -> tuple[str, ...]:
    return tuple(f"{prefix}_{target}" for target in TARGETS)


def make_threshold_curves(author_scores: pd.DataFrame) -> pd.DataFrame:
    validation = author_scores.loc[author_scores["split"] == "val"].copy()
    rows = []
    for target in TARGETS:
        score_col = f"score_gru_{target}"
        y_true = validation[target].to_numpy(dtype=int)
        y_score = validation[score_col].to_numpy(dtype=float)
        for threshold in np.linspace(0.05, 0.95, 181):
            y_pred = (y_score >= threshold).astype(int)
            rows.append(
                {
                    "target": target,
                    "threshold": float(threshold),
                    "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
                    "f1": float(f1_score(y_true, y_pred, zero_division=0)),
                }
            )
    return pd.DataFrame(rows)


def _threshold_lookup(thresholds: pd.DataFrame) -> dict[str, float]:
    return dict(zip(thresholds["target"], thresholds["threshold"], strict=True))


def _mean_threshold_metric(
    frame: pd.DataFrame,
    *,
    score_cols: tuple[str, ...],
    thresholds: dict[str, float],
    metric: str,
) -> float:
    values = []
    for target, score_col in zip(TARGETS, score_cols, strict=True):
        y_true = frame[target].to_numpy(dtype=int)
        if len(np.unique(y_true)) < 2:
            continue
        y_pred = (frame[score_col].to_numpy(dtype=float) >= thresholds[target]).astype(int)
        if metric == "balanced_accuracy":
            values.append(balanced_accuracy_score(y_true, y_pred))
        elif metric == "f1":
            values.append(f1_score(y_true, y_pred, zero_division=0))
        else:
            raise ValueError("metric must be balanced_accuracy or f1")
    return float(np.mean(values)) if values else float("nan")


def make_bootstrap_intervals(
    author_inputs: dict[str, dict[str, Any]],
    *,
    n_bootstrap: int = 2000,
    seed: int = 209066,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    interval_rows = []
    sample_values: dict[str, np.ndarray] = {}

    for model_id, payload in author_inputs.items():
        test = payload["author_scores"].loc[payload["author_scores"]["split"] == "test"].copy()
        thresholds = _threshold_lookup(payload["thresholds"])
        cols = score_columns(payload["score_prefix"])
        n_authors = len(test)
        boot = {"balanced_accuracy": [], "f1": []}
        for _ in range(n_bootstrap):
            sample = test.iloc[rng.integers(0, n_authors, size=n_authors)]
            for metric in boot:
                boot[metric].append(
                    _mean_threshold_metric(
                        sample,
                        score_cols=cols,
                        thresholds=thresholds,
                        metric=metric,
                    )
                )
        for metric, values in boot.items():
            arr = np.asarray(values, dtype=float)
            if metric == "balanced_accuracy":
                sample_values[model_id] = arr
            interval_rows.append(
                {
                    "model_id": model_id,
                    "model_name": DISPLAY_NAMES[model_id],
                    "metric": metric,
                    "point_estimate": _mean_threshold_metric(
                        test,
                        score_cols=cols,
                        thresholds=thresholds,
                        metric=metric,
                    ),
                    "ci_lower": float(np.nanpercentile(arr, 2.5)),
                    "ci_upper": float(np.nanpercentile(arr, 97.5)),
                    "n_bootstrap": n_bootstrap,
                    "n_test_authors": n_authors,
                }
            )

    difference_pairs = [
        ("stage2_text_emotion_gru", "stage2_text_gru_sqrt"),
        ("linear_tfidf_author", "stage2_text_emotion_gru"),
    ]
    difference_rows = []
    for left, right in difference_pairs:
        diff = sample_values[left] - sample_values[right]
        difference_rows.append(
            {
                "comparison": f"{DISPLAY_NAMES[left]} minus {DISPLAY_NAMES[right]}",
                "metric": "balanced_accuracy",
                "point_estimate": float(np.nanmean(diff)),
                "ci_lower": float(np.nanpercentile(diff, 2.5)),
                "ci_upper": float(np.nanpercentile(diff, 97.5)),
                "n_bootstrap": n_bootstrap,
            }
        )
    return pd.DataFrame(interval_rows), pd.DataFrame(difference_rows)


def save_bootstrap_interval_plot(intervals: pd.DataFrame, output_dir: Path) -> Path:
    plot_df = intervals.loc[intervals["metric"] == "balanced_accuracy"].sort_values(
        "point_estimate"
    )
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    ax.errorbar(
        x=plot_df["point_estimate"],
        y=plot_df["model_name"],
        xerr=[
            plot_df["point_estimate"] - plot_df["ci_lower"],
            plot_df["ci_upper"] - plot_df["point_estimate"],
        ],
        fmt="o",
        color=PALETTE["blue"],
        ecolor=PALETTE["muted"],
        capsize=4,
    )
    ax.axvline(0.5, color=PALETTE["muted"], linestyle="--", linewidth=1)
    ax.set_xlim(0.45, max(0.72, float(plot_df["ci_upper"].max()) + 0.02))
    ax.set_xlabel("Mean test balanced accuracy with 95% bootstrap CI")
    ax.set_ylabel("")
    fig.tight_layout()
    path = output_dir / "fig_bootstrap_balanced_accuracy_ci.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return path


def make_threshold_objective_sensitivity(author_scores: pd.DataFrame) -> pd.DataFrame:
    validation = author_scores.loc[author_scores["split"] == "val"].copy()
    test = author_scores.loc[author_scores["split"] == "test"].copy()
    rows = []
    for target in TARGETS:
        score_col = f"score_gru_{target}"
        y_val = validation[target].to_numpy(dtype=int)
        y_val_score = validation[score_col].to_numpy(dtype=float)
        y_test = test[target].to_numpy(dtype=int)
        y_test_score = test[score_col].to_numpy(dtype=float)
        for objective in ("balanced_accuracy", "f1"):
            candidates = []
            for threshold in np.linspace(0.05, 0.95, 181):
                y_val_pred = (y_val_score >= threshold).astype(int)
                if objective == "balanced_accuracy":
                    validation_score = balanced_accuracy_score(y_val, y_val_pred)
                else:
                    validation_score = f1_score(y_val, y_val_pred, zero_division=0)
                candidates.append((validation_score, -abs(threshold - 0.5), threshold))
            _, _, selected = max(candidates)
            y_test_pred = (y_test_score >= selected).astype(int)
            rows.append(
                {
                    "target": target,
                    "threshold_objective": objective,
                    "selected_threshold": float(selected),
                    "validation_objective_score": float(max(candidates)[0]),
                    "test_balanced_accuracy": float(balanced_accuracy_score(y_test, y_test_pred)),
                    "test_f1": float(f1_score(y_test, y_test_pred, zero_division=0)),
                }
            )
    return pd.DataFrame(rows)


def save_threshold_objective_plot(sensitivity: pd.DataFrame, output_dir: Path) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    sns.barplot(
        data=sensitivity,
        x="target",
        y="selected_threshold",
        hue="threshold_objective",
        ax=axes[0],
    )
    axes[0].set_xlabel("")
    axes[0].set_ylabel("Selected threshold")
    axes[0].legend(title="")

    sns.barplot(
        data=sensitivity,
        x="target",
        y="test_balanced_accuracy",
        hue="threshold_objective",
        ax=axes[1],
    )
    axes[1].axhline(0.5, color=PALETTE["muted"], linestyle="--", linewidth=1)
    axes[1].set_xlabel("")
    axes[1].set_ylabel("Test balanced accuracy")
    axes[1].legend(title="")
    fig.tight_layout()
    path = output_dir / "fig_threshold_objective_sensitivity.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return path


def make_token_length_sensitivity() -> pd.DataFrame:
    require_files({"preprocessed_posts": PREPROCESSED_POSTS})
    posts = pd.read_parquet(PREPROCESSED_POSTS, columns=["author", "text_masked"])
    lengths = posts["text_masked"].fillna("").astype(str).str.split().str.len()
    rows = []
    for max_length in (128, 256):
        truncated = lengths > max_length
        author_exposure = truncated.groupby(posts["author"]).max()
        rows.append(
            {
                "max_length": max_length,
                "n_posts": int(len(posts)),
                "n_authors": int(posts["author"].nunique()),
                "median_tokens": float(np.percentile(lengths, 50)),
                "p90_tokens": float(np.percentile(lengths, 90)),
                "p95_tokens": float(np.percentile(lengths, 95)),
                "p99_tokens": float(np.percentile(lengths, 99)),
                "share_posts_over_max": float(truncated.mean()),
                "share_authors_with_any_truncation": float(author_exposure.mean()),
            }
        )
    return pd.DataFrame(rows)


def save_token_length_sensitivity_plot(sensitivity: pd.DataFrame, output_dir: Path) -> Path:
    plot_df = sensitivity.melt(
        id_vars="max_length",
        value_vars=["share_posts_over_max", "share_authors_with_any_truncation"],
        var_name="measure",
        value_name="share",
    )
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    sns.barplot(data=plot_df, x="max_length", y="share", hue="measure", ax=ax)
    ax.set_xlabel("Stage 2 max length")
    ax.set_ylabel("Share")
    ax.yaxis.set_major_formatter(lambda value, _: f"{value:.0%}")
    ax.legend(title="")
    fig.tight_layout()
    path = output_dir / "fig_token_length_sensitivity.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return path


def make_max_length_training_sensitivity() -> pd.DataFrame:
    sources = {
        128: TEXT_GRU_128_DIR,
        256: TEXT_GRU_256_DIR,
    }
    require_files(
        {
            f"max_length_{max_length}_metrics": path / "metrics_stage2_text_gru.csv"
            for max_length, path in sources.items()
        }
        | {
            f"max_length_{max_length}_history": path / "history.csv"
            for max_length, path in sources.items()
        }
        | {
            f"max_length_{max_length}_settings": path / "run_settings.json"
            for max_length, path in sources.items()
        }
    )
    rows = []
    for max_length, path in sources.items():
        metrics = pd.read_csv(path / "metrics_stage2_text_gru.csv")
        history = pd.read_csv(path / "history.csv")
        settings = json.loads((path / "run_settings.json").read_text(encoding="utf-8"))
        test = metrics.loc[metrics["split"] == "test"]
        rows.append(
            {
                "max_length": max_length,
                "run_id": settings["run_id"],
                "epochs_completed": int(len(history)),
                "best_val_loss": float(history["val_loss"].min()),
                "final_train_loss": float(history["train_loss"].iloc[-1]),
                "test_mean_balanced_accuracy": float(test["balanced_accuracy"].mean()),
                "test_mean_f1": float(test["f1"].mean()),
                "parameter_count": int(
                    json.loads((path / "summary.json").read_text(encoding="utf-8"))[
                        "parameter_count"
                    ]
                ),
            }
        )
    return pd.DataFrame(rows)


def save_max_length_training_plot(sensitivity: pd.DataFrame, output_dir: Path) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    sns.barplot(
        data=sensitivity,
        x="max_length",
        y="best_val_loss",
        color=PALETTE["blue"],
        ax=axes[0],
    )
    axes[0].set_xlabel("Stage 2 max length")
    axes[0].set_ylabel("Best validation loss")
    for container in axes[0].containers:
        axes[0].bar_label(container, fmt="%.3f", padding=4)

    sns.barplot(
        data=sensitivity,
        x="max_length",
        y="test_mean_balanced_accuracy",
        color=PALETTE["teal"],
        ax=axes[1],
    )
    axes[1].axhline(0.5, color=PALETTE["muted"], linestyle="--", linewidth=1)
    axes[1].set_ylim(0.48, max(0.64, float(sensitivity["test_mean_balanced_accuracy"].max()) + 0.03))
    axes[1].set_xlabel("Stage 2 max length")
    axes[1].set_ylabel("Mean test balanced accuracy")
    for container in axes[1].containers:
        axes[1].bar_label(container, fmt="%.3f", padding=4)
    fig.tight_layout()
    path = output_dir / "fig_max_length_training_sensitivity.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return path


def read_optional_metrics(run_dir: Path, display_names: dict[str, str]) -> pd.DataFrame:
    frames = []
    if not run_dir.exists():
        return pd.DataFrame(
            columns=[
                "target",
                "n_authors",
                "threshold",
                "balanced_accuracy",
                "f1",
                "minority_precision",
                "minority_recall",
                "roc_auc",
                "average_precision",
                "raw_accuracy",
                "positive_rate_pred",
                "positive_rate_true",
                "model_id",
                "split",
                "model_name",
            ]
        )
    for path in sorted(run_dir.glob("metrics_*.csv")):
        model_id = path.stem.removeprefix("metrics_")
        frame = pd.read_csv(path)
        frame["model_id"] = model_id
        frame["model_name"] = display_names.get(model_id, model_id)
        frames.append(frame)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def set_attention_display_name(model_id: str) -> str:
    for prefix, name in sorted(
        SET_ATTENTION_DISPLAY_PREFIXES.items(),
        key=lambda item: len(item[0]),
        reverse=True,
    ):
        suffix = model_id.removeprefix(prefix)
        if suffix != model_id:
            return f"{name}{suffix.replace('_p', ' p=')}"
    return model_id


def read_optional_set_metrics(run_dir: Path) -> pd.DataFrame:
    if not run_dir.exists():
        return read_optional_metrics(run_dir, {})
    frames = []
    for path in sorted(run_dir.glob("metrics_*.csv")):
        model_id = path.stem.removeprefix("metrics_")
        frame = pd.read_csv(path)
        frame["model_id"] = model_id
        frame["model_name"] = set_attention_display_name(model_id)
        frames.append(frame)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def make_optional_summary(metrics: pd.DataFrame) -> pd.DataFrame:
    if metrics.empty:
        return pd.DataFrame(
            columns=[
                "model_id",
                "model_name",
                "balanced_accuracy",
                "f1",
                "minority_recall",
                "roc_auc",
                "average_precision",
                "post_budget",
            ]
        )
    test = metrics.loc[metrics["split"] == "test"].copy()
    summary = (
        test.groupby(["model_id", "model_name"], as_index=False)[
            ["balanced_accuracy", "f1", "minority_recall", "roc_auc", "average_precision"]
        ]
        .mean()
        .sort_values("balanced_accuracy", ascending=False)
    )
    summary["post_budget"] = summary["model_id"].str.extract(r"_p(\d+)$").astype(float)
    return summary


def make_set_attention_supplemental_summary() -> pd.DataFrame:
    frames = []
    for spec in SUPPLEMENTAL_SET_ATTENTION_RUNS:
        run_dir = spec["run_dir"]
        if not run_dir.exists():
            continue
        metrics = read_optional_set_metrics(run_dir)
        if metrics.empty:
            continue
        summary = make_optional_summary(metrics)
        summary["run_id"] = spec["run_id"]
        summary["analysis"] = spec["analysis"]
        summary["seed"] = spec["seed"]
        summary["epochs"] = spec["epochs"]
        summary["post_budget"] = spec["post_budget"]
        frames.append(summary)
    columns = [
        "analysis",
        "run_id",
        "seed",
        "epochs",
        "post_budget",
        "model_id",
        "model_name",
        "balanced_accuracy",
        "f1",
        "minority_recall",
        "roc_auc",
        "average_precision",
    ]
    if not frames:
        return pd.DataFrame(columns=columns)
    return pd.concat(frames, ignore_index=True)[columns].sort_values(
        ["analysis", "epochs", "seed", "balanced_accuracy"],
        ascending=[True, True, True, False],
    )


def make_set_attention_stability_summary(
    supplemental_summary: pd.DataFrame,
    set_attention_summary: pd.DataFrame,
) -> pd.DataFrame:
    key_models = {
        "set_attention_text_p200",
        "set_attention_text_real_emotion_p200",
        "set_attention_text_shuffled_emotion_p200",
    }
    base = set_attention_summary.loc[set_attention_summary["model_id"].isin(key_models)].copy()
    if not base.empty:
        base["analysis"] = "multi_seed"
        base["run_id"] = "p200_e5_seed209066"
        base["seed"] = 209066
        base["epochs"] = 5
        base["post_budget"] = 200
    extra = supplemental_summary.loc[
        (supplemental_summary["analysis"] == "multi_seed")
        & (supplemental_summary["model_id"].isin(key_models))
    ].copy()
    combined = pd.concat([base, extra], ignore_index=True)
    if combined.empty:
        return pd.DataFrame(
            columns=[
                "model_id",
                "model_name",
                "post_budget",
                "epochs",
                "n_seeds",
                "mean_balanced_accuracy",
                "std_balanced_accuracy",
                "min_balanced_accuracy",
                "max_balanced_accuracy",
            ]
        )
    return (
        combined.groupby(["model_id", "model_name", "post_budget", "epochs"], as_index=False)
        .agg(
            n_seeds=("seed", "nunique"),
            mean_balanced_accuracy=("balanced_accuracy", "mean"),
            std_balanced_accuracy=("balanced_accuracy", "std"),
            min_balanced_accuracy=("balanced_accuracy", "min"),
            max_balanced_accuracy=("balanced_accuracy", "max"),
        )
        .sort_values("mean_balanced_accuracy", ascending=False)
    )


def save_set_attention_stability_plot(stability: pd.DataFrame, output_dir: Path) -> Path:
    fig, ax = plt.subplots(figsize=(9, 4.8))
    plot_df = stability.sort_values("mean_balanced_accuracy", ascending=True)
    ax.barh(
        plot_df["model_name"],
        plot_df["mean_balanced_accuracy"],
        xerr=plot_df["std_balanced_accuracy"].fillna(0.0),
        color=PALETTE["blue"],
        alpha=0.9,
    )
    ax.axvline(0.5, color=PALETTE["muted"], linestyle="--", linewidth=1)
    ax.set_xlabel("Mean test balanced accuracy across seeds")
    ax.set_ylabel("")
    upper = max(0.72, float(plot_df["mean_balanced_accuracy"].max()) + 0.04)
    ax.set_xlim(0.58, upper)
    for y, value in enumerate(plot_df["mean_balanced_accuracy"]):
        ax.text(value + 0.003, y, f"{value:.3f}", va="center")
    fig.tight_layout()
    path = output_dir / "fig_set_attention_seed_stability.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return path


def save_set_attention_epoch_sensitivity_plot(
    supplemental_summary: pd.DataFrame,
    set_attention_summary: pd.DataFrame,
    output_dir: Path,
) -> Path:
    key_models = {
        "set_attention_text_p200",
        "set_attention_text_real_emotion_p200",
        "set_attention_text_shuffled_emotion_p200",
    }
    base = set_attention_summary.loc[set_attention_summary["model_id"].isin(key_models)].copy()
    if not base.empty:
        base["analysis"] = "epoch_sensitivity"
        base["run_id"] = "p200_e5_seed209066"
        base["seed"] = 209066
        base["epochs"] = 5
        base["post_budget"] = 200
    extra = supplemental_summary.loc[
        (supplemental_summary["analysis"] == "epoch_sensitivity")
        & (supplemental_summary["model_id"].isin(key_models))
    ].copy()
    plot_df = pd.concat([base, extra], ignore_index=True)
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    sns.lineplot(
        data=plot_df,
        x="epochs",
        y="balanced_accuracy",
        hue="model_name",
        marker="o",
        ax=ax,
    )
    ax.axhline(0.5, color=PALETTE["muted"], linestyle="--", linewidth=1)
    ax.set_xlabel("Max epochs (early stopping enabled)")
    ax.set_ylabel("Test balanced accuracy")
    ax.set_ylim(0.60, max(0.72, float(plot_df["balanced_accuracy"].max()) + 0.03))
    ax.legend(title="")
    fig.tight_layout()
    path = output_dir / "fig_set_attention_epoch_sensitivity.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return path


def threshold_series(path: Path) -> pd.Series:
    frame = pd.read_csv(path)
    return pd.Series(frame["threshold"].to_numpy(dtype=float), index=frame["target"])


def make_transformer_delta_table(
    *,
    run_dir: Path,
    family: str,
    baseline_model: str,
    real_model: str,
    shuffled_model: str,
    n_bootstrap: int = 2000,
) -> pd.DataFrame:
    required = {
        "baseline_scores": run_dir / f"author_scores_{baseline_model}.csv",
        "real_scores": run_dir / f"author_scores_{real_model}.csv",
        "shuffled_scores": run_dir / f"author_scores_{shuffled_model}.csv",
        "baseline_thresholds": run_dir / f"thresholds_{baseline_model}.csv",
        "real_thresholds": run_dir / f"thresholds_{real_model}.csv",
        "shuffled_thresholds": run_dir / f"thresholds_{shuffled_model}.csv",
    }
    if any(not path.exists() for path in required.values()):
        return pd.DataFrame()

    baseline = pd.read_csv(required["baseline_scores"])
    real = pd.read_csv(required["real_scores"])
    shuffled = pd.read_csv(required["shuffled_scores"])
    if "split" in baseline.columns:
        baseline = baseline.loc[baseline["split"] == "test"].copy()
    if "split" in real.columns:
        real = real.loc[real["split"] == "test"].copy()
    if "split" in shuffled.columns:
        shuffled = shuffled.loc[shuffled["split"] == "test"].copy()
    baseline_thresholds = threshold_series(required["baseline_thresholds"])
    real_thresholds = threshold_series(required["real_thresholds"])
    shuffled_thresholds = threshold_series(required["shuffled_thresholds"])
    baseline_cols = make_score_columns(baseline_model)
    real_cols = make_score_columns(real_model)
    shuffled_cols = make_score_columns(shuffled_model)
    frames = [
        paired_bootstrap_delta(
            baseline,
            real,
            baseline_score_cols=baseline_cols,
            comparison_score_cols=real_cols,
            baseline_thresholds=baseline_thresholds,
            comparison_thresholds=real_thresholds,
            comparison_name=f"{family}: real emotion minus text",
            n_bootstrap=n_bootstrap,
        ),
        paired_bootstrap_delta(
            baseline,
            shuffled,
            baseline_score_cols=baseline_cols,
            comparison_score_cols=shuffled_cols,
            baseline_thresholds=baseline_thresholds,
            comparison_thresholds=shuffled_thresholds,
            comparison_name=f"{family}: shuffled emotion minus text",
            n_bootstrap=n_bootstrap,
        ),
    ]
    return pd.concat(frames, ignore_index=True)


def make_all_transformer_deltas() -> pd.DataFrame:
    frames = []
    frozen = make_transformer_delta_table(
        run_dir=TRANSFORMER_AUTHOR_DIR,
        family="Frozen transformer",
        baseline_model="frozen_text_mean_std",
        real_model="frozen_text_real_emotion",
        shuffled_model="frozen_text_shuffled_emotion",
    )
    if not frozen.empty:
        frames.append(frozen)

    for post_budget in (50, 200):
        suffix = f"_p{post_budget}"
        set_delta = make_transformer_delta_table(
            run_dir=SET_ATTENTION_AUTHOR_DIR,
            family=f"Set attention p={post_budget}",
            baseline_model=f"set_attention_text{suffix}",
            real_model=f"set_attention_text_real_emotion{suffix}",
            shuffled_model=f"set_attention_text_shuffled_emotion{suffix}",
        )
        if not set_delta.empty:
            frames.append(set_delta)
    return (
        pd.concat(frames, ignore_index=True)
        if frames
        else pd.DataFrame(
            columns=[
                "comparison",
                "metric",
                "target",
                "point_estimate",
                "ci_lower",
                "ci_upper",
                "n_bootstrap",
                "n_authors",
            ]
        )
    )


def make_transformer_artifact_status(
    frozen_metrics: pd.DataFrame,
    set_metrics: pd.DataFrame,
    deltas: pd.DataFrame,
) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "artifact_family": "frozen_transformer_author",
                "expected_directory": TRANSFORMER_AUTHOR_DIR.relative_to(CODE_DIR).as_posix(),
                "status": "available" if not frozen_metrics.empty else "missing",
                "n_metric_rows": int(len(frozen_metrics)),
            },
            {
                "artifact_family": "set_attention_author",
                "expected_directory": SET_ATTENTION_AUTHOR_DIR.relative_to(CODE_DIR).as_posix(),
                "status": "available" if not set_metrics.empty else "missing",
                "n_metric_rows": int(len(set_metrics)),
            },
            {
                "artifact_family": "paired_emotion_deltas",
                "expected_directory": "derived from author score artifacts",
                "status": "available" if not deltas.empty else "missing",
                "n_metric_rows": int(len(deltas)),
            },
        ]
    )


def save_optional_figure(fig, output_dir: Path, filename: str) -> Path:
    path = output_dir / filename
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return path


def save_threshold_curve_plot(
    threshold_curves: pd.DataFrame,
    selected_thresholds: pd.DataFrame,
    output_dir: Path,
) -> Path:
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), sharex=True, sharey=True)
    for ax, target in zip(axes.ravel(), TARGETS, strict=True):
        subset = threshold_curves.loc[threshold_curves["target"] == target]
        sns.lineplot(
            data=subset,
            x="threshold",
            y="balanced_accuracy",
            color=PALETTE["blue"],
            ax=ax,
        )
        selected = float(
            selected_thresholds.loc[selected_thresholds["target"] == target, "threshold"].iloc[0]
        )
        ax.axvline(selected, color=PALETTE["red"], linestyle="--", linewidth=1)
        ax.set_title(target)
        ax.set_ylim(0.45, 0.75)
        ax.set_xlabel("Threshold")
        ax.set_ylabel("Validation balanced accuracy")
    fig.tight_layout()
    path = output_dir / "fig_final_threshold_tuning_curves.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return path


def make_confusion_matrices(
    author_scores: pd.DataFrame,
    selected_thresholds: pd.DataFrame,
    *,
    score_cols: tuple[str, ...],
) -> pd.DataFrame:
    test = author_scores.loc[author_scores["split"] == "test"].copy()
    rows = []
    for target, score_col in zip(TARGETS, score_cols, strict=True):
        threshold = float(
            selected_thresholds.loc[selected_thresholds["target"] == target, "threshold"].iloc[0]
        )
        y_true = test[target].to_numpy(dtype=int)
        y_pred = (test[score_col].to_numpy(dtype=float) >= threshold).astype(int)
        matrix = confusion_matrix(y_true, y_pred, labels=[0, 1])
        for true_label in (0, 1):
            for pred_label in (0, 1):
                rows.append(
                    {
                        "target": target,
                        "true_label": true_label,
                        "predicted_label": pred_label,
                        "count": int(matrix[true_label, pred_label]),
                    }
                )
    return pd.DataFrame(rows)


def save_confusion_matrix_plot(confusions: pd.DataFrame, output_dir: Path, filename: str) -> Path:
    fig, axes = plt.subplots(2, 2, figsize=(8.5, 7))
    for ax, target in zip(axes.ravel(), TARGETS, strict=True):
        matrix = confusions.loc[confusions["target"] == target].pivot(
            index="true_label", columns="predicted_label", values="count"
        )
        sns.heatmap(matrix, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax)
        ax.set_title(target)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
    fig.tight_layout()
    path = output_dir / filename
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return path


def make_emotion_distribution() -> pd.DataFrame:
    require_files({"emotion_cache": EMOTION_CACHE})
    emotion_cols = [f"emotion_{label}" for label in EMOTION_LABELS]
    reddit = pd.read_parquet(EMOTION_CACHE, columns=emotion_cols)
    reddit_rows = [
        {
            "source": "Reddit inferred",
            "emotion": label,
            "share": float(reddit[f"emotion_{label}"].mean()),
        }
        for label in EMOTION_LABELS
    ]

    dataset = load_emotion_dataset(RunConfig())
    label_counts: dict[int, int] = {}
    total = 0
    for split in dataset.values():
        labels = split["label"]
        total += len(labels)
        for label_id, count in pd.Series(labels).value_counts().items():
            label_counts[int(label_id)] = label_counts.get(int(label_id), 0) + int(count)
    source_rows = [
        {
            "source": "Emotion source labels",
            "emotion": label,
            "share": label_counts.get(idx, 0) / total if total else 0.0,
        }
        for idx, label in enumerate(EMOTION_LABELS)
    ]
    return pd.DataFrame(source_rows + reddit_rows)


def save_emotion_distribution_plot(distribution: pd.DataFrame, output_dir: Path) -> Path:
    fig, ax = plt.subplots(figsize=(9, 4.8))
    sns.barplot(data=distribution, x="emotion", y="share", hue="source", ax=ax)
    ax.set_xlabel("")
    ax.set_ylabel("Share")
    ax.yaxis.set_major_formatter(lambda value, _: f"{value:.0%}")
    ax.legend(title="")
    fig.tight_layout()
    path = output_dir / "fig_source_vs_reddit_emotion_distribution.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return path


def save_pipeline_diagram(output_dir: Path) -> Path:
    path = output_dir / "fig_ms4_pipeline_diagram.png"
    if path.exists():
        return path

    fig, ax = plt.subplots(figsize=(11, 3.2))
    ax.axis("off")
    boxes = [
        ("Reddit posts\nmasked MBTI terms", 0.08),
        ("Stage 1 emotion\nprobabilities", 0.30),
        ("Stage 2 GRU\ntext + optional emotion", 0.52),
        ("Mean author\nprobability aggregation", 0.74),
        ("Validation thresholds\nfinal author metrics", 0.92),
    ]
    for text, x in boxes:
        ax.text(
            x,
            0.5,
            text,
            ha="center",
            va="center",
            fontsize=10,
            bbox={
                "boxstyle": "round,pad=0.45",
                "facecolor": "#F3F6FA",
                "edgecolor": PALETTE["blue"],
                "linewidth": 1.4,
            },
            transform=ax.transAxes,
        )
    for left, right in zip(boxes[:-1], boxes[1:], strict=True):
        ax.annotate(
            "",
            xy=(right[1] - 0.08, 0.5),
            xytext=(left[1] + 0.08, 0.5),
            xycoords=ax.transAxes,
            textcoords=ax.transAxes,
            arrowprops={"arrowstyle": "->", "color": PALETTE["ink"], "linewidth": 1.4},
        )
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return path


def output_manifest_path(path: Path, output_dir: Path) -> str:
    return path.relative_to(output_dir).as_posix()


def markdown_summary_table(summary: pd.DataFrame) -> str:
    lines = ["| model | test mean balanced accuracy |", "|---|---:|"]
    for row in summary.itertuples(index=False):
        lines.append(f"| {row.model_name} | {row.balanced_accuracy:.4f} |")
    return "\n".join(lines)


def markdown_balanced_accuracy_table(rows: list[tuple[str, float]]) -> str:
    lines = ["| model | test mean balanced accuracy |", "|---|---:|"]
    for name, value in rows:
        lines.append(f"| {name} | {value:.4f} |")
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    set_plot_style()

    metrics = load_metrics()
    histories = load_histories()
    summary = make_summary(metrics)
    gru_diagnostic_author_scores, gru_diagnostic_thresholds = load_gru_diagnostic_author_scores()
    set_attention_author_scores, set_attention_thresholds = (
        load_set_attention_confusion_author_scores()
    )
    author_inputs = load_author_score_inputs()
    threshold_curves = make_threshold_curves(gru_diagnostic_author_scores)
    gru_confusions = make_confusion_matrices(
        gru_diagnostic_author_scores,
        gru_diagnostic_thresholds,
        score_cols=tuple(f"score_gru_{target}" for target in TARGETS),
    )
    set_attention_confusions = make_confusion_matrices(
        set_attention_author_scores,
        set_attention_thresholds,
        score_cols=make_score_columns(SET_ATTENTION_CONFUSION_MODEL_ID),
    )
    emotion_distribution = make_emotion_distribution()
    bootstrap_intervals, bootstrap_differences = make_bootstrap_intervals(author_inputs)
    threshold_sensitivity = make_threshold_objective_sensitivity(gru_diagnostic_author_scores)
    token_length_sensitivity = make_token_length_sensitivity()
    max_length_training_sensitivity = make_max_length_training_sensitivity()
    frozen_transformer_metrics = read_optional_metrics(
        TRANSFORMER_AUTHOR_DIR,
        TRANSFORMER_DISPLAY_NAMES,
    )
    frozen_transformer_summary = make_optional_summary(frozen_transformer_metrics)
    set_attention_metrics = read_optional_set_metrics(SET_ATTENTION_AUTHOR_DIR)
    set_attention_summary = make_optional_summary(set_attention_metrics)
    set_attention_supplemental_summary = make_set_attention_supplemental_summary()
    set_attention_stability_summary = make_set_attention_stability_summary(
        set_attention_supplemental_summary,
        set_attention_summary,
    )
    transformer_deltas = make_all_transformer_deltas()
    transformer_status = make_transformer_artifact_status(
        frozen_transformer_metrics,
        set_attention_metrics,
        transformer_deltas,
    )

    metrics_path = args.output_dir / "all_model_metrics.csv"
    summary_path = args.output_dir / "model_summary.csv"
    histories_path = args.output_dir / "gru_training_histories.csv"
    threshold_curves_path = args.output_dir / "final_threshold_tuning_curves.csv"
    gru_confusions_path = args.output_dir / "gru_baseline_confusion_matrices.csv"
    set_attention_confusions_path = args.output_dir / "set_attention_p200_confusion_matrices.csv"
    emotion_distribution_path = args.output_dir / "emotion_distribution_source_vs_reddit.csv"
    bootstrap_intervals_path = args.output_dir / "bootstrap_model_intervals.csv"
    bootstrap_differences_path = args.output_dir / "bootstrap_pairwise_differences.csv"
    threshold_sensitivity_path = args.output_dir / "threshold_objective_sensitivity.csv"
    token_length_sensitivity_path = args.output_dir / "token_length_sensitivity.csv"
    max_length_training_sensitivity_path = (
        args.output_dir / "max_length_training_sensitivity.csv"
    )
    frozen_transformer_metrics_path = args.output_dir / "frozen_transformer_model_metrics.csv"
    frozen_transformer_summary_path = args.output_dir / "frozen_transformer_model_summary.csv"
    set_attention_metrics_path = args.output_dir / "set_attention_model_metrics.csv"
    set_attention_summary_path = args.output_dir / "set_attention_model_summary.csv"
    set_attention_supplemental_path = (
        args.output_dir / "set_attention_supplemental_summary.csv"
    )
    set_attention_stability_path = args.output_dir / "set_attention_seed_stability.csv"
    transformer_deltas_path = args.output_dir / "transformer_emotion_deltas.csv"
    transformer_status_path = args.output_dir / "transformer_artifact_status.csv"
    metrics.to_csv(metrics_path, index=False)
    summary.to_csv(summary_path, index=False)
    histories.to_csv(histories_path, index=False)
    threshold_curves.to_csv(threshold_curves_path, index=False)
    gru_confusions.to_csv(gru_confusions_path, index=False)
    set_attention_confusions.to_csv(set_attention_confusions_path, index=False)
    emotion_distribution.to_csv(emotion_distribution_path, index=False)
    bootstrap_intervals.to_csv(bootstrap_intervals_path, index=False)
    bootstrap_differences.to_csv(bootstrap_differences_path, index=False)
    threshold_sensitivity.to_csv(threshold_sensitivity_path, index=False)
    token_length_sensitivity.to_csv(token_length_sensitivity_path, index=False)
    max_length_training_sensitivity.to_csv(max_length_training_sensitivity_path, index=False)
    frozen_transformer_metrics.to_csv(frozen_transformer_metrics_path, index=False)
    frozen_transformer_summary.to_csv(frozen_transformer_summary_path, index=False)
    set_attention_metrics.to_csv(set_attention_metrics_path, index=False)
    set_attention_summary.to_csv(set_attention_summary_path, index=False)
    set_attention_supplemental_summary.to_csv(set_attention_supplemental_path, index=False)
    set_attention_stability_summary.to_csv(set_attention_stability_path, index=False)
    transformer_deltas.to_csv(transformer_deltas_path, index=False)
    transformer_status.to_csv(transformer_status_path, index=False)

    figure_paths = [
        save_pipeline_diagram(args.output_dir),
        save_model_summary_plot(summary, args.output_dir),
        save_bootstrap_interval_plot(bootstrap_intervals, args.output_dir),
        save_target_comparison_plot(metrics, args.output_dir),
        save_training_curve_plot(histories, args.output_dir),
        save_emotion_gain_plot(metrics, args.output_dir),
        save_threshold_curve_plot(threshold_curves, gru_diagnostic_thresholds, args.output_dir),
        save_threshold_objective_plot(threshold_sensitivity, args.output_dir),
        save_confusion_matrix_plot(
            gru_confusions,
            args.output_dir,
            "fig_gru_baseline_confusion_matrices.png",
        ),
        save_confusion_matrix_plot(
            set_attention_confusions,
            args.output_dir,
            "fig_set_attention_p200_confusion_matrices.png",
        ),
        save_emotion_distribution_plot(emotion_distribution, args.output_dir),
        save_token_length_sensitivity_plot(token_length_sensitivity, args.output_dir),
        save_max_length_training_plot(max_length_training_sensitivity, args.output_dir),
    ]
    if not frozen_transformer_summary.empty:
        figure_paths.append(
            save_optional_figure(
                plot_frozen_transformer_comparison(frozen_transformer_summary),
                args.output_dir,
                "fig_frozen_transformer_author_models.png",
            )
        )
    if not set_attention_summary.empty:
        figure_paths.append(
            save_optional_figure(
                plot_set_attention_comparison(set_attention_summary),
                args.output_dir,
                "fig_set_attention_author_models.png",
            )
        )
        budget_df = set_attention_summary.dropna(subset=["post_budget"])
        if not budget_df.empty:
            figure_paths.append(
                save_optional_figure(
                    plot_post_budget_ablation(budget_df),
                    args.output_dir,
                    "fig_set_attention_post_budget.png",
                )
            )
    if not transformer_deltas.empty:
        figure_paths.append(
            save_optional_figure(
                plot_emotion_increment_deltas(transformer_deltas),
                args.output_dir,
                "fig_transformer_emotion_deltas.png",
            )
        )
    if not set_attention_stability_summary.empty:
        figure_paths.append(save_set_attention_stability_plot(set_attention_stability_summary, args.output_dir))
    if not set_attention_supplemental_summary.empty:
        figure_paths.append(
            save_set_attention_epoch_sensitivity_plot(
                set_attention_supplemental_summary,
                set_attention_summary,
                args.output_dir,
            )
        )

    transformer_top_rows = []
    for model_id in [
        "set_attention_text_controls_p200",
        "set_attention_text_p200",
        "set_attention_text_shuffled_emotion_p200",
        "set_attention_text_real_emotion_p200",
        "frozen_text_mean_std",
    ]:
        source = (
            set_attention_summary
            if model_id.startswith("set_attention")
            else frozen_transformer_summary
        )
        match = source.loc[source["model_id"] == model_id]
        if not match.empty:
            transformer_top_rows.append(
                (
                    str(match["model_name"].iloc[0]),
                    float(match["balanced_accuracy"].iloc[0]),
                )
            )
    readme = args.output_dir / "README.md"
    readme.write_text(
        "\n".join(
            [
                "# MS4 Report Results",
                "",
                "Generated from local run artifacts by `scripts/aggregate_report_results.py`.",
                "",
                "Tracked CSVs and PNGs here are small report inputs. Large checkpoints,",
                "post-level parquet scores, and full preprocessing caches remain under",
                "`code/artifacts/` and are intentionally ignored by git.",
                "",
                "Top-line test mean balanced accuracy:",
                "",
                markdown_summary_table(summary),
                "",
                "Transformer-author top-line test mean balanced accuracy:",
                "",
                markdown_balanced_accuracy_table(transformer_top_rows)
                if transformer_top_rows
                else "_Transformer-author result artifacts are not available._",
                "",
                "Additional diagnostics included here:",
                "",
                "- MS4 pipeline diagram.",
                "- Bootstrap confidence intervals over test authors.",
                "- GRU text+emotion threshold-tuning curves.",
                "- Threshold objective sensitivity for balanced accuracy vs F1.",
                "- GRU text+emotion baseline confusion matrices.",
                "- Set Attention Text p=200 author-model confusion matrices.",
                "- Source-vs-Reddit emotion distribution comparison.",
                "- Token-length sensitivity audit for 128 vs 256 token limits.",
                "- Fixed text-only GRU 128 vs 256 max-length training sensitivity.",
                "- Transformer-author artifact status, transformer summaries, paired deltas, seed stability, and max-epoch-cap sensitivity.",
                "- Supplemental p200 set/attention checks show that the author-level transformer direction is stable, while the emotion-specific increment is seed/training sensitive.",
                "",
            ]
        ),
        encoding="utf-8",
    )
    write_json(
        args.output_dir / "manifest.json",
        {
            "metrics_path": output_manifest_path(metrics_path, args.output_dir),
            "summary_path": output_manifest_path(summary_path, args.output_dir),
            "histories_path": output_manifest_path(histories_path, args.output_dir),
            "threshold_curves_path": output_manifest_path(threshold_curves_path, args.output_dir),
            "gru_confusions_path": output_manifest_path(gru_confusions_path, args.output_dir),
            "set_attention_confusions_path": output_manifest_path(
                set_attention_confusions_path,
                args.output_dir,
            ),
            "emotion_distribution_path": output_manifest_path(
                emotion_distribution_path, args.output_dir
            ),
            "bootstrap_intervals_path": output_manifest_path(
                bootstrap_intervals_path, args.output_dir
            ),
            "bootstrap_differences_path": output_manifest_path(
                bootstrap_differences_path, args.output_dir
            ),
            "threshold_sensitivity_path": output_manifest_path(
                threshold_sensitivity_path, args.output_dir
            ),
            "token_length_sensitivity_path": output_manifest_path(
                token_length_sensitivity_path, args.output_dir
            ),
            "max_length_training_sensitivity_path": output_manifest_path(
                max_length_training_sensitivity_path, args.output_dir
            ),
            "frozen_transformer_metrics_path": output_manifest_path(
                frozen_transformer_metrics_path, args.output_dir
            ),
            "frozen_transformer_summary_path": output_manifest_path(
                frozen_transformer_summary_path, args.output_dir
            ),
            "set_attention_metrics_path": output_manifest_path(
                set_attention_metrics_path, args.output_dir
            ),
            "set_attention_summary_path": output_manifest_path(
                set_attention_summary_path, args.output_dir
            ),
            "set_attention_supplemental_summary_path": output_manifest_path(
                set_attention_supplemental_path, args.output_dir
            ),
            "set_attention_seed_stability_path": output_manifest_path(
                set_attention_stability_path, args.output_dir
            ),
            "transformer_deltas_path": output_manifest_path(
                transformer_deltas_path, args.output_dir
            ),
            "transformer_status_path": output_manifest_path(
                transformer_status_path, args.output_dir
            ),
            "figures": [output_manifest_path(path, args.output_dir) for path in figure_paths],
            "n_models": int(summary["model_id"].nunique()),
            "n_frozen_transformer_models": int(frozen_transformer_summary["model_id"].nunique())
            if not frozen_transformer_summary.empty
            else 0,
            "n_set_attention_models": int(set_attention_summary["model_id"].nunique())
            if not set_attention_summary.empty
            else 0,
            "n_set_attention_supplemental_runs": int(
                set_attention_supplemental_summary["run_id"].nunique()
            )
            if not set_attention_supplemental_summary.empty
            else 0,
        },
    )
    print(summary.to_string(index=False))
    for path in figure_paths:
        print(path)


if __name__ == "__main__":
    main()
