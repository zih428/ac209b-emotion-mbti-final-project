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
from ms4mbti.viz import PALETTE, set_plot_style


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

FINAL_MODEL_DIR = CODE_DIR / "artifacts" / "runs" / "stage2_text_emotion_gru_full"
FINAL_AUTHOR_SCORES = FINAL_MODEL_DIR / "author_scores_stage2_text_gru.csv"
FINAL_THRESHOLDS = FINAL_MODEL_DIR / "thresholds_stage2_text_gru.csv"
TEXT_GRU_128_DIR = CODE_DIR / "artifacts" / "runs" / "stage2_text_gru_full"
TEXT_GRU_256_DIR = CODE_DIR / "artifacts" / "runs" / "stage2_text_gru_len256_full"
EMOTION_CACHE = CODE_DIR / "artifacts" / "cache" / "emotion_probs_full.parquet"
PREPROCESSED_POSTS = CODE_DIR / "artifacts" / "preprocessed" / "full" / "modeling_posts.parquet"
TARGETS = ("target_E", "target_S", "target_T", "target_J")


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


def load_final_author_scores() -> tuple[pd.DataFrame, pd.DataFrame]:
    require_files({"final_author_scores": FINAL_AUTHOR_SCORES, "final_thresholds": FINAL_THRESHOLDS})
    return pd.read_csv(FINAL_AUTHOR_SCORES), pd.read_csv(FINAL_THRESHOLDS)


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


def make_final_confusion_matrices(
    author_scores: pd.DataFrame,
    selected_thresholds: pd.DataFrame,
) -> pd.DataFrame:
    test = author_scores.loc[author_scores["split"] == "test"].copy()
    rows = []
    for target in TARGETS:
        score_col = f"score_gru_{target}"
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


def save_confusion_matrix_plot(confusions: pd.DataFrame, output_dir: Path) -> Path:
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
    path = output_dir / "fig_final_confusion_matrices.png"
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
    path = output_dir / "fig_ms4_pipeline_diagram.png"
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


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    set_plot_style()

    metrics = load_metrics()
    histories = load_histories()
    summary = make_summary(metrics)
    final_author_scores, final_thresholds = load_final_author_scores()
    author_inputs = load_author_score_inputs()
    threshold_curves = make_threshold_curves(final_author_scores)
    confusions = make_final_confusion_matrices(final_author_scores, final_thresholds)
    emotion_distribution = make_emotion_distribution()
    bootstrap_intervals, bootstrap_differences = make_bootstrap_intervals(author_inputs)
    threshold_sensitivity = make_threshold_objective_sensitivity(final_author_scores)
    token_length_sensitivity = make_token_length_sensitivity()
    max_length_training_sensitivity = make_max_length_training_sensitivity()

    metrics_path = args.output_dir / "all_model_metrics.csv"
    summary_path = args.output_dir / "model_summary.csv"
    histories_path = args.output_dir / "gru_training_histories.csv"
    threshold_curves_path = args.output_dir / "final_threshold_tuning_curves.csv"
    confusions_path = args.output_dir / "final_confusion_matrices.csv"
    emotion_distribution_path = args.output_dir / "emotion_distribution_source_vs_reddit.csv"
    bootstrap_intervals_path = args.output_dir / "bootstrap_model_intervals.csv"
    bootstrap_differences_path = args.output_dir / "bootstrap_pairwise_differences.csv"
    threshold_sensitivity_path = args.output_dir / "threshold_objective_sensitivity.csv"
    token_length_sensitivity_path = args.output_dir / "token_length_sensitivity.csv"
    max_length_training_sensitivity_path = (
        args.output_dir / "max_length_training_sensitivity.csv"
    )
    metrics.to_csv(metrics_path, index=False)
    summary.to_csv(summary_path, index=False)
    histories.to_csv(histories_path, index=False)
    threshold_curves.to_csv(threshold_curves_path, index=False)
    confusions.to_csv(confusions_path, index=False)
    emotion_distribution.to_csv(emotion_distribution_path, index=False)
    bootstrap_intervals.to_csv(bootstrap_intervals_path, index=False)
    bootstrap_differences.to_csv(bootstrap_differences_path, index=False)
    threshold_sensitivity.to_csv(threshold_sensitivity_path, index=False)
    token_length_sensitivity.to_csv(token_length_sensitivity_path, index=False)
    max_length_training_sensitivity.to_csv(max_length_training_sensitivity_path, index=False)

    figure_paths = [
        save_pipeline_diagram(args.output_dir),
        save_model_summary_plot(summary, args.output_dir),
        save_bootstrap_interval_plot(bootstrap_intervals, args.output_dir),
        save_target_comparison_plot(metrics, args.output_dir),
        save_training_curve_plot(histories, args.output_dir),
        save_emotion_gain_plot(metrics, args.output_dir),
        save_threshold_curve_plot(threshold_curves, final_thresholds, args.output_dir),
        save_threshold_objective_plot(threshold_sensitivity, args.output_dir),
        save_confusion_matrix_plot(confusions, args.output_dir),
        save_emotion_distribution_plot(emotion_distribution, args.output_dir),
        save_token_length_sensitivity_plot(token_length_sensitivity, args.output_dir),
        save_max_length_training_plot(max_length_training_sensitivity, args.output_dir),
    ]

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
                "Additional diagnostics included here:",
                "",
                "- MS4 pipeline diagram.",
                "- Bootstrap confidence intervals over test authors.",
                "- Final text+emotion GRU threshold-tuning curves.",
                "- Threshold objective sensitivity for balanced accuracy vs F1.",
                "- Final text+emotion GRU confusion matrices.",
                "- Source-vs-Reddit emotion distribution comparison.",
                "- Token-length sensitivity audit for 128 vs 256 token limits.",
                "- Fixed text-only GRU 128 vs 256 max-length training sensitivity.",
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
            "confusions_path": output_manifest_path(confusions_path, args.output_dir),
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
            "figures": [output_manifest_path(path, args.output_dir) for path in figure_paths],
            "n_models": int(summary["model_id"].nunique()),
        },
    )
    print(summary.to_string(index=False))
    for path in figure_paths:
        print(path)


if __name__ == "__main__":
    main()
