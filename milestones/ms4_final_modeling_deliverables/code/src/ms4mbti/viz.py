"""Visualization helpers for the MS4 notebook."""

from __future__ import annotations

from collections.abc import Iterable

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


PALETTE = {
    "ink": "#243040",
    "muted": "#687385",
    "blue": "#2F6F9F",
    "teal": "#3B8C7A",
    "gold": "#C49A3A",
    "red": "#B85C5C",
    "gray": "#D8DEE7",
}


def set_plot_style() -> None:
    sns.set_theme(
        context="notebook",
        style="whitegrid",
        rc={
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.titleweight": "bold",
            "axes.labelcolor": PALETTE["ink"],
            "xtick.color": PALETTE["ink"],
            "ytick.color": PALETTE["ink"],
            "figure.dpi": 120,
        },
    )


def _finish(fig, title: str | None = None):
    if title:
        fig.suptitle(title, x=0.02, y=1.02, ha="left", fontsize=13, fontweight="bold")
    fig.tight_layout()
    return fig


def plot_leakage_audit(leakage_rows: pd.DataFrame):
    required = {"stage", "post_share_with_any_term", "author_share_with_any_term"}
    missing = required - set(leakage_rows.columns)
    if missing:
        raise ValueError(f"Missing leakage columns: {sorted(missing)}")
    plot_df = leakage_rows.melt(
        id_vars="stage",
        value_vars=["post_share_with_any_term", "author_share_with_any_term"],
        var_name="metric",
        value_name="share",
    )
    labels = {
        "post_share_with_any_term": "posts",
        "author_share_with_any_term": "authors",
    }
    plot_df["metric"] = plot_df["metric"].map(labels)
    fig, ax = plt.subplots(figsize=(7, 4))
    sns.barplot(data=plot_df, x="stage", y="share", hue="metric", ax=ax)
    ax.set_ylabel("Share with explicit MBTI term")
    ax.set_xlabel("")
    ax.yaxis.set_major_formatter(lambda value, _: f"{value:.0%}")
    ax.legend(title="")
    return _finish(fig, "Explicit MBTI Term Leakage Before and After Masking")


def plot_class_balance(class_balance: pd.DataFrame):
    required = {"target", "positive_share"}
    missing = required - set(class_balance.columns)
    if missing:
        raise ValueError(f"Missing class balance columns: {sorted(missing)}")
    fig, ax = plt.subplots(figsize=(7, 4))
    sns.barplot(data=class_balance, x="target", y="positive_share", color=PALETTE["blue"], ax=ax)
    ax.axhline(0.5, color=PALETTE["muted"], linestyle="--", linewidth=1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("")
    ax.set_ylabel("Positive author share")
    ax.yaxis.set_major_formatter(lambda value, _: f"{value:.0%}")
    return _finish(fig, "Author-Level Positive-Class Share by MBTI Dimension")


def plot_posts_per_author(posts_per_author: pd.Series):
    fig, ax = plt.subplots(figsize=(7, 4))
    sns.histplot(posts_per_author, bins=20, color=PALETTE["teal"], ax=ax)
    ax.axvline(posts_per_author.median(), color=PALETTE["ink"], linestyle="--", linewidth=1)
    ax.set_xlabel("Retained posts per author")
    ax.set_ylabel("Authors")
    return _finish(fig, "Author Contribution After Filtering and Cap")


def plot_split_balance(split_balance: pd.DataFrame):
    required = {"split", "target", "positive_share"}
    missing = required - set(split_balance.columns)
    if missing:
        raise ValueError(f"Missing split balance columns: {sorted(missing)}")
    heatmap_df = split_balance.pivot(index="target", columns="split", values="positive_share")
    ordered_cols = [col for col in ("train", "val", "test") if col in heatmap_df.columns]
    heatmap_df = heatmap_df[ordered_cols]
    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    sns.heatmap(
        heatmap_df,
        annot=True,
        fmt=".0%",
        vmin=0,
        vmax=1,
        cmap="Blues",
        cbar_kws={"label": "Positive share"},
        ax=ax,
    )
    ax.set_xlabel("")
    ax.set_ylabel("")
    return _finish(fig, "Author Split Balance")


def plot_token_truncation_by_split(by_split: pd.DataFrame):
    if by_split.empty:
        raise ValueError("No split-level truncation data to plot")
    fig, ax = plt.subplots(figsize=(7, 4))
    sns.barplot(data=by_split, x="split", y="share_over_max", color=PALETTE["gold"], ax=ax)
    ax.set_xlabel("")
    ax.set_ylabel("Share over max length")
    ax.yaxis.set_major_formatter(lambda value, _: f"{value:.0%}")
    return _finish(fig, "Stage 2 Token Truncation Exposure by Split")


def plot_metric_comparison(metrics: pd.DataFrame, *, metric: str = "balanced_accuracy"):
    required = {"model_id", "target", metric}
    missing = required - set(metrics.columns)
    if missing:
        raise ValueError(f"Missing metric comparison columns: {sorted(missing)}")
    fig, ax = plt.subplots(figsize=(8, 4.5))
    sns.barplot(data=metrics, x="target", y=metric, hue="model_id", ax=ax)
    ax.axhline(0.5, color=PALETTE["muted"], linestyle="--", linewidth=1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("")
    ax.set_ylabel(metric.replace("_", " ").title())
    ax.legend(title="", loc="lower right")
    return _finish(fig, f"Model Comparison by {metric.replace('_', ' ').title()}")


def plot_emotion_increment_deltas(deltas: pd.DataFrame):
    required = {"comparison", "target", "point_estimate", "ci_lower", "ci_upper"}
    missing = required - set(deltas.columns)
    if missing:
        raise ValueError(f"Missing delta plot columns: {sorted(missing)}")
    plot_df = deltas.loc[deltas["target"] == "mean"].copy()
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.errorbar(
        plot_df["point_estimate"],
        plot_df["comparison"],
        xerr=[
            plot_df["point_estimate"] - plot_df["ci_lower"],
            plot_df["ci_upper"] - plot_df["point_estimate"],
        ],
        fmt="o",
        color=PALETTE["blue"],
        ecolor=PALETTE["muted"],
        capsize=4,
    )
    ax.axvline(0, color=PALETTE["muted"], linewidth=1)
    ax.set_xlabel("Mean balanced-accuracy delta with 95% paired bootstrap CI")
    ax.set_ylabel("")
    return _finish(fig, "Real and Shuffled Emotion Increment Tests")


def plot_frozen_transformer_comparison(summary: pd.DataFrame):
    required = {"model_name", "balanced_accuracy"}
    missing = required - set(summary.columns)
    if missing:
        raise ValueError(f"Missing frozen transformer summary columns: {sorted(missing)}")
    plot_df = summary.sort_values("balanced_accuracy", ascending=True)
    fig, ax = plt.subplots(figsize=(8, 4.8))
    sns.barplot(data=plot_df, y="model_name", x="balanced_accuracy", color=PALETTE["teal"], ax=ax)
    ax.axvline(0.5, color=PALETTE["muted"], linestyle="--", linewidth=1)
    ax.set_xlabel("Mean test balanced accuracy")
    ax.set_ylabel("")
    return _finish(fig, "Frozen Transformer Author Models")


def plot_set_attention_comparison(summary: pd.DataFrame):
    required = {"model_name", "balanced_accuracy"}
    missing = required - set(summary.columns)
    if missing:
        raise ValueError(f"Missing set/attention summary columns: {sorted(missing)}")
    plot_df = summary.sort_values("balanced_accuracy", ascending=True)
    fig, ax = plt.subplots(figsize=(8, 4.8))
    sns.barplot(data=plot_df, y="model_name", x="balanced_accuracy", color=PALETTE["gold"], ax=ax)
    ax.axvline(0.5, color=PALETTE["muted"], linestyle="--", linewidth=1)
    ax.set_xlabel("Mean test balanced accuracy")
    ax.set_ylabel("")
    return _finish(fig, "Set/Attention Author Models")


def plot_post_budget_ablation(summary: pd.DataFrame):
    required = {"model_id", "post_budget", "balanced_accuracy"}
    missing = required - set(summary.columns)
    if missing:
        raise ValueError(f"Missing post-budget columns: {sorted(missing)}")
    fig, ax = plt.subplots(figsize=(8, 4.8))
    sns.lineplot(
        data=summary,
        x="post_budget",
        y="balanced_accuracy",
        hue="model_id",
        marker="o",
        ax=ax,
    )
    ax.axhline(0.5, color=PALETTE["muted"], linestyle="--", linewidth=1)
    ax.set_xlabel("Retained posts per author")
    ax.set_ylabel("Mean test balanced accuracy")
    ax.legend(title="")
    return _finish(fig, "Author Transformer Post-Budget Sensitivity")


def save_figures(figures: Iterable, output_dir, *, prefix: str = "ms4") -> list:
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for idx, fig in enumerate(figures, start=1):
        path = output_dir / f"{prefix}_{idx:02d}.png"
        fig.savefig(path, dpi=200, bbox_inches="tight")
        paths.append(path)
    return paths
