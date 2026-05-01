#!/usr/bin/env python3
"""Train/evaluate set/attention author models and pooling ablations."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

CODE_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = CODE_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from ms4mbti.author_features import build_author_feature_table
from ms4mbti.config import EMOTION_FEATURE_COLUMNS, TARGET_COLUMNS, RunConfig, ensure_artifact_dirs
from ms4mbti.embeddings import embedding_columns, read_embedding_cache
from ms4mbti.evaluation import evaluate_with_validation_thresholds
from ms4mbti.negative_controls import replace_with_shuffled_features
from ms4mbti.progress import ProgressLogger
from ms4mbti.training import set_global_seed
from ms4mbti.transformer_author import (
    build_author_set_arrays,
    make_score_columns,
    predict_set_attention_scores,
    threshold_frame,
    train_author_probe,
    train_set_attention_model,
)


SET_VARIANTS = {
    "set_attention_text": "text",
    "set_attention_text_shuffled_emotion": "shuffled",
    "set_attention_text_real_emotion": "real",
    "set_attention_text_controls": "controls",
    "set_attention_text_real_emotion_controls": "real_controls",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-parquet",
        type=Path,
        default=CODE_DIR / "artifacts" / "preprocessed" / "full" / "modeling_posts.parquet",
    )
    parser.add_argument("--embedding-cache-dir", type=Path, required=True)
    parser.add_argument(
        "--emotion-feature-path",
        type=Path,
        default=CODE_DIR / "artifacts" / "cache" / "emotion_probs_full.parquet",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=CODE_DIR / "artifacts" / "runs" / "set_attention_author",
    )
    parser.add_argument("--full-run", action="store_true")
    parser.add_argument("--post-budgets", type=int, nargs="+", default=[50, 200])
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-authors-per-split", type=int, default=None)
    parser.add_argument(
        "--set-variants",
        nargs="+",
        choices=sorted(SET_VARIANTS),
        default=sorted(SET_VARIANTS),
        help="Subset of set/attention variants to train.",
    )
    parser.add_argument(
        "--skip-pooling",
        action="store_true",
        help="Skip mean-pooling MLP ablations for supplemental runs.",
    )
    parser.add_argument("--seed", type=int, default=209066)
    return parser.parse_args()


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True, default=str), encoding="utf-8")


def sample_authors(posts: pd.DataFrame, *, max_authors_per_split: int | None, seed: int) -> pd.DataFrame:
    if max_authors_per_split is None:
        return posts
    selected = []
    author_splits = posts[["author", "split"]].drop_duplicates()
    for split_name, group in author_splits.groupby("split", sort=True):
        shuffled = group.sample(frac=1.0, random_state=seed)
        selected.extend(shuffled["author"].head(max_authors_per_split).tolist())
    return posts.loc[posts["author"].isin(selected)].copy()


def write_set_artifacts(model_id: str, scores: pd.DataFrame, output_dir: Path) -> dict[str, Any]:
    score_cols = make_score_columns(model_id)
    eval_result = evaluate_with_validation_thresholds(scores, score_cols=score_cols)
    metrics = pd.concat(
        [
            eval_result["validation_metrics"].assign(model_id=model_id, split="val"),
            eval_result["test_metrics"].assign(model_id=model_id, split="test"),
        ],
        ignore_index=True,
    )
    thresholds = threshold_frame(model_id, eval_result["thresholds"])
    scores_path = output_dir / f"author_scores_{model_id}.csv"
    metrics_path = output_dir / f"metrics_{model_id}.csv"
    thresholds_path = output_dir / f"thresholds_{model_id}.csv"
    scores.to_csv(scores_path, index=False)
    metrics.to_csv(metrics_path, index=False)
    thresholds.to_csv(thresholds_path, index=False)
    return {
        "model_id": model_id,
        "author_scores_path": str(scores_path),
        "metrics_path": str(metrics_path),
        "thresholds_path": str(thresholds_path),
        "test_balanced_accuracy_mean": float(
            metrics.loc[metrics["split"] == "test", "balanced_accuracy"].mean()
        ),
    }


def add_post_controls(posts: pd.DataFrame, *, text_col: str = "text_masked") -> pd.DataFrame:
    out = posts.copy()
    token_length = out[text_col].fillna("").astype(str).str.split().str.len().astype(float)
    out["post_token_length"] = token_length
    out["post_is_over_128"] = (token_length > 128).astype(float)
    out["post_log_token_length"] = np.log1p(token_length)
    return out


def standardize_post_controls_train_only(
    posts: pd.DataFrame,
    *,
    control_cols: tuple[str, ...],
    split_col: str = "split",
    fit_split: str = "train",
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Standardize post-level controls using only the training split."""

    out = posts.copy()
    validate_cols = [split_col, *control_cols]
    missing = [col for col in validate_cols if col not in out.columns]
    if missing:
        raise ValueError(f"Missing columns for post-control standardization: {missing}")
    fit_frame = out.loc[out[split_col] == fit_split, list(control_cols)].astype(float)
    if fit_frame.empty:
        raise ValueError(f"Cannot fit post-control standardization without `{fit_split}` rows")
    mean = fit_frame.mean()
    scale = fit_frame.std(ddof=0).replace(0.0, 1.0)
    out.loc[:, list(control_cols)] = (out[list(control_cols)].astype(float) - mean) / scale
    metadata = {
        "method": "standard_score",
        "fit_split": fit_split,
        "columns": list(control_cols),
        "mean": {col: float(mean[col]) for col in control_cols},
        "scale": {col: float(scale[col]) for col in control_cols},
    }
    return out, metadata


def main() -> None:
    args = parse_args()
    set_global_seed(args.seed)
    config = RunConfig(seed=args.seed)
    ensure_artifact_dirs(config)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    logger = ProgressLogger("set_attention_author", verbose=True)

    if not args.input_parquet.exists():
        raise FileNotFoundError(f"Missing preprocessed parquet: {args.input_parquet}")
    if not args.emotion_feature_path.exists():
        raise FileNotFoundError(f"Missing emotion cache: {args.emotion_feature_path}")

    posts = pd.read_parquet(args.input_parquet).reset_index().rename(columns={"index": "row_index"})
    if not args.full_run:
        max_authors = args.max_authors_per_split or 32
        posts = sample_authors(posts, max_authors_per_split=max_authors, seed=args.seed)
    posts = add_post_controls(posts)
    control_cols = ("post_token_length", "post_is_over_128", "post_log_token_length")
    posts, control_scaling = standardize_post_controls_train_only(
        posts,
        control_cols=control_cols,
    )
    embeddings, _manifest = read_embedding_cache(args.embedding_cache_dir)
    emotion = pd.read_parquet(args.emotion_feature_path)
    merged = posts.merge(
        embeddings[["row_index", "author", "split", *embedding_columns(embeddings)]],
        on=["row_index", "author", "split"],
        how="inner",
        validate="one_to_one",
    ).merge(
        emotion[["row_index", *EMOTION_FEATURE_COLUMNS]],
        on="row_index",
        how="left",
        validate="one_to_one",
    )
    if len(merged) != len(posts):
        raise ValueError("Embedding cache does not cover all selected posts")
    if merged[list(EMOTION_FEATURE_COLUMNS)].isna().any(axis=1).any():
        raise ValueError("Emotion cache does not cover all selected posts")

    summaries = []
    histories = []
    epochs = args.epochs if args.epochs is not None else (5 if args.full_run else 1)

    if not args.skip_pooling:
        # Pooling ablations use the same author feature table and an MLP probe.
        author_features, schema = build_author_feature_table(
            posts,
            embeddings,
            emotion_features=emotion,
            max_length=config.frozen_embedding_max_length,
        )
        for model_id, cols in {
            "mean_pool_mlp_text": schema.text_mean,
            "mean_std_pool_mlp_text": schema.text_mean + schema.text_std,
        }.items():
            logger.step("Training pooling ablation", model_id=model_id, n_features=len(cols))
            result, _models = train_author_probe(
                author_features.fillna(0.0),
                model_id=model_id,
                feature_cols=cols,
                classifier="mlp",
                seed=args.seed,
            )
            summaries.append(write_set_artifacts(model_id, result.author_scores, args.output_dir))
            result.metrics.to_csv(args.output_dir / f"metrics_{model_id}.csv", index=False)
            result.thresholds.to_csv(args.output_dir / f"thresholds_{model_id}.csv", index=False)

    emb_cols = embedding_columns(merged)
    shuffled_posts = replace_with_shuffled_features(
        merged,
        feature_cols=EMOTION_FEATURE_COLUMNS,
        split_col="split",
        seed=args.seed,
    )

    frames = {
        "text": merged,
        "real": merged,
        "shuffled": shuffled_posts,
        "controls": merged,
        "real_controls": merged,
    }
    feature_sets = {
        "text": emb_cols,
        "real": emb_cols + EMOTION_FEATURE_COLUMNS,
        "shuffled": emb_cols + EMOTION_FEATURE_COLUMNS,
        "controls": emb_cols + control_cols,
        "real_controls": emb_cols + EMOTION_FEATURE_COLUMNS + control_cols,
    }

    for post_budget in args.post_budgets:
        for model_id_base in args.set_variants:
            feature_key = SET_VARIANTS[model_id_base]
            model_id = f"{model_id_base}_p{post_budget}"
            frame = frames[feature_key]
            cols = feature_sets[feature_key]
            logger.step(
                "Training set/attention model",
                model_id=model_id,
                post_budget=post_budget,
                n_features=len(cols),
                epochs=epochs,
            )
            arrays = build_author_set_arrays(
                frame,
                feature_cols=cols,
                post_budget=post_budget,
                target_cols=TARGET_COLUMNS,
            )
            model, history = train_set_attention_model(
                arrays,
                model_id=model_id,
                post_budget=post_budget,
                epochs=epochs,
                batch_size=args.batch_size,
                seed=args.seed,
            )
            histories.append(history)
            scores = predict_set_attention_scores(model, arrays, model_id=model_id)
            summaries.append(write_set_artifacts(model_id, scores, args.output_dir))

    if histories:
        pd.concat(histories, ignore_index=True).to_csv(args.output_dir / "history.csv", index=False)
    summary = {
        "dry_run": not args.full_run,
        "n_posts": int(len(merged)),
        "n_authors": int(merged["author"].nunique()),
        "post_budgets": args.post_budgets,
        "epochs": int(epochs),
        "batch_size": int(args.batch_size),
        "seed": int(args.seed),
        "set_variants": args.set_variants,
        "skip_pooling": bool(args.skip_pooling),
        "post_control_scaling": control_scaling,
        "model_summaries": summaries,
        "output_dir": str(args.output_dir),
    }
    write_json(args.output_dir / "summary.json", summary)
    logger.step("Set/attention author models complete", **summary)


if __name__ == "__main__":
    main()
