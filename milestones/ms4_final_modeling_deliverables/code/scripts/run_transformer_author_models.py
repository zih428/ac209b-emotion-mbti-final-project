#!/usr/bin/env python3
"""Train/evaluate frozen-transformer author-level classifiers."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd

CODE_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = CODE_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from ms4mbti.author_features import build_author_feature_table
from ms4mbti.cache import (
    author_feature_metadata,
    read_manifest,
    write_manifest,
)
from ms4mbti.config import EMOTION_FEATURE_COLUMNS, RunConfig, ensure_artifact_dirs
from ms4mbti.embeddings import read_embedding_cache
from ms4mbti.negative_controls import replace_with_shuffled_features
from ms4mbti.progress import ProgressLogger
from ms4mbti.transformer_author import train_author_probe


MODEL_VARIANTS = {
    "frozen_text_mean_std": ("text_mean_std", "text"),
    "frozen_emotion_only": ("emotion_only", "emotion"),
    "frozen_text_shuffled_emotion": ("text_real_emotion", "shuffled"),
    "frozen_text_real_emotion": ("text_real_emotion", "real"),
    "frozen_text_controls": ("text_controls", "controls"),
    "frozen_text_real_emotion_controls": ("text_real_emotion_controls", "real_controls"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-parquet",
        type=Path,
        default=CODE_DIR / "artifacts" / "preprocessed" / "full" / "modeling_posts.parquet",
    )
    parser.add_argument(
        "--embedding-cache-dir",
        type=Path,
        required=True,
        help="Directory containing transformer embedding shards and manifest.json.",
    )
    parser.add_argument(
        "--emotion-feature-path",
        type=Path,
        default=CODE_DIR / "artifacts" / "cache" / "emotion_probs_full.parquet",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=CODE_DIR / "artifacts" / "runs" / "transformer_author",
    )
    parser.add_argument("--classifier", choices=["logistic", "mlp"], default="logistic")
    parser.add_argument("--post-budget", type=int, default=None)
    parser.add_argument("--seed", type=int, default=209066)
    return parser.parse_args()


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True, default=str), encoding="utf-8")


def write_probe_artifacts(result, *, output_dir: Path) -> dict[str, Any]:
    model_id = result.model_id
    scores_path = output_dir / f"author_scores_{model_id}.csv"
    metrics_path = output_dir / f"metrics_{model_id}.csv"
    thresholds_path = output_dir / f"thresholds_{model_id}.csv"
    result.author_scores.to_csv(scores_path, index=False)
    result.metrics.to_csv(metrics_path, index=False)
    result.thresholds.to_csv(thresholds_path, index=False)
    test_mean = float(
        result.metrics.loc[result.metrics["split"] == "test", "balanced_accuracy"].mean()
    )
    return {
        "model_id": model_id,
        "author_scores_path": str(scores_path),
        "metrics_path": str(metrics_path),
        "thresholds_path": str(thresholds_path),
        "test_balanced_accuracy_mean": test_mean,
    }


def main() -> None:
    args = parse_args()
    config = RunConfig(seed=args.seed)
    ensure_artifact_dirs(config)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    logger = ProgressLogger("transformer_author", verbose=True)

    if not args.input_parquet.exists():
        raise FileNotFoundError(f"Missing preprocessed parquet: {args.input_parquet}")
    if not args.emotion_feature_path.exists():
        raise FileNotFoundError(f"Missing emotion features: {args.emotion_feature_path}")

    logger.step("Loading post metadata", input_parquet=args.input_parquet)
    posts = pd.read_parquet(args.input_parquet)
    posts = posts.reset_index().rename(columns={"index": "row_index"})
    logger.step("Loading embedding cache", embedding_cache_dir=args.embedding_cache_dir)
    embeddings, embedding_manifest = read_embedding_cache(args.embedding_cache_dir)
    logger.step("Loading emotion cache", emotion_feature_path=args.emotion_feature_path)
    emotion = pd.read_parquet(args.emotion_feature_path)

    logger.step("Building author feature table", post_budget=args.post_budget)
    features, schema = build_author_feature_table(
        posts,
        embeddings,
        emotion_features=emotion,
        post_budget=args.post_budget,
        max_length=config.frozen_embedding_max_length,
    )
    features = features.fillna(0.0)
    features_path = args.output_dir / "author_features.parquet"
    features.to_parquet(features_path, index=False)

    shuffled = replace_with_shuffled_features(
        features,
        feature_cols=schema.emotion,
        split_col="split",
        seed=args.seed,
    )

    variant_frames = {
        "real": features,
        "shuffled": shuffled,
        "text": features,
        "emotion": features,
        "controls": features,
        "real_controls": features,
    }
    variant_features = {
        "text": schema.text_mean + schema.text_std,
        "emotion": schema.emotion,
        "shuffled": schema.text_mean + schema.text_std + schema.emotion,
        "real": schema.text_mean + schema.text_std + schema.emotion,
        "controls": schema.text_mean + schema.text_std + schema.controls,
        "real_controls": schema.text_mean + schema.text_std + schema.emotion + schema.controls,
    }

    summaries = []
    for model_id, (_variant_name, feature_key) in MODEL_VARIANTS.items():
        frame = variant_frames[feature_key]
        cols = variant_features[feature_key]
        logger.step("Training author probe", model_id=model_id, n_features=len(cols))
        result, _models = train_author_probe(
            frame,
            model_id=model_id,
            feature_cols=cols,
            classifier=args.classifier,
            seed=args.seed,
        )
        summaries.append(write_probe_artifacts(result, output_dir=args.output_dir))

    emotion_manifest_path = args.emotion_feature_path.with_suffix(".manifest.json")
    emotion_manifest_fingerprint = None
    if emotion_manifest_path.exists():
        emotion_manifest_fingerprint = read_manifest(emotion_manifest_path).get("fingerprint")
    metadata = author_feature_metadata(
        embedding_manifest_fingerprint=embedding_manifest.get("fingerprint", ""),
        emotion_manifest_fingerprint=emotion_manifest_fingerprint,
        feature_schema=schema.as_dict(),
        post_budget=args.post_budget,
        extra={
            **config.metadata(),
            "created_by": "scripts/run_transformer_author_models.py",
            "classifier": args.classifier,
            "input_parquet": str(args.input_parquet),
            "embedding_cache_dir": str(args.embedding_cache_dir),
            "emotion_feature_path": str(args.emotion_feature_path),
            "author_features_path": str(features_path),
            "model_summaries": summaries,
        },
    )
    manifest = write_manifest(args.output_dir / "manifest.json", metadata)
    summary = {
        "manifest_fingerprint": manifest["fingerprint"],
        "n_authors": int(features["author"].nunique()),
        "model_summaries": summaries,
        "output_dir": str(args.output_dir),
    }
    write_json(args.output_dir / "summary.json", summary)
    logger.step("Transformer author models complete", **summary)


if __name__ == "__main__":
    main()
