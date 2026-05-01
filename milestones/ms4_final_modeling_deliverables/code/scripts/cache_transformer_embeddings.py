#!/usr/bin/env python3
"""Cache frozen transformer post embeddings for the MS4 author models."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

CODE_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = CODE_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from ms4mbti.config import RunConfig, ensure_artifact_dirs
from ms4mbti.embeddings import EmbeddingCacheSpec, write_embedding_shards
from ms4mbti.progress import ProgressLogger


def safe_model_name(model_id: str) -> str:
    return model_id.replace("/", "__").replace(":", "_")


def parse_args() -> argparse.Namespace:
    config = RunConfig()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-parquet",
        type=Path,
        default=CODE_DIR / "artifacts" / "preprocessed" / "full" / "modeling_posts.parquet",
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--model-id", default=config.frozen_embedding_model_id)
    parser.add_argument("--max-length", type=int, default=config.frozen_embedding_max_length)
    parser.add_argument("--batch-size", type=int, default=config.frozen_embedding_batch_size)
    parser.add_argument("--shard-size", type=int, default=config.embedding_shard_size)
    parser.add_argument("--max-rows", type=int, default=None)
    parser.add_argument(
        "--backend",
        choices=["transformers", "deterministic_hash"],
        default="transformers",
        help="Use deterministic_hash only for tests/smoke checks, not final reported results.",
    )
    parser.add_argument(
        "--local-files-only",
        action="store_true",
        help="Require the Hugging Face model to already exist in the local cache.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = RunConfig(frozen_embedding_model_id=args.model_id)
    ensure_artifact_dirs(config)
    output_dir = args.output_dir or (
        config.cache_dir
        / "transformer_embeddings"
        / f"{safe_model_name(args.model_id)}_max{args.max_length}"
    )
    logger = ProgressLogger("cache_transformer_embeddings", verbose=True)
    if not args.input_parquet.exists():
        raise FileNotFoundError(
            f"Missing preprocessed parquet: {args.input_parquet}. "
            "Run scripts/preprocess_reddit_ms4.py first."
        )

    logger.step("Loading posts", input_parquet=args.input_parquet)
    posts = pd.read_parquet(
        args.input_parquet,
        columns=["author", "split", "text_masked"],
    )
    posts = posts.reset_index().rename(columns={"index": "row_index"})
    if args.max_rows is not None:
        posts = posts.head(args.max_rows).copy()
    logger.step(
        "Loaded embedding input",
        rows=len(posts),
        authors=posts["author"].nunique(),
        backend=args.backend,
        output_dir=output_dir,
    )

    spec = EmbeddingCacheSpec(
        model_id=args.model_id,
        tokenizer_max_length=args.max_length,
        batch_size=args.batch_size,
        shard_size=args.shard_size,
        backend=args.backend,
    )
    manifest = write_embedding_shards(
        posts,
        output_dir=output_dir,
        spec=spec,
        config=config,
        local_files_only=args.local_files_only,
    )
    logger.step("Transformer embedding cache complete", **manifest)


if __name__ == "__main__":
    main()
