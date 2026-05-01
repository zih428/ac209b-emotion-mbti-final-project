#!/usr/bin/env python3
"""Run MS4 author-level majority and TF-IDF logistic baselines."""

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

from ms4mbti.baselines import build_author_documents, train_linear_author_baseline
from ms4mbti.cache import base_compact_metadata, write_manifest
from ms4mbti.config import TARGET_COLUMNS, RunConfig, ensure_artifact_dirs
from ms4mbti.evaluation import (
    evaluate_with_validation_thresholds,
    majority_baseline_author_scores,
)
from ms4mbti.progress import ProgressLogger


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-parquet",
        type=Path,
        default=CODE_DIR / "artifacts" / "preprocessed" / "full" / "modeling_posts.parquet",
        help="Preprocessed post-level parquet from scripts/preprocess_reddit_ms4.py.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=CODE_DIR / "artifacts" / "runs" / "author_baselines",
    )
    parser.add_argument("--max-features", type=int, default=5000)
    return parser.parse_args()


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True, default=str), encoding="utf-8")


def thresholds_to_frame(model_id: str, threshold_result) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "model_id": model_id,
            "target": threshold_result.thresholds.index,
            "threshold": threshold_result.thresholds.to_numpy(dtype=float),
            "validation_balanced_accuracy": threshold_result.validation_scores.to_numpy(
                dtype=float
            ),
        }
    )


def write_model_artifacts(
    *,
    model_id: str,
    author_scores: pd.DataFrame,
    score_cols: tuple[str, ...],
    output_dir: Path,
) -> dict[str, Any]:
    eval_result = evaluate_with_validation_thresholds(author_scores, score_cols=score_cols)
    scores_path = output_dir / f"author_scores_{model_id}.csv"
    metrics_path = output_dir / f"metrics_{model_id}.csv"
    thresholds_path = output_dir / f"thresholds_{model_id}.csv"

    author_scores.drop(columns=["author_document"], errors="ignore").to_csv(
        scores_path, index=False
    )
    pd.concat(
        [
            eval_result["validation_metrics"].assign(model_id=model_id, split="val"),
            eval_result["test_metrics"].assign(model_id=model_id, split="test"),
        ],
        ignore_index=True,
    ).to_csv(metrics_path, index=False)
    thresholds_to_frame(model_id, eval_result["thresholds"]).to_csv(
        thresholds_path, index=False
    )

    metrics = pd.read_csv(metrics_path)
    return {
        "model_id": model_id,
        "author_scores_path": str(scores_path),
        "metrics_path": str(metrics_path),
        "thresholds_path": str(thresholds_path),
        "val_balanced_accuracy_mean": float(
            metrics.loc[metrics["split"] == "val", "balanced_accuracy"].mean()
        ),
        "test_balanced_accuracy_mean": float(
            metrics.loc[metrics["split"] == "test", "balanced_accuracy"].mean()
        ),
    }


def main() -> None:
    args = parse_args()
    config = RunConfig()
    ensure_artifact_dirs(config)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    logger = ProgressLogger("author_baselines", verbose=True)

    if not args.input_parquet.exists():
        raise FileNotFoundError(
            f"Missing preprocessed parquet: {args.input_parquet}. "
            "Run scripts/preprocess_reddit_ms4.py first."
        )

    logger.step("Loading preprocessed posts", input_parquet=args.input_parquet)
    posts = pd.read_parquet(args.input_parquet)
    logger.step(
        "Loaded posts",
        rows=len(posts),
        authors=posts["author"].nunique(),
        splits=posts["split"].value_counts().to_dict(),
    )

    logger.step("Building author documents for majority baseline")
    author_docs = build_author_documents(posts)
    train_authors = author_docs.loc[author_docs["split"] == "train"].copy()
    majority_scores = majority_baseline_author_scores(train_authors, author_docs)
    majority_score_cols = tuple(f"score_majority_{target}" for target in TARGET_COLUMNS)
    majority_summary = write_model_artifacts(
        model_id="majority_author",
        author_scores=majority_scores,
        score_cols=majority_score_cols,
        output_dir=args.output_dir,
    )
    logger.step("Majority baseline complete", **majority_summary)

    logger.step("Training TF-IDF logistic author baseline", max_features=args.max_features)
    linear_scores, _models = train_linear_author_baseline(
        posts,
        max_features=args.max_features,
        seed=config.seed,
    )
    linear_score_cols = tuple(f"score_linear_{target}" for target in TARGET_COLUMNS)
    linear_summary = write_model_artifacts(
        model_id="linear_tfidf_author",
        author_scores=linear_scores,
        score_cols=linear_score_cols,
        output_dir=args.output_dir,
    )
    logger.step("TF-IDF logistic baseline complete", **linear_summary)

    model_summaries = [majority_summary, linear_summary]
    metadata = base_compact_metadata(
        artifact_type="ms4_author_baselines",
        created_by="scripts/run_author_baselines.py",
        split_id="masked_author_split_seed_209066_full",
        masking_status="explicit_mbti_terms_masked",
        label_encoding={target: "1 means configured positive MBTI pole" for target in TARGET_COLUMNS},
        threshold_objective="balanced_accuracy",
        extra={
            **config.metadata(),
            "input_parquet": str(args.input_parquet),
            "output_dir": str(args.output_dir),
            "max_features": args.max_features,
            "n_posts": int(len(posts)),
            "n_authors": int(posts["author"].nunique()),
            "model_summaries": model_summaries,
        },
    )
    manifest = write_manifest(args.output_dir / "manifest.json", metadata)
    summary = {
        "manifest_fingerprint": manifest["fingerprint"],
        "n_posts": int(len(posts)),
        "n_authors": int(posts["author"].nunique()),
        "model_summaries": model_summaries,
        "output_dir": str(args.output_dir),
    }
    write_json(args.output_dir / "summary.json", summary)
    logger.step("Author baselines complete", **summary)


if __name__ == "__main__":
    main()
