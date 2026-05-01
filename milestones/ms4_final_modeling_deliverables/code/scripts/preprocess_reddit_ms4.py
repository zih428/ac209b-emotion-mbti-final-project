#!/usr/bin/env python3
"""Build and cache the MS4 masked Reddit modeling corpus.

This script does not train models. It resolves the KaggleHub Reddit CSV,
builds the MS4 masked/filtered corpus, creates author-level splits, writes
local cache artifacts, and records compact audit tables plus manifests.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

CODE_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = CODE_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from ms4mbti.cache import base_compact_metadata, write_manifest
from ms4mbti.config import DIMENSION_SPECS, TARGET_COLUMNS, RunConfig, ensure_artifact_dirs
from ms4mbti.data import apply_ms3_filters, load_reddit_raw
from ms4mbti.preprocessing import (
    add_masked_text,
    audit_mbti_leakage,
    author_label_conflicts,
    clean_reddit_frame,
    resolve_author_label_conflicts,
)
from ms4mbti.progress import ProgressLogger
from ms4mbti.split import attach_splits, make_author_frame, split_authors, split_balance_table
from ms4mbti.token_audit import token_truncation_audit


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nrows", type=int, default=None, help="Optional row limit for dry runs.")
    parser.add_argument(
        "--conflict-strategy",
        choices=["drop", "modal"],
        default="drop",
        help="How to handle authors with conflicting MBTI labels.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Override output directory. Defaults to artifacts/preprocessed/full or sample.",
    )
    parser.add_argument(
        "--skip-modeling-parquet",
        action="store_true",
        help="Only write compact author/audit artifacts, not the post-level modeling parquet.",
    )
    return parser.parse_args()


def write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")


def main() -> None:
    args = parse_args()
    config = RunConfig()
    ensure_artifact_dirs(config)
    run_name = "sample" if args.nrows else "full"
    output_dir = args.output_dir or (config.artifact_dir / "preprocessed" / run_name)
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = ProgressLogger(f"preprocess-{run_name}", verbose=True)

    logger.step("Starting MS4 Reddit preprocessing", nrows=args.nrows, output_dir=output_dir)

    logger.step("Loading Reddit raw data through KaggleHub")
    raw = load_reddit_raw(config, nrows=args.nrows)
    logger.step("Loaded raw Reddit rows", rows=len(raw), columns=list(raw.columns))

    logger.step("Cleaning schema and deriving MBTI targets")
    cleaned = clean_reddit_frame(raw)
    logger.step(
        "Cleaned Reddit rows",
        rows=len(cleaned),
        authors=cleaned["author"].nunique(),
        dropped_rows=len(raw) - len(cleaned),
    )
    del raw

    logger.step("Auditing author MBTI label conflicts")
    conflicts = author_label_conflicts(cleaned)
    conflicts_path = output_dir / "author_label_conflicts.csv"
    conflicts.to_csv(conflicts_path, index=False)
    logger.step("Conflict audit written", conflicted_authors=len(conflicts), path=conflicts_path)

    logger.step("Resolving author label conflicts", strategy=args.conflict_strategy)
    resolved = resolve_author_label_conflicts(cleaned, strategy=args.conflict_strategy)
    del cleaned
    logger.step("Resolved corpus", rows=len(resolved), authors=resolved["author"].nunique())

    logger.step("Auditing explicit MBTI leakage before masking")
    leakage_before = audit_mbti_leakage(resolved, text_col="text")

    logger.step("Masking explicit MBTI label terms")
    masked = add_masked_text(resolved)
    del resolved

    logger.step("Auditing explicit MBTI leakage after masking")
    leakage_after = audit_mbti_leakage(masked, text_col="text_masked")
    leakage_table = pd.DataFrame(
        [
            {"stage": "before_masking", **leakage_before},
            {"stage": "after_masking", **leakage_after},
        ]
    )
    leakage_path = output_dir / "mbti_leakage_audit.csv"
    leakage_table.to_csv(leakage_path, index=False)
    logger.step("Leakage audit written", path=leakage_path)

    logger.step(
        "Applying MS3 filters",
        min_words=config.min_words,
        min_posts_per_author=config.min_posts_per_author,
        max_posts_per_author=config.max_posts_per_author,
    )
    modeling = apply_ms3_filters(
        masked,
        min_words=config.min_words,
        min_posts_per_author=config.min_posts_per_author,
        max_posts_per_author=config.max_posts_per_author,
        text_col="text_masked",
        seed=config.seed,
    )
    del masked
    logger.step(
        "Built modeling corpus",
        rows=len(modeling),
        authors=modeling["author"].nunique(),
    )

    logger.step("Creating author-level split")
    author_frame = make_author_frame(modeling)
    split_result = split_authors(
        author_frame,
        train_size=config.train_size,
        val_size=config.val_size,
        test_size=config.test_size,
        seed=config.seed,
    )
    author_splits_path = output_dir / "author_splits.csv"
    split_result.authors.to_csv(author_splits_path, index=False)
    logger.step(
        "Author splits written",
        path=author_splits_path,
        method=split_result.method,
        warnings=split_result.warnings,
    )

    logger.step("Attaching splits to post-level modeling corpus")
    modeling = attach_splits(modeling, split_result.authors)

    logger.step("Writing split balance table")
    split_balance = split_balance_table(split_result.authors)
    split_balance_path = output_dir / "split_balance.csv"
    split_balance.to_csv(split_balance_path, index=False)

    logger.step("Running token truncation audit")
    truncation = token_truncation_audit(
        modeling,
        max_length=config.stage2_max_length,
        target_cols=TARGET_COLUMNS,
    )
    pd.DataFrame([truncation["post_summary"]]).to_csv(
        output_dir / "token_truncation_post_summary.csv", index=False
    )
    pd.DataFrame([truncation["author_summary"]]).to_csv(
        output_dir / "token_truncation_author_summary.csv", index=False
    )
    truncation["by_split"].to_csv(output_dir / "token_truncation_by_split.csv", index=False)
    truncation["by_dimension"].to_csv(
        output_dir / "token_truncation_by_dimension.csv", index=False
    )
    logger.step(
        "Token truncation audit written",
        share_over_max=truncation["post_summary"]["share_over_max"],
    )

    modeling_path = None
    if not args.skip_modeling_parquet:
        modeling_path = output_dir / "modeling_posts.parquet"
        logger.step("Writing post-level modeling parquet", path=modeling_path)
        modeling.to_parquet(modeling_path, index=False)
        logger.step("Modeling parquet written", path=modeling_path)

    label_encoding = {
        key: {
            "target_col": spec.target_col,
            "positive_label": spec.positive_label,
            "negative_label": spec.negative_label,
        }
        for key, spec in DIMENSION_SPECS.items()
    }
    metadata = base_compact_metadata(
        artifact_type="ms4_preprocessed_reddit",
        created_by="scripts/preprocess_reddit_ms4.py",
        split_id=f"masked_author_split_seed_{config.seed}_{run_name}",
        masking_status="explicit_mbti_terms_masked",
        label_encoding=label_encoding,
        threshold_objective="balanced_accuracy",
        extra={
            **config.metadata(),
            "run_name": run_name,
            "nrows": args.nrows,
            "conflict_strategy": args.conflict_strategy,
            "split_method": split_result.method,
            "split_warnings": list(split_result.warnings),
            "n_modeling_posts": int(len(modeling)),
            "n_modeling_authors": int(modeling["author"].nunique()),
            "modeling_posts_path": str(modeling_path) if modeling_path else None,
            "author_splits_path": str(author_splits_path),
        },
    )
    manifest_path = output_dir / "manifest.json"
    manifest = write_manifest(manifest_path, metadata)
    logger.step("Manifest written", path=manifest_path, fingerprint=manifest["fingerprint"])

    summary = {
        "output_dir": str(output_dir),
        "n_modeling_posts": int(len(modeling)),
        "n_modeling_authors": int(modeling["author"].nunique()),
        "split_method": split_result.method,
        "split_warnings": list(split_result.warnings),
        "manifest_fingerprint": manifest["fingerprint"],
    }
    write_json(output_dir / "summary.json", summary)
    logger.step("Preprocessing complete", **summary)


if __name__ == "__main__":
    main()
