#!/usr/bin/env python3
"""Cache post-level Stage 1 emotion probabilities for MS4 Stage 2 models."""

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

from ms4mbti.cache import base_compact_metadata, write_manifest
from ms4mbti.config import EMOTION_LABELS, RunConfig, ensure_artifact_dirs
from ms4mbti.emotion import predict_emotion_probabilities
from ms4mbti.progress import ProgressLogger
from ms4mbti.training import resolve_device, torch_environment


DEFAULT_MODEL = "bhadresh-savani/distilbert-base-uncased-emotion"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-parquet",
        type=Path,
        default=CODE_DIR / "artifacts" / "preprocessed" / "full" / "modeling_posts.parquet",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=CODE_DIR / "artifacts" / "cache" / "emotion_probs_full.parquet",
    )
    parser.add_argument("--model-name", default=DEFAULT_MODEL)
    parser.add_argument("--text-col", default="text_masked")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--max-length", type=int, default=64)
    parser.add_argument("--log-every-batches", type=int, default=100)
    parser.add_argument("--limit", type=int, default=None, help="Optional first-N dry run limit.")
    parser.add_argument(
        "--device",
        choices=["auto", "mps", "cuda", "cpu"],
        default="auto",
    )
    return parser.parse_args()


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True, default=str), encoding="utf-8")


def resolve_requested_device(requested: str) -> str:
    if requested == "auto":
        return resolve_device(prefer_mps=True, allow_cpu=True)
    return requested


def load_emotion_model(model_name: str):
    try:
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
    except ImportError as exc:
        raise RuntimeError("Transformers is required for emotion feature caching.") from exc

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    return tokenizer, model


def ordered_emotion_probabilities(
    probabilities: pd.DataFrame,
    *,
    model,
) -> pd.DataFrame:
    """Return probabilities in project label order."""

    id2label = getattr(model.config, "id2label", {}) or {}
    model_labels = [str(id2label[idx]).lower() for idx in range(len(id2label))]
    rename = {f"emotion_{label}": f"emotion_{label}" for label in EMOTION_LABELS}
    if model_labels and set(model_labels) >= set(EMOTION_LABELS):
        probabilities = probabilities.copy()
        probabilities.columns = [f"emotion_{label}" for label in model_labels]
        return probabilities[[rename[col] for col in rename]]
    expected = [f"emotion_{label}" for label in EMOTION_LABELS]
    if list(probabilities.columns) != expected:
        raise ValueError(
            "Emotion model labels do not match project labels. "
            f"model_labels={model_labels}, expected={list(EMOTION_LABELS)}"
        )
    return probabilities


def main() -> None:
    args = parse_args()
    config = RunConfig()
    ensure_artifact_dirs(config)
    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    run_name = "emotion-cache-sample" if args.limit else "emotion-cache-full"
    logger = ProgressLogger(run_name, verbose=True)

    if not args.input_parquet.exists():
        raise FileNotFoundError(
            f"Missing preprocessed parquet: {args.input_parquet}. "
            "Run scripts/preprocess_reddit_ms4.py first."
        )
    device = resolve_requested_device(args.device)
    logger.step("Starting emotion cache", output_path=args.output_path, model=args.model_name)
    logger.step("Torch environment", **torch_environment(), selected_device=device)

    logger.step("Loading posts", input_parquet=args.input_parquet)
    posts = pd.read_parquet(args.input_parquet, columns=[args.text_col])
    posts = posts.reset_index().rename(columns={"index": "row_index"})
    if args.limit is not None:
        posts = posts.head(args.limit).copy()
    logger.step("Loaded posts for emotion inference", rows=len(posts))

    logger.step("Loading emotion model", model=args.model_name)
    tokenizer, model = load_emotion_model(args.model_name)
    logger.step("Emotion model loaded", labels=getattr(model.config, "id2label", {}))

    probabilities = predict_emotion_probabilities(
        posts[args.text_col].tolist(),
        tokenizer=tokenizer,
        model=model,
        labels=EMOTION_LABELS,
        batch_size=args.batch_size,
        max_length=args.max_length,
        device=device,
        log_every_batches=args.log_every_batches,
        logger=logger,
    )
    probabilities = ordered_emotion_probabilities(probabilities, model=model)
    out = pd.concat([posts[["row_index"]].reset_index(drop=True), probabilities], axis=1)
    out.to_parquet(args.output_path, index=False)

    metadata = base_compact_metadata(
        artifact_type="ms4_stage1_emotion_probabilities",
        created_by="scripts/cache_emotion_features.py",
        split_id="masked_author_split_seed_209066_full",
        masking_status="explicit_mbti_terms_masked",
        label_encoding={label: idx for idx, label in enumerate(EMOTION_LABELS)},
        threshold_objective="n/a",
        extra={
            **config.metadata(),
            "input_parquet": str(args.input_parquet),
            "output_path": str(args.output_path),
            "model_name": args.model_name,
            "text_col": args.text_col,
            "batch_size": args.batch_size,
            "max_length": args.max_length,
            "log_every_batches": args.log_every_batches,
            "limit": args.limit,
            "device": device,
            "n_rows": int(len(out)),
        },
    )
    manifest = write_manifest(args.output_path.with_suffix(".manifest.json"), metadata)
    summary = {
        "output_path": str(args.output_path),
        "manifest_fingerprint": manifest["fingerprint"],
        "n_rows": int(len(out)),
        "model_name": args.model_name,
        "device": device,
    }
    write_json(args.output_path.with_suffix(".summary.json"), summary)
    logger.step("Emotion cache complete", **summary)


if __name__ == "__main__":
    main()
