#!/usr/bin/env python3
"""Train or dry-run the MS4 Stage 2 GRU.

The default mode is intentionally a dry run: it uses the real preprocessed
Reddit parquet, samples a fixed author subset, trains briefly, and writes the
same report artifacts as the full run. Pass ``--full-run`` explicitly for the
expensive training path.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

CODE_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = CODE_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from ms4mbti.cache import base_compact_metadata, write_manifest
from ms4mbti.config import EMOTION_LABELS, TARGET_COLUMNS, RunConfig, ensure_artifact_dirs
from ms4mbti.evaluation import aggregate_post_scores, evaluate_with_validation_thresholds
from ms4mbti.models import count_parameters
from ms4mbti.progress import ProgressLogger
from ms4mbti.stage2 import (
    build_stage2_from_arrays,
    make_torch_dataset,
    predict_stage2_probabilities,
    prepare_stage2_split_arrays,
    train_stage2_model,
)
from ms4mbti.training import optional_torch, resolve_device, set_global_seed, torch_environment
from ms4mbti.weighting import author_post_loss_weights, compute_pos_weights


SCORE_COLUMNS = tuple(f"score_gru_{target}" for target in TARGET_COLUMNS)
DEFAULT_EMOTION_COLUMNS = tuple(f"emotion_{label}" for label in EMOTION_LABELS)


@dataclass(frozen=True)
class Stage2RunSettings:
    run_id: str
    dry_run: bool
    input_parquet: str
    output_dir: str
    seed: int
    epochs: int
    batch_size: int
    learning_rate: float
    patience: int
    max_length: int
    vocab_max_size: int
    vocab_min_freq: int
    hidden_dim: int
    dropout: float
    pos_weight_variant: str
    emotion_feature_path: str | None
    emotion_cols: tuple[str, ...]
    max_authors_per_split: int | None
    max_posts_per_author: int | None
    device: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-parquet",
        type=Path,
        default=CODE_DIR / "artifacts" / "preprocessed" / "full" / "modeling_posts.parquet",
        help="Preprocessed post-level parquet from scripts/preprocess_reddit_ms4.py.",
    )
    parser.add_argument(
        "--run-id",
        default=None,
        help="Output run id. Defaults to stage2_text_gru_dryrun or stage2_text_gru_full.",
    )
    parser.add_argument("--output-dir", type=Path, default=None, help="Override run output dir.")
    parser.add_argument(
        "--full-run",
        action="store_true",
        help="Opt into the expensive full training path. Default is a dry run.",
    )
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--patience", type=int, default=2)
    parser.add_argument("--max-length", type=int, default=None)
    parser.add_argument("--vocab-max-size", type=int, default=None)
    parser.add_argument("--vocab-min-freq", type=int, default=2)
    parser.add_argument("--hidden-dim", type=int, default=None)
    parser.add_argument("--dropout", type=float, default=0.20)
    parser.add_argument(
        "--pos-weight-variant",
        choices=["inverse", "sqrt"],
        default="sqrt",
        help="BCE positive-class weighting recipe.",
    )
    parser.add_argument(
        "--emotion-feature-path",
        type=Path,
        default=None,
        help=(
            "Optional post-level emotion probability parquet/csv with row_index and "
            "emotion_* columns."
        ),
    )
    parser.add_argument(
        "--emotion-cols",
        nargs="+",
        default=list(DEFAULT_EMOTION_COLUMNS),
        help="Emotion feature columns to merge from --emotion-feature-path.",
    )
    parser.add_argument(
        "--max-authors-per-split",
        type=int,
        default=None,
        help="Per-split author cap. Defaults to 64 for dry runs and no cap for full runs.",
    )
    parser.add_argument(
        "--max-posts-per-author",
        type=int,
        default=None,
        help="Per-author post cap after author sampling. Defaults to 20 for dry runs.",
    )
    parser.add_argument(
        "--device",
        choices=["auto", "mps", "cuda", "cpu"],
        default="auto",
        help="Torch device. auto prefers MPS, then CUDA, then CPU for dry runs.",
    )
    return parser.parse_args()


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True, default=str), encoding="utf-8")


def resolve_requested_device(requested: str, *, dry_run: bool) -> str:
    if requested == "auto":
        return resolve_device(prefer_mps=True, allow_cpu=dry_run)
    torch = optional_torch()
    if requested == "mps" and not (
        torch is not None and hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
    ):
        raise RuntimeError("Requested device 'mps', but MPS is not available.")
    if requested == "cuda" and not (torch is not None and torch.cuda.is_available()):
        raise RuntimeError("Requested device 'cuda', but CUDA is not available.")
    if requested == "cpu" and not dry_run:
        raise RuntimeError("Full Stage 2 training must run on MPS/CUDA, not CPU.")
    return requested


def sample_authors_for_dry_run(
    posts: pd.DataFrame,
    *,
    max_authors_per_split: int | None,
    max_posts_per_author: int | None,
    seed: int,
) -> pd.DataFrame:
    """Take a deterministic author-balanced subset without crossing splits."""

    if max_authors_per_split is None and max_posts_per_author is None:
        return posts.copy()

    rng = np.random.default_rng(seed)
    selected_authors: list[str] = []
    if max_authors_per_split is None:
        selected_authors = posts["author"].drop_duplicates().tolist()
    else:
        author_splits = posts[["author", "split"]].drop_duplicates()
        for split_name, group in author_splits.groupby("split", sort=True):
            authors = group["author"].sort_values().to_numpy()
            n_take = min(max_authors_per_split, len(authors))
            chosen = rng.choice(authors, size=n_take, replace=False)
            selected_authors.extend(sorted(chosen.tolist()))

    out = posts.loc[posts["author"].isin(selected_authors)].copy()
    if max_posts_per_author is not None:
        out = (
            out.sample(frac=1.0, random_state=seed)
            .groupby("author", group_keys=False)
            .head(max_posts_per_author)
            .sort_values(["split", "author"])
        )
    return out.reset_index(drop=True)


def attach_training_weights(posts: pd.DataFrame, *, variant: str) -> tuple[pd.DataFrame, pd.Series]:
    out = posts.copy()
    train_mask = out["split"] == "train"
    train_posts = out.loc[train_mask].copy()
    train_weights = author_post_loss_weights(train_posts)
    out["post_loss_weight"] = 1.0
    out.loc[train_mask, "post_loss_weight"] = train_weights.reindex(train_posts.index).to_numpy()
    pos_weight = compute_pos_weights(
        train_posts,
        variant=variant,
        sample_weight=train_weights,
    )
    return out, pos_weight


def read_feature_frame(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        return pd.read_parquet(path)
    if suffix == ".csv":
        return pd.read_csv(path)
    raise ValueError("Emotion feature path must be .parquet or .csv")


def attach_emotion_features(
    posts: pd.DataFrame,
    *,
    feature_path: Path | None,
    emotion_cols: tuple[str, ...],
) -> pd.DataFrame:
    if feature_path is None:
        return posts
    if not feature_path.exists():
        raise FileNotFoundError(f"Missing emotion feature cache: {feature_path}")

    features = read_feature_frame(feature_path)
    required = ["row_index", *emotion_cols]
    missing = [col for col in required if col not in features.columns]
    if missing:
        raise ValueError(f"Emotion feature cache is missing columns: {missing}")
    if features["row_index"].duplicated().any():
        raise ValueError("Emotion feature cache contains duplicate row_index values")

    source = posts.reset_index().rename(columns={"index": "row_index"})
    merged = source.merge(
        features[required],
        on="row_index",
        how="left",
        validate="one_to_one",
    )
    missing_features = merged[list(emotion_cols)].isna().any(axis=1)
    if missing_features.any():
        raise ValueError(
            f"Emotion feature cache lacks {int(missing_features.sum())} selected posts"
        )
    return merged.drop(columns=["row_index"])


def thresholds_to_frame(threshold_result) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "target": threshold_result.thresholds.index,
            "threshold": threshold_result.thresholds.to_numpy(dtype=float),
            threshold_result.validation_scores.name: threshold_result.validation_scores.to_numpy(
                dtype=float
            ),
        }
    )


def main() -> None:
    args = parse_args()
    config = RunConfig(run_full_training=args.full_run, smoke_test=not args.full_run)
    ensure_artifact_dirs(config)
    dry_run = not args.full_run
    run_id = args.run_id or ("stage2_text_gru_dryrun" if dry_run else "stage2_text_gru_full")
    output_dir = args.output_dir or (config.artifact_dir / "runs" / run_id)
    output_dir.mkdir(parents=True, exist_ok=True)

    epochs = args.epochs if args.epochs is not None else (1 if dry_run else 8)
    batch_size = args.batch_size if args.batch_size is not None else (128 if dry_run else 512)
    max_length = args.max_length if args.max_length is not None else config.stage2_max_length
    vocab_max_size = args.vocab_max_size if args.vocab_max_size is not None else (
        5000 if dry_run else 30000
    )
    hidden_dim = args.hidden_dim if args.hidden_dim is not None else (64 if dry_run else 128)
    max_authors_per_split = args.max_authors_per_split
    if max_authors_per_split is None and dry_run:
        max_authors_per_split = 64
    max_posts_per_author = args.max_posts_per_author
    if max_posts_per_author is None and dry_run:
        max_posts_per_author = 20
    device = resolve_requested_device(args.device, dry_run=dry_run)

    logger = ProgressLogger(run_id, verbose=True)
    set_global_seed(config.seed)
    logger.step(
        "Starting Stage 2 GRU run",
        dry_run=dry_run,
        output_dir=output_dir,
        emotion_features=args.emotion_feature_path is not None,
    )
    logger.step("Torch environment", **torch_environment(), selected_device=device)

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

    posts = sample_authors_for_dry_run(
        posts,
        max_authors_per_split=max_authors_per_split,
        max_posts_per_author=max_posts_per_author,
        seed=config.seed,
    )
    logger.step(
        "Prepared run corpus",
        rows=len(posts),
        authors=posts["author"].nunique(),
        splits=posts["split"].value_counts().to_dict(),
    )
    emotion_cols = tuple(args.emotion_cols) if args.emotion_feature_path else ()
    if args.emotion_feature_path:
        logger.step(
            "Attaching emotion features",
            feature_path=args.emotion_feature_path,
            emotion_cols=emotion_cols,
        )
        posts = attach_emotion_features(
            posts,
            feature_path=args.emotion_feature_path,
            emotion_cols=emotion_cols,
        )
        logger.step("Emotion features attached", rows=len(posts), emotion_dim=len(emotion_cols))

    posts, pos_weight = attach_training_weights(posts, variant=args.pos_weight_variant)
    logger.step("Computed training weights", pos_weight=pos_weight.to_dict())

    logger.step("Encoding text and building datasets", max_length=max_length)
    arrays_by_split, vocab = prepare_stage2_split_arrays(
        posts,
        max_length=max_length,
        emotion_cols=emotion_cols,
        sample_weight_col="post_loss_weight",
        vocab_max_size=vocab_max_size,
        vocab_min_freq=args.vocab_min_freq,
    )
    for required_split in ("train", "val", "test"):
        if required_split not in arrays_by_split:
            raise ValueError(f"Run corpus lacks required split {required_split!r}")

    train_dataset = make_torch_dataset(arrays_by_split["train"])
    val_dataset = make_torch_dataset(arrays_by_split["val"])
    test_dataset = make_torch_dataset(arrays_by_split["test"])
    model = build_stage2_from_arrays(
        arrays_by_split["train"],
        vocab_size=vocab.size if vocab is not None else vocab_max_size,
        hidden_dim=hidden_dim,
        dropout=args.dropout,
    )
    parameter_count = count_parameters(model)
    logger.step(
        "Built Stage 2 model",
        vocab_size=vocab.size if vocab is not None else None,
        parameter_count=parameter_count,
    )

    settings = Stage2RunSettings(
        run_id=run_id,
        dry_run=dry_run,
        input_parquet=str(args.input_parquet),
        output_dir=str(output_dir),
        seed=config.seed,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=args.learning_rate,
        patience=args.patience,
        max_length=max_length,
        vocab_max_size=vocab_max_size,
        vocab_min_freq=args.vocab_min_freq,
        hidden_dim=hidden_dim,
        dropout=args.dropout,
        pos_weight_variant=args.pos_weight_variant,
        emotion_feature_path=str(args.emotion_feature_path) if args.emotion_feature_path else None,
        emotion_cols=emotion_cols,
        max_authors_per_split=max_authors_per_split,
        max_posts_per_author=max_posts_per_author,
        device=device,
    )
    write_json(output_dir / "run_settings.json", asdict(settings))

    checkpoint_path = output_dir / "checkpoint_best.pt"
    logger.step("Training Stage 2 model", epochs=epochs, batch_size=batch_size)
    history = train_stage2_model(
        model,
        train_dataset,
        val_dataset,
        pos_weight=pos_weight.reindex(TARGET_COLUMNS).to_numpy(dtype=np.float32),
        run_full_training=args.full_run,
        epochs=epochs,
        batch_size=batch_size,
        lr=args.learning_rate,
        patience=args.patience,
        device=device,
        checkpoint_path=checkpoint_path,
        logger=logger,
    )
    history_path = output_dir / "history.csv"
    history.to_csv(history_path, index=False)

    logger.step("Running validation/test inference")
    score_blocks = []
    for split_name, dataset in (("val", val_dataset), ("test", test_dataset)):
        block = predict_stage2_probabilities(
            model,
            dataset,
            score_cols=SCORE_COLUMNS,
            batch_size=batch_size,
            device=device,
            logger=logger,
        )
        block["split"] = split_name
        score_blocks.append(block)
    post_scores = pd.concat(score_blocks, ignore_index=True)
    score_source = posts.reset_index().rename(columns={"index": "row_index"})
    post_scores = post_scores.merge(
        score_source[["row_index", "author", "split", *TARGET_COLUMNS]],
        on=["row_index", "split"],
        how="left",
        validate="one_to_one",
    )
    post_scores_path = output_dir / "post_scores_val_test.parquet"
    post_scores.to_parquet(post_scores_path, index=False)

    logger.step("Aggregating author scores and computing metrics")
    author_scores = aggregate_post_scores(post_scores, score_cols=SCORE_COLUMNS)
    author_scores_path = output_dir / "author_scores_stage2_text_gru.csv"
    author_scores.to_csv(author_scores_path, index=False)
    eval_result = evaluate_with_validation_thresholds(author_scores, score_cols=SCORE_COLUMNS)
    thresholds_path = output_dir / "thresholds_stage2_text_gru.csv"
    metrics_path = output_dir / "metrics_stage2_text_gru.csv"
    thresholds_to_frame(eval_result["thresholds"]).to_csv(thresholds_path, index=False)
    pd.concat(
        [
            eval_result["validation_metrics"].assign(split="val"),
            eval_result["test_metrics"].assign(split="test"),
        ],
        ignore_index=True,
    ).to_csv(metrics_path, index=False)

    metadata = base_compact_metadata(
        artifact_type="ms4_stage2_text_gru_run",
        created_by="scripts/train_stage2_text_gru.py",
        split_id="masked_author_split_seed_209066_full",
        masking_status="explicit_mbti_terms_masked",
        label_encoding={target: "1 means configured positive MBTI pole" for target in TARGET_COLUMNS},
        threshold_objective="balanced_accuracy",
        extra={
            **config.metadata(),
            **asdict(settings),
            "n_run_posts": int(len(posts)),
            "n_run_authors": int(posts["author"].nunique()),
            "parameter_count": parameter_count,
            "checkpoint_path": str(checkpoint_path),
            "history_path": str(history_path),
            "author_scores_path": str(author_scores_path),
            "metrics_path": str(metrics_path),
            "thresholds_path": str(thresholds_path),
        },
    )
    manifest = write_manifest(output_dir / "manifest.json", metadata)
    metrics = pd.read_csv(metrics_path)
    summary = {
        "run_id": run_id,
        "dry_run": dry_run,
        "manifest_fingerprint": manifest["fingerprint"],
        "n_run_posts": int(len(posts)),
        "n_run_authors": int(posts["author"].nunique()),
        "parameter_count": parameter_count,
        "best_val_loss": float(history["val_loss"].min()) if len(history) else None,
        "final_train_loss": float(history["train_loss"].iloc[-1]) if len(history) else None,
        "test_balanced_accuracy_mean": float(
            metrics.loc[metrics["split"] == "test", "balanced_accuracy"].mean()
        ),
        "output_dir": str(output_dir),
    }
    write_json(output_dir / "summary.json", summary)
    logger.step("Stage 2 run complete", **summary)


if __name__ == "__main__":
    main()
