"""Synthetic smoke checks for the MS4 code path."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

from .baselines import train_linear_author_baseline
from .cache import metadata_fingerprint
from .config import RunConfig, TARGET_COLUMNS
from .data import apply_ms3_filters
from .evaluation import (
    aggregate_post_scores,
    evaluate_with_validation_thresholds,
    majority_baseline_author_scores,
    metric_table,
)
from .models import count_parameters
from .preprocessing import (
    add_masked_text,
    audit_mbti_leakage,
    author_label_conflicts,
    clean_reddit_frame,
    resolve_author_label_conflicts,
)
from .progress import ProgressLogger
from .split import attach_splits, make_author_frame, split_authors, split_balance_table
from .stage2 import build_stage2_from_arrays, make_torch_dataset, prepare_stage2_split_arrays
from .token_audit import token_truncation_audit
from .training import optional_torch, require_mps_for_full_training, set_global_seed, torch_environment
from .weighting import author_post_loss_weights, compute_pos_weights


@dataclass(frozen=True)
class SmokeSummary:
    n_raw_posts: int
    n_modeling_posts: int
    n_authors: int
    split_method: str
    split_warnings: tuple[str, ...]
    leakage_before_share: float
    leakage_after_share: float
    token_share_over_max: float
    weight_inverse: dict[str, float]
    weight_sqrt: dict[str, float]
    final_test_balanced_accuracy_mean: float
    linear_test_balanced_accuracy_mean: float
    torch_environment: dict[str, object]
    model_parameter_count: int | None
    cache_fingerprint: str


def make_synthetic_reddit(seed: int = 209066) -> pd.DataFrame:
    """Small deterministic Reddit-like corpus for code-path validation."""

    rng = np.random.default_rng(seed)
    type_counts = {
        "INFP": 30,
        "INFJ": 20,
        "INTP": 16,
        "INTJ": 14,
        "ENFP": 10,
        "ENTP": 8,
        "ISFJ": 8,
        "ISTJ": 7,
        "ISFP": 7,
        "ISTP": 7,
        "ENFJ": 6,
        "ENTJ": 6,
        "ESFP": 6,
        "ESTP": 6,
        "ESFJ": 6,
        "ESTJ": 6,
    }
    mbti_types = [
        mbti for mbti, count in type_counts.items() for _ in range(count)
    ]
    emotion_words = {
        "E": "party friends excited outgoing",
        "I": "quiet reading alone reflective",
        "S": "details practical routine concrete",
        "N": "patterns future abstract ideas",
        "T": "logic analysis objective debate",
        "F": "values empathy caring harmony",
        "J": "plans schedule organized deadline",
        "P": "flexible spontaneous explore open",
    }
    rows = []
    for author_idx, mbti in enumerate(mbti_types):
        author = f"author_{author_idx:02d}"
        for post_idx in range(8):
            words = [
                emotion_words[mbti[0]],
                emotion_words[mbti[1]],
                emotion_words[mbti[2]],
                emotion_words[mbti[3]],
                f"daily note {post_idx}",
            ]
            text = " ".join(words)
            if post_idx == 0 and author_idx % 5 == 0:
                text += f" I once tested as {mbti} on an MBTI forum."
            if post_idx == 1 and author_idx % 7 == 0:
                text += " Myers Briggs labels are weird but memorable."
            if post_idx == 2 and author_idx % 6 == 0:
                text += " " + " ".join(["longcontext"] * int(rng.integers(12, 20)))
            rows.append({"author": author, "body": text, "mbti": mbti})

    rows.append(
        {
            "author": "author_conflict",
            "body": "conflicting label row with enough words for filtering",
            "mbti": "INFP",
        }
    )
    rows.append(
        {
            "author": "author_conflict",
            "body": "second conflicting label row with enough words for filtering",
            "mbti": "ENTJ",
        }
    )
    return pd.DataFrame(rows)


def _synthetic_post_scores(posts: pd.DataFrame, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    out = posts.copy()
    for target in TARGET_COLUMNS:
        signal = out[target].to_numpy(dtype=float)
        noise = rng.normal(0.0, 0.12, size=len(out))
        out[f"score_final_{target}"] = np.clip(0.20 + 0.58 * signal + noise, 0.01, 0.99)
    return out


def _maybe_build_stage2_objects(posts: pd.DataFrame, max_length: int) -> int | None:
    torch = optional_torch()
    if torch is None:
        return None
    small = posts.head(16).copy()
    small = small.assign(split=["train"] * 10 + ["val"] * 3 + ["test"] * 3)
    arrays_by_split, vocab = prepare_stage2_split_arrays(
        small,
        max_length=max_length,
        vocab_max_size=128,
        vocab_min_freq=1,
    )
    arrays = arrays_by_split["train"]
    dataset = make_torch_dataset(arrays)
    model = build_stage2_from_arrays(arrays, vocab_size=vocab.size if vocab else 128)
    first = dataset[0]
    with torch.no_grad():
        _ = model(first["input_ids"].unsqueeze(0))
    return count_parameters(model)


def run_smoke_checks(
    config: RunConfig | None = None,
    *,
    verbose: bool = True,
    run_linear_baseline: bool = True,
) -> SmokeSummary:
    """Exercise the default non-training code path end to end."""

    config = config or RunConfig(
        min_words=5,
        min_posts_per_author=3,
        max_posts_per_author=5,
        stage2_max_length=24,
        train_size=0.625,
        val_size=0.1875,
        test_size=0.1875,
    )
    logger = ProgressLogger("smoke", verbose=verbose)
    set_global_seed(config.seed)
    require_mps_for_full_training(config.run_full_training)

    logger.step("Generating synthetic Reddit-like corpus")
    raw = make_synthetic_reddit(seed=config.seed)

    logger.step("Cleaning schema, deriving MBTI targets, and auditing leakage")
    cleaned = clean_reddit_frame(raw)
    conflicts = author_label_conflicts(cleaned)
    if conflicts.empty:
        raise AssertionError("Synthetic corpus should include one conflict author")
    resolved = resolve_author_label_conflicts(cleaned, strategy="drop")
    leakage_before = audit_mbti_leakage(resolved, text_col="text")

    logger.step("Masking explicit MBTI terms and applying MS3 filters")
    masked = add_masked_text(resolved)
    leakage_after = audit_mbti_leakage(masked, text_col="text_masked")
    modeling = apply_ms3_filters(
        masked,
        min_words=config.min_words,
        min_posts_per_author=config.min_posts_per_author,
        max_posts_per_author=config.max_posts_per_author,
        seed=config.seed,
    )
    if modeling.empty:
        raise AssertionError("Synthetic modeling frame is unexpectedly empty")

    logger.step("Creating author-level split")
    authors = make_author_frame(modeling)
    split_result = split_authors(
        authors,
        train_size=config.train_size,
        val_size=config.val_size,
        test_size=config.test_size,
        seed=config.seed,
    )
    posts = attach_splits(modeling, split_result.authors)
    _ = split_balance_table(split_result.authors)

    logger.step("Running token truncation audit")
    truncation = token_truncation_audit(
        posts, max_length=config.stage2_max_length, target_cols=TARGET_COLUMNS
    )

    logger.step("Computing training weights")
    train_posts = posts.loc[posts["split"] == "train"].copy()
    post_weights = author_post_loss_weights(train_posts)
    inverse_weights = compute_pos_weights(
        train_posts, variant="inverse", sample_weight=post_weights
    )
    sqrt_weights = compute_pos_weights(train_posts, variant="sqrt", sample_weight=post_weights)

    logger.step("Aggregating synthetic post scores and evaluating validation/test metrics")
    scored_posts = _synthetic_post_scores(posts, seed=config.seed)
    score_cols = tuple(f"score_final_{target}" for target in TARGET_COLUMNS)
    author_scores = aggregate_post_scores(scored_posts, score_cols=score_cols)
    eval_result = evaluate_with_validation_thresholds(author_scores, score_cols=score_cols)
    test_metrics = eval_result["test_metrics"]

    logger.step("Evaluating majority baseline")
    train_authors = split_result.authors.loc[split_result.authors["split"] == "train"]
    majority_scores = majority_baseline_author_scores(train_authors, split_result.authors)
    majority_score_cols = tuple(f"score_majority_{target}" for target in TARGET_COLUMNS)
    _ = metric_table(
        majority_scores.loc[majority_scores["split"] == "test"],
        score_cols=majority_score_cols,
        thresholds={target: 0.5 for target in TARGET_COLUMNS},
    )

    linear_mean = float("nan")
    if run_linear_baseline:
        logger.step("Training lightweight TF-IDF author baseline on synthetic data")
        linear_scores, _ = train_linear_author_baseline(posts, seed=config.seed)
        linear_score_cols = tuple(f"score_linear_{target}" for target in TARGET_COLUMNS)
        linear_eval = evaluate_with_validation_thresholds(
            linear_scores, score_cols=linear_score_cols
        )
        linear_mean = float(linear_eval["test_metrics"]["balanced_accuracy"].mean())

    logger.step("Checking optional torch Stage 2 dataset/model construction")
    parameter_count = _maybe_build_stage2_objects(posts, config.stage2_max_length)
    torch_env = torch_environment()

    metadata = {
        **config.metadata(),
        "masking_status": "masked",
        "split_method": split_result.method,
        "n_modeling_posts": int(len(modeling)),
    }
    fingerprint = metadata_fingerprint(metadata)
    summary = SmokeSummary(
        n_raw_posts=int(len(raw)),
        n_modeling_posts=int(len(modeling)),
        n_authors=int(authors["author"].nunique()),
        split_method=split_result.method,
        split_warnings=split_result.warnings,
        leakage_before_share=float(leakage_before["post_share_with_any_term"]),
        leakage_after_share=float(leakage_after["post_share_with_any_term"]),
        token_share_over_max=float(truncation["post_summary"]["share_over_max"]),
        weight_inverse={key: float(value) for key, value in inverse_weights.items()},
        weight_sqrt={key: float(value) for key, value in sqrt_weights.items()},
        final_test_balanced_accuracy_mean=float(test_metrics["balanced_accuracy"].mean()),
        linear_test_balanced_accuracy_mean=linear_mean,
        torch_environment=torch_env,
        model_parameter_count=parameter_count,
        cache_fingerprint=fingerprint,
    )
    logger.step("Smoke checks completed", **asdict(summary))
    return summary


def smoke_summary_json(summary: SmokeSummary) -> str:
    return json.dumps(asdict(summary), indent=2, sort_keys=True)
