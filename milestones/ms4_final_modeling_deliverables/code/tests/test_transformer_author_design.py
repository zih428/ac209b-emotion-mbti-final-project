from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ms4mbti.author_features import build_author_feature_table
from ms4mbti.config import EMOTION_FEATURE_COLUMNS, TARGET_COLUMNS
from ms4mbti.embeddings import embedding_frame_from_array
from ms4mbti.evaluation import paired_bootstrap_delta
from ms4mbti.models import SetAttentionAuthorConfig, build_set_attention_author_model
from ms4mbti.negative_controls import (
    assert_split_preserving_shuffle,
    replace_with_shuffled_features,
)
from ms4mbti.transformer_author import (
    build_author_set_arrays,
    predict_set_attention_scores,
    train_set_attention_model,
)


CODE_DIR = Path(__file__).resolve().parents[1]


def _set_attention_script():
    spec = importlib.util.spec_from_file_location(
        "run_set_attention_author_models",
        CODE_DIR / "scripts" / "run_set_attention_author_models.py",
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _posts() -> pd.DataFrame:
    rows = []
    for idx in range(12):
        split = "train" if idx < 6 else ("val" if idx < 9 else "test")
        author = f"a{idx // 2}"
        rows.append(
            {
                "row_index": idx,
                "author": author,
                "split": split,
                "text_masked": f"masked post text {idx}",
                "target_E": int(idx % 2 == 0),
                "target_S": int(idx % 3 == 0),
                "target_T": int(idx % 4 < 2),
                "target_J": int(idx % 5 < 2),
            }
        )
    return pd.DataFrame(rows)


def _emotion(posts: pd.DataFrame) -> pd.DataFrame:
    out = posts[["row_index", "split"]].copy()
    for offset, col in enumerate(EMOTION_FEATURE_COLUMNS):
        out[col] = (posts["row_index"].to_numpy() + offset + 1) / 100.0
    return out


def _set_attention_posts() -> pd.DataFrame:
    rows = []
    author_splits = {
        "a0": "train",
        "a1": "train",
        "a2": "train",
        "a3": "train",
        "a4": "val",
        "a5": "val",
        "a6": "test",
        "a7": "test",
    }
    for author_idx, (author, split) in enumerate(author_splits.items()):
        for post_idx in range(3):
            row_index = author_idx * 10 + post_idx
            rows.append(
                {
                    "row_index": row_index,
                    "author": author,
                    "split": split,
                    "f0": float(author_idx) / 10.0,
                    "f1": float(post_idx) / 10.0,
                    "target_E": int(author_idx % 2 == 0),
                    "target_S": int(author_idx in {1, 2, 5, 6}),
                    "target_T": int(author_idx in {0, 3, 4, 7}),
                    "target_J": int(author_idx in {2, 3, 6, 7}),
                }
            )
    return pd.DataFrame(rows)


def test_author_feature_table_includes_text_emotion_and_controls() -> None:
    posts = _posts()
    embeddings = embedding_frame_from_array(
        posts[["row_index", "author", "split"]],
        np.arange(len(posts) * 4, dtype=np.float32).reshape(len(posts), 4),
    )

    features, schema = build_author_feature_table(
        posts,
        embeddings,
        emotion_features=_emotion(posts),
        post_budget=2,
    )

    assert len(features) == posts["author"].nunique()
    assert set(TARGET_COLUMNS).issubset(features.columns)
    assert len(schema.text_mean) == 4
    assert len(schema.text_std) == 4
    assert len(schema.emotion) == len(EMOTION_FEATURE_COLUMNS) * 4
    assert "control_n_posts" in features.columns
    assert features["control_n_posts"].max() == 2


def test_shuffled_emotion_preserves_split_membership_and_marginals() -> None:
    frame = _emotion(_posts())
    shuffled = replace_with_shuffled_features(
        frame,
        feature_cols=EMOTION_FEATURE_COLUMNS,
        split_col="split",
        seed=1,
    )

    assert_split_preserving_shuffle(
        frame,
        shuffled,
        feature_cols=EMOTION_FEATURE_COLUMNS,
        split_col="split",
    )
    assert shuffled["split"].equals(frame["split"])


def test_set_attention_author_model_has_no_temporal_positional_encoding() -> None:
    model = build_set_attention_author_model(
        SetAttentionAuthorConfig(input_dim=5, model_dim=8, num_heads=2)
    )

    assert getattr(model, "uses_positional_encoding") is False
    assert not any("pos" in name.lower() for name, _ in model.named_parameters())


def test_paired_bootstrap_delta_schema() -> None:
    authors = pd.DataFrame(
        {
            "author": [f"a{i}" for i in range(8)],
            "target_E": [0, 0, 1, 1, 0, 1, 0, 1],
            "target_S": [0, 1, 0, 1, 0, 0, 1, 1],
            "target_T": [1, 1, 0, 0, 1, 0, 1, 0],
            "target_J": [0, 1, 1, 0, 0, 1, 1, 0],
        }
    )
    baseline = authors.copy()
    comparison = authors.copy()
    for target in TARGET_COLUMNS:
        baseline[f"score_base_{target}"] = 0.45
        comparison[f"score_comp_{target}"] = comparison[target] * 0.8 + 0.1

    result = paired_bootstrap_delta(
        baseline,
        comparison,
        baseline_score_cols=tuple(f"score_base_{target}" for target in TARGET_COLUMNS),
        comparison_score_cols=tuple(f"score_comp_{target}" for target in TARGET_COLUMNS),
        baseline_thresholds={target: 0.5 for target in TARGET_COLUMNS},
        comparison_thresholds={target: 0.5 for target in TARGET_COLUMNS},
        comparison_name="comparison minus baseline",
        n_bootstrap=25,
        seed=3,
    )

    assert {
        "comparison",
        "metric",
        "target",
        "point_estimate",
        "ci_lower",
        "ci_upper",
        "n_bootstrap",
        "n_authors",
    }.issubset(result.columns)
    assert set(result["target"]) == {*TARGET_COLUMNS, "mean"}


def test_post_controls_are_standardized_from_train_split_only() -> None:
    script = _set_attention_script()
    posts = pd.DataFrame(
        {
            "split": ["train", "train", "val", "test"],
            "post_token_length": [10.0, 20.0, 110.0, 210.0],
            "post_is_over_128": [0.0, 1.0, 0.0, 1.0],
            "post_log_token_length": [2.0, 4.0, 20.0, 40.0],
        }
    )
    scaled, metadata = script.standardize_post_controls_train_only(
        posts,
        control_cols=("post_token_length", "post_is_over_128", "post_log_token_length"),
    )

    train = scaled.loc[scaled["split"] == "train"]
    control_cols = ["post_token_length", "post_is_over_128", "post_log_token_length"]
    assert np.allclose(train[control_cols].mean(), 0.0)
    assert np.allclose(train[control_cols].std(ddof=0), 1.0)
    assert metadata["fit_split"] == "train"
    assert metadata["mean"]["post_token_length"] == 15.0


def test_set_attention_training_is_reproducible_with_seed() -> None:
    pytest.importorskip("torch")
    arrays = build_author_set_arrays(
        _set_attention_posts(),
        feature_cols=("f0", "f1"),
        post_budget=3,
    )

    model_a, history_a = train_set_attention_model(
        arrays,
        model_id="seeded",
        post_budget=3,
        epochs=2,
        batch_size=2,
        device="cpu",
        seed=123,
    )
    model_b, history_b = train_set_attention_model(
        arrays,
        model_id="seeded",
        post_budget=3,
        epochs=2,
        batch_size=2,
        device="cpu",
        seed=123,
    )
    scores_a = predict_set_attention_scores(model_a, arrays, model_id="seeded", device="cpu")
    scores_b = predict_set_attention_scores(model_b, arrays, model_id="seeded", device="cpu")

    pd.testing.assert_frame_equal(history_a, history_b)
    score_cols = [col for col in scores_a.columns if col.startswith("score_seeded_")]
    assert np.allclose(scores_a[score_cols], scores_b[score_cols])
