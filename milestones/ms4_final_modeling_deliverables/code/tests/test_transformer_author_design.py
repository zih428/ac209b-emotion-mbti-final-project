from __future__ import annotations

import numpy as np
import pandas as pd

from ms4mbti.author_features import build_author_feature_table
from ms4mbti.config import EMOTION_FEATURE_COLUMNS, TARGET_COLUMNS
from ms4mbti.embeddings import embedding_frame_from_array
from ms4mbti.evaluation import paired_bootstrap_delta
from ms4mbti.models import SetAttentionAuthorConfig, build_set_attention_author_model
from ms4mbti.negative_controls import (
    assert_split_preserving_shuffle,
    replace_with_shuffled_features,
)


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
