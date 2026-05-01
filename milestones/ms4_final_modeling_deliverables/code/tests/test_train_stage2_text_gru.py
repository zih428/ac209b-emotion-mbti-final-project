from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pandas as pd


CODE_DIR = Path(__file__).resolve().parents[1]
SCRIPT_PATH = CODE_DIR / "scripts" / "train_stage2_text_gru.py"


def _load_script_module():
    spec = importlib.util.spec_from_file_location("train_stage2_text_gru", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_dry_run_author_sampling_caps_posts_and_authors() -> None:
    module = _load_script_module()
    rows = []
    for split in ("train", "val", "test"):
        for author_idx in range(5):
            author = f"{split}_author_{author_idx}"
            for post_idx in range(author_idx + 1):
                rows.append(
                    {
                        "author": author,
                        "split": split,
                        "text_masked": f"post {post_idx}",
                        "target_E": author_idx % 2,
                        "target_S": author_idx % 2,
                        "target_T": author_idx % 2,
                        "target_J": author_idx % 2,
                    }
                )
    posts = pd.DataFrame(rows)

    sampled = module.sample_authors_for_dry_run(
        posts,
        max_authors_per_split=3,
        max_posts_per_author=2,
        seed=209066,
    )

    assert sampled.groupby("split")["author"].nunique().max() <= 3
    assert sampled.groupby("author").size().max() <= 2
    assert (sampled.groupby("author")["split"].nunique() == 1).all()


def test_attach_emotion_features_aligns_by_row_index(tmp_path) -> None:
    module = _load_script_module()
    posts = pd.DataFrame(
        {
            "author": ["a", "b"],
            "split": ["train", "val"],
            "text_masked": ["one", "two"],
            "target_E": [0, 1],
            "target_S": [0, 1],
            "target_T": [1, 0],
            "target_J": [1, 0],
        },
        index=[10, 20],
    )
    features = pd.DataFrame(
        {
            "row_index": [20, 10],
            "emotion_joy": [0.8, 0.2],
            "emotion_sadness": [0.1, 0.7],
        }
    )
    path = tmp_path / "emotion.parquet"
    features.to_parquet(path, index=False)

    merged = module.attach_emotion_features(
        posts,
        feature_path=path,
        emotion_cols=("emotion_joy", "emotion_sadness"),
    )

    assert merged.loc[0, "emotion_joy"] == 0.2
    assert merged.loc[1, "emotion_joy"] == 0.8
