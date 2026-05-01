"""Configuration constants for the MS4 pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


PROJECT_NUMBER = 66
TEAM_MEMBERS = ("Harry Hu", "Tom Shan", "Wendy Wang", "Kemeng Zhang")

DIMENSIONS = ("EI", "NS", "FT", "JP")
EMOTION_LABELS = ("sadness", "joy", "love", "anger", "fear", "surprise")
EMOTION_FEATURE_COLUMNS = tuple(f"emotion_{label}" for label in EMOTION_LABELS)
DEFAULT_FROZEN_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
FROZEN_EMBEDDING_FEATURE_GROUPS = (
    "text_embedding_mean",
    "text_embedding_std",
    "real_emotion_aggregates",
    "shuffled_emotion_aggregates",
    "activity_length_controls",
)
AUTHOR_POST_BUDGETS = (50, 200)


@dataclass(frozen=True)
class DimensionSpec:
    """One binary MBTI dimension.

    The positive labels are chosen to match the expected minority side in the
    Reddit MBTI corpus. This makes BCE pos_weight semantics align with the
    minority-class recall goal.
    """

    key: str
    display: str
    mbti_position: int
    positive_label: str
    negative_label: str
    target_col: str


DIMENSION_SPECS: dict[str, DimensionSpec] = {
    "EI": DimensionSpec("EI", "E/I", 0, "E", "I", "target_E"),
    "NS": DimensionSpec("NS", "N/S", 1, "S", "N", "target_S"),
    "FT": DimensionSpec("FT", "F/T", 2, "T", "F", "target_T"),
    "JP": DimensionSpec("JP", "J/P", 3, "J", "P", "target_J"),
}

TARGET_COLUMNS = tuple(DIMENSION_SPECS[key].target_col for key in DIMENSIONS)

CODE_DIR = Path(__file__).resolve().parents[2]
MS4_DIR = CODE_DIR.parent
PROJECT_ROOT = CODE_DIR.parents[2]


@dataclass(frozen=True)
class RunConfig:
    """Runtime settings shared by notebook, scripts, and tests."""

    seed: int = 209066
    run_full_training: bool = False
    smoke_test: bool = True
    min_words: int = 5
    min_posts_per_author: int = 20
    max_posts_per_author: int = 200
    stage1_max_length: int = 64
    stage2_max_length: int = 128
    frozen_embedding_model_id: str = DEFAULT_FROZEN_EMBEDDING_MODEL
    frozen_embedding_max_length: int = 256
    frozen_embedding_batch_size: int = 256
    embedding_shard_size: int = 50000
    author_post_budgets: tuple[int, ...] = AUTHOR_POST_BUDGETS
    cache_version: str = "ms4_transformer_author_v1"
    train_size: float = 0.70
    val_size: float = 0.15
    test_size: float = 0.15
    code_dir: Path = CODE_DIR
    artifact_dir: Path = CODE_DIR / "artifacts"
    compact_artifact_dir: Path = CODE_DIR / "artifacts" / "compact"
    cache_dir: Path = CODE_DIR / "artifacts" / "cache"
    model_dir: Path = CODE_DIR / "artifacts" / "models"
    reddit_dataset: str = "minhaozhang1/reddit-mbti-dataset"
    reddit_file: str = "reddit_post.csv"
    emotion_dataset: str = "AdamCodd/emotion-balanced"
    extra_metadata: dict[str, Any] = field(default_factory=dict)

    def metadata(self) -> dict[str, Any]:
        """Stable metadata for caches and compact artifacts."""

        return {
            "project_number": PROJECT_NUMBER,
            "seed": self.seed,
            "min_words": self.min_words,
            "min_posts_per_author": self.min_posts_per_author,
            "max_posts_per_author": self.max_posts_per_author,
            "stage1_max_length": self.stage1_max_length,
            "stage2_max_length": self.stage2_max_length,
            "frozen_embedding_model_id": self.frozen_embedding_model_id,
            "frozen_embedding_max_length": self.frozen_embedding_max_length,
            "embedding_shard_size": self.embedding_shard_size,
            "author_post_budgets": list(self.author_post_budgets),
            "cache_version": self.cache_version,
            "train_size": self.train_size,
            "val_size": self.val_size,
            "test_size": self.test_size,
            "positive_labels": {
                key: spec.positive_label for key, spec in DIMENSION_SPECS.items()
            },
            "target_columns": list(TARGET_COLUMNS),
            **self.extra_metadata,
        }


def ensure_artifact_dirs(config: RunConfig) -> None:
    """Create local artifact directories used by code paths that write outputs."""

    for path in (
        config.artifact_dir,
        config.compact_artifact_dir,
        config.cache_dir,
        config.model_dir,
    ):
        path.mkdir(parents=True, exist_ok=True)
