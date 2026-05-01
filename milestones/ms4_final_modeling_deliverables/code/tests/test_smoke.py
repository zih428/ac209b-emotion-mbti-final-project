from __future__ import annotations

import sys
from pathlib import Path


CODE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE_DIR / "src"))

from ms4mbti.config import RunConfig
from ms4mbti.quality import run_smoke_checks


def test_non_training_smoke_checks_pass() -> None:
    summary = run_smoke_checks(
        RunConfig(
            min_words=5,
            min_posts_per_author=3,
            max_posts_per_author=5,
            stage2_max_length=24,
            train_size=0.625,
            val_size=0.1875,
            test_size=0.1875,
        ),
        verbose=False,
    )
    assert summary.n_modeling_posts > 0
    assert summary.n_authors > 0
    assert summary.leakage_after_share == 0.0
    assert summary.final_test_balanced_accuracy_mean >= 0.5
    assert summary.cache_fingerprint
