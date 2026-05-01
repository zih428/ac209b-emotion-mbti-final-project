"""Console entry points for MS4 helper commands."""

from __future__ import annotations

import argparse

from .config import RunConfig
from .quality import run_smoke_checks, smoke_summary_json


def _smoke_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run MS4 non-training smoke checks.")
    parser.add_argument("--quiet", action="store_true", help="Suppress progress lines.")
    parser.add_argument(
        "--skip-linear",
        action="store_true",
        help="Skip the lightweight TF-IDF baseline smoke check.",
    )
    return parser


def smoke_main() -> None:
    args = _smoke_parser().parse_args()
    config = RunConfig(
        min_words=5,
        min_posts_per_author=3,
        max_posts_per_author=5,
        stage2_max_length=24,
        train_size=0.625,
        val_size=0.1875,
        test_size=0.1875,
    )
    summary = run_smoke_checks(
        config,
        verbose=not args.quiet,
        run_linear_baseline=not args.skip_linear,
    )
    print(smoke_summary_json(summary))
