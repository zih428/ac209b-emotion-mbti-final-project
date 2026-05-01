"""Data loading and MS3/MS4 filtering helpers."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from .config import RunConfig
from .preprocessing import (
    add_masked_text,
    add_word_count,
    clean_reddit_frame,
    resolve_author_label_conflicts,
)


def load_reddit_raw(
    config: RunConfig,
    *,
    nrows: int | None = None,
    usecols: list[str] | None = None,
) -> pd.DataFrame:
    """Load the Kaggle Reddit MBTI CSV through KaggleHub."""

    try:
        import kagglehub
    except ImportError as exc:
        raise RuntimeError(
            "KaggleHub is required for full Reddit loading. Install `kagglehub` "
            "or use the synthetic smoke-test path."
        ) from exc

    path = Path(kagglehub.dataset_download(config.reddit_dataset, path=config.reddit_file))
    read_kwargs: dict[str, object] = {"usecols": usecols or ["author", "body", "mbti"]}
    if nrows is not None:
        read_kwargs["nrows"] = nrows
    return pd.read_csv(path, **read_kwargs)


def load_reddit_sample_via_kagglehub(
    config: RunConfig,
    *,
    nrows: int = 1000,
) -> pd.DataFrame:
    """Lightweight KaggleHub API read for notebook/data-access smoke checks.

    This uses KaggleHub's pandas adapter and `nrows`, so it can verify access
    without resolving and reading the full 2.7 GB CSV.
    """

    try:
        import kagglehub
        from kagglehub import KaggleDatasetAdapter
    except ImportError as exc:
        raise RuntimeError(
            "KaggleHub is required for the Reddit API smoke path. Install the "
            "`full` optional dependencies with `uv sync --extra full --extra dev`."
        ) from exc

    return kagglehub.dataset_load(
        KaggleDatasetAdapter.PANDAS,
        config.reddit_dataset,
        config.reddit_file,
        pandas_kwargs={"nrows": nrows, "usecols": ["author", "body", "mbti"]},
    )


def load_emotion_dataset(config: RunConfig):
    """Load the Hugging Face emotion dataset for Stage 1 training."""

    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError(
            "The `datasets` package is required for full emotion-data loading. "
            "Install it or keep RUN_FULL_TRAINING=False."
        ) from exc
    return load_dataset(config.emotion_dataset)


def emotion_dataset_overview(config: RunConfig, *, max_rows_per_split: int = 5) -> dict:
    """Return a lightweight schema/count overview for the emotion dataset."""

    dataset = load_emotion_dataset(config)
    overview = {}
    for split_name, split in dataset.items():
        preview_count = min(max_rows_per_split, len(split))
        overview[split_name] = {
            "n_rows": int(len(split)),
            "columns": list(split.column_names),
            "preview": split.select(range(preview_count)).to_pandas()
            if preview_count
            else pd.DataFrame(),
        }
    return overview


def apply_ms3_filters(
    df: pd.DataFrame,
    *,
    min_words: int,
    min_posts_per_author: int,
    max_posts_per_author: int,
    text_col: str = "text_masked",
    seed: int = 209066,
) -> pd.DataFrame:
    """Apply the MS3 word floor, author floor, and per-author cap."""

    out = add_word_count(df, text_col=text_col)
    out = out.loc[out["word_count"] >= min_words].copy()

    author_counts = out["author"].value_counts()
    eligible_authors = author_counts.loc[author_counts >= min_posts_per_author].index
    out = out.loc[out["author"].isin(eligible_authors)].copy()

    if max_posts_per_author > 0:
        random_order = pd.Series(
            pd.util.hash_pandas_object(out[["author", "text", "mbti"]], index=False)
            .astype("uint64")
            .to_numpy()
            ^ seed,
            index=out.index,
            name="_cap_order",
        )
        out = (
            out.assign(_cap_order=random_order)
            .sort_values(["author", "_cap_order"])
            .groupby("author", group_keys=False)
            .head(max_posts_per_author)
            .drop(columns="_cap_order")
        )

    return out.sort_values(["author", "mbti", "text"]).reset_index(drop=True)


def build_modeling_frame(
    reddit_raw: pd.DataFrame,
    config: RunConfig,
    *,
    conflict_strategy: str = "drop",
    mask_text: bool = True,
) -> pd.DataFrame:
    """Build the masked, filtered MS4 modeling frame from raw Reddit rows."""

    out = clean_reddit_frame(reddit_raw)
    out = resolve_author_label_conflicts(out, strategy=conflict_strategy)
    if mask_text:
        out = add_masked_text(out, text_col="text", output_col="text_masked")
    else:
        out = out.copy()
        out["text_masked"] = out["text"]
    return apply_ms3_filters(
        out,
        min_words=config.min_words,
        min_posts_per_author=config.min_posts_per_author,
        max_posts_per_author=config.max_posts_per_author,
        text_col="text_masked",
        seed=config.seed,
    )
