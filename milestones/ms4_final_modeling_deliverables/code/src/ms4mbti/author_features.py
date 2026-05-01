"""Author-level feature construction for transformer-author experiments."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from .config import EMOTION_FEATURE_COLUMNS, TARGET_COLUMNS
from .embeddings import embedding_columns
from .preprocessing import normalize_text, validate_required_columns


CONTROL_COLUMNS = (
    "control_n_posts",
    "control_mean_tokens",
    "control_median_tokens",
    "control_p90_tokens",
    "control_total_tokens",
    "control_share_over_stage2_max",
)


@dataclass(frozen=True)
class AuthorFeatureSchema:
    text_mean: tuple[str, ...]
    text_std: tuple[str, ...]
    emotion: tuple[str, ...]
    controls: tuple[str, ...]
    targets: tuple[str, ...] = TARGET_COLUMNS

    def as_dict(self) -> dict[str, list[str]]:
        return {
            "text_mean": list(self.text_mean),
            "text_std": list(self.text_std),
            "emotion": list(self.emotion),
            "controls": list(self.controls),
            "targets": list(self.targets),
        }


@dataclass
class TrainOnlyScaler:
    columns: tuple[str, ...]
    scaler: StandardScaler
    fit_split: str = "train"


def token_lengths(posts: pd.DataFrame, *, text_col: str = "text_masked") -> pd.Series:
    if "token_length" in posts.columns:
        return posts["token_length"].astype(float)
    return posts[text_col].map(lambda value: len(normalize_text(value).split())).astype(float)


def select_post_budget(
    posts: pd.DataFrame,
    *,
    author_col: str = "author",
    row_id_col: str = "row_index",
    post_budget: int | None = None,
) -> pd.DataFrame:
    """Apply a deterministic per-author post budget without implying chronology."""

    if post_budget is None:
        return posts.copy()
    sort_cols = [author_col, row_id_col] if row_id_col in posts.columns else [author_col]
    return (
        posts.sort_values(sort_cols)
        .groupby(author_col, group_keys=False)
        .head(post_budget)
        .reset_index(drop=True)
    )


def aggregate_embedding_features(
    merged: pd.DataFrame,
    *,
    author_col: str = "author",
    emb_cols: tuple[str, ...] | None = None,
) -> tuple[pd.DataFrame, tuple[str, ...], tuple[str, ...]]:
    emb_cols = emb_cols or embedding_columns(merged)
    if not emb_cols:
        raise ValueError("No embedding columns found")
    grouped = merged.groupby(author_col)[list(emb_cols)]
    means = grouped.mean().add_prefix("text_mean_")
    stds = grouped.std(ddof=0).fillna(0.0).add_prefix("text_std_")
    out = pd.concat([means, stds], axis=1).reset_index()
    return out, tuple(means.columns), tuple(stds.columns)


def aggregate_emotion_features(
    merged: pd.DataFrame,
    *,
    author_col: str = "author",
    emotion_cols: tuple[str, ...] = EMOTION_FEATURE_COLUMNS,
) -> tuple[pd.DataFrame, tuple[str, ...]]:
    present = tuple(col for col in emotion_cols if col in merged.columns)
    if not present:
        return pd.DataFrame({author_col: merged[author_col].drop_duplicates()}), ()
    grouped = merged.groupby(author_col)[list(present)]
    frames = [
        grouped.mean().add_prefix("emotion_mean_"),
        grouped.std(ddof=0).fillna(0.0).add_prefix("emotion_std_"),
        grouped.max().add_prefix("emotion_max_"),
        grouped.quantile(0.90).add_prefix("emotion_p90_"),
    ]
    out = pd.concat(frames, axis=1).reset_index()
    feature_cols = tuple(col for frame in frames for col in frame.columns)
    return out, feature_cols


def aggregate_control_features(
    merged: pd.DataFrame,
    *,
    author_col: str = "author",
    text_col: str = "text_masked",
    max_length: int = 128,
) -> pd.DataFrame:
    work = merged[[author_col, text_col]].copy()
    work["token_length"] = token_lengths(merged, text_col=text_col)
    work["is_over_max"] = work["token_length"] > max_length
    grouped = work.groupby(author_col)
    return (
        grouped.agg(
            control_n_posts=("token_length", "size"),
            control_mean_tokens=("token_length", "mean"),
            control_median_tokens=("token_length", "median"),
            control_p90_tokens=("token_length", lambda values: float(np.percentile(values, 90))),
            control_total_tokens=("token_length", "sum"),
            control_share_over_stage2_max=("is_over_max", "mean"),
        )
        .reset_index()
    )


def author_targets(
    posts: pd.DataFrame,
    *,
    author_col: str = "author",
    split_col: str = "split",
    target_cols: tuple[str, ...] = TARGET_COLUMNS,
) -> pd.DataFrame:
    validate_required_columns(posts, [author_col, split_col, *target_cols])
    return (
        posts.groupby(author_col)[[split_col, *target_cols]]
        .first()
        .reset_index()
    )


def build_author_feature_table(
    posts: pd.DataFrame,
    embeddings: pd.DataFrame,
    *,
    emotion_features: pd.DataFrame | None = None,
    row_id_col: str = "row_index",
    author_col: str = "author",
    split_col: str = "split",
    text_col: str = "text_masked",
    target_cols: tuple[str, ...] = TARGET_COLUMNS,
    emotion_cols: tuple[str, ...] = EMOTION_FEATURE_COLUMNS,
    post_budget: int | None = None,
    max_length: int = 128,
) -> tuple[pd.DataFrame, AuthorFeatureSchema]:
    """Build one row per author with text, emotion, control, and target columns."""

    validate_required_columns(posts, [author_col, split_col, text_col, *target_cols])
    validate_required_columns(embeddings, [row_id_col, author_col, split_col])
    source = posts.copy()
    if row_id_col not in source.columns:
        source = source.reset_index().rename(columns={"index": row_id_col})
    source = select_post_budget(
        source,
        author_col=author_col,
        row_id_col=row_id_col,
        post_budget=post_budget,
    )

    merge_cols = [row_id_col, author_col, split_col, *embedding_columns(embeddings)]
    merged = source.merge(
        embeddings[merge_cols],
        on=[row_id_col, author_col, split_col],
        how="inner",
        validate="one_to_one",
    )
    if len(merged) != len(source):
        raise ValueError("Embedding cache does not cover all selected posts")

    if emotion_features is not None:
        validate_required_columns(emotion_features, [row_id_col, *emotion_cols])
        merged = merged.merge(
            emotion_features[[row_id_col, *emotion_cols]],
            on=row_id_col,
            how="left",
            validate="one_to_one",
        )
        if merged[list(emotion_cols)].isna().any(axis=1).any():
            raise ValueError("Emotion feature cache does not cover all selected posts")

    targets = author_targets(
        merged,
        author_col=author_col,
        split_col=split_col,
        target_cols=target_cols,
    )
    text_features, text_mean_cols, text_std_cols = aggregate_embedding_features(
        merged,
        author_col=author_col,
    )
    emotion_table, emotion_feature_cols = aggregate_emotion_features(
        merged,
        author_col=author_col,
        emotion_cols=emotion_cols,
    )
    controls = aggregate_control_features(
        merged,
        author_col=author_col,
        text_col=text_col,
        max_length=max_length,
    )

    table = targets.merge(text_features, on=author_col, how="left")
    table = table.merge(emotion_table, on=author_col, how="left")
    table = table.merge(controls, on=author_col, how="left")
    table = table.sort_values([split_col, author_col]).reset_index(drop=True)
    schema = AuthorFeatureSchema(
        text_mean=text_mean_cols,
        text_std=text_std_cols,
        emotion=emotion_feature_cols,
        controls=CONTROL_COLUMNS,
        targets=target_cols,
    )
    return table, schema


def fit_train_only_scaler(
    features: pd.DataFrame,
    columns: tuple[str, ...],
    *,
    split_col: str = "split",
    fit_split: str = "train",
) -> TrainOnlyScaler:
    validate_required_columns(features, [split_col, *columns])
    train = features.loc[features[split_col] == fit_split, list(columns)]
    if train.empty:
        raise ValueError(f"No rows found for scaler fit split {fit_split!r}")
    scaler = StandardScaler()
    scaler.fit(train.to_numpy(dtype=float))
    return TrainOnlyScaler(columns=columns, scaler=scaler, fit_split=fit_split)


def apply_train_only_scaler(
    features: pd.DataFrame,
    fitted: TrainOnlyScaler,
    *,
    suffix: str = "_z",
) -> pd.DataFrame:
    out = features.copy()
    scaled = fitted.scaler.transform(out[list(fitted.columns)].to_numpy(dtype=float))
    for idx, column in enumerate(fitted.columns):
        out[f"{column}{suffix}"] = scaled[:, idx]
    return out


def feature_columns_for_variant(
    schema: AuthorFeatureSchema,
    variant: str,
) -> tuple[str, ...]:
    lookup = {
        "text_mean": schema.text_mean,
        "text_mean_std": schema.text_mean + schema.text_std,
        "emotion_only": schema.emotion,
        "text_real_emotion": schema.text_mean + schema.text_std + schema.emotion,
        "text_controls": schema.text_mean + schema.text_std + schema.controls,
        "text_real_emotion_controls": (
            schema.text_mean + schema.text_std + schema.emotion + schema.controls
        ),
    }
    if variant not in lookup:
        raise ValueError(f"Unknown feature variant {variant!r}")
    return lookup[variant]
