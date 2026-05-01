"""Fast author-level linear baseline."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from .config import TARGET_COLUMNS
from .preprocessing import normalize_text, validate_required_columns


def build_author_documents(
    posts: pd.DataFrame,
    *,
    text_col: str = "text_masked",
    author_col: str = "author",
    split_col: str = "split",
    target_cols: tuple[str, ...] = TARGET_COLUMNS,
) -> pd.DataFrame:
    validate_required_columns(posts, [text_col, author_col, split_col, *target_cols])
    rows = []
    for author, group in posts.groupby(author_col):
        row = {
            author_col: author,
            split_col: group[split_col].iloc[0],
            "author_document": " ".join(group[text_col].map(normalize_text)),
        }
        for target in target_cols:
            row[target] = int(group[target].iloc[0])
        rows.append(row)
    return pd.DataFrame(rows)


def train_linear_author_baseline(
    posts: pd.DataFrame,
    *,
    text_col: str = "text_masked",
    author_col: str = "author",
    split_col: str = "split",
    target_cols: tuple[str, ...] = TARGET_COLUMNS,
    max_features: int = 5000,
    seed: int = 209066,
) -> tuple[pd.DataFrame, dict[str, Pipeline | DummyClassifier]]:
    """Train one TF-IDF + logistic regression classifier per dimension."""

    author_docs = build_author_documents(
        posts,
        text_col=text_col,
        author_col=author_col,
        split_col=split_col,
        target_cols=target_cols,
    )
    train = author_docs.loc[author_docs[split_col] == "train"].copy()
    if train.empty:
        raise ValueError("Linear baseline requires at least one training author")

    scored = author_docs.copy()
    models: dict[str, Pipeline | DummyClassifier] = {}
    for target in target_cols:
        y_train = train[target].to_numpy(dtype=int)
        if len(np.unique(y_train)) < 2:
            model: Pipeline | DummyClassifier = DummyClassifier(strategy="prior")
            model.fit(train[["author_document"]], y_train)
            positive_prob = float(y_train.mean())
            scored[f"score_linear_{target}"] = positive_prob
        else:
            model = Pipeline(
                steps=[
                    (
                        "tfidf",
                        TfidfVectorizer(
                            min_df=1,
                            max_features=max_features,
                            ngram_range=(1, 2),
                            strip_accents="unicode",
                        ),
                    ),
                    (
                        "clf",
                        LogisticRegression(
                            class_weight="balanced",
                            max_iter=500,
                            random_state=seed,
                            solver="liblinear",
                        ),
                    ),
                ]
            )
            model.fit(train["author_document"], y_train)
            scored[f"score_linear_{target}"] = model.predict_proba(
                scored["author_document"]
            )[:, 1]
        models[target] = model
    return scored, models
