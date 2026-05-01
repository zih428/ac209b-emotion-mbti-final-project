"""Author-level frozen-transformer classifier helpers."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .config import TARGET_COLUMNS
from .evaluation import evaluate_with_validation_thresholds
from .models import SetAttentionAuthorConfig, build_set_attention_author_model
from .preprocessing import validate_required_columns
from .training import optional_torch, resolve_device


@dataclass(frozen=True)
class AuthorProbeResult:
    model_id: str
    author_scores: pd.DataFrame
    metrics: pd.DataFrame
    thresholds: pd.DataFrame


def make_score_columns(model_id: str, target_cols: tuple[str, ...] = TARGET_COLUMNS) -> tuple[str, ...]:
    return tuple(f"score_{model_id}_{target}" for target in target_cols)


def threshold_frame(model_id: str, threshold_result) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "model_id": model_id,
            "target": threshold_result.thresholds.index,
            "threshold": threshold_result.thresholds.to_numpy(dtype=float),
            threshold_result.validation_scores.name: threshold_result.validation_scores.to_numpy(
                dtype=float
            ),
        }
    )


def train_author_probe(
    features: pd.DataFrame,
    *,
    model_id: str,
    feature_cols: tuple[str, ...],
    classifier: str = "logistic",
    target_cols: tuple[str, ...] = TARGET_COLUMNS,
    author_col: str = "author",
    split_col: str = "split",
    seed: int = 209066,
) -> tuple[AuthorProbeResult, dict[str, Pipeline | DummyClassifier]]:
    """Train one author-level binary probe per MBTI dimension."""

    validate_required_columns(features, [author_col, split_col, *feature_cols, *target_cols])
    train = features.loc[features[split_col] == "train"].copy()
    if train.empty:
        raise ValueError("Author probe requires training rows")

    scored = features[[author_col, split_col, *target_cols]].copy()
    score_cols = make_score_columns(model_id, target_cols)
    models: dict[str, Pipeline | DummyClassifier] = {}
    for target, score_col in zip(target_cols, score_cols, strict=True):
        y_train = train[target].to_numpy(dtype=int)
        if len(np.unique(y_train)) < 2:
            model: Pipeline | DummyClassifier = DummyClassifier(strategy="prior")
            model.fit(train[list(feature_cols)], y_train)
            scored[score_col] = float(y_train.mean())
        else:
            if classifier == "logistic":
                estimator = LogisticRegression(
                    class_weight="balanced",
                    max_iter=500,
                    random_state=seed,
                    solver="liblinear",
                )
            elif classifier == "mlp":
                estimator = MLPClassifier(
                    hidden_layer_sizes=(128, 64),
                    activation="relu",
                    alpha=1e-4,
                    batch_size=128,
                    early_stopping=True,
                    max_iter=120,
                    random_state=seed,
                )
            else:
                raise ValueError("classifier must be `logistic` or `mlp`")
            model = Pipeline(
                steps=[
                    ("scale", StandardScaler()),
                    ("clf", estimator),
                ]
            )
            model.fit(train[list(feature_cols)], y_train)
            scored[score_col] = model.predict_proba(features[list(feature_cols)])[:, 1]
        models[target] = model

    eval_result = evaluate_with_validation_thresholds(scored, score_cols=score_cols)
    metrics = pd.concat(
        [
            eval_result["validation_metrics"].assign(model_id=model_id, split="val"),
            eval_result["test_metrics"].assign(model_id=model_id, split="test"),
        ],
        ignore_index=True,
    )
    thresholds = threshold_frame(model_id, eval_result["thresholds"])
    return AuthorProbeResult(model_id, scored, metrics, thresholds), models


@dataclass(frozen=True)
class AuthorSetArrays:
    author: np.ndarray
    split: np.ndarray
    post_features: np.ndarray
    post_mask: np.ndarray
    targets: np.ndarray


def build_author_set_arrays(
    posts: pd.DataFrame,
    *,
    feature_cols: tuple[str, ...],
    post_budget: int,
    author_col: str = "author",
    split_col: str = "split",
    target_cols: tuple[str, ...] = TARGET_COLUMNS,
) -> AuthorSetArrays:
    """Create padded unordered post sets for author-level attention models."""

    validate_required_columns(posts, [author_col, split_col, *feature_cols, *target_cols])
    authors = []
    splits = []
    targets = []
    features = []
    masks = []
    for author, group in posts.sort_values([author_col]).groupby(author_col, sort=True):
        group = group.head(post_budget)
        values = group[list(feature_cols)].to_numpy(dtype=np.float32)
        mask = np.zeros(post_budget, dtype=bool)
        padded = np.zeros((post_budget, len(feature_cols)), dtype=np.float32)
        n_take = min(len(values), post_budget)
        padded[:n_take] = values[:n_take]
        mask[:n_take] = True
        authors.append(author)
        splits.append(group[split_col].iloc[0])
        targets.append(group[list(target_cols)].iloc[0].to_numpy(dtype=np.float32))
        features.append(padded)
        masks.append(mask)
    return AuthorSetArrays(
        author=np.asarray(authors),
        split=np.asarray(splits),
        post_features=np.asarray(features, dtype=np.float32),
        post_mask=np.asarray(masks, dtype=bool),
        targets=np.asarray(targets, dtype=np.float32),
    )


def make_author_set_dataset(arrays: AuthorSetArrays, split_name: str):
    torch = optional_torch()
    if torch is None:
        raise RuntimeError("PyTorch is required for author set datasets")
    idx = np.flatnonzero(arrays.split == split_name)

    class AuthorSetDataset(torch.utils.data.Dataset):
        def __len__(self):
            return len(idx)

        def __getitem__(self, item):
            source_idx = idx[item]
            return {
                "features": torch.tensor(arrays.post_features[source_idx], dtype=torch.float32),
                "mask": torch.tensor(arrays.post_mask[source_idx], dtype=torch.bool),
                "targets": torch.tensor(arrays.targets[source_idx], dtype=torch.float32),
                "author_index": torch.tensor(source_idx, dtype=torch.long),
            }

    return AuthorSetDataset()


def train_set_attention_model(
    arrays: AuthorSetArrays,
    *,
    model_id: str,
    post_budget: int,
    epochs: int = 5,
    batch_size: int = 64,
    lr: float = 1e-3,
    patience: int = 2,
    device: str | None = None,
):
    """Train a small order-agnostic set/attention author model."""

    torch = optional_torch()
    if torch is None:
        raise RuntimeError("PyTorch is required for set/attention training")
    device = device or resolve_device(prefer_mps=True, allow_cpu=True)
    model = build_set_attention_author_model(
        SetAttentionAuthorConfig(input_dim=arrays.post_features.shape[-1])
    ).to(device)
    train_ds = make_author_set_dataset(arrays, "train")
    val_ds = make_author_set_dataset(arrays, "val")
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_ds, batch_size=batch_size)
    pos = arrays.targets[arrays.split == "train"].mean(axis=0)
    pos_weight = np.sqrt((1.0 - pos) / np.clip(pos, 1e-6, None)).astype(np.float32)
    criterion = torch.nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor(pos_weight, dtype=torch.float32, device=device)
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    history = []
    best_state = None
    best_val = float("inf")
    bad_epochs = 0
    for epoch in range(1, epochs + 1):
        model.train()
        train_losses = []
        for batch in train_loader:
            optimizer.zero_grad(set_to_none=True)
            logits = model(batch["features"].to(device), batch["mask"].to(device))
            loss = criterion(logits, batch["targets"].to(device))
            loss.backward()
            optimizer.step()
            train_losses.append(float(loss.detach().cpu()))
        model.eval()
        val_losses = []
        with torch.no_grad():
            for batch in val_loader:
                logits = model(batch["features"].to(device), batch["mask"].to(device))
                loss = criterion(logits, batch["targets"].to(device))
                val_losses.append(float(loss.detach().cpu()))
        row = {
            "model_id": model_id,
            "post_budget": post_budget,
            "epoch": epoch,
            "train_loss": float(np.mean(train_losses)),
            "val_loss": float(np.mean(val_losses)),
        }
        history.append(row)
        if row["val_loss"] < best_val:
            best_val = row["val_loss"]
            bad_epochs = 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, pd.DataFrame(history)


def predict_set_attention_scores(
    model,
    arrays: AuthorSetArrays,
    *,
    model_id: str,
    batch_size: int = 128,
    device: str | None = None,
) -> pd.DataFrame:
    torch = optional_torch()
    if torch is None:
        raise RuntimeError("PyTorch is required for set/attention inference")
    device = device or resolve_device(prefer_mps=True, allow_cpu=True)
    model = model.to(device)
    model.eval()
    score_cols = make_score_columns(model_id)
    rows = []
    for split_name in ("val", "test"):
        dataset = make_author_set_dataset(arrays, split_name)
        loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size)
        with torch.no_grad():
            for batch in loader:
                logits = model(batch["features"].to(device), batch["mask"].to(device))
                probs = torch.sigmoid(logits).detach().cpu().numpy()
                source_idx = batch["author_index"].detach().cpu().numpy()
                block = pd.DataFrame(probs, columns=score_cols)
                block["author"] = arrays.author[source_idx]
                block["split"] = arrays.split[source_idx]
                for target_idx, target in enumerate(TARGET_COLUMNS):
                    block[target] = arrays.targets[source_idx, target_idx].astype(int)
                rows.append(block)
    return pd.concat(rows, ignore_index=True)
