"""Stage 2 GRU dataset, training, and inference utilities."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .config import TARGET_COLUMNS
from .evaluation import sigmoid
from .models import Stage2ModelConfig, build_stage2_gru
from .progress import ProgressLogger
from .text import BasicVocab, build_basic_vocab, encode_texts
from .training import optional_torch, require_mps_for_full_training, resolve_device


@dataclass(frozen=True)
class Stage2Arrays:
    input_ids: np.ndarray
    targets: np.ndarray
    emotion_features: np.ndarray | None
    sample_weights: np.ndarray | None
    row_index: np.ndarray


def make_stage2_arrays(
    posts: pd.DataFrame,
    *,
    text_col: str = "text_masked",
    target_cols: tuple[str, ...] = TARGET_COLUMNS,
    emotion_cols: tuple[str, ...] = (),
    sample_weight_col: str | None = None,
    max_length: int = 128,
    tokenizer=None,
    vocab: BasicVocab | None = None,
) -> Stage2Arrays:
    """Convert a post dataframe into arrays for the GRU dataset."""

    if vocab is None and tokenizer is None:
        vocab = build_basic_vocab(posts[text_col].tolist(), max_size=30000, min_freq=2)
    input_ids = encode_texts(
        posts[text_col].tolist(),
        max_length=max_length,
        tokenizer=tokenizer,
        vocab=vocab,
    )
    targets = posts[list(target_cols)].to_numpy(dtype=np.float32)
    emotion_features = (
        posts[list(emotion_cols)].to_numpy(dtype=np.float32) if emotion_cols else None
    )
    sample_weights = (
        posts[sample_weight_col].to_numpy(dtype=np.float32)
        if sample_weight_col is not None
        else None
    )
    return Stage2Arrays(
        input_ids=input_ids,
        targets=targets,
        emotion_features=emotion_features,
        sample_weights=sample_weights,
        row_index=posts.index.to_numpy(),
    )


def prepare_stage2_split_arrays(
    posts: pd.DataFrame,
    *,
    split_col: str = "split",
    text_col: str = "text_masked",
    target_cols: tuple[str, ...] = TARGET_COLUMNS,
    emotion_cols: tuple[str, ...] = (),
    sample_weight_col: str | None = None,
    max_length: int = 128,
    tokenizer=None,
    vocab: BasicVocab | None = None,
    vocab_max_size: int = 30000,
    vocab_min_freq: int = 2,
) -> tuple[dict[str, Stage2Arrays], BasicVocab | None]:
    """Prepare arrays for all splits using one train-fitted vocabulary.

    If a Hugging Face tokenizer is supplied, no BasicVocab is built. Otherwise
    the fallback vocabulary is fit on training text only and reused unchanged
    for validation/test encoding.
    """

    if split_col not in posts.columns:
        raise ValueError(f"posts must contain split column {split_col!r}")
    if tokenizer is None and vocab is None:
        train_text = posts.loc[posts[split_col] == "train", text_col].tolist()
        if not train_text:
            raise ValueError("Cannot build Stage 2 vocabulary without training rows")
        vocab = build_basic_vocab(
            train_text,
            max_size=vocab_max_size,
            min_freq=vocab_min_freq,
        )

    arrays_by_split: dict[str, Stage2Arrays] = {}
    for split_name, group in posts.groupby(split_col):
        arrays_by_split[str(split_name)] = make_stage2_arrays(
            group,
            text_col=text_col,
            target_cols=target_cols,
            emotion_cols=emotion_cols,
            sample_weight_col=sample_weight_col,
            max_length=max_length,
            tokenizer=tokenizer,
            vocab=vocab,
        )
    return arrays_by_split, vocab


def make_torch_dataset(arrays: Stage2Arrays):
    torch = optional_torch()
    if torch is None:
        raise RuntimeError("PyTorch is required for Stage 2 datasets.")

    class Stage2Dataset(torch.utils.data.Dataset):
        def __init__(self, values: Stage2Arrays):
            self.values = values

        def __len__(self):
            return len(self.values.input_ids)

        def __getitem__(self, idx):
            item = {
                "input_ids": torch.tensor(self.values.input_ids[idx], dtype=torch.long),
                "targets": torch.tensor(self.values.targets[idx], dtype=torch.float32),
                "row_index": torch.tensor(self.values.row_index[idx], dtype=torch.long),
            }
            if self.values.emotion_features is not None:
                item["emotion_features"] = torch.tensor(
                    self.values.emotion_features[idx], dtype=torch.float32
                )
            if self.values.sample_weights is not None:
                item["sample_weights"] = torch.tensor(
                    self.values.sample_weights[idx], dtype=torch.float32
                )
            return item

    return Stage2Dataset(arrays)


def build_stage2_from_arrays(
    arrays: Stage2Arrays,
    *,
    vocab_size: int,
    hidden_dim: int = 128,
    dropout: float = 0.20,
):
    emotion_dim = 0 if arrays.emotion_features is None else arrays.emotion_features.shape[1]
    return build_stage2_gru(
        Stage2ModelConfig(
            vocab_size=vocab_size,
            hidden_dim=hidden_dim,
            emotion_dim=emotion_dim,
            output_dim=arrays.targets.shape[1],
            dropout=dropout,
        )
    )


def _batch_to_device(batch: dict, device: str) -> dict:
    return {
        key: value.to(device) if hasattr(value, "to") else value
        for key, value in batch.items()
    }


def train_stage2_model(
    model,
    train_dataset,
    val_dataset,
    *,
    pos_weight: np.ndarray,
    run_full_training: bool,
    epochs: int = 10,
    batch_size: int = 512,
    lr: float = 1e-3,
    patience: int = 2,
    device: str | None = None,
    checkpoint_path: Path | None = None,
    logger: ProgressLogger | None = None,
) -> pd.DataFrame:
    """Train a Stage 2 GRU model after an explicit full-training opt-in."""

    require_mps_for_full_training(run_full_training)
    torch = optional_torch()
    if torch is None:
        raise RuntimeError("PyTorch is required for Stage 2 training.")

    device = device or resolve_device(prefer_mps=True, allow_cpu=False)
    logger = logger or ProgressLogger("stage2", verbose=True)
    model = model.to(device)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True
    )
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size)
    pos_weight_tensor = torch.tensor(pos_weight, dtype=torch.float32, device=device)
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor, reduction="none")
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    history = []
    best_val = float("inf")
    bad_epochs = 0
    best_state = None
    for epoch in range(1, epochs + 1):
        model.train()
        train_losses = []
        for batch in train_loader:
            batch = _batch_to_device(batch, device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(batch["input_ids"], batch.get("emotion_features"))
            loss_matrix = criterion(logits, batch["targets"])
            if "sample_weights" in batch:
                loss_matrix = loss_matrix * batch["sample_weights"].unsqueeze(1)
            loss = loss_matrix.mean()
            loss.backward()
            optimizer.step()
            train_losses.append(float(loss.detach().cpu()))

        model.eval()
        val_losses = []
        with torch.no_grad():
            for batch in val_loader:
                batch = _batch_to_device(batch, device)
                logits = model(batch["input_ids"], batch.get("emotion_features"))
                val_loss = criterion(logits, batch["targets"]).mean()
                val_losses.append(float(val_loss.detach().cpu()))

        row = {
            "epoch": epoch,
            "train_loss": float(np.mean(train_losses)),
            "val_loss": float(np.mean(val_losses)),
        }
        history.append(row)
        logger.step("Stage 2 epoch complete", **row)

        if row["val_loss"] < best_val:
            best_val = row["val_loss"]
            bad_epochs = 0
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
            if checkpoint_path is not None:
                checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save(best_state, checkpoint_path)
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                logger.step("Early stopping", epoch=epoch, best_val_loss=best_val)
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return pd.DataFrame(history)


def predict_stage2_probabilities(
    model,
    dataset,
    *,
    score_cols: tuple[str, ...],
    batch_size: int = 1024,
    device: str | None = None,
    logger: ProgressLogger | None = None,
) -> pd.DataFrame:
    """Run Stage 2 inference and return row indices plus probability columns."""

    torch = optional_torch()
    if torch is None:
        raise RuntimeError("PyTorch is required for Stage 2 inference.")
    device = device or resolve_device(prefer_mps=True, allow_cpu=True)
    logger = logger or ProgressLogger("stage2", verbose=False)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size)
    model = model.to(device)
    model.eval()
    rows = []
    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            batch = _batch_to_device(batch, device)
            logits = model(batch["input_ids"], batch.get("emotion_features"))
            probs = sigmoid(logits.detach().cpu().numpy())
            block = pd.DataFrame(probs, columns=score_cols)
            block["row_index"] = batch["row_index"].detach().cpu().numpy()
            rows.append(block)
            logger.step("Stage 2 inference batch", batch=batch_idx, n_rows=len(block))
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(columns=score_cols)
