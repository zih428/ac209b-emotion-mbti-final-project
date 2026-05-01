"""Stage 1 emotion-model helpers.

These functions are inert unless the full training or inference path imports
Transformers and PyTorch. The default smoke path does not call them.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd

from .config import EMOTION_LABELS
from .progress import ProgressLogger
from .training import optional_torch, resolve_device


def _require_transformers():
    try:
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
    except ImportError as exc:
        raise RuntimeError(
            "Transformers is required for the Stage 1 DistilBERT emotion path. "
            "Install `transformers` or keep RUN_FULL_TRAINING=False."
        ) from exc
    return AutoModelForSequenceClassification, AutoTokenizer


def build_emotion_model(
    *,
    model_name: str = "distilbert-base-uncased",
    labels: Sequence[str] = EMOTION_LABELS,
):
    """Create an unfine-tuned sequence-classification model and tokenizer."""

    AutoModelForSequenceClassification, AutoTokenizer = _require_transformers()
    id2label = {idx: label for idx, label in enumerate(labels)}
    label2id = {label: idx for idx, label in id2label.items()}
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=len(labels),
        id2label=id2label,
        label2id=label2id,
    )
    return tokenizer, model


def predict_emotion_probabilities(
    texts: Sequence[str],
    *,
    tokenizer,
    model,
    labels: Sequence[str] = EMOTION_LABELS,
    batch_size: int = 64,
    max_length: int = 64,
    device: str | None = None,
    log_every_batches: int = 1,
    logger: ProgressLogger | None = None,
) -> pd.DataFrame:
    """Run batched emotion inference and return six probability columns."""

    torch = optional_torch()
    if torch is None:
        raise RuntimeError("PyTorch is required for emotion inference.")

    text_list = list(texts)
    device = device or resolve_device(prefer_mps=True, allow_cpu=True)
    model = model.to(device)
    model.eval()
    logger = logger or ProgressLogger("emotion", verbose=False)

    chunks = []
    n_texts = len(text_list)
    with torch.no_grad():
        for batch_idx, start in enumerate(range(0, n_texts, batch_size)):
            end = min(start + batch_size, n_texts)
            should_log = (
                batch_idx == 0
                or end == n_texts
                or (log_every_batches > 0 and batch_idx % log_every_batches == 0)
            )
            if should_log:
                logger.step(
                    "Emotion inference batch",
                    batch=batch_idx,
                    start=start,
                    end=end,
                    total=n_texts,
                )
            batch = tokenizer(
                text_list[start:end],
                truncation=True,
                padding=True,
                max_length=max_length,
                return_tensors="pt",
            )
            batch = {key: value.to(device) for key, value in batch.items()}
            logits = model(**batch).logits
            probs = torch.softmax(logits, dim=-1).detach().cpu().numpy()
            chunks.append(probs)

    array = np.vstack(chunks) if chunks else np.empty((0, len(labels)))
    return pd.DataFrame(array, columns=[f"emotion_{label}" for label in labels])


def save_emotion_probabilities(
    probabilities: pd.DataFrame,
    path: Path,
    *,
    metadata: dict,
) -> None:
    """Save probabilities with a sibling metadata JSON file."""

    from .cache import write_dataframe, write_manifest

    write_dataframe(probabilities, path)
    write_manifest(path.with_suffix(path.suffix + ".manifest.json"), metadata)
