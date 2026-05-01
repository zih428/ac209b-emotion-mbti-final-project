"""Text tokenization helpers for Stage 2 GRU models."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Sequence

import numpy as np

from .preprocessing import normalize_text


PAD_TOKEN = "[PAD]"
UNK_TOKEN = "[UNK]"


@dataclass(frozen=True)
class BasicVocab:
    token_to_id: dict[str, int]
    max_size: int
    min_freq: int

    @property
    def pad_id(self) -> int:
        return self.token_to_id[PAD_TOKEN]

    @property
    def unk_id(self) -> int:
        return self.token_to_id[UNK_TOKEN]

    @property
    def size(self) -> int:
        return len(self.token_to_id)


def basic_tokenize(text: object) -> list[str]:
    return normalize_text(text).lower().split()


def build_basic_vocab(
    texts: Sequence[str],
    *,
    max_size: int = 30000,
    min_freq: int = 2,
) -> BasicVocab:
    counts: Counter[str] = Counter()
    for text in texts:
        counts.update(basic_tokenize(text))

    token_to_id = {PAD_TOKEN: 0, UNK_TOKEN: 1}
    for token, freq in counts.most_common(max(0, max_size - 2)):
        if freq < min_freq:
            continue
        token_to_id[token] = len(token_to_id)
    return BasicVocab(token_to_id=token_to_id, max_size=max_size, min_freq=min_freq)


def encode_with_basic_vocab(
    texts: Sequence[str],
    vocab: BasicVocab,
    *,
    max_length: int,
) -> np.ndarray:
    encoded = np.full((len(texts), max_length), vocab.pad_id, dtype=np.int64)
    for row_idx, text in enumerate(texts):
        ids = [vocab.token_to_id.get(token, vocab.unk_id) for token in basic_tokenize(text)]
        ids = ids[:max_length]
        encoded[row_idx, : len(ids)] = ids
    return encoded


def encode_texts(
    texts: Sequence[str],
    *,
    max_length: int,
    tokenizer=None,
    vocab: BasicVocab | None = None,
) -> np.ndarray:
    """Encode texts with a Hugging Face tokenizer or a BasicVocab fallback."""

    text_list = list(texts)
    if tokenizer is not None:
        batch = tokenizer(
            text_list,
            truncation=True,
            padding="max_length",
            max_length=max_length,
            return_tensors=None,
        )
        return np.asarray(batch["input_ids"], dtype=np.int64)
    if vocab is None:
        raise ValueError("Either tokenizer or vocab must be provided")
    return encode_with_basic_vocab(text_list, vocab, max_length=max_length)
