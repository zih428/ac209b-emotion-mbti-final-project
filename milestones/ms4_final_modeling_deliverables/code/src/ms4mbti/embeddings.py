"""Frozen transformer post embedding cache utilities."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from .cache import metadata_fingerprint, read_manifest, transformer_embedding_metadata, write_manifest
from .config import DEFAULT_FROZEN_EMBEDDING_MODEL, RunConfig
from .preprocessing import normalize_text, validate_required_columns
from .training import optional_torch, resolve_device


EMBEDDING_PREFIX = "emb_"


@dataclass(frozen=True)
class EmbeddingCacheSpec:
    model_id: str = DEFAULT_FROZEN_EMBEDDING_MODEL
    tokenizer_max_length: int = 256
    batch_size: int = 256
    shard_size: int = 50000
    text_col: str = "text_masked"
    row_id_col: str = "row_index"
    author_col: str = "author"
    split_col: str = "split"
    backend: str = "transformers"


def embedding_columns(frame: pd.DataFrame) -> tuple[str, ...]:
    return tuple(col for col in frame.columns if col.startswith(EMBEDDING_PREFIX))


def embedding_frame_from_array(
    metadata: pd.DataFrame,
    embeddings: np.ndarray,
    *,
    row_id_col: str = "row_index",
    author_col: str = "author",
    split_col: str = "split",
) -> pd.DataFrame:
    """Attach dense embedding columns to row/author/split metadata."""

    validate_required_columns(metadata, [row_id_col, author_col, split_col])
    if len(metadata) != len(embeddings):
        raise ValueError("metadata and embeddings must have the same row count")
    emb_df = pd.DataFrame(
        embeddings.astype(np.float32),
        columns=[f"{EMBEDDING_PREFIX}{idx:03d}" for idx in range(embeddings.shape[1])],
    )
    return pd.concat(
        [metadata[[row_id_col, author_col, split_col]].reset_index(drop=True), emb_df],
        axis=1,
    )


def preprocessing_fingerprint(posts: pd.DataFrame, *, columns: Iterable[str]) -> str:
    """Hash stable row identifiers and key preprocessing columns for cache provenance."""

    present = [col for col in columns if col in posts.columns]
    if not present:
        raise ValueError("At least one fingerprint column must be present")
    digest = hashlib.sha256()
    digest.update(str(len(posts)).encode("utf-8"))
    sample = posts[present].head(1000).tail(1000)
    digest.update(pd.util.hash_pandas_object(sample, index=True).to_numpy().tobytes())
    return digest.hexdigest()[:16]


def deterministic_hash_embeddings(
    texts: Iterable[object],
    *,
    dim: int = 64,
    seed: int = 209066,
) -> np.ndarray:
    """Small deterministic embedding backend for tests and smoke runs.

    This is not a scientific replacement for transformer embeddings. It exists
    so schema, sharding, and downstream model code can be tested without network
    access or a local Hugging Face model cache.
    """

    rows = []
    for text in texts:
        value = normalize_text(text)
        vector = np.zeros(dim, dtype=np.float32)
        for token in value.split():
            digest = hashlib.blake2b(
                f"{seed}:{token}".encode("utf-8"),
                digest_size=8,
            ).digest()
            bucket = int.from_bytes(digest[:4], "little") % dim
            sign = 1.0 if int.from_bytes(digest[4:], "little") % 2 == 0 else -1.0
            vector[bucket] += sign
        norm = np.linalg.norm(vector)
        if norm:
            vector /= norm
        rows.append(vector)
    return np.vstack(rows) if rows else np.zeros((0, dim), dtype=np.float32)


def mean_pool_transformer_output(model_output, attention_mask):
    """Mean-pool token embeddings while respecting the attention mask."""

    torch = optional_torch()
    if torch is None:
        raise RuntimeError("PyTorch is required for transformer pooling")
    token_embeddings = model_output.last_hidden_state
    mask = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    summed = (token_embeddings * mask).sum(dim=1)
    counts = mask.sum(dim=1).clamp(min=1e-9)
    return summed / counts


def load_transformer_encoder(model_id: str, *, local_files_only: bool = False):
    """Load a Hugging Face encoder/tokenizer pair lazily."""

    try:
        from transformers import AutoModel, AutoTokenizer
    except ImportError as exc:
        raise RuntimeError("transformers is required for frozen transformer embeddings") from exc

    tokenizer = AutoTokenizer.from_pretrained(model_id, local_files_only=local_files_only)
    model = AutoModel.from_pretrained(model_id, local_files_only=local_files_only)
    return tokenizer, model


def encode_texts_with_transformer(
    texts: list[str],
    *,
    model_id: str,
    tokenizer_max_length: int,
    batch_size: int,
    device: str | None = None,
    local_files_only: bool = False,
) -> np.ndarray:
    """Encode texts with a frozen transformer encoder and mean pooling."""

    torch = optional_torch()
    if torch is None:
        raise RuntimeError("PyTorch is required for transformer embeddings")
    tokenizer, model = load_transformer_encoder(model_id, local_files_only=local_files_only)
    device = device or resolve_device(prefer_mps=True, allow_cpu=True)
    model = model.to(device)
    model.eval()

    blocks = []
    with torch.no_grad():
        for start in range(0, len(texts), batch_size):
            batch_texts = texts[start : start + batch_size]
            encoded = tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=tokenizer_max_length,
                return_tensors="pt",
            )
            encoded = {key: value.to(device) for key, value in encoded.items()}
            output = model(**encoded)
            pooled = mean_pool_transformer_output(output, encoded["attention_mask"])
            blocks.append(pooled.detach().cpu().numpy().astype(np.float32))
    if not blocks:
        return np.zeros((0, 0), dtype=np.float32)
    return np.vstack(blocks)


def write_embedding_shards(
    posts: pd.DataFrame,
    *,
    output_dir: Path,
    spec: EmbeddingCacheSpec,
    config: RunConfig | None = None,
    max_rows: int | None = None,
    local_files_only: bool = False,
    deterministic_dim: int = 64,
) -> dict:
    """Generate resumable post-embedding parquet shards plus a manifest."""

    config = config or RunConfig()
    validate_required_columns(posts, [spec.text_col, spec.author_col, spec.split_col])
    output_dir.mkdir(parents=True, exist_ok=True)

    work = posts.copy()
    if spec.row_id_col not in work.columns:
        work = work.reset_index().rename(columns={"index": spec.row_id_col})
    if max_rows is not None:
        work = work.head(max_rows).copy()

    fingerprint = preprocessing_fingerprint(
        work,
        columns=[spec.row_id_col, spec.author_col, spec.split_col, spec.text_col],
    )
    shard_paths = []
    embedding_dim = None
    for shard_id, start in enumerate(range(0, len(work), spec.shard_size)):
        shard = work.iloc[start : start + spec.shard_size].copy()
        path = output_dir / f"embeddings_{shard_id:05d}.parquet"
        if path.exists():
            shard_paths.append(path)
            if embedding_dim is None:
                existing_cols = embedding_columns(pd.read_parquet(path).head(1))
                embedding_dim = len(existing_cols)
            continue

        texts = shard[spec.text_col].map(normalize_text).tolist()
        if spec.backend == "deterministic_hash":
            vectors = deterministic_hash_embeddings(texts, dim=deterministic_dim, seed=config.seed)
        elif spec.backend == "transformers":
            vectors = encode_texts_with_transformer(
                texts,
                model_id=spec.model_id,
                tokenizer_max_length=spec.tokenizer_max_length,
                batch_size=spec.batch_size,
                local_files_only=local_files_only,
            )
        else:
            raise ValueError("backend must be `transformers` or `deterministic_hash`")

        embedding_dim = int(vectors.shape[1])
        out = embedding_frame_from_array(
            shard,
            vectors,
            row_id_col=spec.row_id_col,
            author_col=spec.author_col,
            split_col=spec.split_col,
        )
        out["shard_id"] = shard_id
        out.to_parquet(path, index=False)
        shard_paths.append(path)

    if embedding_dim is None:
        embedding_dim = 0
    metadata = transformer_embedding_metadata(
        model_id=spec.model_id,
        tokenizer_max_length=spec.tokenizer_max_length,
        preprocessing_fingerprint=fingerprint,
        input_rows=len(work),
        embedding_dim=embedding_dim,
        shard_size=spec.shard_size,
        backend=spec.backend,
        extra={
            "row_id_col": spec.row_id_col,
            "author_col": spec.author_col,
            "split_col": spec.split_col,
            "text_col": spec.text_col,
            "shards": [path.name for path in shard_paths],
            **config.metadata(),
        },
    )
    return write_manifest(output_dir / "manifest.json", metadata)


def read_embedding_cache(cache_dir: Path) -> tuple[pd.DataFrame, dict]:
    """Read all embedding shards listed in the cache manifest."""

    manifest = read_manifest(cache_dir / "manifest.json")
    shards = manifest.get("shards", [])
    if not shards:
        raise ValueError(f"No embedding shards listed in {cache_dir / 'manifest.json'}")
    frames = [pd.read_parquet(cache_dir / shard) for shard in shards]
    frame = pd.concat(frames, ignore_index=True)
    expected_rows = int(manifest.get("input_rows", len(frame)))
    if len(frame) != expected_rows:
        raise ValueError(f"Embedding cache row mismatch: {len(frame)} != {expected_rows}")
    if len(embedding_columns(frame)) != int(manifest.get("embedding_dim", 0)):
        raise ValueError("Embedding cache dimension does not match manifest")
    return frame, manifest


def embedding_cache_fingerprint(cache_dir: Path) -> str:
    return str(read_manifest(cache_dir / "manifest.json").get("fingerprint", ""))
