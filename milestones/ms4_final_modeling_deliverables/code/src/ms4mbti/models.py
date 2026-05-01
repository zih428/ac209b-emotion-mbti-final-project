"""Stage 2 GRU model definitions."""

from __future__ import annotations

from dataclasses import dataclass

from .config import TARGET_COLUMNS
from .training import optional_torch


@dataclass(frozen=True)
class Stage2ModelConfig:
    vocab_size: int = 30000
    embedding_dim: int = 128
    hidden_dim: int = 128
    dense_dim: int = 64
    output_dim: int = len(TARGET_COLUMNS)
    emotion_dim: int = 0
    padding_idx: int = 0
    dropout: float = 0.20


@dataclass(frozen=True)
class AuthorMLPConfig:
    input_dim: int
    hidden_dim: int = 128
    output_dim: int = len(TARGET_COLUMNS)
    dropout: float = 0.20


@dataclass(frozen=True)
class SetAttentionAuthorConfig:
    input_dim: int
    model_dim: int = 128
    num_heads: int = 4
    output_dim: int = len(TARGET_COLUMNS)
    dropout: float = 0.20


def build_stage2_gru(config: Stage2ModelConfig):
    """Build a GRU MBTI head. Import torch lazily for lightweight smoke paths."""

    torch = optional_torch()
    if torch is None:
        raise RuntimeError("PyTorch is required to build the Stage 2 GRU model.")

    nn = torch.nn

    class TextGruMbtiModel(nn.Module):
        def __init__(self, cfg: Stage2ModelConfig):
            super().__init__()
            self.cfg = cfg
            self.embedding = nn.Embedding(
                cfg.vocab_size, cfg.embedding_dim, padding_idx=cfg.padding_idx
            )
            self.gru = nn.GRU(
                input_size=cfg.embedding_dim,
                hidden_size=cfg.hidden_dim,
                batch_first=True,
            )
            self.dropout = nn.Dropout(cfg.dropout)
            self.head = nn.Sequential(
                nn.Linear(cfg.hidden_dim + cfg.emotion_dim, cfg.dense_dim),
                nn.ReLU(),
                nn.Dropout(cfg.dropout),
                nn.Linear(cfg.dense_dim, cfg.output_dim),
            )

        def forward(self, input_ids, emotion_features=None):
            embedded = self.embedding(input_ids)
            _, hidden = self.gru(embedded)
            pooled = hidden[-1]
            if self.cfg.emotion_dim:
                if emotion_features is None:
                    raise ValueError("emotion_features are required for this model")
                pooled = torch.cat([pooled, emotion_features], dim=1)
            return self.head(self.dropout(pooled))

    return TextGruMbtiModel(config)


def build_author_mlp(config: AuthorMLPConfig):
    """Build a reusable author-level MLP for pooled feature tables."""

    torch = optional_torch()
    if torch is None:
        raise RuntimeError("PyTorch is required to build the author MLP.")
    nn = torch.nn
    return nn.Sequential(
        nn.Linear(config.input_dim, config.hidden_dim),
        nn.ReLU(),
        nn.Dropout(config.dropout),
        nn.Linear(config.hidden_dim, config.hidden_dim),
        nn.ReLU(),
        nn.Dropout(config.dropout),
        nn.Linear(config.hidden_dim, config.output_dim),
    )


def build_set_attention_author_model(config: SetAttentionAuthorConfig):
    """Build an order-agnostic attention model over an author's post set.

    The module intentionally has no positional or temporal encoding. Self
    attention is permutation equivariant, and masked mean pooling makes the
    final author representation permutation invariant.
    """

    torch = optional_torch()
    if torch is None:
        raise RuntimeError("PyTorch is required to build the set/attention model.")
    nn = torch.nn

    class SetAttentionAuthorModel(nn.Module):
        uses_positional_encoding = False

        def __init__(self, cfg: SetAttentionAuthorConfig):
            super().__init__()
            self.cfg = cfg
            self.input_projection = nn.Linear(cfg.input_dim, cfg.model_dim)
            self.attention = nn.MultiheadAttention(
                embed_dim=cfg.model_dim,
                num_heads=cfg.num_heads,
                dropout=cfg.dropout,
                batch_first=True,
            )
            self.norm = nn.LayerNorm(cfg.model_dim)
            self.dropout = nn.Dropout(cfg.dropout)
            self.head = nn.Sequential(
                nn.Linear(cfg.model_dim, cfg.model_dim),
                nn.ReLU(),
                nn.Dropout(cfg.dropout),
                nn.Linear(cfg.model_dim, cfg.output_dim),
            )

        def forward(self, post_features, post_mask=None):
            hidden = self.input_projection(post_features)
            key_padding_mask = None
            if post_mask is not None:
                key_padding_mask = ~post_mask.bool()
            attended, _weights = self.attention(
                hidden,
                hidden,
                hidden,
                key_padding_mask=key_padding_mask,
                need_weights=False,
            )
            hidden = self.norm(hidden + self.dropout(attended))
            if post_mask is None:
                pooled = hidden.mean(dim=1)
            else:
                mask = post_mask.unsqueeze(-1).to(hidden.dtype)
                pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)
            return self.head(pooled)

    return SetAttentionAuthorModel(config)


def count_parameters(model) -> int:
    return int(sum(parameter.numel() for parameter in model.parameters()))
