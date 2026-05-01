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


def count_parameters(model) -> int:
    return int(sum(parameter.numel() for parameter in model.parameters()))
