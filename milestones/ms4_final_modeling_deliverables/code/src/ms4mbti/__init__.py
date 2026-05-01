"""MS4 code helpers for Project 66.

The package is intentionally importable without KaggleHub, Transformers, or
Hugging Face Datasets installed. Full data loading and training paths raise
clear errors only when those optional dependencies are used.
"""

from .config import (
    DIMENSIONS,
    DIMENSION_SPECS,
    EMOTION_LABELS,
    TARGET_COLUMNS,
    PROJECT_NUMBER,
    RunConfig,
)
from .text import BasicVocab, build_basic_vocab, encode_texts

__all__ = [
    "BasicVocab",
    "DIMENSIONS",
    "DIMENSION_SPECS",
    "EMOTION_LABELS",
    "PROJECT_NUMBER",
    "TARGET_COLUMNS",
    "RunConfig",
    "build_basic_vocab",
    "encode_texts",
]
