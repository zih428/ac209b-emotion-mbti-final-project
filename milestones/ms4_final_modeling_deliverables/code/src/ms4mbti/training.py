"""Training utilities and device guards.

This module does not start training. It only centralizes checks that training
code can call before expensive work begins.
"""

from __future__ import annotations

import random
from typing import Any

import numpy as np


def optional_torch():
    try:
        import torch
    except ImportError:
        return None
    return torch


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch = optional_torch()
    if torch is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)


def torch_environment() -> dict[str, Any]:
    torch = optional_torch()
    if torch is None:
        return {"torch_available": False}
    return {
        "torch_available": True,
        "torch_version": torch.__version__,
        "cuda_available": bool(torch.cuda.is_available()),
        "mps_available": bool(
            hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
        ),
    }


def resolve_device(*, prefer_mps: bool = True, allow_cpu: bool = True) -> str:
    torch = optional_torch()
    if torch is None:
        if allow_cpu:
            return "cpu"
        raise RuntimeError("PyTorch is not installed")
    if prefer_mps and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    if allow_cpu:
        return "cpu"
    raise RuntimeError("No acceptable torch device is available")


def require_mps_for_full_training(run_full_training: bool) -> None:
    """Fail only when the user explicitly asks for full training without MPS."""

    if not run_full_training:
        return
    torch = optional_torch()
    if torch is None:
        raise RuntimeError("RUN_FULL_TRAINING=True requires PyTorch.")
    if not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
        raise RuntimeError(
            "RUN_FULL_TRAINING=True requires MPS for this project plan. "
            "Fix the Python/torch/Jupyter environment or set RUN_FULL_TRAINING=False."
        )
