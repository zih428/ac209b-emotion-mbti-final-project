#!/usr/bin/env python3
"""Run the MS4 non-training smoke checks."""

from __future__ import annotations

import sys
from pathlib import Path


CODE_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = CODE_DIR / "src"
sys.path.insert(0, str(SRC_DIR))

from ms4mbti.cli import smoke_main


if __name__ == "__main__":
    smoke_main()
