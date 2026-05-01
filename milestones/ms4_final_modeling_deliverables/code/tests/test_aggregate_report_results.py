from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pandas as pd


CODE_DIR = Path(__file__).resolve().parents[1]
SCRIPT_PATH = CODE_DIR / "scripts" / "aggregate_report_results.py"


def _load_script_module():
    spec = importlib.util.spec_from_file_location("aggregate_report_results", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_summary_sorts_by_test_balanced_accuracy() -> None:
    module = _load_script_module()
    metrics = pd.DataFrame(
        {
            "model_id": ["a", "b", "a", "b"],
            "model_name": ["A", "B", "A", "B"],
            "split": ["test", "test", "val", "val"],
            "balanced_accuracy": [0.6, 0.7, 0.9, 0.1],
            "f1": [0.1, 0.2, 0.3, 0.4],
            "minority_recall": [0.1, 0.2, 0.3, 0.4],
            "roc_auc": [0.1, 0.2, 0.3, 0.4],
            "average_precision": [0.1, 0.2, 0.3, 0.4],
        }
    )

    summary = module.make_summary(metrics)

    assert summary["model_id"].tolist() == ["b", "a"]
    assert "| B | 0.7000 |" in module.markdown_summary_table(summary)
