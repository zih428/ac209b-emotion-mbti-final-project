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


def test_set_attention_display_name_prefers_longest_prefix() -> None:
    module = _load_script_module()

    name = module.set_attention_display_name("set_attention_text_real_emotion_p200")

    assert name == "Set Attention Text + Real Emotion p=200"


def test_transformer_delta_table_uses_test_split_only(tmp_path: Path) -> None:
    module = _load_script_module()
    targets = module.TARGETS
    authors = pd.DataFrame(
        {
            "author": ["train0", "train1", "test0", "test1"],
            "split": ["train", "train", "test", "test"],
            **{target: [0, 1, 0, 1] for target in targets},
        }
    )

    baseline = authors.copy()
    real = authors.copy()
    shuffled = authors.copy()
    for target in targets:
        baseline[f"score_base_{target}"] = [0.9, 0.1, 0.1, 0.9]
        real[f"score_real_{target}"] = [0.1, 0.9, 0.9, 0.1]
        shuffled[f"score_shuffled_{target}"] = [0.1, 0.9, 0.1, 0.9]

    baseline.to_csv(tmp_path / "author_scores_base.csv", index=False)
    real.to_csv(tmp_path / "author_scores_real.csv", index=False)
    shuffled.to_csv(tmp_path / "author_scores_shuffled.csv", index=False)
    for model_id in ["base", "real", "shuffled"]:
        pd.DataFrame(
            {
                "model_id": model_id,
                "target": targets,
                "threshold": [0.5] * len(targets),
            }
        ).to_csv(tmp_path / f"thresholds_{model_id}.csv", index=False)

    deltas = module.make_transformer_delta_table(
        run_dir=tmp_path,
        family="Synthetic",
        baseline_model="base",
        real_model="real",
        shuffled_model="shuffled",
        n_bootstrap=10,
    )

    real_target = deltas.loc[
        (deltas["comparison"] == "Synthetic: real emotion minus text")
        & (deltas["target"] == targets[0])
    ].iloc[0]
    assert real_target["n_authors"] == 2
    assert real_target["point_estimate"] == -1.0
