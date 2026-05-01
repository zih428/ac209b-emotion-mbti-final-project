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


def test_set_attention_baseline_deltas_use_test_authors(
    tmp_path: Path,
    monkeypatch,
) -> None:
    module = _load_script_module()
    targets = module.TARGETS
    authors = pd.DataFrame(
        {
            "author": ["train0", "test0", "test1"],
            "split": ["train", "test", "test"],
            **{target: [0, 0, 1] for target in targets},
        }
    )

    baseline = authors.copy()
    set_attention = authors.copy()
    for target in targets:
        baseline[f"score_base_{target}"] = [0.0, 0.9, 0.1]
        set_attention[f"score_set_attention_text_p200_{target}"] = [0.0, 0.1, 0.9]

    baseline_path = tmp_path / "author_scores_baseline.csv"
    baseline_thresholds_path = tmp_path / "thresholds_baseline.csv"
    set_path = tmp_path / "author_scores_set_attention_text_p200.csv"
    set_thresholds_path = tmp_path / "thresholds_set_attention_text_p200.csv"
    baseline.to_csv(baseline_path, index=False)
    set_attention.to_csv(set_path, index=False)
    for path in [baseline_thresholds_path, set_thresholds_path]:
        pd.DataFrame({"target": targets, "threshold": [0.5] * len(targets)}).to_csv(
            path,
            index=False,
        )

    baseline_ids = [
        "linear_tfidf_author",
        "stage2_text_emotion_gru",
        "stage2_text_gru_sqrt",
    ]
    monkeypatch.setattr(module, "SET_ATTENTION_CONFUSION_AUTHOR_SCORES", set_path)
    monkeypatch.setattr(module, "SET_ATTENTION_CONFUSION_THRESHOLDS", set_thresholds_path)
    monkeypatch.setattr(
        module,
        "AUTHOR_SCORE_SOURCES",
        {
            model_id: {
                "path": baseline_path,
                "thresholds": baseline_thresholds_path,
                "score_prefix": "score_base",
            }
            for model_id in baseline_ids
        },
    )
    monkeypatch.setattr(
        module,
        "DISPLAY_NAMES",
        {model_id: model_id for model_id in baseline_ids},
    )

    author_inputs = {
        model_id: {
            "author_scores": pd.read_csv(baseline_path),
            "thresholds": pd.read_csv(baseline_thresholds_path),
            "score_prefix": "score_base",
        }
        for model_id in baseline_ids
    }

    deltas = module.make_set_attention_baseline_deltas(author_inputs, n_bootstrap=10)

    means = deltas.loc[deltas["target"] == "mean"]
    assert means["n_authors"].eq(2).all()
    assert means["point_estimate"].eq(1.0).all()
