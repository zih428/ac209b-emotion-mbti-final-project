"""Cache metadata and compact-artifact utilities."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd


COMPACT_ARTIFACT_REQUIRED_KEYS = (
    "artifact_type",
    "created_by",
    "split_id",
    "masking_status",
    "label_encoding",
    "threshold_objective",
)


def compact_artifact_contract() -> pd.DataFrame:
    """Describe the compact artifacts expected by the submitted notebook path."""

    rows = [
        {
            "artifact": "author_splits.csv",
            "level": "author",
            "required_columns": "author, split, mbti, target_E, target_S, target_T, target_J",
            "purpose": "Reconstruct split balance and join compact predictions.",
            "submit": True,
        },
        {
            "artifact": "model_author_scores_<model_id>.csv",
            "level": "author",
            "required_columns": "author, split, targets, score_<target> columns",
            "purpose": "Regenerate validation threshold tuning and test metrics without post logits.",
            "submit": True,
        },
        {
            "artifact": "metrics_<model_id>.csv",
            "level": "author/model",
            "required_columns": "model_id, split, target, balanced_accuracy, f1, minority_recall, average_precision, roc_auc",
            "purpose": "Final report tables and sanity checks.",
            "submit": True,
        },
        {
            "artifact": "thresholds_<model_id>.csv",
            "level": "model",
            "required_columns": "model_id, target, threshold, objective, validation_score",
            "purpose": "Show validation-only decision rules applied to test.",
            "submit": True,
        },
        {
            "artifact": "stage1_emotion_probs_postlevel.*",
            "level": "post",
            "required_columns": "post id/index, six emotion probabilities",
            "purpose": "Large intermediate cache for local reruns; document but do not submit by default.",
            "submit": False,
        },
        {
            "artifact": "stage2_logits_postlevel_<model_id>.*",
            "level": "post",
            "required_columns": "post id/index, four logits/probabilities",
            "purpose": "Large intermediate cache for local reruns; document but do not submit by default.",
            "submit": False,
        },
        {
            "artifact": "model_checkpoints/",
            "level": "model",
            "required_columns": "n/a",
            "purpose": "Training restart and audit only; exclude from zip unless staff requests.",
            "submit": False,
        },
    ]
    return pd.DataFrame(rows)


def base_compact_metadata(
    *,
    artifact_type: str,
    created_by: str,
    split_id: str,
    masking_status: str,
    label_encoding: dict,
    threshold_objective: str = "balanced_accuracy",
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build required compact-artifact metadata."""

    metadata = {
        "artifact_type": artifact_type,
        "created_by": created_by,
        "split_id": split_id,
        "masking_status": masking_status,
        "label_encoding": label_encoding,
        "threshold_objective": threshold_objective,
    }
    if extra:
        metadata.update(extra)
    return metadata


def stable_json_dumps(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)


def metadata_fingerprint(metadata: dict[str, Any]) -> str:
    return hashlib.sha256(stable_json_dumps(metadata).encode("utf-8")).hexdigest()[:16]


def write_manifest(path: Path, metadata: dict[str, Any]) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    manifest = dict(metadata)
    manifest["fingerprint"] = metadata_fingerprint(metadata)
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return manifest


def read_manifest(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def assert_manifest_compatible(
    manifest: dict[str, Any],
    expected_metadata: dict[str, Any],
    *,
    required_keys: tuple[str, ...],
) -> None:
    mismatches = []
    for key in required_keys:
        if manifest.get(key) != expected_metadata.get(key):
            mismatches.append((key, manifest.get(key), expected_metadata.get(key)))
    if mismatches:
        formatted = "; ".join(
            f"{key}: cache={old!r}, expected={new!r}" for key, old, new in mismatches
        )
        raise ValueError(f"Cache metadata mismatch: {formatted}")


def write_dataframe(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        df.to_parquet(path, index=False)
    elif suffix == ".csv":
        df.to_csv(path, index=False)
    else:
        raise ValueError("Supported dataframe cache formats are .parquet and .csv")


def read_dataframe(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        return pd.read_parquet(path)
    if suffix == ".csv":
        return pd.read_csv(path)
    raise ValueError("Supported dataframe cache formats are .parquet and .csv")
