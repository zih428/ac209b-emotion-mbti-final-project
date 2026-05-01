# MS4 Code Workspace

Use this folder for the final main notebook and supporting scripts.

Recommended main notebook name:

```text
cs1090b_ms4_main_group66.ipynb
```

Current implementation entry points:

| Path | Purpose |
|---|---|
| `cs1090b_ms4_main_group66.ipynb` | Executed report-facing MS4 notebook with outputs embedded. It loads compact tracked results and does not launch long training. |
| `src/ms4mbti/` | Importable helper package for preprocessing, splitting, weighting, metrics, cache metadata, baseline models, Stage 2 modeling, and progress reporting. |
| `scripts/preprocess_reddit_ms4.py` | Full Reddit preprocessing, MBTI masking, author split, leakage audit, and token truncation audit. |
| `scripts/cache_emotion_features.py` | DistilBERT emotion probability cache for Reddit posts. |
| `scripts/run_author_baselines.py` | Majority and TF-IDF author-level baselines. |
| `scripts/train_stage2_text_gru.py` | Text-only and text-plus-emotion Stage 2 GRU training/evaluation entry point. |
| `scripts/aggregate_report_results.py` | Builds tracked report-ready CSV/PNG artifacts from local run outputs. |
| `scripts/run_smoke_checks.py` | Command-line synthetic smoke test for the default non-training path. |
| `tests/test_smoke.py` | Pytest wrapper for the same non-training smoke path. |
| `pyproject.toml` / `uv.lock` | Reproducible dependency and package management through `uv`. |

Minimum sections:

1. Project metadata and dependency notes.
2. Data paths and loading.
3. Preprocessing and author-level split.
4. Baseline reproduction or loaded MS3 baseline reference.
5. Final model implementation.
6. Training and validation.
7. Author-level aggregation and threshold tuning.
8. Test evaluation.
9. Ablations.
10. Interpretation and reproducibility notes.

Current notebook coverage:

- uv/package/hardware environment check
- preprocessing, leakage, split-balance, and token-truncation audits
- pipeline diagram and report artifact manifest
- author-level model summary and per-dimension test metrics
- bootstrap confidence intervals over test authors
- emotion feature gain visualization
- source-vs-Reddit emotion distribution comparison
- GRU validation-loss curves
- final text-plus-emotion GRU threshold tuning and confusion matrices
- threshold-objective sensitivity for balanced accuracy versus F1
- token-length audit and 128 versus 256 GRU training sensitivity
- run-level commands, interpretation, references, and disclosure

## Environment Management

Use `uv` from this directory as the dependency source of truth:

```bash
cd milestones/ms4_final_modeling_deliverables/code
uv sync
```

For development checks:

```bash
uv sync --extra dev
uv run --extra dev pytest
```

For the full training/data path, install the optional full dependencies:

```bash
uv sync --extra full --extra dev
```

Default quality check:

```bash
cd milestones/ms4_final_modeling_deliverables/code
uv run ms4-smoke
```

This command uses synthetic Reddit-like data and should not download data or train neural models. It verifies that the code can:

- standardize and mask Reddit text
- detect and resolve author label conflicts
- apply MS3 filters
- create author-level stratified splits
- compute truncation and leakage audits
- compute author-balanced loss weights and BCE `pos_weight` variants
- aggregate post-level probabilities to author-level scores
- tune validation thresholds and produce test metrics
- train the lightweight TF-IDF author baseline on synthetic data
- construct the Stage 2 GRU model if PyTorch is available

Use KaggleHub for the Reddit dataset:

```python
from pathlib import Path

import kagglehub
import pandas as pd

reddit_path = Path(
    kagglehub.dataset_download(
        "minhaozhang1/reddit-mbti-dataset",
        path="reddit_post.csv",
    )
)
reddit_raw = pd.read_csv(reddit_path, usecols=["author", "body", "mbti"])
```

Full local artifact build order:

```bash
uv run --extra full python scripts/preprocess_reddit_ms4.py
uv run --extra full python scripts/run_author_baselines.py
uv run --extra full python scripts/cache_emotion_features.py --output-path artifacts/cache/emotion_probs_full.parquet --batch-size 256 --log-every-batches 100
uv run --extra full python scripts/train_stage2_text_gru.py --full-run
uv run --extra full python scripts/train_stage2_text_gru.py --full-run --run-id stage2_text_gru_inverse_full --pos-weight-variant inverse
uv run --extra full python scripts/train_stage2_text_gru.py --full-run --run-id stage2_text_emotion_gru_full --emotion-feature-path artifacts/cache/emotion_probs_full.parquet
uv run --extra full python scripts/train_stage2_text_gru.py --full-run --run-id stage2_text_gru_len256_full --max-length 256 --pos-weight-variant sqrt
uv run --extra full python scripts/aggregate_report_results.py
```
