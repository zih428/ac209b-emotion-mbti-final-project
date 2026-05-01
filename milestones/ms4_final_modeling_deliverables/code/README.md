# MS4 Code Workspace

Use this folder for the final main notebook and supporting scripts.

Recommended main notebook name:

```text
cs1090b_ms4_main_group66.ipynb
```

Current implementation entry points for the corrected baseline layer:

| Path | Purpose |
|---|---|
| `cs1090b_ms4_main_group66.ipynb` | Executed baseline-layer notebook with outputs embedded. It loads compact tracked GRU/TF-IDF results and does not yet include the updated transformer-author result path. |
| `src/ms4mbti/` | Importable helper package for preprocessing, splitting, weighting, metrics, cache metadata, baseline models, Stage 2 modeling, and progress reporting. |
| `scripts/preprocess_reddit_ms4.py` | Full Reddit preprocessing, MBTI masking, author split, leakage audit, and token truncation audit. |
| `scripts/cache_emotion_features.py` | DistilBERT emotion probability cache for Reddit posts. |
| `scripts/run_author_baselines.py` | Majority and TF-IDF author-level baselines. |
| `scripts/train_stage2_text_gru.py` | Text-only and text-plus-emotion Stage 2 GRU training/evaluation entry point. |
| `scripts/aggregate_report_results.py` | Builds tracked report-ready CSV/PNG artifacts from local run outputs. |
| `scripts/run_smoke_checks.py` | Command-line synthetic smoke test for the default non-training path. |
| `tests/test_smoke.py` | Pytest wrapper for the same non-training smoke path. |
| `pyproject.toml` / `uv.lock` | Reproducible dependency and package management through `uv`. |

Updated design entry points to add before the transformer-author results are report-facing:

| Path | Purpose |
|---|---|
| `src/ms4mbti/embeddings.py` | Frozen transformer post-embedding inference and cache metadata. |
| `src/ms4mbti/author_features.py` | Author-level text, emotion, activity, length, and truncation feature construction. |
| `src/ms4mbti/negative_controls.py` | Deterministic split-preserving shuffled-emotion feature tables. |
| `src/ms4mbti/transformer_author.py` | Mean-pooling, mean-plus-std, and set/attention author transformer training helpers. |
| `scripts/cache_transformer_embeddings.py` | Full-corpus frozen transformer embedding cache job. |
| `scripts/run_transformer_author_models.py` | Frozen transformer author classifiers: text, emotion-only, shuffled emotion, real emotion, controls, and real emotion plus controls. |
| `scripts/run_set_attention_author_models.py` | Set/attention author transformer, pooling ablations, and 50 versus 200 post-budget sensitivity. |

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

Current notebook coverage for the tracked baseline-layer results:

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

Updated notebook sections required by the current design:

- explicit statement that emotion probabilities are text-derived transferred representations, not independent emotion measurements or causal mediators
- frozen transformer embedding cache and author-feature construction
- emotion-only author baseline
- frozen transformer author models with text-only, shuffled emotion, real emotion, controls, and real emotion plus controls
- set/attention author transformer over unordered post embeddings with matched shuffled-emotion and control variants
- mean-pooling and mean-plus-std pooling ablations
- 50 versus 200 retained-post sensitivity
- paired bootstrap intervals for real-emotion-minus-text and shuffled-emotion-minus-text deltas
- future-work note excluding supervised post-level transformer fine-tuning from the MS4 mainline

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

The build order above regenerates the currently tracked baseline-layer artifacts. After the transformer-author scripts are implemented, the full build order should insert transformer embedding caching, frozen author classifiers, set/attention author classifiers, and the updated aggregation step before `scripts/aggregate_report_results.py`.
