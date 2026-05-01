# MS4 Code Workspace

Use this folder for the final main notebook and supporting scripts.

Recommended main notebook name:

```text
cs1090b_ms4_main_group66.ipynb
```

Current implementation entry points:

| Path | Purpose |
|---|---|
| `cs1090b_ms4_main_group66.ipynb` | Main MS4 orchestration notebook. Default path runs smoke checks only and does not train. |
| `src/ms4mbti/` | Importable helper package for preprocessing, splitting, weighting, metrics, cache metadata, baseline models, and progress reporting. |
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

- runtime flags for default, real-data smoke, and full-training paths
- uv/package/hardware environment checks
- MS3 failure diagnosis and MS4 controlled-experiment framing
- optional real KaggleHub/Hugging Face data smoke cells
- default synthetic modeling frame that uses the same preprocessing APIs
- MBTI leakage, author-label conflict, split-balance, class-balance, and token-truncation audits
- report-style visualization cells for leakage, class balance, retained posts per author, split balance, truncation exposure, and baseline metric comparison
- author-balanced post weights and BCE `pos_weight` calculations
- planned model comparison table
- Stage 1 emotion classifier plan and diagnostic metric contract
- majority and TF-IDF author baseline metric skeleton
- main results schema and historical-vs-controlled table separation
- Stage 2 GRU array/model construction dry run without training
- validation-only class-weight recipe selection API demonstration
- compact artifact contract for submitted author-level results
- local smoke figure export to `artifacts/figures_smoke/`, ignored by git because those are reproducibility checks rather than final results
- interpretation frame, limitations, broader impact notes, references, and AI assistance disclosure
- explicit full-training launch order and guard

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

Optional real data smoke checks can then be run by opening the notebook and setting:

```python
RUN_REAL_DATA_SMOKE = True
RUN_FULL_TRAINING = False
```

This validates tiny public-data reads only; it does not start model training.

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
