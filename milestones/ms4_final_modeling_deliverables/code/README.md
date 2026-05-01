# MS4 Code Workspace

Use this folder for the final main notebook and supporting scripts.

Recommended main notebook name:

```text
cs1090b_ms4_main_group66.ipynb
```

Current implementation entry points:

| Path | Purpose |
|---|---|
| `cs1090b_ms4_main_group66.ipynb` | Executed MS4 notebook with outputs embedded. It tells the final scientific story, loads tracked GRU/TF-IDF baseline-layer results and MiniLM transformer-author results, and explains the full script-based reproduction path. |
| `src/ms4mbti/` | Importable helper package for preprocessing, splitting, weighting, metrics, cache metadata, baseline models, Stage 2 modeling, frozen transformer embeddings, author features, negative controls, transformer-author models, visualization, and progress reporting. |
| `scripts/preprocess_reddit_ms4.py` | Full Reddit preprocessing, MBTI masking, author split, leakage audit, and token truncation audit. |
| `scripts/cache_emotion_features.py` | DistilBERT emotion probability cache for Reddit posts. |
| `scripts/run_author_baselines.py` | Majority and TF-IDF author-level baselines. |
| `scripts/train_stage2_text_gru.py` | Text-only and text-plus-emotion Stage 2 GRU training/evaluation entry point. |
| `scripts/cache_transformer_embeddings.py` | Frozen transformer post-embedding cache job with resumable shards and manifest metadata. |
| `scripts/run_transformer_author_models.py` | Frozen transformer author classifiers: text, emotion-only, shuffled emotion, real emotion, controls, and real emotion plus controls. |
| `scripts/run_set_attention_author_models.py` | Mean-pooling, mean-plus-std, set/attention author transformer, and 50 versus 200 post-budget sensitivity. |
| `scripts/aggregate_report_results.py` | Builds tracked report-ready CSV/PNG artifacts from local run outputs. |
| `scripts/run_smoke_checks.py` | Command-line synthetic smoke test for the default non-training path. |
| `tests/test_smoke.py` | Pytest wrapper for the same non-training smoke path. |
| `pyproject.toml` / `uv.lock` | Reproducible dependency and package management through `uv`. |

Transformer-author helper modules:

| Path | Purpose |
|---|---|
| `src/ms4mbti/embeddings.py` | Frozen transformer post-embedding inference and cache metadata. Supports real transformer inference and deterministic smoke caches. |
| `src/ms4mbti/author_features.py` | Author-level text, emotion, activity, length, and truncation feature construction. |
| `src/ms4mbti/negative_controls.py` | Deterministic split-preserving shuffled-emotion feature tables. |
| `src/ms4mbti/transformer_author.py` | Mean-pooling, mean-plus-std, and set/attention author transformer training helpers. |

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

- project framing, group metadata, dependency notes, and references
- author-level scientific framing with emotion features treated as text-derived transferred representations
- preprocessing, leakage, split-balance, token-truncation, and post-budget audits
- author-level baseline results for majority, TF-IDF, and corrected GRU models
- per-dimension metrics, bootstrap uncertainty, threshold tuning, and GRU token-length sensitivity
- source-vs-Reddit emotion distribution comparison
- frozen transformer author result table from `scripts/run_transformer_author_models.py`
- set/attention author result table from `scripts/run_set_attention_author_models.py`
- paired bootstrap delta table for real-emotion-minus-text and shuffled-emotion-minus-text comparisons
- paired bootstrap delta table for the clean p200 set/attention text model versus TF-IDF and corrected GRU baselines
- supplemental p200 seed-stability and max-epoch-cap sensitivity tables/figures for text, real-emotion, and shuffled-emotion set/attention variants
- full reproduction commands showing which scripts generate the training outputs consumed by the notebook
- final takeaways suitable for report and video planning

The set/attention training entry point fixes the requested seed before model initialization, uses a seeded PyTorch `DataLoader` generator for shuffled training batches, and standardizes post-level control columns with training-split statistics inside each post-budget setting. The set/attention truncation control uses a 256-token indicator to match the frozen MiniLM embedding cache.

Post-budget comparisons use a deterministic seed/hash order within each author, so the p50 setting is a stable pseudo-random retained-post budget rather than the first 50 rows after text sorting.

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

The notebook itself is intentionally not the training driver. Full preprocessing, caching, model training, and report aggregation are script-based so that the notebook remains a readable final deliverable and `Restart Kernel + Run All` does not depend on rerunning long neural jobs. The commands below regenerate the result files that the notebook reads from `../report/results`.

```bash
uv run --extra full python scripts/preprocess_reddit_ms4.py
uv run --extra full python scripts/run_author_baselines.py
uv run --extra full python scripts/cache_emotion_features.py --output-path artifacts/cache/emotion_probs_full.parquet --batch-size 256 --log-every-batches 100
uv run --extra full python scripts/train_stage2_text_gru.py --full-run
uv run --extra full python scripts/train_stage2_text_gru.py --full-run --run-id stage2_text_gru_inverse_full --pos-weight-variant inverse
uv run --extra full python scripts/train_stage2_text_gru.py --full-run --run-id stage2_text_emotion_gru_full --emotion-feature-path artifacts/cache/emotion_probs_full.parquet
uv run --extra full python scripts/train_stage2_text_gru.py --full-run --run-id stage2_text_gru_len256_full --max-length 256 --pos-weight-variant sqrt
uv run --extra full python scripts/cache_transformer_embeddings.py
uv run --extra full python scripts/run_transformer_author_models.py --embedding-cache-dir artifacts/cache/transformer_embeddings/sentence-transformers__all-MiniLM-L6-v2_max256
uv run --extra full python scripts/run_set_attention_author_models.py --full-run --embedding-cache-dir artifacts/cache/transformer_embeddings/sentence-transformers__all-MiniLM-L6-v2_max256
uv run --extra full python scripts/run_set_attention_author_models.py --full-run --embedding-cache-dir artifacts/cache/transformer_embeddings/sentence-transformers__all-MiniLM-L6-v2_max256 --output-dir artifacts/runs/set_attention_author_p200_e5_seed209067 --post-budgets 200 --epochs 5 --seed 209067 --skip-pooling --set-variants set_attention_text set_attention_text_real_emotion set_attention_text_shuffled_emotion
uv run --extra full python scripts/run_set_attention_author_models.py --full-run --embedding-cache-dir artifacts/cache/transformer_embeddings/sentence-transformers__all-MiniLM-L6-v2_max256 --output-dir artifacts/runs/set_attention_author_p200_e5_seed209068 --post-budgets 200 --epochs 5 --seed 209068 --skip-pooling --set-variants set_attention_text set_attention_text_real_emotion set_attention_text_shuffled_emotion
uv run --extra full python scripts/run_set_attention_author_models.py --full-run --embedding-cache-dir artifacts/cache/transformer_embeddings/sentence-transformers__all-MiniLM-L6-v2_max256 --output-dir artifacts/runs/set_attention_author_p200_e10_seed209066 --post-budgets 200 --epochs 10 --seed 209066 --skip-pooling --set-variants set_attention_text set_attention_text_real_emotion set_attention_text_shuffled_emotion
uv run --extra full python scripts/run_set_attention_author_models.py --full-run --embedding-cache-dir artifacts/cache/transformer_embeddings/sentence-transformers__all-MiniLM-L6-v2_max256 --output-dir artifacts/runs/set_attention_author_p200_e20_seed209066 --post-budgets 200 --epochs 20 --seed 209066 --skip-pooling --set-variants set_attention_text set_attention_text_real_emotion set_attention_text_shuffled_emotion
uv run --extra full python scripts/aggregate_report_results.py
```

The build order above regenerates the tracked baseline-layer artifacts and MiniLM transformer-author report tables. For offline smoke checks, `scripts/cache_transformer_embeddings.py --backend deterministic_hash --max-rows ...` can validate the cache/model plumbing, but deterministic hash embeddings must not be reported as final transformer results.
