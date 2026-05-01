# MS4 Implemented Modeling and Notebook Design

Main notebook:

```text
code/cs1090b_ms4_main_group66.ipynb
```

Tracked result artifacts:

```text
report/results/
```

## Final Scientific Story

The project predicts four binary MBTI dimensions from Reddit writing at the
**author level**. The author is the correct unit because MBTI labels are
self-reported per user; individual posts only inherit that label noisily.

The final MS4 story has two parts:

1. **Robust modeling result:** author-level set/attention aggregation over many
   frozen MiniLM post embeddings is the strongest modeling direction.
2. **Cautious emotion result:** emotion-derived features are informative, but
   they are not robustly incremental beyond matched text representations.

The clean main semantic model is:

```text
Set Attention Text, 200-post budget
```

The highest point estimate is:

```text
Set Attention Text + Controls, 200-post budget
```

That controls model is treated as diagnostic/control-augmented, not as the clean
main semantic model, because it includes activity and length information.

## Research Questions

The notebook answers two questions:

1. Can author-level transformer aggregation improve MBTI prediction over TF-IDF
   and corrected GRU baselines?
2. Do transferred emotion representations add robust incremental signal beyond
   matched text representations?

The implemented answer is:

- **Yes** for author-level set/attention modeling.
- **No robust evidence** for an incremental real-emotion gain beyond text.

## Data and Safeguards

Reddit data is loaded through KaggleHub in the full preprocessing script, not
stored as a raw project CSV. The default notebook path reads compact tracked
artifacts from `report/results/`.

Implemented safeguards:

- mask explicit MBTI leakage terms before modeling
- filter very short posts
- require enough retained posts per author
- cap retained posts at 200 per author
- split train/validation/test by author, never by post
- derive four binary targets: `E/I`, `N/S`, `F/T`, `J/P`
- use validation-tuned thresholds and test-only final reporting

Tracked preprocessing artifacts used by the notebook:

- `preprocessing_summary.csv`
- `preprocessing_mbti_leakage_audit.csv`
- `preprocessing_split_balance.csv`

Full preprocessing audit:

- before masking: 216,846 of 13,028,455 posts contain an MBTI-related term
  (`1.664%`)
- before masking: 9,232 of 11,773 authors have at least one such post
  (`78.42%`)
- before masking: 112,294 posts mention the author's own type (`0.862%`)
- after masking: all tracked leakage counts are zero

## Model Families

### Baseline Layer

The baseline layer is retained as context and comparison:

- majority author baseline
- author-level TF-IDF logistic regression
- corrected GRU text model
- corrected GRU text + emotion model
- GRU inverse-weight and max-length sensitivity diagnostics

The GRU emotion model improves over corrected GRU text, but TF-IDF remains
stronger than the GRU family. This motivates the move to transformer author
representations.

### Frozen Transformer Author Probes

Frozen MiniLM embeddings are cached per post, then summarized to author-level
features. The frozen-probe variants include:

- text-only
- emotion-only
- text + shuffled emotion
- text + real emotion
- text + controls
- text + real emotion + controls

These probes test whether simple summary statistics over frozen transformer
embeddings are sufficient. They are useful, but they do not produce the strongest
final model.

### Set/Attention Author Models

The main model family represents each author as an unordered set of retained
post embeddings. The model uses learned attention over posts, without temporal
positional encoding, because the dataset does not provide reliable chronology.

Implemented variants:

- mean-pooling MLP
- mean-plus-std pooling MLP
- set/attention text-only
- set/attention text + shuffled emotion
- set/attention text + real emotion
- set/attention text + controls
- set/attention text + real emotion + controls
- 50-post and 200-post budget sensitivity
- supplemental p200 seed stability
- supplemental p200 10/20 max-epoch-cap sensitivity with early stopping enabled

The p50 budget uses deterministic seed/hash post ordering within each author, so
it is a stable pseudo-random retained-post budget rather than a text-sorting
artifact.

## Controls and Emotion Interpretation

Emotion probabilities are transferred text-derived representations from an
emotion classifier. They are not independent measurements of a user's emotional
state and should not be interpreted causally.

Primary emotion estimand:

```text
text + real emotion - text-only
```

Negative-control estimand:

```text
text + shuffled emotion - text-only
```

The shuffled control preserves split membership and marginal emotion
distributions while breaking author-emotion alignment. If shuffled emotion
matches or exceeds real emotion, the emotion-specific claim is not supported.

Observed result:

- emotion-only features perform above the majority baseline, so they contain
  author-level information
- real emotion is below text-only in the main p200 set/attention comparison
- shuffled emotion and text-only match or exceed real emotion in key robustness
  checks
- the final claim is therefore: emotion-derived features are informative but not
  robustly incremental beyond text representations

## Headline Results

Mean test balanced accuracy:

| model | mean balanced accuracy |
|---|---:|
| Set Attention Text + Controls p=200 | 0.6879 |
| Set Attention Text p=200 | 0.6784 |
| Set Attention Text + Shuffled Emotion p=200 | 0.6716 |
| Set Attention Text + Real Emotion p=200 | 0.6715 |
| TF-IDF Logistic | 0.6512 |
| Frozen MiniLM Text | 0.6293 |
| GRU Text + Emotion | 0.6223 |
| GRU Text | 0.5964 |
| Majority | 0.5000 |

Paired bootstrap comparisons for the clean p200 text-only set/attention model:

| comparison | mean delta | 95% CI |
|---|---:|---|
| Set Attention Text p=200 minus TF-IDF Logistic | +0.0272 | [+0.0089, +0.0441] |
| Set Attention Text p=200 minus GRU Text + Emotion | +0.0561 | [+0.0365, +0.0739] |
| Set Attention Text p=200 minus GRU Text | +0.0820 | [+0.0634, +0.1001] |

Main emotion delta:

| comparison | mean delta | 95% CI |
|---|---:|---|
| Set Attention p=200 real emotion minus text | -0.0069 | [-0.0191, +0.0041] |

## Notebook Organization

The notebook is organized as a final scientific narrative:

1. project framing and executive summary
2. MS4 pipeline diagram
3. data and preprocessing checks
4. post-budget and token-length motivation
5. baseline models
6. per-dimension baseline performance
7. GRU emotion baseline
8. emotion source-to-Reddit transfer check
9. frozen transformer author probes
10. set/attention author transformer results
11. p200 text-only set/attention error profile
12. emotion increment analysis
13. p200 stability checks
14. baseline uncertainty, set/attention paired bootstrap comparisons, and
    threshold diagnostics
15. final takeaways
16. reproduction commands and references

The notebook is intentionally not the training driver. It reads compact tracked
CSV/PNG artifacts from `report/results` so `Restart Kernel + Run All` is fast and
review-friendly.

## Reproduction Path

Full preprocessing, caching, training, evaluation, and report aggregation are
script-based from `code/`:

```bash
uv sync --extra full --extra dev
uv run --extra full python scripts/preprocess_reddit_ms4.py
uv run --extra full python scripts/run_author_baselines.py
uv run --extra full python scripts/cache_emotion_features.py --output-path artifacts/cache/emotion_probs_full.parquet --batch-size 256 --log-every-batches 100
uv run --extra full python scripts/cache_transformer_embeddings.py --posts-path artifacts/preprocessed/full/modeling_posts.parquet --output-dir artifacts/cache/transformer_embeddings --batch-size 256 --log-every-shards 10
uv run --extra full python scripts/train_stage2_text_gru.py --full-run
uv run --extra full python scripts/train_stage2_text_gru.py --full-run --run-id stage2_text_emotion_gru_full --emotion-feature-path artifacts/cache/emotion_probs_full.parquet
uv run --extra full python scripts/run_transformer_author_models.py
uv run --extra full python scripts/run_set_attention_author_models.py
uv run --extra full python scripts/aggregate_report_results.py
```

Large local artifacts under `code/artifacts/` remain ignored. Small report-ready
CSV/PNG artifacts under `report/results/` are tracked so the notebook is
self-contained for review.

## Report and Video Guidance

Use the following wording discipline:

- Say **author-level set/attention aggregation is the robust modeling result**.
- Say **emotion-derived features are informative but not robustly incremental
  beyond text representations**.
- Do not describe emotion probabilities as independent emotional-state
  measurements or causal mediators.
- Do not call `Text + Controls p=200` the clean final semantic model; it is the
  highest diagnostic/control-augmented run.
- Use `Set Attention Text p=200` for the clean main error profile and paired
  comparison against TF-IDF/GRU baselines.
