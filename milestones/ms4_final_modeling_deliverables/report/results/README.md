# MS4 Report Results

Generated from local run artifacts by `scripts/aggregate_report_results.py`.

Tracked CSVs and PNGs here are small report inputs. Large checkpoints,
post-level parquet scores, and full preprocessing caches remain under
`code/artifacts/` and are intentionally ignored by git.

Top-line test mean balanced accuracy:

| model | test mean balanced accuracy |
|---|---:|
| Set Attention Text + Real Emotion, 200-post budget | 0.6865 |
| Set Attention Text, 200-post budget | 0.6778 |
| TF-IDF Logistic | 0.6512 |
| Frozen MiniLM Text | 0.6293 |
| GRU Text + Emotion | 0.6223 |
| GRU Text | 0.5964 |
| GRU Text Inverse Weight | 0.5855 |
| Majority | 0.5000 |

Matched emotion-control deltas over test authors:

| comparison | mean balanced accuracy delta |
|---|---:|
| Frozen transformer real emotion minus text | -0.0060 |
| Frozen transformer shuffled emotion minus text | -0.0019 |
| Set attention p=50 real emotion minus text | -0.0156 |
| Set attention p=50 shuffled emotion minus text | 0.0067 |
| Set attention p=200 real emotion minus text | 0.0087 |
| Set attention p=200 shuffled emotion minus text | -0.0026 |

Additional diagnostics included here:

- MS4 pipeline diagram.
- Bootstrap confidence intervals over test authors.
- Final text+emotion GRU threshold-tuning curves.
- Threshold objective sensitivity for balanced accuracy vs F1.
- Final text+emotion GRU confusion matrices.
- Source-vs-Reddit emotion distribution comparison.
- Token-length sensitivity audit for 128 vs 256 token limits.
- Fixed text-only GRU 128 vs 256 max-length training sensitivity.
- Transformer-author artifact status, transformer summaries, paired deltas, and transformer comparison figures.
