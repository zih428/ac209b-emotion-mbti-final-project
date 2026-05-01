# MS4 Report Results

Generated from local run artifacts by `scripts/aggregate_report_results.py`.

Tracked CSVs and PNGs here are small report inputs. Large checkpoints,
post-level parquet scores, and full preprocessing caches remain under
`code/artifacts/` and are intentionally ignored by git.

Top-line test mean balanced accuracy:

| model | test mean balanced accuracy |
|---|---:|
| TF-IDF Logistic | 0.6512 |
| GRU Text + Emotion | 0.6223 |
| GRU Text | 0.5964 |
| GRU Text Inverse Weight | 0.5855 |
| Majority | 0.5000 |

Transformer-author top-line test mean balanced accuracy:

| model | test mean balanced accuracy |
|---|---:|
| Set Attention Text + Real Emotion p=200 | 0.6865 |
| Set Attention Text p=200 | 0.6778 |
| Frozen Text | 0.6293 |

Additional diagnostics included here:

- MS4 pipeline diagram.
- Bootstrap confidence intervals over test authors.
- Final text+emotion GRU threshold-tuning curves.
- Threshold objective sensitivity for balanced accuracy vs F1.
- Final text+emotion GRU confusion matrices.
- Source-vs-Reddit emotion distribution comparison.
- Token-length sensitivity audit for 128 vs 256 token limits.
- Fixed text-only GRU 128 vs 256 max-length training sensitivity.
- Transformer-author artifact status, transformer summaries, paired deltas, seed stability, and epoch sensitivity.
- Supplemental p200 set/attention checks show that the author-level transformer direction is stable, while the emotion-specific increment is seed/training sensitive.
