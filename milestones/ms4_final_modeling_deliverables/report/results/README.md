# MS4 Report Results

Generated from local run artifacts by `scripts/aggregate_report_results.py`.

Tracked CSVs and PNGs here are small report inputs. Large checkpoints,
post-level parquet scores, and full preprocessing caches remain under
`code/artifacts/` and are intentionally ignored by git.

Current scope: these artifacts cover the corrected GRU/TF-IDF baseline layer. They do not yet include the updated transformer author-representation design from `../../tentative_modeling_notebook_design.md`.

Top-line test mean balanced accuracy:

| model | test mean balanced accuracy |
|---|---:|
| TF-IDF Logistic | 0.6512 |
| GRU Text + Emotion | 0.6223 |
| GRU Text | 0.5964 |
| GRU Text Inverse Weight | 0.5855 |
| Majority | 0.5000 |

Additional diagnostics included here:

- MS4 pipeline diagram.
- Bootstrap confidence intervals over test authors.
- Final text+emotion GRU threshold-tuning curves.
- Threshold objective sensitivity for balanced accuracy vs F1.
- Final text+emotion GRU confusion matrices.
- Source-vs-Reddit emotion distribution comparison.
- Token-length sensitivity audit for 128 vs 256 token limits.
- Fixed text-only GRU 128 vs 256 max-length training sensitivity.

Updated design artifacts still to add:

- frozen transformer author classifier metrics
- emotion-only author baseline metrics
- shuffled-emotion negative-control metrics
- activity/length control variants
- set/attention author transformer metrics
- mean-pooling and mean-plus-std pooling ablations
- 50 versus 200 retained-post budget sensitivity
- paired bootstrap intervals for real-emotion-minus-text and shuffled-emotion-minus-text deltas
