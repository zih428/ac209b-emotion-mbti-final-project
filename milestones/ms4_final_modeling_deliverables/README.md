# MS4 - Final Modeling and Deliverables

## Status

Corrected GRU/TF-IDF baseline layer implemented locally and summarized for report/video use. The transformer author-representation code path is now implemented and smoke-tested, with notebook/status cells that make full-transformer result availability explicit. Full MiniLM transformer result artifacts are not yet generated in the tracked report outputs.

## Source Boundary

Requirements and deadlines below summarize `requirements.md`, which was transcribed from the MS4 requirement screenshot. The modeling plan is a recommendation derived from MS3 results, MS3 future-pipeline notes, TA feedback, and the team note that the MS3 person-by-person work split was tentative.

## Files and Work Areas

| Path | Purpose |
|---|---|
| `requirements.md` | Markdown summary of the MS4 requirement page. |
| `tentative_modeling_notebook_design.md` | Current MS4 experiment and notebook design, including transformer-author experiments. |
| `artifacts/requirements_ms4_final_modeling_deliverables.png` | Screenshot of the MS4 requirement page. |
| `report/results/` | Tracked report-ready CSV/PNG result artifacts used by the executed notebook. |
| `video/` | Video script, recording notes, and video link notes. |
| `code/` | Executed final notebook, uv environment, training/evaluation scripts, and helper package. |

## Deadlines

- MS4 graded deliverables due: Tuesday, May 12, 2026 at 9:59pm Eastern Time.
- Peer evaluation due: Wednesday, May 13, 2026 at 9:59pm Eastern Time.
- Optional feedback meeting with TF should happen before Friday, May 8, 2026.

## Graded Deliverables

| Deliverable | Points | Notes |
|---|---:|---|
| Final report | 35 | Paper-style report, target 2000-2500 words, single PDF. |
| Video presentation | 25 | Six minutes or less, normal speed, all team members appear and speak unless accommodated. |
| Code notebook | 8 | Main pipeline and final results demonstrated end-to-end in one clearly named notebook. |
| Peer evaluations | Required | Separate Canvas assignment, individual. |

## Final Report Required Structure

The exact section names may vary, but the report must clearly cover:

1. Title, group Canvas number, and group member names.
2. Background and motivation.
3. Problem statement.
4. Data and EDA.
5. Methods and models.
6. Results and interpretation.
7. Conclusions and discussion.
8. Broader impact.
9. References.

Important: do not re-paste the full MS2/MS3 EDA. Distill only the findings that materially influenced final modeling decisions.

## Video Required Arc

The video should cover, roughly in order:

1. Introductions of all team members.
2. Background, motivation, and problem statement.
3. Data and key EDA findings.
4. Modeling approach and training details.
5. Results and interpretation.
6. Conclusions and future work.

## Code Notebook Requirements

- Name the main notebook clearly, for example `cs1090b_ms4_main_group66.ipynb`.
- It must be readable enough for someone who has not seen the project.
- Use clear section headings and Markdown explanations.
- Include group Canvas number and member names at the top.
- Document dependencies, data paths, API keys, and hardware assumptions.
- "Restart kernel + Run All" should succeed given documented dependencies and data.
- The main notebook may import helper scripts or supporting notebooks.
- Do not include raw datasets larger than a few hundred MB in a submission zip. Document how to obtain them instead.

## Implemented MS4 Modeling Summary

The expanded experiment design is preserved in `tentative_modeling_notebook_design.md`.
The executed notebook and tracked result artifacts include the corrected baseline layer:

- masked Reddit preprocessing through KaggleHub
- author-level train/validation/test split
- majority and TF-IDF author baselines
- fixed text-only GRU with class-weighted BCE, soft author aggregation, and validation-tuned thresholds
- text-plus-emotion GRU using cached DistilBERT emotion probabilities
- bootstrap confidence intervals over test authors
- threshold-objective sensitivity for balanced accuracy versus F1
- token-length audit plus a real fixed text-only GRU 128 versus 256 max-length sensitivity run
- report-ready figures, tables, and interpretation in `code/cs1090b_ms4_main_group66.ipynb`

Headline test mean balanced accuracy:

- TF-IDF Logistic: 0.6512
- GRU Text + Emotion: 0.6223
- GRU Text: 0.5964
- GRU Text Inverse Weight: 0.5855
- Majority: 0.5000

These tracked numeric results should be treated as the baseline layer, not as completed transformer-author evidence. They support the MS3 diagnosis and show that corrected GRU plus DistilBERT emotion improves over corrected GRU text, while author-level TF-IDF logistic remains the strongest currently tracked model.

## Updated Transformer Author Design

The current implementation supports the final scientific claim by adding transformer author representations and stricter emotion controls:

- emotion probabilities are framed as text-derived transferred representations, not independent measurements or causal mediators
- primary estimand: matched `text + real emotion` minus `text-only` at the author level
- negative control: matched `text + shuffled emotion` minus `text-only`
- activity/length controls: retained post count, post length, truncation exposure, and total retained token count
- emotion-only author baseline to distinguish standalone signal, complementarity, and compressed text proxy behavior
- frozen transformer author classifiers using post-embedding summaries
- set/attention author transformer over unordered post embeddings
- mean-pooling and mean-plus-std pooling ablations for the author transformer
- 50 versus 200 retained-post budget sensitivity
- supervised post-level transformer fine-tuning excluded from the MS4 mainline because it changes the estimand and reintroduces post-label noise

Implemented code entry points:

- `code/scripts/cache_transformer_embeddings.py`
- `code/scripts/run_transformer_author_models.py`
- `code/scripts/run_set_attention_author_models.py`
- `code/scripts/aggregate_report_results.py`

The main notebook includes executed cells for the design, artifact status, schema checks, and report-facing placeholders. Once the full MiniLM embedding cache and transformer author runs are available locally, rerunning aggregation and the notebook will populate the transformer summary and paired-delta tables.

## Suggested Success Metrics

Report per dimension:

- balanced accuracy
- F1
- precision
- recall
- ROC-AUC when using continuous author-level scores
- PR-AUC / average precision for skewed dimensions such as `N/S`

Do not rely on raw accuracy alone because the MBTI dimensions are imbalanced, especially `N/S`.
