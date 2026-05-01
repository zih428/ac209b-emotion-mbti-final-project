# MS4 - Final Modeling and Deliverables

## Status

Corrected GRU/TF-IDF baseline layer and MiniLM transformer-author experiments are implemented locally and summarized for report/video use. The executed notebook now includes tracked report-ready transformer result tables, figures, and paired emotion-control deltas.

## Source Boundary

Requirements and deadlines below summarize `requirements.md`, which was transcribed from the MS4 requirement screenshot. The modeling design now reflects the implemented MS4 experiments and tracked result artifacts; older MS3 future-pipeline notes and person-by-person work splits should be treated only as historical context.

## Files and Work Areas

| Path | Purpose |
|---|---|
| `requirements.md` | Markdown summary of the MS4 requirement page. |
| `ms4_experiment_summary.md` | Implemented MS4 experiment summary, notebook organization, headline results, and report/video guidance. |
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

The implemented experiment summary is preserved in `ms4_experiment_summary.md`.
The executed notebook and tracked result artifacts include the corrected baseline layer and transformer-author layer:

- masked Reddit preprocessing through KaggleHub
- author-level train/validation/test split
- majority and TF-IDF author baselines
- fixed text-only GRU with class-weighted BCE, soft author aggregation, and validation-tuned thresholds
- text-plus-emotion GRU using cached DistilBERT emotion probabilities
- bootstrap confidence intervals over test authors
- threshold-objective sensitivity for balanced accuracy versus F1
- token-length audit plus a real fixed text-only GRU 128 versus 256 max-length sensitivity run
- frozen MiniLM post embeddings aggregated to author features
- frozen transformer author probes with emotion-only, real-emotion, shuffled-emotion, and control variants
- set/attention author transformer over unordered post sets with 50 versus 200 post-budget sensitivity
- paired bootstrap comparisons of the clean p200 set/attention text model against TF-IDF and corrected GRU baselines
- report-ready figures, tables, and interpretation in `code/cs1090b_ms4_main_group66.ipynb`

Headline test mean balanced accuracy:

- Set Attention Text + Controls, 200-post budget: 0.6879
- Set Attention Text, 200-post budget: 0.6784
- Set Attention Text + Shuffled Emotion, 200-post budget: 0.6716
- Set Attention Text + Real Emotion, 200-post budget: 0.6715
- TF-IDF Logistic: 0.6512
- Frozen MiniLM Text: 0.6293
- GRU Text + Emotion: 0.6223
- GRU Text: 0.5964
- GRU Text Inverse Weight: 0.5855
- Majority: 0.5000

These tracked results support the updated MS4 direction: the robust model family is the 200-post set/attention author transformer, not the GRU. The clean p200 set/attention text model is higher than TF-IDF by +0.0272 mean balanced accuracy with a paired bootstrap 95% CI of [+0.0089, +0.0441]. The revised seeded runs and post-budget-specific train-only standardized controls make the emotion story more cautious: real emotion is below text-only in the main p200 matched comparison by point estimate, and shuffled emotion or text-only can match or exceed it under some seeds/training lengths. The final writeup therefore emphasizes author-level transformer modeling as the robust gain and describes emotion-derived features as informative but not robustly incremental beyond text representations.

## Updated Transformer Author Design

The current implementation supports the final scientific claim by adding transformer author representations and stricter emotion controls:

- emotion probabilities are framed as text-derived transferred representations, not independent measurements or causal mediators
- primary estimand: matched `text + real emotion` minus `text-only` at the author level
- negative control: matched `text + shuffled emotion` minus `text-only`
- activity/length controls: retained post count, post length, truncation exposure, total retained token count, and post-budget-specific train-only standardized post-level length controls for set/attention variants; the set/attention truncation proxy uses 256 tokens to match the frozen MiniLM cache
- emotion-only author baseline to distinguish standalone signal, complementarity, and compressed text proxy behavior
- frozen transformer author classifiers using post-embedding summaries
- set/attention author transformer over unordered post embeddings
- mean-pooling and mean-plus-std pooling ablations for the author transformer
- 50 versus 200 retained-post budget sensitivity using a deterministic seed/hash post order within each author
- supplemental p200 seed stability and 10/20 max-epoch-cap sensitivity with early stopping for text-only, real-emotion, and shuffled-emotion variants
- supervised post-level transformer fine-tuning excluded from the MS4 mainline because it changes the estimand and reintroduces post-label noise

Implemented code entry points:

- `code/scripts/cache_transformer_embeddings.py`
- `code/scripts/run_transformer_author_models.py`
- `code/scripts/run_set_attention_author_models.py`
- `code/scripts/aggregate_report_results.py`

The main notebook is now organized as a report-facing scientific narrative: framing, data safeguards, baseline models, transformer author models, emotion controls, stability checks, final takeaways, and references. Large post-level embedding caches and run directories remain outside the submission path; compact result tables and figures are tracked under `report/results/`.

## Suggested Success Metrics

Report per dimension:

- balanced accuracy
- F1
- precision
- recall
- ROC-AUC when using continuous author-level scores
- PR-AUC / average precision for skewed dimensions such as `N/S`

Do not rely on raw accuracy alone because the MBTI dimensions are imbalanced, especially `N/S`.
