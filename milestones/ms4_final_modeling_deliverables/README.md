# MS4 - Final Modeling and Deliverables

## Status

Not started.

## Source Boundary

Requirements and deadlines below summarize `requirements.md`, which was transcribed from the MS4 requirement screenshot. The modeling plan is a recommendation derived from MS3 results, MS3 future-pipeline notes, TA feedback, and the team note that the MS3 person-by-person work split was tentative.

## Files and Work Areas

| Path | Purpose |
|---|---|
| `requirements.md` | Markdown summary of the MS4 requirement page. |
| `tentative_modeling_notebook_design.md` | Tentative MS4 experiment and notebook design; not completed results. |
| `artifacts/requirements_ms4_final_modeling_deliverables.png` | Screenshot of the MS4 requirement page. |
| `report/` | Final report assets and drafts. |
| `video/` | Video script, recording notes, and video link notes. |
| `code/` | Main final notebook and supporting code. |

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

## Recommended MS4 Modeling Plan

For the expanded tentative experiment and notebook design, see `tentative_modeling_notebook_design.md`.

Do not inherit the named person-by-person MS3 task split as a real constraint. That split was tentative and presentation-driven. Reassign work based on availability, implementation needs, and compute constraints.

Start from the MS3 baseline diagnosis:

1. Reuse the author-level split and modeling DataFrame decisions from MS3.
2. Implement class-weighted BCE on the Stage 2 MBTI head.
3. Replace hard majority vote with soft author aggregation:
   - average post-level probabilities by author
   - tune one threshold per MBTI dimension on validation
4. Add ROC-AUC where continuous author-level scores exist.
5. Fine-tune or otherwise improve the Stage 1 emotion model if compute allows.
6. Cache Stage 1 emotion probabilities before Stage 2 experiments.
7. Report ablations:
   - no class weighting
   - hard vote instead of soft aggregation
   - RNN emotion model instead of DistilBERT emotion model

## Suggested Success Metrics

Report per dimension:

- balanced accuracy
- F1
- precision
- recall
- ROC-AUC when using continuous author-level scores

Do not rely on raw accuracy alone because the MBTI dimensions are imbalanced, especially `N/S`.
