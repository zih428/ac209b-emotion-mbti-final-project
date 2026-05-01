# Agent Task Router

Use this file to choose the right MS4 document or workspace for a task. This is
not a project summary; it is a routing guide.

## Source of Truth

| Need | Go to |
|---|---|
| Course requirements, due dates, grading weights | `milestones/ms4_final_modeling_deliverables/requirements.md` |
| Scientific story, headline interpretation, report/video wording | `milestones/ms4_final_modeling_deliverables/ms4_experiment_summary.md` |
| Generated result numbers, CSVs, and figures | `milestones/ms4_final_modeling_deliverables/report/results/` |
| Code environment, scripts, tests, reproduction commands | `milestones/ms4_final_modeling_deliverables/code/README.md` |
| Main executed notebook | `milestones/ms4_final_modeling_deliverables/code/cs1090b_ms4_main_group66.ipynb` |
| Report writing structure | `milestones/ms4_final_modeling_deliverables/report/README.md` |
| Video timing/script structure | `milestones/ms4_final_modeling_deliverables/video/README.md` |

## Common Tasks

| Task | Read first | Edit here |
|---|---|---|
| Update final result interpretation | `milestones/ms4_final_modeling_deliverables/ms4_experiment_summary.md`, `milestones/ms4_final_modeling_deliverables/report/results/README.md` | `milestones/ms4_final_modeling_deliverables/ms4_experiment_summary.md` |
| Update generated result numbers | `milestones/ms4_final_modeling_deliverables/code/README.md`, `milestones/ms4_final_modeling_deliverables/report/results/manifest.json` | Run `milestones/ms4_final_modeling_deliverables/code/scripts/aggregate_report_results.py`; do not hand-edit generated CSV/PNG/manifest files unless fixing generation is impossible |
| Update notebook narrative | `milestones/ms4_final_modeling_deliverables/ms4_experiment_summary.md`, notebook section around the change | `milestones/ms4_final_modeling_deliverables/code/cs1090b_ms4_main_group66.ipynb` |
| Update training/reproduction commands | `milestones/ms4_final_modeling_deliverables/code/README.md` | `milestones/ms4_final_modeling_deliverables/code/README.md`, notebook "Full Reproduction Commands" if needed |
| Update report plan | `milestones/ms4_final_modeling_deliverables/ms4_experiment_summary.md`, `milestones/ms4_final_modeling_deliverables/report/README.md` | `milestones/ms4_final_modeling_deliverables/report/README.md` or report draft files |
| Update video plan | `milestones/ms4_final_modeling_deliverables/ms4_experiment_summary.md`, `milestones/ms4_final_modeling_deliverables/video/README.md` | `milestones/ms4_final_modeling_deliverables/video/` |
| Add or modify experiments | `milestones/ms4_final_modeling_deliverables/code/README.md`, relevant scripts, `milestones/ms4_final_modeling_deliverables/ms4_experiment_summary.md` | `milestones/ms4_final_modeling_deliverables/code/scripts/`, `milestones/ms4_final_modeling_deliverables/code/src/ms4mbti/`, tests, then regenerate `milestones/ms4_final_modeling_deliverables/report/results/` |
| Check course compliance | `milestones/ms4_final_modeling_deliverables/requirements.md`, `milestones/ms4_final_modeling_deliverables/README.md` | Relevant deliverable docs only |

## Rules

- Do not duplicate headline result numbers across README files.
- Treat `milestones/ms4_final_modeling_deliverables/report/results/` as the numeric source of truth.
- Treat `milestones/ms4_final_modeling_deliverables/ms4_experiment_summary.md` as the interpretation source of truth.
- Treat `milestones/ms4_final_modeling_deliverables/code/README.md` as the source of truth for commands and environment setup.
- The notebook is report-facing; do not turn it into the full training driver.
- Large local artifacts under `milestones/ms4_final_modeling_deliverables/code/artifacts/` are not submission sources unless explicitly requested.
- Treat MS2/MS3 files as historical context unless the task explicitly asks about earlier milestones.
- For MS4 scope decisions, prefer `milestones/ms4_final_modeling_deliverables/ms4_experiment_summary.md` over MS3 slide/script notes.
- If generated artifacts and prose disagree, verify `milestones/ms4_final_modeling_deliverables/report/results/` first, then update prose.
- When unsure, read `milestones/ms4_final_modeling_deliverables/README.md` first, then follow its source-of-truth table.
