# AC 209b Final Project - Project 66

This folder is organized for both human review and LLM-assisted work. Agents should read `AGENTS.md` first. Markdown files are the default project memory; original notebooks, slides, PDFs, screenshots, and DOCX files live under `artifacts/` for verification.

## Provenance

The Markdown layer is based on:

- original course/project artifacts under `artifacts/`
- submitted MS2/MS3 notebooks and slides
- presenter notes and slide planning notes already present in the folder
- TA feedback and team clarifications
- the KaggleHub API access check recorded in `data/README.md`

Sections labeled as recommendations, carry-forward notes, or agent instructions are synthesis for future work, not claims of original course wording.

## Project Snapshot

- **Course:** AC 209b / CS 1090b, Spring 2026
- **Canvas project number:** 66
- **Team:** Harry Hu, Tom Shan, Wendy Wang, Kemeng Zhang
- **Topic:** Emotion-informed MBTI prediction from Reddit writing
- **Current state:** Milestones 0-3 are historical/submitted materials. Milestone 4 has not started.

## Core Research Question

Do text plus emotion-informed features improve author-level prediction of the four binary MBTI dimensions (`E/I`, `N/S`, `F/T`, `J/P`) over direct text-only baselines?

## Human Read Order

1. `docs/llm_project_context.md` - compact project briefing.
2. `data/README.md` - dataset sources and API loading instructions.
3. `milestones/README.md` - milestone-by-milestone status.
4. The relevant milestone README, `requirements.md`, and `notebook_summary.md` when present.

## Folder Map

| Path | Purpose |
|---|---|
| `data/` | Dataset source notes and API loading instructions. |
| `docs/course/` | Course-level notes; original syllabus and milestone overview screenshot are under `docs/course/artifacts/`. |
| `milestones/ms0_project_proposal/` | Proposal Markdown, MS0 requirements Markdown, and original artifacts. |
| `milestones/ms1_group_formation/` | Group formation notes, requirements Markdown, and original artifacts. |
| `milestones/ms2_data_wrangling_project_redefinition/` | MS2 summaries, presenter notes, TA feedback, requirements, and original artifacts. |
| `milestones/ms3_eda_baseline_pipeline/` | MS3 summaries, slide text, presenter script, TA feedback, requirements, and original artifacts. |
| `milestones/ms4_final_modeling_deliverables/` | Empty working area for final report, video, and code notebook. |

## Version Rules

- Markdown files are the main project memory.
- `artifacts/submitted/` contains original submitted notebooks and slide files.
- `artifacts/drafts/` contains earlier iterations.
- `artifacts/requirements_*.png` are requirement screenshots kept for visual verification.
- `docs/course/artifacts/` contains original course-level PDF/PNG references.
- For MS2, `notes/ms2_presenter_script_part_1_2.md` is important because slides omit spoken detail.
- For MS3, `notes/ms3_presenter_script.md` and `notes/ms3_slide_plan.md` are important context even though they were not submitted.

## MS4 Immediate Next Step

Before writing final code or report text, read:

- `milestones/ms4_final_modeling_deliverables/README.md`
- `milestones/ms3_eda_baseline_pipeline/README.md`
- `data/README.md`

The MS3 diagnosis says the vanilla baselines collapsed to the majority class. The final model should prioritize class-weighted BCE, soft author aggregation, tuned thresholds, and a stronger emotion encoder before expanding into many unrelated ablations.
