# Agent Instructions

This file gives agent reading and editing rules. It is not a course artifact summary.

Start here. Do **not** recursively inspect the whole folder by default.

## Default Reading Order

1. `README.md`
2. `docs/llm_project_context.md`
3. `data/README.md`
4. `milestones/README.md`
5. The relevant milestone's `README.md`
6. That milestone's `requirements.md`
7. That milestone's `notebook_summary.md`, if present
8. Other Markdown notes only if they are relevant to the task

## Artifacts Policy

Ignore `artifacts/` by default.

Files under `artifacts/` are original evidence: PDFs, PNG screenshots, PPTX decks, DOCX notes, and IPYNB notebooks. Open them only when:

- a Markdown summary is missing, incomplete, or contradictory
- exact original wording is needed
- submitted artifacts must be verified
- notebook implementation details or code need to be audited
- visual layout or screenshot content matters

Do not scan every notebook, PDF, slide deck, screenshot, or DOCX just because it exists.

## Project-Specific Cautions

- The MS3 person-by-person work split was tentative and presentation-driven. Do not treat it as the real MS4 division of labor.
- MS3 slides and notebook differ slightly on whether Stage 2 DistilBERT is core scope or a stretch goal. Decide this explicitly before implementation.
- For MS4, build from the MS3 baseline diagnosis: class-weighted BCE, soft author aggregation, validation-tuned thresholds, and improved emotion features.
- Load Reddit data through KaggleHub. Do not assume a project-local raw CSV exists.

## Editing Guidance

- Keep Markdown files as the project memory and navigation layer.
- Keep original binary files under `artifacts/`.
- If you add new generated summaries, place them next to the relevant milestone README rather than inside `artifacts/`.
- If a new binary deliverable is created, put it under the appropriate milestone's `artifacts/` or MS4 work area and document it in Markdown.
