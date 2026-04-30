# MS3 - EDA, Baseline Modeling, and Pipeline Development

## Status

Submitted. TA score: **19.8 / 20**.

## Files

| Path | Role |
|---|---|
| `requirements.md` | Markdown summary of MS3 requirements. |
| `notebook_summary.md` | LLM-friendly summary of the submitted MS3 notebook. |
| `submitted/ms3_submitted_slides_text.md` | Extracted slide text for LLM-friendly reading. |
| `notes/ms3_presenter_script.md` | Presenter script. Important because the slide deck is concise. |
| `notes/ms3_slide_plan.md` | Slide planning document and visual asset checklist. |
| `artifacts/submitted/ms3_slides_submitted.pptx` | Original submitted MS3 slide deck. |
| `artifacts/submitted/ms3_eda_baseline_pipeline_submitted.ipynb` | Original submitted MS3 notebook. |
| `artifacts/notes/ms3_presenter_script_docx_original.docx` | Original DOCX form of the presenter script. |
| `artifacts/drafts/ms3_slides_prior_or_autosave.pptx` | Earlier or auto-saved slide version. Text appears equivalent to the submitted deck, but hash differs. |
| `artifacts/requirements_ms3_eda_baseline_pipeline.png` | Screenshot of MS3 requirements. |

## Requirement Summary

Dates:

- Slides and accompanying notebook due at presentation time or by Friday, April 24, 2026 at 9:59pm, whichever came first.
- Presentation with TF due by Sunday, April 26, 2026 EOD.

Deliverables:

- 8-10 minute presentation with the TF.
- Slides used in presentation, submitted as PDF.
- `.ipynb` notebook used for wrangling, EDA, visualizations, and preliminary models.

Required content:

- Problem statement refinement and introduction.
- Comprehensive EDA review.
- Baseline model selection and justification.
- Results interpretation and analysis.
- Final model pipeline setup.

Effective notebook guidelines:

- Clear problem statement.
- Logical headings and subheadings.
- Table of contents required.
- Explain code, choices, and results using comments and Markdown.
- Use clear, labeled visualizations.
- Keep code readable and reproducible.

## Submitted Story

MS3 extended the MS2 findings into concrete modeling choices:

- Drop posts with fewer than 5 words.
- Require at least 20 posts per author.
- Cap each author at 200 posts.
- Use soft six-dimensional emotion probabilities.
- Use author-level train/validation/test splits.
- Compare matched baselines:
  - B1: text-only GRU.
  - B2: text plus emotion-probability GRU.

## Baseline Result

- Stage 1 RNN emotion classifier worked reasonably well: about 0.897 validation accuracy and 0.908 test accuracy.
- B1 and B2 had nearly identical validation BCE around 0.520.
- E/I, N/S, and J/P collapsed to the majority class with balanced accuracy 0.500.
- F/T showed a small signal: high minority precision but recall around 2-4 percent.
- Adding emotion features in the vanilla B2 setup did not meaningfully improve over B1.

## TA Feedback

Overall: **19.8 / 20**

- Intro and motivation: 1.0 / 1.0.
- Problem redefinition: 1.0 / 1.0.
- MS2 feedback addressed: 1.0 / 1.0.
- Data and EDA: 1.0 / 1.0.
- Baseline implementation: 1.0 / 1.0.
- Baseline justification: 1.0 / 1.0.
- Results: 1.0 / 1.0.
- Interpretations: 1.0 / 1.0.
- Future pipeline: 1.0 / 1.0.
- Insights: 1.0 / 1.0.
- Rescope: 1.0 / 1.0.
- Storytelling: 0.9 / 1.0. Some cross-references do not align with notebook structure. Example: section 3.1.4 is cited twice in Section 5.1, but Section 3.1 has no sub-subsections.
- Presentation and slides: 1.0 / 1.0.

## Carry Forward to MS4

The strongest final project story is not "emotion failed." It is:

1. Vanilla emotion concatenation failed because the MBTI head and hard aggregation collapsed to the majority class.
2. The remedy is targeted:
   - class-weighted BCE
   - soft author-level aggregation
   - validation-tuned thresholds
   - stronger or better-calibrated emotion probabilities
3. Report ablations that isolate each remedy.
4. Fix broken notebook cross-references before final submission.

## Team-Work Note

The named person-by-person work split in the MS3 slides, presenter script, and slide plan was tentative. It was included to satisfy the MS3 presentation requirement to delineate next steps by team member. It should **not** be treated as the actual MS4 implementation plan or as a binding division of labor.

## Internal Consistency Note

The MS3 materials are slightly inconsistent about the final model scope:

- The submitted slide deck and presenter script mention four upgrades, including a DistilBERT Stage 2 text encoder.
- The submitted notebook's Section 5 frames the final pipeline as three bundled upgrades and treats Stage 2 DistilBERT as a stretch goal.

For MS4, decide this explicitly based on compute. The lower-risk path is to keep the Stage 2 GRU fixed and first test class weighting, soft aggregation, threshold tuning, and improved Stage 1 emotion probabilities. Add Stage 2 DistilBERT only if there is enough time to evaluate it cleanly.
