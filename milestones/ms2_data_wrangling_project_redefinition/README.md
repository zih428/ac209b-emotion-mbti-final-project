# MS2 - Data Wrangling and Project Redefinition

## Status

Submitted. TA score: **9.7 / 10**.

## Files

| Path | Role |
|---|---|
| `requirements.md` | Markdown summary of MS2 requirements. |
| `notebook_summary.md` | LLM-friendly summary of the submitted MS2 notebook. |
| `submitted/ms2_submitted_summary.md` | Concise submitted-story summary. |
| `notes/ms2_presenter_script_part_1_2.md` | Presenter script. Not submitted, but important because slides omit spoken detail. |
| `artifacts/submitted/ms2_slides_submitted.pdf` | Original submitted MS2 slide deck. |
| `artifacts/submitted/ms2_eda_submitted.ipynb` | Original submitted MS2 notebook. |
| `artifacts/drafts/` | Earlier notebook and slide iterations. |
| `artifacts/requirements_ms2_data_wrangling_project_redefinition.png` | Screenshot of MS2 requirements. |

## Requirement Summary

Dates:

- Project assigned: Friday, March 27, 2026.
- Slides and accompanying notebook due at presentation time or by Friday, April 10, 2026 at 9:59pm, whichever came first.
- Presentation with TF due by Sunday, April 12, 2026 EOD.

Deliverables:

- 8-10 minute presentation with the TF.
- Slides used in presentation, submitted as PDF.
- `.ipynb` notebook used to access data, do wrangling, EDA, and visualizations.

Suggested content:

- Introduction and motivation.
- Data source, access, wrangling, preparation, visualizations, and insights.
- Project redefinition and rescoping.
- Next steps by team member.
- Future considerations, model classes, challenges, and questions for TF.

## Submitted Story

MS2 established:

- The emotion dataset is small, clean, balanced, and suitable as source supervision.
- The Reddit MBTI dataset is large but imbalanced and author-concentrated.
- The project should avoid post-level random splits because the same author appears many times.
- A direct 16-way MBTI classifier is not a good first target.
- The project should focus on four binary MBTI dimensions and author-level evaluation.

Revised MS2 research question:

> Do emotion-informed features improve author-level prediction of the four binary MBTI dimensions over direct text-to-MBTI baselines?

## TA Feedback

Overall: **9.7 / 10**

- Problem definition: 0.9 / 1.0. Well-scoped with a clear revised research question, but success framing should name imbalance-aware metrics such as ROC-AUC or F1.
- Motivation: 1.0 / 1.0.
- Data access and provenance: 1.0 / 1.0.
- Data loading: 1.0 / 1.0.
- Dataset understanding: 1.0 / 1.0.
- Summary / initial EDA: 1.0 / 1.0.
- Preprocessing / data preparation: 1.0 / 1.0.
- Analysis / patterns: 1.0 / 1.0.
- Visualization quality: 1.0 / 1.0.
- Insights and next steps: 0.9 / 1.0. Next steps should be more concrete on baseline choices and comparisons.
- Rescope / adaptation: 1.0 / 1.0.
- Presentation and slides: 0.8 / 1.0. Slides were organized and paced well, but some text overflowed slide boundaries, slides were text-heavy, and slides should be uploaded before presentation next time.

## Carry Forward to MS3/MS4

- Explicitly name success metrics: balanced accuracy, F1, precision, recall, and ROC-AUC where continuous scores exist.
- Keep slides visual and low-text.
- Make baseline model choices concrete and controlled.
- Submit slides before presentation.
