# MS3 Notebook Summary

Source artifact: `artifacts/submitted/ms3_eda_baseline_pipeline_submitted.ipynb`

## Purpose

The MS3 notebook extends EDA, implements matched baselines, interprets their failure modes, and proposes the final modeling pipeline.

## Metadata

- Canvas project number: 66.
- Team: Harry Hu, Tom Shan, Wendy Wang, Kemeng Zhang.
- Notebook title: AC 209b Milestone 3: EDA, Baseline Modeling, and Pipeline Development.

## Structure

1. Problem statement refinement and introduction.
2. Comprehensive EDA review.
3. Baseline model selection and justification.
4. Results interpretation and analysis.
5. Final model pipeline and future directions.

## Refined Research Question

Do text plus emotion-informed features improve author-level prediction of the four binary MBTI dimensions (`E/I`, `N/S`, `F/T`, `J/P`) over direct text-based baselines?

## Extended EDA Decisions

MS3 converted the MS2 qualitative findings into explicit modeling thresholds:

- Drop posts with fewer than 5 words.
- Require at least 20 posts per author.
- Cap at 200 posts per author.
- Use soft six-dimensional emotion probabilities instead of hard emotion labels.

Rationale:

- 1-4 word posts are dominated by short reactions and cross-author duplicates.
- Author-level features become more stable as post count grows; 20 posts is a practical reliability floor.
- A few prolific authors dominate the uncapped corpus; the 200-post cap reduces dominance while retaining about 1.7M posts.
- Source-target transfer from the emotion dataset to Reddit is plausible but uneven, so downstream models should preserve emotion uncertainty.

## Data Preparation

- Load Hugging Face emotion dataset.
- Load Kaggle Reddit MBTI dataset.
- Standardize schemas.
- Drop blank Reddit rows.
- Derive four MBTI dimension labels.
- Create a modeling DataFrame after word-count, author-count, and author-cap filters.
- Split train/validation/test by author to prevent leakage.

## Baseline Models

Stage 1:

- RNN emotion classifier trained on the balanced emotion dataset.
- Outputs six softmax emotion probabilities.

Baseline 1:

- Text-only GRU over Reddit posts.
- Predicts four MBTI logits.

Baseline 2:

- Same GRU text pathway as Baseline 1.
- Concatenates six Stage 1 emotion probabilities before the dense prediction head.
- Predicts four MBTI logits.

Evaluation:

- Author-level majority vote from post predictions.
- Metrics include accuracy, balanced accuracy, precision, recall, and F1.
- A majority-class baseline is reported for comparison.

## Key Results

- Stage 1 emotion RNN achieved about 0.897 validation accuracy and 0.908 test accuracy.
- B1 and B2 training curves were very similar.
- Both MBTI baselines plateaued around validation BCE 0.520.
- E/I, N/S, and J/P collapsed to majority-class prediction at author-level balanced accuracy 0.500.
- F/T showed a small above-floor signal, with high minority precision but recall around 2-4 percent.
- B2 did not meaningfully outperform B1; simple emotion-probability concatenation did not help in the vanilla setup.

## Interpretation

The negative result is informative:

- Vanilla BCE encourages class-prior solutions under strong label imbalance.
- Hard majority vote discards weak but real minority signals.
- Static emotion concatenation cannot rescue a collapsed MBTI head.
- The model treats posts independently, so within-author structure is not used beyond the final vote.

## Proposed Final Pipeline

The notebook prioritizes:

1. Class-weighted BCE.
2. Soft author-level aggregation by averaging post probabilities.
3. Per-dimension validation-tuned thresholds.
4. Improved Stage 1 emotion features, especially DistilBERT.

The notebook treats Stage 2 DistilBERT as a future/stretch option, while the submitted slides mention it as a fourth targeted upgrade. This scope should be resolved explicitly for MS4.

## Team-Work Note

Any named team-member assignments from MS3 are presentation artifacts, not binding MS4 responsibilities. They were tentative and written to meet the milestone requirement for next-step delineation.

## Known Issues

- TA noted broken cross-references. Example: Section 5.1 refers to 3.1.4, but Section 3.1 has no sub-subsections.
- Notebook and slides differ slightly on whether Stage 2 DistilBERT is core MS4 scope or a stretch goal.

## MS4 Relevance

This is the most important historical notebook for MS4. It provides:

- thresholds for the modeling dataset
- baseline design
- baseline failure diagnosis
- final model priorities
- evaluation metric direction

Do not rerun or rewrite every MS3 analysis unless necessary. Use it as the starting point for targeted MS4 fixes.
