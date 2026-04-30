# Original Project Proposal - Markdown Rewrite

Source PDF: `artifacts/original_project_proposal.pdf`

## Author

Ambika Grover (`ambikagrover@college.harvard.edu`)

## Data

1. **Reddit MBTI Dataset** - public Reddit posts labeled with self-reported MBTI personality types.
2. **Emotion-Balanced Twitter Dataset** - public short texts labeled with six emotions: sadness, joy, love, anger, fear, and surprise.

Project-local interpretation:

- Reddit source used by the team: Kaggle `minhaozhang1/reddit-mbti-dataset`.
- Emotion source used by the team: Hugging Face `AdamCodd/emotion-balanced`.
- KaggleHub API access for the Reddit dataset has been verified in `../../data/README.md`; the project does not rely on a local raw CSV.

## Background and Motivation

Personality theory suggests that individuals differ in stable emotional tendencies. Rather than directly predicting personality from text, this project proposes first learning a robust model of emotional expression and then using it to measure how emotional language differs across personality types.

This approach treats emotion as a measurable intermediate representation rather than a secondary outcome, allowing the project to study how personality is reflected in affective language patterns.

## Problem

The project aims to explore whether emotional distributions differ systematically across personality categories and whether emotional features improve personality modeling.

The broad plan is:

1. Train a language model to classify emotional content.
2. Apply it to Reddit posts from users with known MBTI types.
3. Aggregate emotional signals at the user level.
4. Analyze whether personality types exhibit distinct emotional language profiles.

## Scope and Methods

The proposal planned to use course-covered methods including:

- transformer-based language models
- transfer learning
- multi-class classification
- representation analysis
- clustering

## Concerns and Limitations

- MBTI labels are self-reported and imperfect measures of personality.
- Emotion classifiers trained on one platform may not perfectly generalize to another.
- Inference is based on published text, which may not translate perfectly to an individual's true emotions.
- The emotion dataset is effectively anonymized.
- The Reddit MBTI dataset is pseudonymized.
- No attempt should be made to identify users.
- Results should be reported only at aggregate level.

## Later Rescope

MS2 and MS3 narrowed the proposal into a cleaner comparative question:

> Do text plus emotion-informed features improve author-level prediction of the four binary MBTI dimensions over text-only baselines?

This rescope was driven by the severe 16-class MBTI imbalance, repeated authors, and the need for author-level evaluation.
