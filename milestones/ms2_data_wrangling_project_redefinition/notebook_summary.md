# MS2 Notebook Summary

Source artifact: `artifacts/submitted/ms2_eda_submitted.ipynb`

## Purpose

The MS2 notebook validates access to both datasets, performs initial cleaning and EDA, and justifies the project rescope from direct 16-type MBTI classification to author-level binary-dimension prediction.

## Metadata

- Canvas project number: 66.
- Team: Harry Hu, Tom Shan, Wendy Wang, Kemeng Zhang.
- Notebook title: AC 209b Milestone 2: Data Access, Wrangling, and EDA.

## Data Access

The notebook uses:

- Hugging Face `AdamCodd/emotion-balanced`.
- Kaggle `minhaozhang1/reddit-mbti-dataset`.

The original notebook used `kagglehub.dataset_download(...)` for Reddit and pandas/Hugging Face Hub for the emotion dataset.

## Preprocessing

Emotion data:

- Loaded train, validation, and test JSONL splits.
- Mapped labels to six emotion names.
- Computed word and character counts.

Reddit data:

- Loaded `author`, `body`, `mbti`.
- Renamed `body` to `text` and `mbti` to `type`.
- Trimmed author names and upper-cased MBTI labels.
- Removed 180 blank Reddit posts.
- Found 0 invalid MBTI labels.
- Derived binary dimensions: `E/I`, `N/S`, `F/T`, `J/P`.
- Used the full dataset for count-level summaries.
- Used a fixed 250,000-post sample for text-length EDA.

## Key EDA Findings

- The emotion dataset is nearly perfectly balanced across sadness, joy, love, anger, fear, and surprise.
- The Reddit MBTI dataset is highly imbalanced across 16 types.
- INFP is the largest type, about 22.94 percent of all posts.
- ESFP is the smallest type, about 0.18 percent of all posts.
- Binary MBTI dimensions are more tractable than the full 16-way label space, although `N/S` remains heavily skewed.
- Reddit posts are longer and more variable than emotion texts, but both datasets are short-form social text.
- The same authors appear many times. The dataset has 11,773 authors and over 13 million usable posts.
- Author contribution is highly uneven, creating leakage risk for post-level random train/test splits.

## Modeling Implications

The notebook argues for:

- Author-level splits, not row-level splits.
- Four binary MBTI dimensions instead of 16-way type prediction.
- Author-level aggregation rather than single-post personality prediction.
- A comparative design: direct text-to-MBTI baselines versus emotion-informed models.

## Final MS2 Research Question

Do emotion-informed features improve author-level prediction of the four binary MBTI dimensions over direct text-to-MBTI baselines?

## Known Limitations

- MBTI labels are self-reported and noisy.
- Reddit source lacks subreddit, timestamp, and topic metadata.
- Platform mismatch exists between emotion data and Reddit data.
- Text is an incomplete proxy for emotion.
- The Stage 2 MBTI model depends on Stage 1 emotion classifier quality.

## MS4 Relevance

For MS4, this notebook mainly matters for data provenance, initial preprocessing, and the rescope rationale. Use MS3 for baseline results and final pipeline diagnosis.
