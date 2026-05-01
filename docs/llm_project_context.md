# LLM Project Context

Use this file as the compact project briefing. For agent behavior and read-order rules, see `../AGENTS.md`.

## Source Basis

This briefing summarizes the original artifacts, submitted MS2/MS3 materials, presenter notes, TA feedback, team clarifications, the KaggleHub API access check recorded in `../data/README.md`, and the completed MS4 modeling artifacts.

## Identity

- **Project:** Project 66, AC 209b final project
- **Team:** Harry Hu, Tom Shan, Wendy Wang, Kemeng Zhang
- **Original proposal author:** Ambika Grover. The team adopted an existing proposal rather than writing a new one for MS0.
- **High-level topic:** Learn emotion signals from a labeled emotion dataset, apply them to Reddit posts labeled by self-reported MBTI type, and test whether emotion features improve MBTI prediction.

## Data Sources

1. **Emotion source dataset:** `AdamCodd/emotion-balanced` on Hugging Face.
   - Roughly 20,000 short social texts.
   - Six labels: sadness, joy, love, anger, fear, surprise.
   - Local note: this Python environment currently does not have `datasets` installed, but the dataset can be read through `datasets`, pandas `hf://`, or Hugging Face Hub files.

2. **Reddit MBTI target dataset:** `minhaozhang1/reddit-mbti-dataset` on Kaggle.
   - Default access path: KaggleHub API.
   - Tested API call: `kagglehub.dataset_load(KaggleDatasetAdapter.PANDAS, "minhaozhang1/reddit-mbti-dataset", "reddit_post.csv", pandas_kwargs={"nrows": 3})`.
   - API test succeeded on 2026-04-30.
   - Dataset columns used by this project: `author`, `body`, `mbti`.
   - Expected scale from project notebooks: 13,028,455 usable rows after dropping 180 blank bodies; 11,773 authors.

## Current Modeling Story

The project started as direct MBTI prediction but was rescoped after MS2 EDA:

- Direct 16-way MBTI classification is too imbalanced.
- Repeated authors make post-level random splits invalid.
- Single posts are noisy, so evaluation should happen at the author level.
- Emotion features should be tested as an additive signal, not assumed useful.

Current research question:

> Do text plus emotion-informed features improve author-level prediction of the four binary MBTI dimensions over text-only baselines?

## MS2 Takeaways

- Emotion data is clean and balanced.
- Reddit data is very large but imbalanced and author-concentrated.
- Use author-level splits.
- Focus on four binary dimensions: `E/I`, `N/S`, `F/T`, `J/P`.
- Compare direct text baselines against emotion-informed models.
- TA score: 9.7/10.
- TA requested clearer success metrics and more concrete baseline model choices.
- TA also noted text overflow / text-heavy slides and late slide upload issue.

## MS3 Takeaways

- Extended EDA set thresholds:
  - Drop posts with fewer than 5 words.
  - Require at least 20 posts per author.
  - Cap at 200 posts per author.
  - Use soft six-dimensional emotion probabilities, not hard emotion labels.
- Baseline setup:
  - Stage 1 emotion classifier.
  - Baseline 1: text-only GRU for four MBTI logits.
  - Baseline 2: same GRU plus six emotion probabilities.
  - Author-level majority vote aggregation.
- Result:
  - Emotion RNN reached about 0.897 validation accuracy and 0.908 test accuracy.
  - B1 and B2 plateaued around validation BCE 0.520.
  - E/I, N/S, and J/P collapsed to majority class at balanced accuracy 0.500.
  - F/T showed high minority precision but near-zero recall.
  - B2 did not meaningfully outperform B1.
- TA score: 19.8/20.
- TA issue: notebook narrative has broken cross-references, for example references to section 3.1.4 even though section 3.1 has no sub-subsections.
- Internal consistency note: MS3 slides mention a DistilBERT Stage 2 text encoder as part of the final pipeline, while the submitted notebook treats Stage 2 DistilBERT as a stretch goal. The current MS4 design resolves this by excluding supervised post-level Stage 2 transformer fine-tuning from the report-facing mainline and using frozen transformer author representations instead.
- Team-work note: any person-by-person task split shown in MS3 slides/scripts was tentative and written to satisfy presentation requirements. Do not treat it as the actual MS4 division of labor.

## MS4 Direction

MS4 has completed corrected GRU/TF-IDF baseline results and MiniLM transformer-author results in `../milestones/ms4_final_modeling_deliverables/code/` and `../milestones/ms4_final_modeling_deliverables/report/results/`. The current implemented experiment summary is recorded in `../milestones/ms4_final_modeling_deliverables/ms4_experiment_summary.md`.

The final deliverables should still build from the MS3 diagnosis:

- Class-weighted BCE for the legacy GRU MBTI head.
- Soft author-level aggregation by averaging post-level probabilities.
- Per-dimension thresholds tuned on validation, with F1 and balanced accuracy tracked.
- Cached DistilBERT emotion probabilities so repeated MBTI runs do not recompute Stage 1.
- Report author-level metrics: balanced accuracy, F1, precision, recall, ROC-AUC, and PR-AUC/average precision where continuous scores exist.
- Load Reddit data with KaggleHub in code.

The updated MS4 design adds a stricter transformer author-representation layer:

- Treat emotion probabilities as text-derived transferred representations, not independent emotion measurements or causal mediators.
- Use matched comparisons: `text + real emotion` minus `text-only`.
- Add shuffled-emotion negative controls to test whether emotion gains depend on aligned author-level emotion features.
- Add activity/length controls to check whether gains are proxies for verbosity, retained-post count, or truncation exposure.
- Add an emotion-only author baseline to decide whether emotion is standalone signal, complementary signal, or compressed text proxy.
- Compare frozen transformer author features and set/attention author transformers against mean-pooling baselines.
- Exclude supervised post-level transformer fine-tuning from the MS4 mainline because it changes the estimand and reintroduces post-label noise.

Avoid treating the updated design as a model-capacity contest. The strongest story is a controlled author-level test of whether transferred emotion representations add incremental information beyond matched text representations. The current result is that emotion-derived features are informative but not robustly incremental beyond text representations.
