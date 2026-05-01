# MS4 Tentative Modeling and Notebook Design

Status: **tentative design, not completed MS4 results**.

This document records the proposed scientific plan for the MS4 code notebook. It is meant to guide implementation, report writing, and video planning, but should be revised if experiments reveal that a different design is more defensible.

## Design Thesis

The strongest MS4 story is not simply that the team will use larger models. The stronger and more scientifically defensible story is:

> MS3 did not prove that emotion features are useless. It showed that vanilla class-imbalanced training and hard majority-vote author aggregation can hide minority-class signal. MS4 should test whether targeted fixes make emotion-informed features useful for author-level MBTI prediction.

This design directly follows the MS3 diagnosis:

- MS3 Baseline 1 and Baseline 2 plateaued around validation BCE 0.520.
- `E/I`, `N/S`, and `J/P` collapsed to the majority class at balanced accuracy 0.500.
- `F/T` showed high precision but near-zero recall, indicating weak minority signal existed but was discarded.
- Adding six emotion probabilities alone did not help under vanilla BCE and hard majority vote.

Therefore, MS4 should first fix the training and aggregation conditions that caused collapse, then test whether emotion features add value over a corrected text-only model. The key final comparison is not "new model versus old broken baseline"; it is "fixed text-only versus fixed text-plus-emotion under the same preprocessing, split, loss family, aggregation, and threshold-tuning protocol."

## Hardware and Runtime Assumption

Use the local Mac with MPS GPU support as the default compute path.

Local check on 2026-04-30:

- MacBook Pro `Mac17,6`
- Apple M5 Max
- 18 CPU cores
- 40-core Apple GPU
- 64 GB memory
- macOS 26.4.1
- Clean Python 3.12.11 arm64 environment
- torch 2.11.0
- `torch.backends.mps.is_available() == True`
- Tensor matmul on `mps:0` succeeded

Implication for the notebook:

- The submitted notebook should not hard-fail on machines without MPS when it is only loading cached predictions, running smoke tests, or regenerating tables and figures.
- If `RUN_FULL_TRAINING = True`, assert MPS availability near the top of the training path.
- If MPS is unavailable for full training, stop and fix the Python/torch/Jupyter environment rather than silently training large neural models on CPU.
- CPU is acceptable for lightweight data diagnostics, dataframe summaries, plotting, cached evaluation, and smoke-test execution.

PyTorch's MPS backend documentation should be cited for the device setup: <https://docs.pytorch.org/docs/stable/notes/mps>.

## Data Design

Use the two public dataset APIs rather than local raw data files:

1. Emotion source dataset: Hugging Face `AdamCodd/emotion-balanced`.
2. Reddit MBTI target dataset: Kaggle `minhaozhang1/reddit-mbti-dataset`, file `reddit_post.csv`.

Retain the MS3 modeling filters:

- Drop blank Reddit posts.
- Drop posts with fewer than 5 words.
- Require at least 20 posts per author.
- Cap at 200 posts per author.
- Derive four binary MBTI dimensions: `E/I`, `N/S`, `F/T`, `J/P`.
- Split train/validation/test by author, never by post.
- Prefer stratified author-level splitting by 16-type MBTI label because the 16-type label implies all four binary dimensions.
  - If rare types make simple stratification unstable after filtering, use iterative stratification over the four binary labels or another documented author-level approximation.
  - Always show a split balance heatmap for all four binary dimensions.

Add a Stage 2 token truncation audit before training:

- Tokenize the masked Reddit modeling corpus with the same tokenizer used by the Stage 2 GRU models.
- Report token-length median, p90, p95, p99, and the share of posts longer than the planned Stage 2 maximum length of 128.
- Report author-level truncation exposure, including the share of authors with at least one truncated post and the average within-author truncated-post share.
- Check whether truncation is materially uneven across train/validation/test splits or MBTI dimensions.
- Keep Stage 2 max length at 128 for the core controlled comparison unless the audit shows severe truncation.
- If truncation is severe and time permits, run a small validation-only sensitivity check comparing max length 128 versus 256 on the fixed text-only GRU only. Do not expand this into a full model grid unless 256 clearly improves validation metrics enough to justify the added compute.

Add MS4-specific preprocessing audits:

- Detect and mask explicit MBTI label terms in Reddit post text before modeling.
- Use a conservative regular expression for masking.
- Terms should include the 16 four-letter MBTI types and explicit label phrases such as `MBTI`, `Myers-Briggs`, and `Myers Briggs`.
- Do not initially mask broad ordinary-language words such as `introvert`, `extrovert`, `feeling`, `thinking`, `judging`, or `perceiving`; those may be real language signal rather than direct label leakage.
- A quick 200,000-row sample check found:
  - 2.76% of posts contained at least one MBTI-related label term.
  - 18.46% of authors had at least one such post.
  - 1.34% of posts directly mentioned the author's own MBTI type.
- Check whether any author appears with conflicting self-reported MBTI labels.
  - If conflicts exist, document the count and use a deterministic rule before splitting, such as dropping conflicted authors or assigning the author's modal valid type.
  - Do not allow the same author to contribute different labels across rows after preprocessing.

This leakage is not large enough to invalidate the dataset, but it is large enough that ignoring it would weaken the final report. Masking makes the downstream modeling claim cleaner.

All headline MS4 models should use the same masked corpus, same author split, same label encoding, and same metric pipeline. MS3 results can be reported as historical diagnosis, but the final comparison table should not mix unmasked historical baselines with masked final-model results unless that distinction is clearly labeled.

## Primary Research Question

Do emotion-informed features improve author-level prediction of the four binary MBTI dimensions over text-only baselines after correcting for class imbalance and aggregation loss?

The phrase "after correcting" matters. MS3 already showed that naive emotion concatenation does not help under a collapsed training setup. MS4 should not repeat that same test and call it final.

Operationally, the causal comparison should be:

> corrected text-only GRU versus corrected GRU plus emotion probabilities.

This requires a fixed text-only baseline under the same weighted-loss, soft-aggregation, and threshold-tuning protocol as the emotion-informed models.

## Scope for MS4

The plan should be executable by the May 12, 2026 MS4 deadline. The core scope is deliberately narrower than the full list of possible model upgrades.

Core MS4 work:

- Build one masked, author-split modeling corpus with conservative MBTI leakage masking and split-balance diagnostics.
- Train or load the fixed text-only GRU baseline.
- Train or load the fixed RNN-emotion GRU model.
- Train or load the fixed DistilBERT-emotion GRU model.
- Include a cheap author-level TF-IDF plus logistic regression or linear SVM sanity baseline.
- Report author-level controlled comparisons with cached predictions, threshold tuning, and imbalance-aware metrics.

Optional if time permits:

- Paired bootstrap confidence intervals over test authors.
- One or two targeted ablations that isolate the largest observed effect.
- Additional threshold-sensitivity plots if balanced-accuracy and F1 thresholds differ materially.

Future work, not core MS4:

- Stage 2 DistilBERT MBTI text encoder.
- Learned author-level aggregators.
- Reddit-side emotion calibration.
- Larger post caps or transformer-heavy ablation grids.

## Model Families to Compare

### 1. Majority Baseline

Predict the training-split majority label for each MBTI dimension.

Purpose:

- Establish the imbalance floor.
- Prevent raw accuracy from looking deceptively strong, especially on `N/S`.

### 2. Linear Author-Level Sanity Baseline

Train a fast author-level TF-IDF plus logistic regression or linear SVM model on the same MS4 train/validation/test split.

Suggested representation:

- Aggregate each author's retained posts into one author document, or average post-level TF-IDF vectors by author.
- Train one binary linear classifier per MBTI dimension.
- Tune regularization on validation data only.

Purpose:

- Provides a cheap but strong non-neural baseline.
- Checks whether the GRU family is adding value beyond a simple lexical model.
- Gives the report an interpretable sanity check that can often be trained quickly even when neural experiments are slow.

### 3. MS3 Baseline 1: Text-Only GRU

Reuse or reproduce the MS3 text-only GRU with hard majority-vote author aggregation, clearly labeled as a historical baseline.

Purpose:

- Historical baseline, not a row in the main MS4 controlled comparison if it is not rerun on the masked MS4 corpus.
- Shows what direct text modeling did before the MS4 fixes.

### 4. MS3 Baseline 2: GRU Plus RNN Emotion Probabilities

Reuse or reproduce the MS3 GRU plus six-dimensional RNN emotion probabilities, again with hard majority-vote aggregation and clearly labeled as a historical baseline.

Purpose:

- Historical emotion baseline, not a row in the main MS4 controlled comparison if it is not rerun on the masked MS4 corpus.
- Shows that emotion concatenation alone was insufficient.

### 5. Fixed Text-Only Baseline: GRU Plus Weighted BCE and Soft Aggregation

Train the same Stage 2 GRU text pathway on the MS4 masked corpus, but apply the MS4 training and aggregation fixes:

- Stage 2 loss: class-weighted `BCEWithLogitsLoss(pos_weight=...)`.
- Author aggregation: mean post-level probabilities per author.
- Decision rule: one validation-tuned threshold per MBTI dimension.

Purpose:

- This is the most important control model.
- It answers how much improvement comes from fixing class imbalance and aggregation before adding emotion features.
- Without this row, MS4 cannot cleanly claim that emotion probabilities add incremental value.

### 6. Fixed RNN-Emotion Model: RNN Emotion Plus Weighted BCE and Soft Aggregation

Keep the MS3 RNN emotion probabilities and Stage 2 GRU architecture, but change:

- Stage 2 loss: class-weighted `BCEWithLogitsLoss(pos_weight=...)`.
- Author aggregation: mean post-level probabilities per author.
- Decision rule: one validation-tuned threshold per MBTI dimension.

Purpose:

- Compare against the fixed text-only baseline to estimate the incremental value of the original RNN emotion channel under corrected training conditions.
- Isolate whether MS3's negative emotion result was mainly caused by the loss/aggregation setup rather than by the RNN emotion encoder itself.

### 7. Final Model: DistilBERT Emotion Plus Fixed Stage 2

Use a fine-tuned DistilBERT emotion classifier for Stage 1, then feed six soft emotion probabilities into the same Stage 2 GRU setup used above:

- Stage 1: DistilBERT emotion classifier trained on `AdamCodd/emotion-balanced`.
- Stage 2: GRU text encoder plus six emotion probabilities.
- Loss: class-weighted BCE.
- Aggregation: soft author-level mean probability.
- Thresholds: tuned on validation only.

Purpose:

- Tests whether stronger emotion probabilities add value over both the fixed text-only baseline and the fixed RNN-emotion model.
- Keeps the Stage 2 text pathway fixed so improvements are not conflated with a stronger text encoder.
- Treat Stage 1 emotion test accuracy as a quality-control diagnostic, not as evidence that downstream MBTI prediction improved. Downstream author-level comparison against the fixed text-only baseline is the deciding evidence.

### 8. Future Work: Stage 2 DistilBERT Text Encoder

Treat a transformer Stage 2 MBTI text encoder as future work, not a core MS4 experiment.

Reason:

- It may improve performance, but it changes the text encoder and confounds the question "does emotion help?"
- It is more compute-intensive on the 1.6M-post modeling corpus.
- It is likely to distract from the controlled MS4 comparison given the deadline.

## Why This Is Scientifically Defensible

This design is defensible because it follows controlled-experiment logic.

First, it keeps the evaluation unit aligned with the prediction claim. MBTI labels are author-level self-reports, so evaluation should be author-level. Post-level random splits would leak author identity and overstate performance.

Second, it addresses class imbalance using metrics and loss functions appropriate to the problem. Raw accuracy is not meaningful when `N/S` is roughly 93/7 at the author level. Balanced accuracy, ROC-AUC, F1, precision, and recall are more informative. Class-weighted BCE directly targets the majority-class collapse observed in MS3.

Third, it separates model-capacity changes from training-procedure changes. Before concluding that emotion features help or fail, MS4 tests whether the collapsed Stage 2 head can be repaired using weighted loss and soft aggregation in a text-only model. Only then does it add the RNN emotion channel and swap in a stronger emotion encoder.

Fourth, it tunes thresholds only on validation data. Test data should be used once for final reporting. This prevents accidental test-set optimization. The threshold objective should be fixed before looking at test results, with balanced accuracy as the default primary objective and F1 reported alongside it.

Fifth, it adds a leakage audit. Masking explicit MBTI terms makes the final claim about language and emotion patterns stronger than a claim that may partly depend on users naming their type in posts.

Sixth, it keeps the training unit closer to the evaluation unit. Since labels are author-level but Stage 2 trains on posts, the implementation should reduce prolific-author dominance using either author-balanced sampling or per-post loss weights proportional to `1 / posts_per_author` within the training split.

The imbalance corrections should be treated as a coherent training recipe rather than three independent knobs tuned until test performance looks good:

- Class-weighted BCE handles label skew.
- Author-balanced sampling or per-author post loss weights handle prolific-author dominance.
- Validation threshold tuning calibrates final decisions after continuous author-level scores are produced.

Predeclare the recipe-selection protocol before opening test results:

- Compare at most two class-weight variants on validation: strict inverse-frequency `pos_weight = neg / pos` versus milder square-root weights `pos_weight = sqrt(neg / pos)`.
- Keep the author-balancing choice fixed while comparing the two class-weight variants, so the comparison isolates the strength of class weighting rather than changing multiple training choices at once.
- Select one recipe using validation mean balanced accuracy across the four MBTI dimensions as the primary criterion.
- Use macro F1 and minority-class recall/precision tradeoff only as tie-breakers or diagnostics if the balanced-accuracy difference is negligible.
- After selecting the recipe, apply it unchanged to all fixed GRU models and evaluate the test set once.

If per-post loss weights are used, class weights should be computed from the same effective training distribution used by the loss, not from an incompatible full-dataset or unweighted post count. If author-balanced sampling is used instead, document the sampling rule and compute class weights from the corresponding training-author distribution.

Seventh, it supports either positive or negative final outcomes:

- If fixed emotion models outperform the fixed text-only baseline, the conclusion can be that emotion features add incremental author-level signal once imbalance and aggregation are handled.
- If fixed emotion models do not outperform the fixed text-only baseline, the conclusion can still be strong: targeted fixes recovered some minority signal, but transferred emotion probabilities add limited incremental value beyond text under controlled author-level evaluation.

## Metrics

Report all metrics at the author level.

Primary metrics:

- Balanced accuracy per MBTI dimension.
- ROC-AUC per MBTI dimension using continuous author-level scores.
- PR-AUC or average precision per MBTI dimension, especially for skewed dimensions such as `N/S`.
- F1 per MBTI dimension.
- Minority-class precision and recall per MBTI dimension.

Secondary metrics:

- Raw accuracy, clearly marked as secondary.
- Confusion matrices.
- Bootstrap confidence intervals over authors if time permits.

For thresholded predictions:

- Tune one threshold per dimension on validation.
- Primary threshold objective: validation balanced accuracy.
- Secondary threshold sensitivity check: validation F1, reported if the selected thresholds differ materially.
- Report the selected thresholds.
- Apply those thresholds unchanged to test.

For uncertainty:

- If time permits, report paired bootstrap confidence intervals over test authors for the final model versus the fixed text-only baseline.
- Bootstrap differences are more useful than standalone intervals because the headline question is whether emotion improves over the corrected text-only control.

For result tables:

- Keep historical MS3 diagnosis separate from the MS4 controlled comparison.
- Historical table: majority baseline, MS3 B1, and MS3 B2, clearly labeled as unmasked historical references if they are not rerun on the MS4 masked corpus.
- Main MS4 table: linear author-level sanity baseline, fixed text-only GRU, fixed RNN-emotion GRU, and fixed DistilBERT-emotion GRU.
- The main table should be compact: per dimension, report balanced accuracy, F1, minority recall, PR-AUC or average precision, and ROC-AUC.
- Avoid ranking models primarily by raw accuracy.

## Recommended Notebook Structure

The main notebook should be readable as a complete final-project artifact. Helper functions should live in `.py` files and be imported by the notebook.

1. **Title and Reproducibility Header**
   - Course, project number, team members.
   - Short statement of the final research question.
   - Runtime assumptions and package versions.

2. **Hardware and Environment Check**
   - Print whether `torch.backends.mps.is_available()`.
   - Print Python, torch, transformers, datasets, pandas, sklearn versions.
   - Set seeds.
   - Fail loudly if MPS is unavailable only when `RUN_FULL_TRAINING = True`.

3. **Project Framing**
   - One concise paragraph explaining the MS3 failure diagnosis.
   - One diagram of the Stage 1 -> Stage 2 -> author aggregation pipeline.

4. **Data Access**
   - Load Reddit through KaggleHub.
   - Load emotion dataset through Hugging Face.
   - Show dataset schema and row counts.

5. **Preprocessing**
   - Clean text.
   - Mask explicit MBTI terms.
   - Apply MS3 filters.
   - Derive binary labels.
   - Audit author-level MBTI consistency and resolve or drop conflicted authors before splitting.
   - Build stratified author-level split.

6. **Compact EDA Carry-Forward**
   - Do not repeat all MS2/MS3 EDA.
   - Show only the EDA that materially justifies MS4 decisions:
     - class imbalance by dimension
     - posts per author before and after cap
     - split balance heatmap
     - MBTI label leakage audit
     - author-label consistency audit if conflicts are found
     - Stage 2 token truncation audit for max length 128

7. **Stage 1 Emotion Classifier**
   - Fine-tune DistilBERT on the emotion dataset.
   - Report validation/test accuracy and macro-F1.
   - State explicitly that emotion-source accuracy is diagnostic only; downstream MBTI comparison is the final evidence.
   - Show confusion matrix.
   - Generate and cache Reddit emotion probabilities.
   - Compare Reddit mean emotion distribution to source emotion labels.

8. **Stage 2 MBTI Models**
   - Define the model comparison table.
   - Train or load the author-level TF-IDF plus logistic regression or linear SVM sanity baseline.
   - Train or load MS3 historical B1/B2 references, fixed text-only baseline, fixed RNN-emotion model, and final DistilBERT-emotion model.
   - Compare inverse-frequency versus square-root class weights on validation only, then lock one recipe before test evaluation.
   - Compute class weights from the training split's effective author-level distribution, not from the full dataset.
   - Use author-balanced sampling or per-post loss weights to reduce prolific-author dominance during post-level training, and keep that choice fixed across fixed GRU models.
   - Plot loss curves.
   - Show validation threshold tuning curves.
   - If the truncation audit is severe and time permits, run a fixed text-only GRU validation sensitivity check for max length 128 versus 256 before locking the core max length.

9. **Main Results**
   - Historical diagnosis table for MS3 B1/B2 and the majority-class floor.
   - Main controlled author-level metric table for the MS4 masked-corpus models.
   - One model-comparison plot by MBTI dimension.
   - Confusion matrices for the final model.
   - ROC or precision-recall curves, especially for imbalanced dimensions.

10. **Ablations**
    - Treat these as candidate ablations; run the highest-value one or two if the core models finish first.
    - Remove class weighting.
    - Use hard vote instead of soft aggregation.
    - Use RNN emotion probabilities instead of DistilBERT emotion probabilities.
    - No emotion features under the fixed protocol.

11. **Interpretation**
    - Explain which change moved the model off the majority-class floor.
    - Discuss whether emotion adds incremental signal over the fixed text-only baseline.
    - Report limitations: self-reported labels, domain shift, missing subreddit/time metadata, MBTI construct limitations.

12. **Reproducibility and References**
    - List data sources.
    - List core libraries.
    - Cite PyTorch MPS, Hugging Face Transformers/Datasets, KaggleHub, and relevant papers.

## Suggested Helper Modules

Place implementation helpers under the MS4 code area, for example `code/src/`:

- `config.py`: constants, labels, paths, seeds, runtime flags.
- `data.py`: KaggleHub/Hugging Face loading, filtering, split creation.
- `preprocessing.py`: text cleaning, MBTI term masking, label derivation, author-label consistency checks.
- `emotion.py`: Stage 1 training, inference, and probability caching.
- `models.py`: Stage 2 GRU architecture and model builders.
- `training.py`: MPS device checks, training loops, early stopping.
- `evaluation.py`: author aggregation, threshold tuning, metric calculation, bootstrap intervals.
- `cache.py`: cache naming, metadata, and loading checks for model outputs and intermediate tables.
- `viz.py`: plotting theme and reusable figure functions.

The notebook should orchestrate the analysis, not contain large reusable functions.

## Runtime Usability Requirements

The implementation should be designed for long-running local MPS experiments.

The submitted notebook should support two paths:

- Default path: load cached predictions and metrics, run smoke tests, and regenerate all final tables and figures.
- Full-training path: train or rerun inference when `RUN_FULL_TRAINING = True`; this path may require MPS and longer wall-clock time.
- Use a visible runtime flag such as `RUN_FULL_TRAINING = False` for the submitted notebook and document how to enable full retraining.
- Include a small `SMOKE_TEST = True` path or equivalent that validates data loading, preprocessing, model construction, aggregation, and metric code on a tiny subset.
- Training and long inference steps should be resumable after interruption.
- Expensive intermediate outputs, especially Reddit emotion probabilities and Stage 2 post-level logits, should be cached and reused.
- Cache files should include enough metadata to prevent accidental reuse across incompatible preprocessing or model versions, including at minimum split id, masking status, emotion model id, and label encoding.
- The notebook should show live progress for long steps, such as training epochs, validation passes, Reddit inference shards, and ablation runs.
- Progress reporting should make it clear which experiment is running, how far it has progressed, and where outputs are being written.

These are design requirements for the MS4 implementation, not completed features.

## Code Notebook Submission Package

The MS4 code deliverable can be packaged as a zipped folder if the notebook depends on helper scripts and compact final artifacts. The package should be organized so a grader can identify the main notebook immediately and run it without hunting through the project tree.

Submission package principles:

- Include the clearly named main notebook, helper code it imports, dependency/runtime notes, and any compact final artifacts needed for the default cached-evaluation path.
- Keep submitted artifacts compact and author-level where possible, so the notebook can regenerate final tables and figures without rerunning full Reddit inference or model training.
- Include a small manifest or README note describing where the final artifacts came from, which preprocessing/split/model settings they correspond to, and how to rerun the expensive path if needed.
- Exclude raw datasets and large intermediate caches unless course staff explicitly requests them. Large local caches such as post-level emotion probabilities, post-level logits, tokenized corpora, and model checkpoints should be documented but not treated as required submitted files.
- The default submitted-notebook path should remain `RUN_FULL_TRAINING = False`: load compact final artifacts, run smoke checks, and regenerate results. Full training should be documented as optional.

## Visualization Plan

Use a small number of polished, high-information figures.

Recommended figures:

1. Class imbalance bar chart for the four binary MBTI dimensions.
2. Author contribution distribution before and after the 200-post cap.
3. Train/validation/test split balance heatmap.
4. MBTI leakage audit bar chart before masking.
5. Author-label consistency audit summary, if any conflicts are found.
6. Stage 2 token truncation audit summary for max length 128.
7. Linear TF-IDF baseline comparison against the fixed GRU models.
8. Stage 1 emotion confusion matrix.
9. Source-vs-Reddit emotion distribution comparison.
10. Stage 2 loss curves.
11. Validation threshold tuning curves by dimension.
12. Main model comparison plot by metric and dimension, with fixed text-only as the key control.
13. Final model confusion matrices.
14. Ablation plot showing the effect of each targeted fix, if time permits.

Avoid text-heavy figures. Captions should explain the modeling decision supported by each visualization.

## Compute Plan

Local MPS should be sufficient for the core plan:

- DistilBERT Stage 1 on about 20,000 emotion rows.
- Batched emotion inference over the filtered Reddit modeling corpus.
- GRU Stage 2 training on the capped author-level dataset.

Recommended local settings:

- Stage 1 DistilBERT max length: 64.
- Stage 1 batch size: start with 32 or 64 and adjust for MPS memory.
- Stage 1 epochs: 3-5 with early stopping.
- Stage 2 GRU max length: 128 for the core comparison, with a reported truncation audit and an optional fixed text-only 128 versus 256 validation sensitivity check if truncation is severe.
- Stage 2 batch size: start with 512 or 1024 and adjust for memory.
- Cache Stage 1 Reddit emotion probabilities immediately after inference.

Rent GPU only if:

- The team wants multiple transformer ablations.
- Local MPS inference over the full capped Reddit corpus is too slow for the MS4 timeline.

The core MS4 plan should not require rented GPU compute if implemented efficiently. Stage 2 DistilBERT should remain future work rather than a reason to rent GPU for the core submission.

## Risks and Mitigations

| Risk | Mitigation |
|---|---|
| MPS unavailable in notebook | Allow cached evaluation and smoke tests on CPU; require MPS only when `RUN_FULL_TRAINING = True`. |
| Plan becomes too large for the deadline | Keep Stage 2 DistilBERT, learned aggregators, Reddit-side calibration, and large ablation grids as future work. |
| Stage 2 max length 128 truncates too much Reddit text | Report post-level and author-level truncation rates; if severe, run a fixed text-only 128 versus 256 validation sensitivity check before locking the core max length. |
| Weighted BCE over-corrects `N/S` | Compare strict inverse-frequency weights with milder square-root weights on validation only, select by mean balanced accuracy with documented tie-breakers, and keep one locked recipe for test. |
| Imbalance corrections become inconsistent | Define one training recipe: class weights from the effective training distribution, fixed author-balancing or loss-weighting, and validation-only threshold tuning. |
| Apparent final-model gain comes only from weighted BCE or soft aggregation | Include the fixed text-only baseline and compare emotion models against it under the same protocol. |
| Post-level training overweights prolific authors | Use author-balanced sampling or per-post loss weights based on each author's retained post count. |
| Thresholds look arbitrary | Predeclare validation balanced accuracy as the primary threshold objective, report thresholds, and apply them unchanged to test. |
| DistilBERT emotion model improves source test accuracy but not Reddit downstream performance | Treat as evidence of source-target domain shift; report honestly. |
| Neural models underperform a simple lexical baseline | Include the author-level TF-IDF linear baseline and report the result honestly. |
| Cached predictions become stale after preprocessing changes | Store cache metadata for masking status, split id, model id, and label encoding; invalidate mismatches. |
| Notebook becomes too long | Move helpers into `.py` modules and keep only orchestration, prose, and outputs in the notebook. |
| Final result is negative | Frame as a controlled negative result: fixes addressed MS3 collapse, but emotion transfer had limited incremental value. |

## References to Cite in MS4

- PyTorch MPS backend: <https://docs.pytorch.org/docs/stable/notes/mps>
- PyTorch `BCEWithLogitsLoss`: <https://docs.pytorch.org/docs/stable/generated/torch.nn.BCEWithLogitsLoss.html>
- Hugging Face text classification: <https://huggingface.co/docs/transformers/master/en/tasks/sequence_classification>
- Hugging Face dataset loading: <https://huggingface.co/docs/datasets/loading>
- Reddit MBTI dataset: <https://www.kaggle.com/datasets/minhaozhang1/reddit-mbti-dataset/data>
- Hugging Face emotion dataset: <https://huggingface.co/datasets/AdamCodd/emotion-balanced>
- scikit-learn balanced accuracy: <https://scikit-learn.org/stable/modules/generated/sklearn.metrics.balanced_accuracy_score.html>
- scikit-learn ROC-AUC: <https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_auc_score.html>
- scikit-learn TF-IDF vectorizer: <https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html>
- scikit-learn logistic regression: <https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html>
- DistilBERT paper: <https://arxiv.org/abs/1910.01108>
