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

Therefore, MS4 should first fix the training and aggregation conditions that caused collapse, then test whether emotion features add value over corrected text-only models. However, MS4 should not remain a GRU-centered project. The final notebook should treat the existing GRU and TF-IDF work as the corrected baseline layer, then introduce transformer-based author representations as the main new modeling contribution.

The key final comparison is not "new model versus old broken baseline." It is:

> text-only versus text-plus-emotion under the same preprocessing, split, aggregation, and threshold-tuning protocol, repeated across increasingly strong text representations.

This creates three scientific layers:

1. **Corrected legacy layer:** fixed GRU text versus fixed GRU plus emotion.
2. **Frozen transformer representation layer:** transformer text embeddings versus transformer text embeddings plus emotion aggregates.
3. **Set/attention author transformer layer:** learned author-level aggregation of unordered post embeddings, again tested with and without emotion.

The optional full fine-tuned Stage 2 transformer should be treated as a supervised transformer ceiling, not the center of the project.

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

Do emotion-informed features improve author-level prediction of the four binary MBTI dimensions over text-only baselines after correcting for class imbalance, aggregation loss, and author-level evaluation constraints?

The phrase "after correcting" matters. MS3 already showed that naive emotion concatenation does not help under a collapsed training setup. MS4 should not repeat that same test and call it final.

The updated MS4 question should be operationalized across model families:

> Does explicit emotion add incremental author-level signal beyond text-only representations in corrected GRU, frozen-transformer, and set/attention-transformer settings?

This preserves the original emotion-informed MBTI question while making transformer modeling the main new contribution.

## Scope for MS4

The plan should be executable by the May 12, 2026 MS4 deadline while making the final project substantively transformer-centered. Treat the currently completed GRU/TF-IDF work as a baseline layer with experiment weight about `10`, then add transformer work with experiment weight about `15`.

Baseline layer to retain:

- Masked, author-split modeling corpus with leakage, conflict, split-balance, and truncation audits.
- Majority baseline.
- Author-level TF-IDF plus logistic regression baseline.
- Fixed text-only GRU.
- Fixed GRU plus emotion features.
- GRU weighting and length sensitivity diagnostics.
- Threshold tuning, bootstrap intervals, cached notebook outputs, and final metric pipeline.

New transformer-centered core:

- Frozen post-level transformer embedding cache for the masked Reddit modeling corpus.
- Author-level transformer feature table built from post embeddings.
- Frozen transformer author classifiers with and without emotion aggregates.
- Emotion-only author baseline using aggregate emotion probabilities.
- Set/attention author transformer over each author's unordered post embeddings, again with and without emotion.
- Activity/length-control variants for the key author-level emotion comparisons.
- Unified emotion-increment analysis across GRU, frozen-transformer, and set/attention-transformer families.

Stretch scope:

- One supervised post-level MiniLM or DistilBERT MBTI fine-tuning run as a transformer ceiling.
- A matching emotion-head version of that run only if compute and time remain.

Out of scope for the core:

- Multiple full fine-tuned transformer ablations.
- DeBERTa-base or other heavier encoders on the full Reddit corpus.
- Uncapped 13M-post transformer training or inference.
- Large multi-seed transformer grids.

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
- Checks whether neural models add value beyond a simple lexical model.
- Gives the report an interpretable sanity check that can often be trained quickly even when neural experiments are slow.

### 3. Corrected GRU Text Baseline

Train the Stage 2 GRU text pathway on the MS4 masked corpus using the corrected MS4 protocol:

- Author-balanced post loss or author-balanced sampling.
- Class-weighted `BCEWithLogitsLoss(pos_weight=...)`.
- Soft author-level aggregation by mean post probability.
- One validation-tuned threshold per MBTI dimension.

Purpose:

- Preserves continuity with MS3.
- Shows how much of the MS3 failure was fixed by better training and aggregation.
- Serves as the corrected legacy text-only control.

### 4. Corrected GRU Text Plus Emotion

Use the same corrected GRU setup but concatenate six emotion probabilities to the text representation.

Emotion options:

- Use the already cached DistilBERT emotion probabilities as the primary emotion channel.
- Keep any RNN-emotion version as a historical bridge or secondary diagnostic if available.

Purpose:

- Tests whether emotion adds value within the corrected GRU family.
- Keeps Stage 2 architecture fixed so the GRU text-versus-emotion comparison remains clean.

### 5. Frozen Transformer Text Embedding Baseline

Generate post-level text embeddings using a pretrained sentence or compact transformer encoder, then aggregate embeddings to author-level features.

Recommended starting point:

- `sentence-transformers/all-MiniLM-L6-v2`, because it maps sentences and short paragraphs to 384-dimensional vectors and is designed for embedding-style use.

Author features:

- Mean post embedding.
- Standard deviation of post embeddings.
- Optional post-count and length controls.

Classifier:

- Logistic regression, linear probe, or a small MLP trained at the author level.

Purpose:

- Introduces a modern transformer text representation without repeatedly fine-tuning a large transformer over 1.65M posts.
- Aligns training with author-level labels.
- Directly challenges the strong TF-IDF baseline.

### 6. Emotion-Only Author Baseline

Train an author-level classifier using only aggregate emotion features.

Suggested features:

- Mean and standard deviation of the six post-level emotion probabilities.
- Optional max or quantile summaries if the same feature set is used consistently across train/validation/test.

Purpose:

- Tests whether transferred emotion probabilities contain standalone author-level signal.
- Helps interpret later text-plus-emotion gains.
- If emotion-only is weak but text-plus-emotion improves over text-only, emotion is complementary rather than sufficient.
- If emotion-only is strong, the report should treat emotion features as a substantial proxy representation of the underlying text, not merely a small auxiliary signal.

### 7. Frozen Transformer Text Plus Emotion

Use the same author-level transformer text features and add emotion aggregates.

Emotion features:

- Mean and standard deviation of six post-level emotion probabilities.
- Optional emotion-distribution summaries such as max or quantiles if useful and kept consistent across train/validation/test.

Purpose:

- Cleanly tests whether explicit emotion adds value beyond a modern transformer text representation.
- This comparison is more important than comparing the final emotion model only against GRU.

### 8. Frozen Transformer Text Plus Emotion Plus Controls

Add simple author-level activity and length controls to the frozen transformer text-plus-emotion model.

Suggested controls:

- Retained post count.
- Mean or median post length.
- Share of posts above the model's token limit.
- Optional total retained token count.

Purpose:

- Checks whether any emotion gain is actually a proxy for verbosity, retained-post count, or truncation exposure.
- Provides a low-cost robustness check for the most important author-level transformer emotion comparison.

### 9. Set/Attention Author Transformer Text

Represent each author as an unordered set of retained post embeddings:

```text
author = {post_embedding_1, post_embedding_2, ..., post_embedding_k}
```

Train a small attention-based author aggregator over the post-embedding set, with one output head for the four MBTI dimensions.

Important design constraint:

- Reddit post order is not available in the current dataset, so the model should not rely on arbitrary positional encoding as if posts formed a temporal sequence.
- If a transformer encoder is used, it should be described as a permutation-aware or order-agnostic set/attention aggregator. Any positional feature should be omitted or justified as a non-temporal sampling/index artifact, not as chronology.

Purpose:

- Models within-author post structure rather than reducing each author to a simple mean vector.
- Trains directly at the author level, matching the label and evaluation unit.
- Provides the strongest architecture-level transformer contribution without raw-text transformer fine-tuning over every post.

### 10. Set/Attention Author Transformer Text Plus Emotion

Use the same set/attention author transformer, but append the six emotion probabilities to each post embedding before author-level aggregation.

Purpose:

- Tests whether emotion helps when the model can learn how to weight and combine an author's posts.
- Provides the cleanest transformer-based emotion-increment comparison:

```text
Set/Attention Transformer Text + Emotion
minus
Set/Attention Transformer Text
```

### 11. Set/Attention Author Transformer Text Plus Emotion Plus Controls

Add author-level activity and length controls to the set/attention transformer text-plus-emotion model.

Purpose:

- Mirrors the frozen-transformer robustness check.
- Helps distinguish genuine emotion complementarity from activity or length effects.

### 12. Mean-Pooling Ablation for the Set/Attention Transformer

Compare the set/attention transformer against a simpler mean-pooling or mean-plus-std MLP model using the same post embeddings.

Purpose:

- Determines whether learned author-level aggregation helps beyond fixed pooling.
- Prevents overclaiming if the set/attention transformer is no better than simple pooled embeddings.

### 13. Post-Budget Sensitivity for Author Transformers

Compare at least two retained-post budgets for transformer author models if time permits, for example 50 versus 200 posts per author.

Purpose:

- Tests whether the author transformer needs many posts or whether a smaller sampled budget preserves most of the signal.
- Helps manage compute and supports a practical final recommendation.

### 14. Supervised Post-Level Transformer Ceiling

Optionally fine-tune one compact transformer, such as MiniLM or DistilBERT, to predict post-level MBTI logits, then aggregate post-level probabilities to authors.

Recommended use:

- Run one text-only supervised transformer if the frozen and set/attention transformer branches finish.
- Add a text-plus-emotion classifier-head variant only if time remains.

Purpose:

- Provides a supervised transformer ceiling.
- Tests whether direct post-level MBTI supervision improves over frozen embedding author models.

Risk:

- It is compute-heavy and reintroduces the post-label noise problem because each post inherits an author-level MBTI label.
- It should not be allowed to crowd out the cleaner frozen and set/attention author-level transformer experiments.

## Why This Is Scientifically Defensible

This design is defensible because it follows controlled-experiment logic.

First, it keeps the evaluation unit aligned with the prediction claim. MBTI labels are author-level self-reports, so evaluation should be author-level. Post-level random splits would leak author identity and overstate performance.

Second, it addresses class imbalance using metrics and loss functions appropriate to the problem. Raw accuracy is not meaningful when `N/S` is roughly 93/7 at the author level. Balanced accuracy, ROC-AUC, F1, precision, and recall are more informative. Class-weighted BCE directly targets the majority-class collapse observed in MS3.

Third, it separates training-procedure fixes from representation-strength changes. The corrected GRU branch tests whether MS3's collapsed Stage 2 head can be repaired using weighted loss, author-balanced training, and soft aggregation. The transformer branches then ask a different question: once text is represented by a modern pretrained encoder, do explicit emotion features still add incremental value?

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

Seventh, it makes transformer modeling scientifically meaningful rather than decorative. The frozen transformer branch tests modern text representation without noisy post-level MBTI fine-tuning. The set/attention author transformer branch then tests whether learned author-level post aggregation improves over fixed mean or mean-plus-std pooling. Both branches use author-level labels at the author level.

Eighth, it respects the data's lack of post order. The Reddit dataset used here does not provide a reliable chronological sequence of posts per author. Therefore, learned author aggregation should be framed as set or multiple-instance learning over retained posts, not as temporal sequence modeling. This is why mean-pooling baselines and order-agnostic attention are important controls.

Ninth, it includes controls that make the emotion claim harder to attack. The emotion-only baseline tests whether emotion probabilities carry standalone author-level signal. Activity and length controls test whether apparent emotion gains are merely proxies for retained post count, verbosity, or truncation exposure.

Tenth, it supports multiple honest final outcomes:

- If emotion improves GRU but not transformer models, the conclusion can be that emotion helps weak sequence models but becomes redundant once semantic text representations are strong.
- If emotion improves both GRU and transformer models, the conclusion can be that emotion carries robust incremental author-level signal.
- If transformer text models beat GRU and TF-IDF but emotion adds little, the conclusion can emphasize representation strength and source-target limits of transferred emotion.
- If TF-IDF remains competitive or strongest, the conclusion can honestly discuss MBTI prediction as a lexical self-expression task where sparse author-level language markers are powerful.

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

- Report paired bootstrap confidence intervals over test authors for emotion-minus-text deltas within each model family when time permits.
- Bootstrap differences are more useful than standalone intervals because the headline question is whether emotion improves over the matched text-only control.

For result tables:

- Keep historical MS3 diagnosis separate from the MS4 controlled comparison.
- Historical table: majority baseline, MS3 B1, and MS3 B2, clearly labeled as unmasked historical references if they are not rerun on the MS4 masked corpus.
- Main MS4 table: TF-IDF, corrected GRU text, corrected GRU plus emotion, emotion-only, frozen transformer text, frozen transformer plus emotion, set/attention transformer text, and set/attention transformer plus emotion.
- Optional ceiling table: supervised post-level transformer text and supervised post-level transformer plus emotion, if run.
- The main table should be compact: per dimension, report balanced accuracy, F1, minority recall, PR-AUC or average precision, and ROC-AUC.
- Avoid ranking models primarily by raw accuracy.

Recommended appendix or robustness placement:

- GRU inverse-weight sensitivity belongs in a robustness or appendix section unless it becomes the selected recipe.
- GRU max-length 256 sensitivity belongs in robustness, not the main model table.
- Threshold-objective sensitivity belongs in robustness unless it materially changes the final conclusion.
- Detailed truncation tables and smoke-test outputs should stay out of the main paper body.
- TF-IDF should remain in the main body because it is a strong classical baseline, not a minor diagnostic.

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
   - One concise paragraph explaining why MS4 adds transformer author representations.
   - One diagram showing the three branches: corrected GRU, frozen transformer author features, and set/attention author transformer.

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

7. **Existing Baseline Layer**
   - Present majority, TF-IDF, corrected GRU text, and corrected GRU plus emotion.
   - Keep this section compact.
   - Explain that GRU emotion improves over corrected GRU text, but TF-IDF remains a strong lexical baseline if that is observed.
   - Use this section to motivate transformer representations rather than to make GRU the project centerpiece.

8. **Stage 1 Emotion Features**
   - Load or generate DistilBERT emotion probabilities for Reddit posts.
   - Report the source emotion model only as feature provenance and quality control.
   - Compare source emotion distribution to Reddit inferred emotion distribution.
   - State explicitly that downstream author-level MBTI results are the evidence for or against emotion usefulness.

9. **Transformer Embedding Pipeline**
   - Introduce the selected frozen transformer encoder.
   - Cache post-level embeddings for the masked Reddit modeling corpus.
   - Build author-level embedding features using mean and dispersion summaries.
   - Document cache metadata, model id, embedding dimension, and runtime at a high level.

10. **Frozen Transformer Author Models**
    - Train or load transformer text-only author classifiers.
    - Train or load emotion-only author classifiers.
    - Train or load transformer text-plus-emotion author classifiers.
    - Train or load transformer text-plus-emotion-plus-controls variants if included.
    - Compare against TF-IDF and corrected GRU models using the same author-level metric pipeline.
    - Treat this as the first major transformer result block.

11. **Set/Attention Author Transformer**
    - Present the author-as-unordered-set-of-post-embeddings design.
    - Explain why arbitrary temporal sequence assumptions are not justified without timestamps.
    - Train or load set/attention transformer text-only and text-plus-emotion models.
    - Train or load text-plus-emotion-plus-controls variants if included.
    - Include the mean-pooling or mean-plus-std ablation using the same embeddings.
    - Include post-budget sensitivity if run.
    - Treat this as the second major transformer result block.

12. **Optional Supervised Transformer Ceiling**
    - Include only if run successfully.
    - Present one compact supervised MiniLM or DistilBERT post-level model.
    - Compare author-level aggregated performance against frozen and set/attention transformer models.
    - Keep this section clearly marked as a ceiling or stretch analysis.

13. **Unified Results**
    - One headline model-family table:
      - classical baseline
      - corrected GRU
      - frozen transformer
      - set/attention transformer
      - optional supervised transformer
    - One per-dimension metric table.
    - One model-comparison plot by MBTI dimension.
    - Precision-recall or average-precision view for skewed dimensions such as `N/S`.
    - Confusion matrices for the selected final model.

14. **Emotion Increment Analysis**
    - Report matched deltas:
      - GRU plus emotion minus GRU text
      - frozen transformer plus emotion minus frozen transformer text
      - set/attention transformer plus emotion minus set/attention transformer text
      - optional supervised transformer plus emotion minus supervised transformer text
    - Include emotion-only performance to contextualize whether emotion is standalone signal, complementary signal, or mostly redundant.
    - Use paired bootstrap intervals where available.
    - This section should answer the main research question.

15. **Compute and Robustness**
    - Summarize local MPS runtime, cache size, and resumability.
    - Report threshold-objective sensitivity if relevant.
    - Report token-length or post-budget sensitivity if run.
    - Explain why the plan avoids a large full fine-tuned transformer grid.

16. **Interpretation**
    - Explain whether emotion adds incremental signal, and under which representation family.
    - Discuss whether transformer semantic features make explicit emotion redundant or complementary.
    - Address TF-IDF if it remains competitive or strongest.
    - Report limitations: self-reported labels, MBTI construct limitations, source-target emotion shift, missing subreddit/time metadata, and post-level label noise.

17. **Reproducibility and References**
    - List data sources.
    - List core libraries and model cards.
    - Cite PyTorch MPS, Hugging Face Transformers/Datasets, KaggleHub, sentence-transformer or MiniLM model cards, and relevant papers.

## Suggested Helper Modules

Place implementation helpers under the MS4 code area, for example `code/src/`:

- `config.py`: constants, labels, paths, seeds, runtime flags.
- `data.py`: KaggleHub/Hugging Face loading, filtering, split creation.
- `preprocessing.py`: text cleaning, MBTI term masking, label derivation, author-label consistency checks.
- `emotion.py`: Stage 1 training, inference, and probability caching.
- `embeddings.py`: frozen transformer embedding inference and author-level embedding aggregation.
- `models.py`: GRU, author-level MLP, and set/attention transformer model builders.
- `transformer_author.py`: set/attention author transformer training and prediction helpers.
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
- Expensive intermediate outputs, especially Reddit emotion probabilities, transformer post embeddings, author-level embedding tables, and post-level logits, should be cached and reused.
- Cache files should include enough metadata to prevent accidental reuse across incompatible preprocessing or model versions, including at minimum split id, masking status, embedding model id, emotion model id, and label encoding.
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
6. Token truncation or embedding-input length audit.
7. Baseline layer comparison: TF-IDF, corrected GRU text, corrected GRU plus emotion.
8. Source-vs-Reddit emotion distribution comparison.
9. Transformer embedding pipeline summary, including cache size and author-feature construction.
10. Emotion-only baseline summary.
11. Frozen transformer author-model comparison, text versus text plus emotion, with controls if included.
12. Set/attention author transformer architecture diagram emphasizing unordered post aggregation.
13. Set/attention transformer results, text versus text plus emotion, with controls if included.
14. Emotion-increment plot across model families.
15. Unified model comparison plot by metric and dimension.
16. Bootstrap confidence intervals for matched emotion-minus-text deltas, if available.
17. Final model confusion matrices.
18. Post-budget or pooling ablation plot for set/attention transformer, if run.

Avoid text-heavy figures. Captions should explain the modeling decision supported by each visualization.

## Compute Plan

Local MPS should be sufficient for the expanded transformer-centered plan if full fine-tuned transformer grids are avoided.

Core local work:

- Cached DistilBERT emotion inference over the filtered Reddit modeling corpus.
- Corrected GRU text and text-plus-emotion runs.
- Frozen transformer post-embedding inference over the filtered Reddit modeling corpus.
- Author-level frozen-transformer classifiers.
- Emotion-only and control-feature author baselines.
- Small set/attention author transformers over cached post embeddings.

Recommended local settings:

- Stage 1 DistilBERT max length: 64.
- Stage 1 batch size: start with 32 or 64 and adjust for MPS memory.
- Stage 1 emotion fine-tuning, if performed locally, should be limited to 3-5 epochs with early stopping.
- Stage 2 GRU max length: 128 for the core comparison, with a reported truncation audit and an optional fixed text-only 128 versus 256 validation sensitivity check if truncation is severe.
- Stage 2 batch size: start with 512 or 1024 and adjust for memory.
- Cache Stage 1 Reddit emotion probabilities immediately after inference.
- Frozen transformer embeddings should be cached once and reused for all author-level transformer models.
- Use a compact encoder such as `sentence-transformers/all-MiniLM-L6-v2` or MiniLM before considering larger encoders.
- Store embedding caches in a compact numeric format with row ids, authors, split ids, model id, and preprocessing fingerprint.
- Set/attention transformer training should operate on cached post embeddings and author-level batches, not raw text.
- Do not treat post order as temporal unless reliable timestamps are introduced, which the current dataset does not provide.

Rent GPU only if:

- The team decides to run multiple full supervised post-level transformer fine-tunes.
- Local MPS embedding inference over the full capped Reddit corpus is too slow for the MS4 timeline.
- A heavier transformer such as DeBERTa is moved from future work into the actual experiment plan.

The core MS4 plan should not require rented GPU compute if implemented efficiently. One supervised MiniLM or DistilBERT ceiling run may fit locally, but it should not become a prerequisite for a strong MS4 submission.

## Risks and Mitigations

| Risk | Mitigation |
|---|---|
| MPS unavailable in notebook | Allow cached evaluation and smoke tests on CPU; require MPS only when `RUN_FULL_TRAINING = True`. |
| Plan becomes too large for the deadline | Prioritize frozen transformer author models first, set/attention author transformers second, and supervised post-level transformer fine-tuning last. |
| Stage 2 max length 128 truncates too much Reddit text | Report post-level and author-level truncation rates; if severe, run a fixed text-only 128 versus 256 validation sensitivity check before locking the core max length. |
| Weighted BCE over-corrects `N/S` | Compare strict inverse-frequency weights with milder square-root weights on validation only, select by mean balanced accuracy with documented tie-breakers, and keep one locked recipe for test. |
| Imbalance corrections become inconsistent | Define one training recipe: class weights from the effective training distribution, fixed author-balancing or loss-weighting, and validation-only threshold tuning. |
| Apparent final-model gain comes only from weighted BCE or soft aggregation | Include the fixed text-only baseline and compare emotion models against it under the same protocol. |
| Post-level training overweights prolific authors | Use author-balanced sampling or per-post loss weights based on each author's retained post count. |
| Transformer text features make emotion look unnecessary | Treat this as a valid finding: explicit emotion may be redundant once semantic text representations are strong. |
| Frozen transformer embeddings underperform TF-IDF | Report honestly; TF-IDF may capture sparse lexical self-expression better than generic semantic embeddings for MBTI labels. |
| Set/attention transformer overfits only 10k authors | Compare against mean-pooling baselines, use validation monitoring, and keep model size small. |
| Arbitrary post order contaminates author transformer results | Use order-agnostic pooling or attention; do not use temporal positional encodings without timestamps. |
| Emotion gains are actually activity or length effects | Include activity/length controls in the key author-level emotion comparisons. |
| Full supervised transformer becomes a compute sink | Keep it as a one-run ceiling or omit it if frozen and set/attention transformer results already answer the research question. |
| Thresholds look arbitrary | Predeclare validation balanced accuracy as the primary threshold objective, report thresholds, and apply them unchanged to test. |
| DistilBERT emotion model improves source test accuracy but not Reddit downstream performance | Treat as evidence of source-target domain shift; report honestly. |
| Neural models underperform a simple lexical baseline | Include the author-level TF-IDF linear baseline and report the result honestly. |
| Cached predictions become stale after preprocessing changes | Store cache metadata for masking status, split id, model id, embedding model id, emotion model id, and label encoding; invalidate mismatches. |
| Notebook becomes too long | Move helpers into `.py` modules and keep only orchestration, prose, and outputs in the notebook. |
| Final result is negative | Frame as a controlled result: transformer text representations may capture the available signal while transferred emotion contributes limited incremental value. |

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
- Sentence-transformers `all-MiniLM-L6-v2` model card: <https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2>
- Microsoft MiniLM model card: <https://huggingface.co/microsoft/MiniLM-L12-H384-uncased>
- Hugging Face Apple Silicon / MPS training note: <https://huggingface.co/docs/transformers/perf_train_special>
- DistilBERT paper: <https://arxiv.org/abs/1910.01108>
