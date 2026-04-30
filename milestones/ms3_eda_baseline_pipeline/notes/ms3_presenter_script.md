# MS3 Speaker Script — Humanized & Shortened Version

**Target: ~7–7.5 minutes**  
**Speakers:** HH = Harry Hu · TS = Tom Shan · WW = Wendy Wang · KZ = Kemeng Zhang

---

## Section 1 — Summary of Milestone 2 Progress

### Slide 1 — Title

**HH · ~8 s**

> Good afternoon. We’re Project 66 — Harry, Tom, Wendy, and Kemeng. Today we’ll walk through our Milestone 3 progress: EDA, baselines, and the updated modeling pipeline.

**Transition:** Let me start with the main question.

---

### Slide 2 — The Question and Two Datasets

**HH · ~35 s**

> Our main question is: can emotion-aware language signals help predict MBTI personality?
>
> We use two datasets. The first is a balanced six-class emotion dataset with about twenty thousand short texts from HuggingFace. This is our source dataset, and we use it to train a small emotion classifier.
>
> The second is a much larger Reddit dataset, where authors self-report their MBTI type. This is our target dataset.
>
> The overall idea is simple: we first learn emotion patterns from the source data, then test whether those emotion features improve MBTI prediction on Reddit.

**Transition:** In Milestone 2, we identified four issues that shaped our design.

---

### Slide 3 — Four MS2 Concerns → Four Design Decisions

**HH · ~35 s**

> We made four main decisions after Milestone 2.
>
> First, the 16 MBTI types are very imbalanced. For example, INFP is much more common than ESFP. So instead of predicting all 16 types directly, we predict the four MBTI dimensions separately: E versus I, N versus S, F versus T, and J versus P.
>
> Second, because the same author can write many posts, we split the data by author, not by post.
>
> Third, single posts are often too short, so we aggregate predictions at the author level.
>
> Fourth, to test whether emotion actually helps, we compare two matched baselines: one without emotion and one with emotion.

**Transition:** These choices lead to our refined research question.

---

### Slide 4 — Refined Research Question

**HH · ~18 s**

> Our refined question is: do text-plus-emotion features improve author-level prediction of the four MBTI dimensions compared with a text-only baseline?
>
> To make this fair, both models use the same authors, the same split, and the same evaluation metrics.

**Transition:** Tom will now walk through the extended EDA.

---

## Section 2 — Extended EDA

### Slide 5 — Short-Post Tail Is Noise

**TS · ~35 s**

> The extended EDA gave us three important thresholds.
>
> The first one is the word-count threshold. On Reddit, posts with one to four words are about three percent of the corpus, but many of them are repeated across authors — things like “lol,” “thanks,” or “same.”
>
> These posts add noise and can create leakage, because the same short phrases appear everywhere.
>
> Once we keep posts with at least five words, duplication drops sharply. So we remove posts shorter than five words before feature extraction.

**Transition:** Next, we looked at how many posts each author needs.

---

### Slide 6 — Author Feature Stability → 20-Post Floor

**TS · ~30 s**

> For author-level prediction, we need each author’s features to be reasonably stable.
>
> The left plot shows that feature variation decreases as the number of posts increases, roughly following a one-over-square-root pattern.
>
> The right plot shows split-half reproducibility. At twenty posts, every feature reaches a correlation above 0.25.
>
> Having a twenty posts bottom line greatly reduces noise, while only removing about five percent of authors. So we set the minimum at twenty posts per author.

**Transition:** We also needed a cap, because a few authors write too much.

---

### Slide 7 — Author Concentration → 200-Post Cap

**TS · ~35 s**

> Without a cap, the dataset is dominated by a small number of authors. The top one percent of authors write about nineteen percent of all posts, and the Gini coefficient is 0.75.
>
> That suggests a strong power-law pattern, and some high-volume accounts may be bots.
>
> We tested different caps. A cap between twenty and one thousand posts already reduces the dominance problem.
>
> We chose two hundred posts because it still leaves us with about 1.7 million training posts, while limiting the influence of the most extreme authors.

**Transition:** The last EDA question is whether the emotion classifier can transfer to Reddit.

---

### Slide 8 — Source → Target Transfer Is Feasible

**TS · ~35 s**

> This slide checks whether emotion information can transfer from the source dataset to Reddit.
>
> On the left, the main mismatch is length. Reddit posts can be much longer than the emotion-training texts.
>
> Because of this, we do not use a hard emotion label. Instead, we keep the full six-dimensional soft probability vector. This lets the downstream model see uncertainty from the emotion classifier.
>
> On the right, token coverage improves as Reddit posts get longer. Very short posts have lower coverage, but after applying the five-word floor, coverage is much stronger.

**Transition:** These EDA results directly define our modeling setup.

---

### Slide 9 — EDA → Decisions

**WW · ~18 s**

> To summarize the EDA decisions: we use a five-word post minimum, a twenty-post per author minimum, a two-hundred-post author cap, and soft six-dimensional emotion probabilities.
>
> These choices also imply three modeling needs: sequence-aware encoders, author-level aggregation, and a fair baseline comparison which we will describe next.

**Transition:** Kemeng will now walk through the model architecture.

---

## Section 3 — Baseline Models

### Slide 10 — Model Input, Output, and Architecture

**KZ · ~25 s**

> This slide shows our three-model setup. In general, we use WordPiece tokenizer and GRU as the base model.
>
> Stage 1 is the emotion classifier. It takes short texts from the HuggingFace emotion dataset and outputs six emotion probabilities.
>
> Baseline 1 is our text-only MBTI model. It uses Reddit posts as input and predicts four MBTI dimension logits.
>
> Baseline 2 is the same as Baseline 1, but with one extra input: the six emotion probabilities from Stage 1.
>
> So the comparison is controlled. Both baselines use the same split, loss, encoder, and aggregation. The only difference is that Baseline 2 adds the emotion channel before the final dense head.

**Transition:** Now let’s see how training went.

---

### Slide 11 — Training Curves + Stage-1 Emotion

**KZ · ~30 s**

> On the left, the MBTI loss curves for B1 and B2 are almost the same. Both plateau at around 0.520 validation BCE after six or seven epochs.
>
> On the right, the Stage-1 emotion classifier performs much better. It reaches about 0.897 validation accuracy and 0.908 test accuracy.
>
> So the emotion classifier itself is not the obvious bottleneck. But the MBTI baselines converge to the same loss floor.

**Transition:** Now let’s take a closer look at the author-level results.

---

### Slide 12 — Author-Level Results: Majority-Class Collapse

**KZ · ~40 s**

> This is the main result, and it is a negative result.
>
> For three of the four dimensions — E-I, N-S, and J-P — both baselines collapse to the majority class. Balanced accuracy is exactly 0.500, which means the model predicts every test author as the majority label.
>
> Also, B2 does not meaningfully outperform B1, so adding emotion features does not help under this setup.
>
> The only partial exception is F-T. There, the model has high precision for the minority class T, around eighty-six to eighty-eight percent, but recall is only two to four percent.
>
> So the model sometimes recognizes T correctly, but it almost never predicts it.

**Transition:** The next slide explains where that signal is lost.

---

### Slide 13 — Interpretation: Where the Signal Dies

**KZ · ~30 s**

> We think two problems are happening at the same time.
>
> First, vanilla BCE does not handle the strong label imbalance well. The model can reduce loss by leaning toward the majority class.
>
> Second, majority-vote aggregation is too strict. Even if a model predicts the minority class for many posts, the author only gets that label if more than half of their posts cross the 0.5 threshold.
>
> So weak but real minority signals can be erased during aggregation.

**Transition:** That diagnosis motivates our final pipeline changes. Wendy?

---

## Section 4 — Future Directions & Next Steps

### Slide 14 — Final Pipeline: 4 Targeted Upgrades

**WW · ~25 s**

> Our final pipeline makes four focused upgrades.
>
> First, we use class-weighted BCE to handle label imbalance.
>
> Second, we replace hard majority vote with soft author aggregation: we average post-level probabilities and tune one threshold per MBTI dimension.
>
> Third, we upgrade the Stage-1 emotion model from an RNN to DistilBERT.
>
> Fourth, we also replace the Stage-2 GRU with a DistilBERT fine-tuned on Reddit. We keep the prediction head mostly the same, so improvements are easier to trace.

**Transition:** Before next steps, here are the main risks and questions for our TF.

---

### Slide 15 — Potential Challenges & Open Questions for TF

**WW · ~18 s**

> We expect four challenges.
>
> First is compute. Running DistilBERT on both stages and 1.5 million Reddit posts is expensive.
>
> Second is domain shift. The emotion dataset is not Reddit, so some emotion scores may need recalibration.
>
> Third is label noise. MBTI self-reports are imperfect, so model performance has a natural ceiling.
>
> Our TF questions are: For the N-S imbalance, should we use full inverse-frequency weights or a milder square-root version? And after soft aggregation, is AUC-ROC okay as the main metric?

**Transition:** Finally, here is our work split.

---

### Slide 16 — Next Steps

**WW · ~18 s**

> Here is the remaining work.
>
> Harry will implement class-weighted BCE and soft author aggregation on Baseline 2.
>
> Tom will fine-tune DistilBERT for emotion classification and cache Reddit emotion probabilities.
>
> Wendy will fine-tune DistilBERT on Reddit as the new Stage-2 text encoder.
>
> Kemeng will run the author-level evaluation, ablations, and interpretability analysis.
>
> Thank you — we’re happy to take questions.

---

## Rehearsal Notes

- Aim for a natural pace. Do not rush the technical numbers.
- If you are short on time, shorten Slide 7 and Slide 11 first.
- If you need to cut more, skip the detailed Gini explanation and just say: “a few authors dominate the dataset.”
- Make Slide 12 clear: the main result is not that the model learned nothing; it is that the current training and aggregation setup hides the minority-class signal.
- Slide 10 (model I/O and architecture) is a good candidate to trim if time is tight: you can drop the exact hidden sizes and the dense-head dimensions and keep only the three rows and the two orange arrows at a high level.
- On Slide 14, emphasize that lever (4) — DistilBERT Stage-2 text encoder — is projected down to 128-d so the head architecture stays identical to Baseline 2. That’s what makes the ablation fair.
- On Slide 15, if time is short, present only the three challenges (compute, domain shift, ablation sprawl) and the three TF questions; skip the label-noise bullet and the "two checkpoints" question.
- Slide 16 is presented by HH alone. Keep it tight — roughly four sentences, one per teammate.

