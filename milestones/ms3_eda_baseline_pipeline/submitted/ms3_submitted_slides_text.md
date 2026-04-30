# MS3 Submitted Slide Text

Source PPTX: `../artifacts/submitted/ms3_slides_submitted.pptx`

This is extracted slide text for LLM-friendly reading. For the full spoken story, read `../notes/ms3_presenter_script.md`.

## Slide 1 - Title

AC 209b Milestone 3. Emotion-Informed MBTI Prediction. Project 66. Harry Hu, Tom Shan, Wendy Wang, Kemeng Zhang.

## Slide 2 - The Question and Two Datasets

Can emotion-aware language signals predict MBTI?

- Emotion source: 20k rows, six balanced emotion classes, Hugging Face corpus.
- Reddit MBTI target: about 13M posts, 16 self-reported types, Kaggle Reddit corpus.
- Transfer from emotion source to Reddit target.

## Slide 3 - Four MS2 Concerns to Four Design Decisions

1. 16-type skew -> four binary dimensions.
2. Authors repeated -> author-level split.
3. Posts too short -> author aggregation.
4. Does emotion help? -> matched B1 vs B2.

## Slide 4 - Refined Research Question

Do text plus emotion features beat text-only baselines on author-level MBTI?

Target dimensions: E/I, N/S, F/T, J/P.

## Slide 5 - Short-Post Tail Is Noise

- Decision: drop posts under five words.
- 1-4 words: 3.4 percent of sample.
- Duplication: 1.1 percent -> 0.2 percent.
- Examples: lol, thanks, same.

## Slide 6 - Author Feature Stability to 20-Post Floor

- Bootstrap standard deviation tracks 1/sqrt(n).
- 20 posts clears correlation >= 0.25.
- Only about 8 percent author loss.

## Slide 7 - Author Concentration to 200-Post Cap

- Uncapped Gini: 0.75.
- Top 1 percent: 18.9 percent of posts.
- Cap 200: Gini 0.27.
- About 1.7M posts retained.

## Slide 8 - Source to Target Transfer Is Feasible

- Coverage rises: 59 percent -> 92 percent.
- Reddit 95th percentile is about 3x source length.
- Use soft six-dimensional probabilities.

## Slide 9 - EDA to Decisions

Every threshold is EDA-derived.

Data choices:

- word floor >= 5
- author floor >= 20
- author cap <= 200
- emotion as soft probabilities

Modeling choices:

- sequence encoder for variable text
- author aggregation from post to author
- matched pair B1 vs B2

## Slide 10 - Model Input, Output, and Architecture

Stage 1 emotion classifier:

- short text
- Hugging Face emotion 20k
- WordPiece vocab 30k
- max length 64
- embedding 30k -> 128-d
- GRU hidden = 128
- dense -> softmax 128 -> 64 -> 6

Baseline 1 text only:

- Reddit text
- post, 5-200 words
- WordPiece vocab 30k
- max length 128
- embedding 30k -> 128-d
- GRU hidden = 128
- dense head 128 -> 64 -> 4 MBTI logits

Baseline 2 text plus emotion:

- Same text path as B1.
- Adds six-dimensional emotion probabilities from Stage 1.
- Concatenates 128 + 6 -> 134.
- Dense head 134 -> 64 -> 4 logits.

## Slide 11 - Training Curves plus Stage-1 Emotion

- Emotion classifier validation accuracy: 0.897.
- Emotion classifier test accuracy: 0.908.
- B1/B2 plateau: validation BCE about 0.520.
- Stage 1 is not the bottleneck; MBTI baselines converge to the same loss floor.

## Slide 12 - Author-Level Results

Majority-class collapse.

- E/I, N/S, J/P collapse to the prior.
- B2 minus B1 gap is about zero.
- F/T precision: 86-88 percent.
- F/T recall: only 2-4 percent.

## Slide 13 - Interpretation

Where the signal dies:

- per-post probabilities
- hard 50 percent majority vote
- author-level label

Diagnosis:

- Vanilla BCE -> class-prior solution.
- Hard vote discards high-precision minority signal.

## Slide 14 - Final Pipeline

Four targeted upgrades:

1. class-weighted BCE
2. soft author aggregation
3. DistilBERT emotion encoder
4. DistilBERT text encoder

## Slide 15 - Potential Challenges and Open Questions for TF

Challenges:

- compute budget
- domain shift
- label noise

Open questions:

- For heavily skewed N/S, around 93/7, what weighting recommendation should we use?
- Once soft aggregation produces continuous author-level scores, is AUC-ROC per dimension acceptable as a headline metric alongside balanced accuracy?

## Slide 16 - Next Steps

Note: this person-by-person split was tentative and included for the MS3 presentation requirement. It should not be treated as the actual MS4 work assignment.

- HH: weighted BCE and soft aggregation.
- TS: DistilBERT Stage 1 and probability cache.
- WW: DistilBERT Stage 2 fine-tune.
- KZ: evaluation and ablations.
