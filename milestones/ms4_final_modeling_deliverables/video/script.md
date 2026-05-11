# Video Script — Emotion-Informed MBTI Prediction

> **Target runtime:** 5:50 (buffer to the 6:00 hard limit).
> **Tone:** Conversational, enthusiastic, clear. Not reading — practiced natural delivery.
> **Tip:** Each speaker should practice their section 3 times. Use the script as a guide, not a teleprompter.

---

## HARRY — Slides 1–4 (~1:20)

### Slide 1 — Title & Introductions (15 sec)

> Hi everyone! We're Group 66. I'm Harry — and with me are Tom, Wendy, and Kemeng. Today we're presenting our project: **Emotion-Informed MBTI Prediction**.

### Slide 2 — The Hook (15 sec)

> Here's a question: if you read a hundred of someone's Reddit posts, could you figure out whether they're an introvert or an extrovert? A thinker or a feeler? That's exactly what we set out to test — whether writing style alone can predict Myers-Briggs personality type.

### Slide 3 — Why This Matters (25 sec)

> Why does this matter? MBTI inference is already being used in the real world — in HR screening, recommendation systems, and adaptive learning platforms. But there's a catch: how much can text *actually* tell us about personality, and how much is just noise? We wanted to rigorously test that, and specifically ask: do emotion features from text add anything beyond what the words themselves already capture?

### Slide 4 — The Problem with Prior Work (25 sec)

> Most prior work treats this as a post-level task — take a Reddit post, predict MBTI. But that's methodologically broken. MBTI labels are self-reported *per user*, not per post. If you split posts randomly, the same author's posts leak into both training and test. Our fix: we reframe this as an **author-level** prediction task. All posts from one author stay together, and we make one prediction per person.

---

## TOM — Slides 5–7 (~1:15)

### Slide 5 — Data at a Glance (25 sec)

> Thanks, Harry. Let me walk you through the data. We start with two public datasets. The Reddit MBTI corpus gives us 13 million posts from about 12,000 authors who self-reported their personality types. After filtering for quality — at least 5 words per post, at least 20 posts per author, and capping at 200 posts — we end up with 1.65 million posts from 10,414 authors. We also use a separate emotion-balanced dataset of 20,000 texts to train our emotion classifier, which we then transfer onto the Reddit data.

### Slide 6 — Leakage Masking (20 sec)

> One critical step: leakage masking. We found that 78% of authors mention their own MBTI type somewhere in their posts — things like "As an INTJ, I think..." That's a trivial shortcut for any model. So we mask all MBTI-related tokens with a placeholder before any modeling. After masking, leakage drops to zero.

### Slide 7 — Class Imbalance & EDA (30 sec)

> Now, the four MBTI dimensions are *not* balanced. On Reddit, only 21% of users identify as Extroverts and just 7% as Sensors. This severe imbalance is why we use balanced accuracy as our primary metric — it weights both classes equally regardless of size. On the right, you can see the emotion distribution: when we apply our emotion classifier to Reddit posts, joy and anger dominate, which looks very different from the balanced training source. That's why we treat these as *transferred* text features, not actual mood measurements.

---

## WENDY — Slides 8–12 (~1:40)

### Slide 8 — Pipeline Overview (20 sec)

> Thanks, Tom. Here's our end-to-end pipeline. Reddit posts come in, get masked and filtered, then split by author. Each post gets two representations: a frozen MiniLM sentence embedding — that's 384 dimensions — and a six-class emotion probability vector from our transferred DistilBERT classifier. These feed into our model families.

### Slide 9 — Model Progression (20 sec)

> We built five model families, each testing a different idea. Starting at the bottom: a majority-class baseline at 0.50. Then a corrected GRU — corrected because the MS3 version collapsed to majority on three of four dimensions. Then TF-IDF logistic regression. Frozen MiniLM probes. And at the top: our set-attention author transformer. Details of the baseline models are introduced in the paper. I'll only explain our main set-attention architecture here.

### Slide 10 — Set-Attention Architecture (20 sec)

> One key insight is that an author is a *set* of posts, not a sequence. Post order on Reddit is unreliable, so we don't use positional encoding. We project each post embedding through a linear layer, apply one block of multi-head self-attention with four heads, then masked-mean-pool across posts to get a single author vector. It's permutation-invariant by design.

### Slide 11 — Training Details (25 sec)

> For training specifics: we use AdamW with a learning rate of 10 to the negative 3 and weight decay of 10 to the negative 2. The loss is class-weighted binary cross-entropy, using square-root inverse-frequency weights — we found raw inverse-frequency over-shot minority recall and collapsed precision. Set-attention runs train for 5 epochs with early stopping on validation loss. The architecture uses 4 attention heads, model dimension 128, and dropout of 0.2 with a post budget of 200 per author. We verified stability across three random seeds and 5, 10, and 20 epoch caps — all reproduce the same model ranking. See the report for full details.

### Slide 12 — The Emotion Experiment (15 sec)

> For the emotion question, we designed a controlled experiment with three variants: text-only, text plus real emotion, and text plus *shuffled* emotion — our negative control. If shuffled emotion performs the same as real, the signal isn't from author-emotion alignment.

---

## KEMENG — Slides 13–17 (~1:35)

### Slide 13 — Headline Results (20 sec)

> Thanks, Wendy. Here are the headline results. Our set-attention model with a 200-post budget achieves a mean balanced accuracy of **0.678** — that's +2.7 points over TF-IDF and +8.2 over the corrected GRU. Both improvements are statistically significant: paired bootstrap 95% confidence intervals exclude zero. Another observatino is that Emotion features *do* carry personality information — the Frozen MiniLM  emotion-only model beats majority. The limitation with emotion will be discussed later.

### Slide 14 — Per-Dimension Breakdown (20 sec)

> Breaking it down by dimension, set-attention improves on *every* dimension compared to all baselines. The biggest gains are exactly where we needed them most — on the hardest, most imbalanced dimensions. On Sensing, we go from 0.53 with the GRU to 0.67. On Extraversion, from 0.60 to 0.69.

### Slide 15 — Emotion: Informative but Redundant (20 sec)

> Now, the emotion question. This figure shows the deltas: real emotion minus text-only, and shuffled emotion minus text-only. Both confidence intervals straddle zero, and they completely overlap each other. The negative control matches the real signal. Emotion features *do* carry personality information but once you have a good text encoder, emotion adds nothing new.

### Slide 16 — Three Takeaways (15 sec)

> Three takeaways. **One:** predict per author, not per post — post-level splits inflate results. **Two:** set-attention is the right architecture — it respects the unordered nature of an author's post collection. **Three:** transferred emotion is informative on its own but redundant given text.

### Slide 17 — Future Work & Thank You (20 sec)

> Looking ahead, two directions. First: replace the emotion classifier with a low-rank residual on MiniLM, to test if there's anything *orthogonal* to text that emotion captures. Second: scale up with Set Transformer inducing points to handle more than 200 posts per author. Thank you for watching — we're happy to take any questions!

---

# Timing Summary

| Speaker | Slides | Time |
|---|---|---|
| Harry Hu | 1–4 | ~1:20 |
| Tom Shan | 5–7 | ~1:15 |
| Wendy Wang | 8–12 | ~1:40 |
| Kemeng Zhang | 13–17 | ~1:35 |
| **Total** | **17 slides** | **~5:50** |

---

# Speaker Transition Cues

- **Harry → Tom:** Harry finishes Slide 4 with "...one prediction per person." Tom picks up with "Thanks, Harry. Let me walk you through the data."
- **Tom → Wendy:** Tom finishes Slide 7 with "...not actual mood measurements." Wendy picks up with "Thanks, Tom. Here's our end-to-end pipeline."
- **Wendy → Kemeng:** Wendy finishes Slide 12 with "...the signal isn't from author-emotion alignment." Kemeng picks up with "Thanks, Wendy. Here are the headline results."

---

# Practice Notes

1. **Read-through 1:** Read aloud, time each section. Adjust wording where you stumble.
2. **Read-through 2:** Practice with slides advancing. Make sure you're *looking at the camera*, not reading.
3. **Read-through 3:** Full team run-through with recording. Check audio, transitions, and pacing.
4. **Key numbers to memorize:** 1.65M posts, 10,414 authors, 0.678 BA, +2.7% over TF-IDF, lr 10⁻³, 5 epochs, 4 heads, d=128, 78% leakage, 7% S minority.
5. **Energy:** Smile. Vary your intonation. Pause briefly after key numbers to let them land. End strong.
