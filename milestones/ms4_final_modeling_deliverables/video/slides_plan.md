# Slides Plan — Emotion-Informed MBTI Prediction

> 17 slides, ~5:50 target runtime. Visual-first design: every slide has a dominant visual element, minimal bullet text. Consistent dark-navy + accent-teal palette, slide numbers bottom-right.

---

## Slide 1 — Title (on screen during introductions)

| Element | Details |
|---|---|
| **Visual** | Large project title centered. **[MEME PLACEHOLDER]** — a fun/relevant meme about personality types or MBTI on the right side (e.g., "Tell me your MBTI and I'll tell you..." format, or a personality-related meme that hooks the audience). |
| **Title** | **Emotion-Informed MBTI Prediction** |
| **Subtitle** | Group 66 · AC 209b / CS 1090b · Spring 2026 · Harvard |
| **Bottom** | Four small circular headshots in a row: Harry, Tom, Wendy, Kemeng |

---

## Slide 2 — The Hook

| Element | Details |
|---|---|
| **Visual** | A Reddit-style post card mockup (anonymous, blurred username). Below it, four binary MBTI dimension toggles: E↔I, N↔S, F↔T, J↔P — all showing "?" |
| **Text** | *"Can your Reddit posts reveal your personality type?"* (large, centered) |

**Design note:** This is a question slide — no answers yet. Build curiosity.

---

## Slide 3 — Why This Matters

| Element | Details |
|---|---|
| **Visual** | Three icon panels side by side: (1) briefcase icon → "HR Screening", (2) laptop icon → "Recommendation Systems", (3) graduation cap → "Adaptive Learning". A caution symbol underneath. |
| **Key text** | One sentence: *"MBTI inference is already deployed — but how much can text really tell us?"* |

**Design note:** Rule of three. Three real-world use cases, one cautionary framing.

---

## Slide 4 — The Problem with Prior Work

| Element | Details |
|---|---|
| **Visual** | Animated two-panel comparison diagram. **Left panel (red X):** "Post-Level Splits" — arrows from a single author's posts going into both train and test sets (data leakage). **Right panel (green check):** "Author-Level Splits" — all posts from one author stay in the same split. |
| **Key text** | *"Post-level splits let models memorize writing style — we predict per author instead."* |

**Design note:** This is the core methodological reframing. Make the visual instantly clear.

---

## Slide 5 — Data at a Glance

| Element | Details |
|---|---|
| **Visual** | Three large stat callout boxes in a row: **13M** raw posts → **1.65M** filtered posts → **10,414** authors. Arrow flow between them. Below: small logos for "Reddit MBTI Corpus" and "Emotion-Balanced Corpus". |
| **Key text** | *"Two datasets. Filtered by word count, author activity, and a 200-post cap per author."* |

---

## Slide 6 — Leakage Masking

| Element | Details |
|---|---|
| **Visual** | A before/after post example. **Before:** "As an INTJ, I find that ..." with "INTJ" highlighted in red. **After:** "As an [TYPE], I find that ..." with "[TYPE]" in green. Below: a mini-table showing leakage audit: 78.4% of authors mention MBTI → 0% after masking. |
| **Key text** | *"78% of authors mention their type — we mask all MBTI tokens to prevent shortcuts."* |

---

## Slide 7 — Class Imbalance & EDA

| Element | Details |
|---|---|
| **Visual** | Horizontal stacked bars showing the four dichotomies with minority %: E 21%, S 7%, T 40%, J 40%. Color-code minority vs. majority. On the right: the source-vs-Reddit emotion distribution chart (`fig_source_vs_reddit_emotion_distribution.png`). |
| **Key text** | *"Severe imbalance on E and S — balanced accuracy is the right metric."* |
| **Figure file** | `fig_source_vs_reddit_emotion_distribution.png` (right half) |

---

## Slide 8 — Pipeline Overview

| Element | Details |
|---|---|
| **Visual** | Full pipeline diagram (`fig_ms4_pipeline_diagram.png`), sized to fill the slide. |
| **Key text** | Slide title only: *"End-to-End Pipeline"* |
| **Figure file** | `fig_ms4_pipeline_diagram.png` |

**Design note:** Let the diagram speak. Walk through it verbally.

---

## Slide 9 — Model Progression

| Element | Details |
|---|---|
| **Visual** | Vertical "staircase" graphic showing the model ladder, each step higher than the last: **Step 1:** Majority (0.50) → **Step 2:** GRU Baseline (0.60) → **Step 3:** TF-IDF Logistic (0.65) → **Step 4:** Frozen MiniLM Probes (0.63) → **Step 5:** Set-Attention (0.68). Each step has a tiny icon (dice, neural net, magnifying glass, snowflake, attention heads). |
| **Key text** | *"Five model families, each testing a different hypothesis."* |

**Design note:** No table — a visual progression. The staircase makes the improvement intuitive.

---

## Slide 10 — Set-Attention Architecture

| Element | Details |
|---|---|
| **Visual** | Clean architecture diagram: post embeddings (384-d each, shown as colored bars) → linear projection → multi-head self-attention block (4 heads, d=128) → masked mean pool → MLP → 4 binary outputs. Label "permutation-invariant" with a shuffle icon. |
| **Key text** | *"An author is an unordered set of posts — attention respects that."* |

**Design note:** Simplified version of the equations in the paper. Visually show that post order doesn't matter.

---

## Slide 11 — Training Details

| Element | Details |
|---|---|
| **Visual** | Clean spec card or dashboard-style layout with grouped parameters. Three columns: **Optimizer** (icon: gear), **Architecture** (icon: layers), **Evaluation** (icon: chart). |
| **Optimizer column** | AdamW · lr = 10⁻³ · weight decay = 10⁻² · class-weighted BCE (√ inverse-frequency) |
| **Architecture column** | 4 attention heads · d_model = 128 · dropout = 0.2 · post budget P = 200 · embedding dim = 384 (frozen MiniLM) |
| **Evaluation column** | 5 epochs (early stopping on val loss) · 70/15/15 author split (7,289 / 1,562 / 1,563) · threshold-tuned on val BA · paired bootstrap CIs (B = 2,000) |
| **Key text** | *"See the report for full hyperparameter sweeps and seed/epoch sensitivity."* |

**Design note:** Three-column card layout keeps it scannable. No paragraphs — just key-value pairs with icons.

---

## Slide 12 — The Emotion Experiment

| Element | Details |
|---|---|
| **Visual** | Three-panel experimental design: **Panel A:** Text-only (baseline). **Panel B:** Text + Real Emotion (6 emotion probabilities appended). **Panel C:** Text + Shuffled Emotion (negative control — emotions randomly reassigned within split). Arrows show the comparison: B vs A = Δ_real, C vs A = Δ_shuf. |
| **Key text** | *"If shuffled emotion matches real emotion, the signal isn't from emotion — it's already in the text."* |

---

## Slide 13 — Headline Results

| Element | Details |
|---|---|
| **Visual** | The set-attention author models bar chart (`fig_set_attention_author_models.png`), with the text-only P=200 bar highlighted/glowing. |
| **Callout box** | Large: **0.678 mean balanced accuracy** · "+2.7% over TF-IDF" · "+8.2% over GRU" |
| **Figure file** | `fig_set_attention_author_models.png` |

---

## Slide 14 — Per-Dimension Breakdown

| Element | Details |
|---|---|
| **Visual** | The per-dimension balanced accuracy grouped bar chart (`fig_target_balanced_accuracy.png`). |
| **Key text** | *"Set-attention improves every dimension — biggest gains on the hardest ones (E, S)."* |
| **Figure file** | `fig_target_balanced_accuracy.png` |

---

## Slide 15 — Emotion: Informative but Redundant

| Element | Details |
|---|---|
| **Visual** | The emotion deltas figure with bootstrap CIs (`fig_transformer_emotion_deltas.png`). Annotate the key finding: both real and shuffled CIs straddle zero. |
| **Key text** | *"Emotion carries personality signal — but text already captures it."* |
| **Figure file** | `fig_transformer_emotion_deltas.png` |

---

## Slide 16 — Three Takeaways

| Element | Details |
|---|---|
| **Visual** | Three large numbered cards: **1** "Predict per author, not per post" (icon: person silhouette), **2** "Set-attention > TF-IDF > GRU" (icon: trophy), **3** "Emotion is redundant given text" (icon: overlapping circles). |
| **Key text** | Slide title: *"What We Learned"* |

**Design note:** Rule of three. Clean, memorable. No paragraphs.

---

## Slide 17 — Future Work & Thank You

| Element | Details |
|---|---|
| **Visual** | Two future-work bullets with icons: (1) rocket icon → *"Low-rank emotion residual on MiniLM"* (2) expand icon → *"Set Transformer with inducing points to scale beyond 200 posts"*. Below: *"Thank you! Questions?"* with team headshots. **[MEME PLACEHOLDER]** — a fun closing meme (e.g., a reaction meme about emotions being redundant, or "me explaining MBTI to my friends" format). |

---

# Figures Used (from `report/final/figures/`)

| Slide | Figure file |
|---|---|
| 7 | `fig_source_vs_reddit_emotion_distribution.png` |
| 8 | `fig_ms4_pipeline_diagram.png` |
| 13 | `fig_set_attention_author_models.png` |
| 14 | `fig_target_balanced_accuracy.png` |
| 15 | `fig_transformer_emotion_deltas.png` |

All other slides use custom diagrams, icons, or stat callouts (to be created in the slide tool of choice).

---

# Speaker Assignment Summary

| Speaker | Slides | Approx. Time |
|---|---|---|
| **Harry Hu** | 1 (intro), 2, 3, 4 | ~1:20 |
| **Tom Shan** | 5, 6, 7 | ~1:15 |
| **Wendy Wang** | 8, 9, 10, 11, 12 | ~1:40 |
| **Kemeng Zhang** | 13, 14, 15, 16, 17 | ~1:35 |
| **Total** | 17 slides | **~5:50** |
