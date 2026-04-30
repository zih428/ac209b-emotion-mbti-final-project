# MS3 Slide Plan — Sections 1 & 2 Presentation
**Project 66 · Harry Hu, Tom Shan, Wendy Wang, Kemeng Zhang · Target: 8 minutes · ~13 slides**

Design principle: **visuals first, text minimal**. Every slide is anchored on a single figure or diagram. Bullets are short phrases, not sentences. Numbers appear on the slide only when the speaker will cite them.

All figure paths below are relative to the notebook folder: `figures_ms3_sections12/` for exported PNGs; training / confusion-matrix figures are the ones inline in `MS3_Sections_1_2_Refined_Problem_and_Comprehensive_EDA.ipynb` (export to PNG before building the deck).

Speaker legend: **HH** = Harry Hu, **TS** = Tom Shan, **WW** = Wendy Wang, **KZ** = Kemeng Zhang.

---

## Section 1 — Summary of Milestone 2 Progress (≈1:45 total)

### Slide 1 — Title (00:00–00:10, 10 s) — HH
- **Visual:** team photo or MBTI × emotion word-cloud motif.
- **On-slide text (bare minimum):**
  - *AC 209b · Milestone 3 — Emotion-Informed MBTI Prediction*
  - Names · Project 66.
- **Purpose:** open the talk in one breath.

### Slide 2 — The Question & Two Datasets (00:10–00:55, 45 s) — HH
- **Visual:** two side-by-side dataset cards.
  - Left card: "Emotion (source)" — 20 k rows · 6 balanced classes · HuggingFace.
  - Right card: "Reddit MBTI (target)" — ~13 M posts · 16 types · Kaggle.
  - Arrow from emotion → Reddit labeled "transfer".
- **On-slide text:** one headline — *Can emotion-aware language signals predict MBTI?*
- **Optional small inset:** 6-class pie (balanced) vs 16-type bar (skewed), to foreshadow the imbalance.

### Slide 3 — Four MS2 Concerns → Four Design Decisions (00:55–01:35, 40 s) — HH
- **Visual:** 2×2 icon grid, each cell = one concern → arrow → one decision. No paragraphs.
  - 16-type skew → **4 binary dimensions (E/I, N/S, F/T, J/P)**.
  - Authors repeated → **author-level split**.
  - Single posts too short → **author-level aggregation**.
  - Does emotion help? → **matched-pair B1 vs B2 design**.
- Keep each cell ≤ 6 words.

### Slide 4 — Refined Research Question (01:35–01:55, 20 s) — HH
- **Visual:** one centered sentence, large font.
  - *Do text + emotion features beat text-only baselines on author-level MBTI — on all four binary dimensions?*
- **Lower strip:** 3 tiny tags — *controlled · author-level · comparative*.

---

## Section 2 — Extended EDA (≈2:45 total)

### Slide 5 — Short-Post Tail Is Noise (01:55–02:35, 40 s) — TS
- **Visual (primary):** `figures_ms3_sections12/post_length_cdf.png` — word-count CDF with a vertical dashed line at **5 words**.
- **Visual (optional inset top-right):** small table of duplicate rates by word-bucket (1-4 vs 5+), taken from code cell 7.
- **On-slide text:**
  - *Posts < 5 words ≈ "lol", "thanks", "same"*
  - *Decision: drop `word_count < 5`*
- Purpose: motivate the first subsampling threshold.

### Slide 6 — Author Feature Stability → 20-Post Floor (02:35–03:10, 35 s) — TS
- **Visual:** `figures_ms3_sections12/author_feature_stability.png` — bootstrap-std vs posts-per-author (left panel) with the 1/√n curve; split-half r (right panel).
- **On-slide text:**
  - *Bootstrap std ~ 1/√n, no elbow*
  - *Floor 20 halves noise vs 10, costs only ~5% of authors*
  - *Decision: `posts_per_author ≥ 20`*

### Slide 7 — Author Concentration → 200-Post Cap (03:10–03:50, 40 s) — TS
- **Visual:** `figures_ms3_sections12/author_concentration.png` side-by-side with `figures_ms3_sections12/author_concentration_lorenz.png`.
- **Callouts on figure:**
  - "Uncapped Gini ≈ 0.75" · "Top 1% ≈ 19% of posts".
  - Red band highlighting the chosen cap = 200 on the top-100-share curve.
- **On-slide text:**
  - *Cap = 200 → Gini 0.27 · retains ~1.7 M posts · clips top quartile*

### Slide 8 — Source → Target Transfer Is Feasible (03:50–04:30, 40 s) — WW
- **Visual (left):** `figures_ms3_sections12/source_target_transfer.png` — coverage-by-length-bucket bars.
- **Visual (right):** `figures_ms3_sections12/source_target_length_comparison.png` — source vs target length CDFs with 95th-pctile line.
- **On-slide text (two short phrases, each anchored to one panel):**
  - *Coverage rises with length: 59% → 92%*
  - *Reddit 95th pct ≈ 3× emotion corpus*
  - *Decision: soft 6-d probabilities, not hard labels*

### Slide 9 — EDA → Decisions (one-glance) (04:30–04:50, 20 s) — WW
- **Visual:** 2-row flow graphic (no table paragraphs):
  - Row A (Data): word-floor · post-floor · post-cap · soft-probs.
  - Row B (Modeling): sequence encoder · author-level aggregation · matched-pair B1/B2.
- **On-slide text:** header — *Every threshold is EDA-derived*.
- **Optional small inset:** `figures_ms3_sections12/label_balance_overview.png` (thumb) to remind the imbalance target.

---

## Section 3 — Baseline Models (≈2:40 total)

### Slide 10 — Two Baselines, One Fair Comparison (04:50–05:30, 40 s) — KZ
- **Visual:** side-by-side block diagram (redraw clean):
  - B1: tokens → Embed → GRU → Dense → 4 logits.
  - B2: same path **+ 6-d emotion probs concat** → Dense → 4 logits.
  - Shared "author-level majority vote" box on the right.
- **On-slide text (three tags under diagram):**
  - *Identical text encoder · single 6-d delta · BCE loss*
  - *Eval: author-level balanced accuracy*

### Slide 11 — Training Curves + Stage-1 Emotion (05:30–06:10, 40 s) — KZ
- **Visual (left):** B1 / B2 train + val loss curves (export code cells 31 & 37 figures).
- **Visual (right):** Stage-1 emotion classifier val-accuracy curve (code cell 35 figure), annotated *val_acc = 0.897, test = 0.908*.
- **On-slide text:**
  - *Both B1 and B2 plateau at val BCE ≈ 0.520*
  - *Stage-1 ≫ 1/6 chance → emotion probs carry signal*

### Slide 12 — Author-Level Results: Majority-Class Collapse (06:10–06:55, 45 s) — KZ
- **Visual (primary):** 2×4 confusion-matrix grid (code cell 43 figure — B1 top row, B2 bottom row, four dimensions).
- **Visual (side callout):** tiny 3-row bar comparing balanced accuracy — Majority / B1 / B2 — across the 4 dimensions (all four sit at ~0.50).
- **On-slide text:**
  - *Both baselines collapse to the prior on E/I, N/S, J/P*
  - *F/T: precision 86–88% when model commits T → signal is there, aggregator throws it away*

### Slide 13 — Interpretation: Where the Signal Dies (06:55–07:30, 35 s) — KZ
- **Visual:** a single annotated schematic — *per-post probs → hard majority vote → per-author label*, with a red "X" over the 50% threshold and a note that high-precision T posts never cross it.
- **On-slide text (two short lines):**
  - *Vanilla BCE ⇒ class-prior solution*
  - *Hard 50% vote ⇒ discards learned minority signal*

---

## Section 4 — Future Directions & Next Steps (≈0:50 total)

### Slide 14 — Final Pipeline: 3 Targeted Upgrades (07:30–07:55, 25 s) — WW (lead) + KZ (handoff)
- **Visual:** clean 3-stage pipeline diagram (Stage 1 DistilBERT · Stage 2 GRU + 6-d concat with class-weighted BCE · Stage 3 soft author aggregation with per-dim thresholds).
- **On-slide text (three chips above the stages):**
  - *(1) Class-weighted BCE*
  - *(2) Soft author aggregation*
  - *(3) DistilBERT emotion encoder*
- **Bottom-right tag:** *ablate one lever at a time*.

### Slide 15 — Next Steps & Open Questions for TF (07:55–08:00, 5 s handoff; body read on slide 14 transition) — All four
- **Visual:** 4 avatar tiles, each with a 3-word task:
  - **HH** — class-weighted BCE + soft aggregation on B2.
  - **TS** — DistilBERT fine-tune on emotion dataset.
  - **WW** — cache Reddit emotion probs · final Stage-2 training.
  - **KZ** — author-level evaluation · headline ablations · interpretability.
- **Lower strip — Open questions for TF (max 3 short bullets):**
  - *Raise 200-post cap for DistilBERT run?*
  - *Sqrt vs full inverse-freq weights on N/S (93/7)?*
  - *AUC-ROC reporting once soft-aggregation is in — acceptable?*

---

## Timing audit (cumulative)

| Slide | End time | Cumulative | Speaker |
|---|---|---|---|
| 1 | 0:10 | 0:10 | HH |
| 2 | 0:55 | 0:55 | HH |
| 3 | 1:35 | 1:35 | HH |
| 4 | 1:55 | 1:55 | HH |
| 5 | 2:35 | 2:35 | TS |
| 6 | 3:10 | 3:10 | TS |
| 7 | 3:50 | 3:50 | TS |
| 8 | 4:30 | 4:30 | WW |
| 9 | 4:50 | 4:50 | WW |
| 10 | 5:30 | 5:30 | KZ |
| 11 | 6:10 | 6:10 | KZ |
| 12 | 6:55 | 6:55 | KZ |
| 13 | 7:30 | 7:30 | KZ |
| 14 | 7:55 | 7:55 | WW |
| 15 | 8:00 | 8:00 | All (closing) |

Workload by person: HH ≈ 1:55, TS ≈ 1:55, WW ≈ 1:05, KZ ≈ 2:40 + closing. Rebalance if desired (e.g., move Slide 13 to WW to even out KZ's block).

---

## Visual-asset checklist (before building the PPTX)

Already on disk in `figures_ms3_sections12/`:
- `post_length_cdf.png` — Slide 5.
- `author_feature_stability.png` — Slide 6.
- `author_concentration.png`, `author_concentration_lorenz.png` — Slide 7.
- `source_target_transfer.png`, `source_target_length_comparison.png` — Slide 8.
- `label_balance_overview.png`, `binary_dimension_balance.png` — thumbnails on Slides 2 / 9.
- `feature_signal_analysis.png` — optional backup for Slide 5 / 9.

Export from the notebook (not yet PNGs — save via `fig.savefig(...)` during the next run):
- Code cell **31** — B1 loss curve.
- Code cell **37** — B2 loss curve.
- Code cell **35** — Stage-1 emotion val curves.
- Code cell **43** — 2×4 confusion-matrix grid.

Redraw clean (vector, not from the notebook's ASCII blocks):
- B1 vs B2 architecture diagram — Slide 10.
- Final 3-stage pipeline diagram — Slide 14.
- Majority-vote "signal dies here" schematic — Slide 13.

## Style notes (to avoid last milestone's text-density deduction)

- Max 6 short bullets or 1 short sentence per slide, font ≥ 24 pt.
- No full tables on slides. Any table in the notebook that matters gets distilled into 2–3 numbers as callouts on a figure.
- Every slide answers in one glance: *what is the visual, what is the one takeaway*.
- Do not include equations; 1/√n can be verbalized, not printed.
- Put section dividers (1, 2, 3, 4) as tiny progress dots in the footer instead of full divider slides — saves time.
