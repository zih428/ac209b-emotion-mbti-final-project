# MS2 Submitted Summary

Source artifacts:

- `../artifacts/submitted/ms2_slides_submitted.pdf`
- `../artifacts/submitted/ms2_eda_submitted.ipynb`
- `../notes/ms2_presenter_script_part_1_2.md`

## Main Points

1. Project question: can emotion-related language help explain self-reported MBTI patterns in Reddit writing?
2. Data:
   - Hugging Face emotion dataset: 20,000 short texts, six balanced emotion labels, about 2.35 MB.
   - Kaggle Reddit MBTI dataset: 13,028,455 usable posts, 11,773 authors, 16 self-reported MBTI types, about 2.70 GB.
3. Wrangling:
   - Standardized labels and schema.
   - Removed 180 blank Reddit posts.
   - Derived four binary dimensions: `E/I`, `N/S`, `F/T`, `J/P`.
   - Used full Reddit data for count summaries and a fixed 250,000-post sample for expensive text-level EDA.
4. EDA:
   - Emotion source task is clean and balanced.
   - Reddit target task is highly imbalanced: INFP is about 22.94 percent of posts, ESFP about 0.18 percent.
   - Author contributions are concentrated: median author has about 272 posts, top 1 percent creates about 19 percent of the corpus.
5. Rescope:
   - Move away from direct 16-type MBTI prediction.
   - Use author-level splits and author-level evaluation.
   - Compare text-only baselines against emotion-informed models.

## Final MS2 Research Question

Do emotion-informed features improve author-level prediction of the four binary MBTI dimensions over direct text-to-MBTI baselines?

## Reading Note

The submitted PDF is useful for presentation framing, but the notebook and presenter script contain more detail than the slides. For later work, use this summary plus `ms2_eda_submitted.ipynb` and `../notes/ms2_presenter_script_part_1_2.md`.
