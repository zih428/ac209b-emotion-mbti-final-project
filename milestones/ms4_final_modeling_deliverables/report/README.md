# MS4 Report Workspace

Use this folder for final report drafts, figures selected for the report, and reference notes.

Recommended report arc:

1. Motivation and problem statement.
2. Data and only the EDA findings that directly drove modeling.
3. MS3 baseline failure diagnosis.
4. Scientific framing of emotion as a text-derived transferred representation.
5. Corrected baseline layer: TF-IDF, corrected GRU text, and corrected GRU plus emotion.
6. Transformer author layer: frozen transformer features and set/attention author aggregation.
7. Emotion increment analysis: real emotion versus text-only, checked against shuffled-emotion negative controls and activity/length controls.
8. Paired bootstrap comparison of the clean p200 set/attention text model against TF-IDF and corrected GRU baselines.
9. Interpretation, limitations, broader impact, and future work.

Avoid dumping notebook output. The report should read like a concise paper.

Do not frame emotion probabilities as independent measurements or causal mediators. The defensible claim is that the project tests whether emotion-derived transferred representations add incremental predictive information beyond matched text representations under author-level evaluation; the current result is informative but not robustly incremental.
