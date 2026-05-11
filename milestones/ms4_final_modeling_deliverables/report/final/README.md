# MS4 Final Report

Single-PDF, paper-style report for AC 209b / CS 1090b Project 66.

## Files

| Path | Purpose |
|---|---|
| `cs1090b_ms4_report_group66.pdf` | Submission PDF, 11 pages (body + appendix). |
| `cs1090b_ms4_report_group66.tex` | LaTeX source. |
| `neurips_2026.sty` | NeurIPS 2026 single-column style file (copied from `../template/`). |
| `figures/` | All embedded figures, copied from `../results/`. |

## How to rebuild

```bash
cd milestones/ms4_final_modeling_deliverables/report/final
pdflatex cs1090b_ms4_report_group66.tex
pdflatex cs1090b_ms4_report_group66.tex   # second pass for references
pdflatex cs1090b_ms4_report_group66.tex   # third pass for natbib citations
```

The PDF compiles with `pdflatex` against TeX Live 2022 or newer. No external
bibliography file is needed: references are inline in the `thebibliography`
environment at the end of the source.

## Numbers source-of-truth

All result numbers, tables, and figures in the report come from the tracked
artifacts in `../results/`. The numeric source of truth and headline
interpretation are documented in `../../ms4_experiment_summary.md`.

## Structure (per MS4 requirement)

The body addresses each required section in order:

1. Title page (project number, team).
2. Abstract.
3. Background and motivation.
4. Problem statement.
5. Data and EDA.
6. Methods and models (includes mathematical notation for loss, set-attention,
   and the bootstrap estimands; the AC 209b "method not covered in class" is
   permutation-invariant set/attention author aggregation).
7. Results and interpretation.
8. Conclusions and discussion.
9. Broader impact.
10. References.

Appendix A holds supplementary EDA visuals (token-length, emotion-source vs.
Reddit distributions), baseline diagnostics, per-dimension breakdowns,
set-attention stability/seed/epoch checks, and the end-to-end pipeline diagram.
