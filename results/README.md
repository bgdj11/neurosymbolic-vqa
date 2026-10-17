# Reported thesis results

`thesis_reported.json` records the final values from `DIPLOMSKI_RAD_BOGDAN_PDF_JUL.pdf`, Tables 6.2 and 6.4 through 6.8, cross-checked against slides 13 through 15 of the defense presentations. The PDF's SHA-256 is included to identify the source version.

These are manually transcribed research results. The final Kaggle output files were not retained, as confirmed by the author. The JSON is not a newly generated evaluation summary, and exact correct-answer counts must not be inferred from rounded percentages.

The README training figure is taken directly from thesis Figure 6.1, with translated labels. Its original and translated versions are documented in [the figure sources](../docs/assets/README.md).

Earlier summaries and plots have been moved to `.local/results/historical/` and `.local/figures/historical/`. They remain on the author's machine and are excluded from the public result presentation. Earlier committed versions remain available in Git history.

New evaluations write to ignored directories such as `results/runs/clevr_baseline/`. Keep per-question outputs, checkpoints, and graph caches local. To publish a future run, select and review its summary and provenance explicitly rather than force-adding the entire results directory.
