# CMT publication package

This directory contains the frozen publication-layer outputs for the CMT manuscript:

**Effect of Computational Workflow Stage on Machine Learning Prediction of Computed Insertion Electrode Properties**

The publication scripts do not fit or tune machine-learning models. They read the frozen Stage A-F analysis outputs and render or verify the manuscript figures and machine-readable publication tables.

## Canonical layout

```text
papers/cmt/publication/
|-- FIGURE_SOURCE_MAP.csv
|-- make_cmt_submission_figures.py
|-- verify_cmt_publication_outputs.py
|-- README.md
|-- outputs/
|   |-- Fig2_plot_data.csv
|   |-- Table1_target_level_summary.csv
|   |-- TableS1_stage_c_full_skill_bootstrap.csv
|   |-- TableS2_threshold_sensitivity.csv
|   |-- TableS3_domain_robustness_summary.csv
|   |-- TableS4_estimator_sensitivity.csv
|   `-- source_data/
|       |-- Fig5_applicability_domain_source.csv
|       |-- Fig6_dft_voltage_source.csv
|       `-- Fig6_sodium_candidates_source.csv
`-- submission_artwork/
    |-- Fig1_*_submission.{pdf,svg,tif}
    |-- Fig1_*_submission_preview.png
    |-- ...
    |-- Fig6_*_submission.{pdf,svg,tif}
    |-- Fig6_*_submission_preview.png
    |-- submission_artwork_run_manifest.json
    `-- submission_artwork_sha256.csv
```

`outputs/` contains machine-readable publication tables and the preserved source tables required for the supporting applicability-domain and sodium/DFT-handoff figures.

`submission_artwork/` contains the final journal artwork. PDF and SVG are vector outputs, TIFF files are 600 dpi raster backups, and PNG files are 300 dpi previews.

## Re-render the final figures

From the repository root on Windows:

```bat
conda activate cmt-insertion-electrode-workflow
python -u papers\cmt\publication\make_cmt_submission_figures.py --repo-root "%CD%" --check-only
python -u papers\cmt\publication\make_cmt_submission_figures.py --repo-root "%CD%"
```

On Linux/macOS:

```bash
python -u papers/cmt/publication/make_cmt_submission_figures.py --repo-root "$(pwd)" --check-only
python -u papers/cmt/publication/make_cmt_submission_figures.py --repo-root "$(pwd)"
```

The renderer uses frozen source data only. Re-rendering the figures does not refit the models.

## Verify the publication package

From the repository root:

```bash
python papers/cmt/publication/verify_cmt_publication_outputs.py
```

The verifier checks all six final figures in PDF, SVG, TIFF, and PNG-preview formats; artwork SHA256 values; frozen input hashes from the artwork run manifest; SVG structural validity; the main publication table, four supplementary tables, Figure 2 plot data, and the three preserved supporting source-data tables.

The verifier does not import Matplotlib and does not perform model fitting.

## Main figures

1. **Figure 1:** target-specific information provenance and stage eligibility.
2. **Figure 2:** stage-dependent learnability.
3. **Figure 3:** incremental information value.
4. **Figure 4:** chemical-domain robustness of the net P1-to-P3 gain.
5. **Figure 5:** P1 composition-space applicability-domain diagnostic across validation regimes.
6. **Figure 6:** sodium screening landscape and fixed-geometry PBE handoff.

Figures 1-4 summarize the core stage-dependent analysis. Figure 5 is a supporting applicability-domain diagnostic. Figure 6 is a downstream computational handoff and is not statistical or experimental validation of the machine-learning models.

## Figure provenance

`FIGURE_SOURCE_MAP.csv` records the analysis source for each figure. Figures 2-4 are generated from frozen CMT Stage C/D/F result tables. Figure 1 combines frozen descriptor counts with the pre-specified workflow-stage definitions. Figures 5-6 use the preserved machine-readable tables under `outputs/source_data/`.

The canonical final submission renderer is:

```text
papers/cmt/publication/make_cmt_submission_figures.py
```
