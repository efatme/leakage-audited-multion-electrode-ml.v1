# CMT publication outputs

This folder contains the **frozen publication figures and tables** for the CMT analysis. No model fitting is performed here.

## Windows / broken Matplotlib font installations

The supplied PDF, SVG, TIFF, and PNG-preview figures are already rendered and checksum-frozen. **You do not need to re-render them on the Windows server.**

Run only:

```bash
python papers/cmt/publication/verify_cmt_publication_outputs.py
```

or, for compatibility with the earlier command:

```bash
python papers/cmt/publication/make_cmt_publication_outputs.py
```

Both commands perform verification only and do **not** import Matplotlib.

The original renderer for Figures 1–4 is preserved at:

`papers/cmt/publication/reference_rendering/make_cmt_publication_outputs_matplotlib_reference.py`

The exact-data renderer for supporting Figures 5–6 is preserved at:

`papers/cmt/publication/reference_rendering/make_cmt_figures_5_6_exact_source.py`

The analysis environment does not need to be changed to use or verify the frozen figures.

## Figure standard

The main figures are supplied as vector PDF/SVG plus 600 dpi TIFF and 300 dpi PNG previews. Artwork uses publication-scale lettering, restrained line weights, compact legends, and numerical encodings that can be checked against machine-readable source tables. Captions remain outside the artwork.

## Main figures

1. **Figure 1:** target-specific information provenance and stage eligibility.
2. **Figure 2:** stage-dependent learnability.
3. **Figure 3:** incremental information value.
4. **Figure 4:** chemical-domain robustness of the net P1-to-P3 gain.
5. **Figure 5:** P1 applicability-domain diagnostic across validation regimes.
6. **Figure 6:** sodium screening landscape with the fixed-geometry PBE handoff candidate.

Figures 1–4 are the core CMT analysis figures. Figures 5–6 are supporting main-text figures retained by manuscript-design choice. They do not redefine the paper's novelty: Figure 5 is a diagnostic, and Figure 6 is a downstream computational handoff rather than statistical validation of the ML analysis.

## Exact-data provenance for Figures 5–6

Figures 5–6 were rebuilt from the preserved machine-readable source tables rather than from style-transfer or AI-redrawn approximations. Copies of those small source tables are stored under `outputs/source_data/`, with their original repository locations recorded in `FIGURE_SOURCE_MAP.csv`.

## Text-overlap control

`figure_captions_overlap_controlled.md` is written in the CMT information-provenance/stage-learnability framing. `TEXT_OVERLAP_FIREWALL.md` records the conceptual separation from the submitted PLOS ONE paper. The existing exact 8-word audit file predates the addition of Figures 5–6; the complete manuscript-level overlap audit should be rerun once the final six-figure manuscript text is frozen.

## Submission files

Use the PDF figures as the primary vector artwork unless the journal portal requests raster artwork. TIFF files are provided as 600 dpi backups. PNG files are repository/manuscript previews.
