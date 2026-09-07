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

The original Matplotlib renderer is preserved for reproducibility in a clean plotting environment at:

`papers/cmt/publication/reference_rendering/make_cmt_publication_outputs_matplotlib_reference.py`

The reference artifacts in this package were generated in a clean environment with Python 3.13.5, Matplotlib 3.10.8, NumPy 2.3.5, and pandas 2.2.3. The user's analysis environment does not need to be changed to use or verify the figures.

## Figure standard

The main figures are designed at 190 mm full width and supplied as vector PDF/SVG plus 600 dpi TIFF and 300 dpi PNG previews. Artwork uses final-size lettering of approximately 7 pt or larger, restrained line weights, colorblind-aware quantitative encoding, numeric redundancy in heatmaps, and marker-shape redundancy in interval plots.

## Main figures

- Figure 1: target-specific information provenance and stage eligibility
- Figure 2: stage-dependent learnability
- Figure 3: incremental information value
- Figure 4: chemical-domain robustness of the net P1-to-P3 gain

## Text-overlap control

`figure_captions_overlap_controlled.md` was written from the frozen CMT claims register. `TEXT_OVERLAP_FIREWALL.md` records the separation from the submitted PLOS ONE paper. `caption_exact_overlap_audit.csv` is expected to contain no flagged exact contiguous 8-word matches.

## Submission files

Use the PDF figures as the primary vector artwork unless the journal portal requests raster artwork. TIFF files are provided as 600 dpi backups. Figure captions should remain outside the artwork.
