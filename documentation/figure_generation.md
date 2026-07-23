# Figure-generation policy and instructions

Publication figures are generated only from locked repository outputs.

## Reproduction

From the repository root:

```bash
python software/manuscript_figures.py
```

or run:

```text
notebooks/14_manuscript_figure_generation.ipynb
```

The generator writes:

- `figures/main/`: six manuscript figures in PDF and 600-dpi PNG
- `figures/supplementary/`: ten supplementary figures in PDF and 600-dpi PNG
- `figures/source_data/`: figure-level source tables and SHA256 hashes
- `figures/figure_manifest.csv`: generated-file manifest
- `figures/figure_generation_receipt.json`: generation receipt

## Locked controls

Figure generation must not:

- fit or tune models;
- change train/test splits;
- alter descriptor protocols;
- select or re-rank candidates;
- run Quantum ESPRESSO;
- use unconverged relaxed endpoints in the voltage calculation.

The figure generator reads only selected CSV/TSV outputs already present in the clean-room repository. Every figure must be visually inspected before manuscript submission. PDF is the preferred vector manuscript format; PNG files are supplied for review and compatibility.
