# Integrating the manuscript figure package

1. Extract the patch ZIP into the root of the local repository. Allow it to create the new `figures/`, `provenance/notebook_14/`, and `results/audits/notebook_14/` paths and to replace `documentation/figure_generation.md`.
2. Run `python software/manuscript_figures.py` from the repository root, or run `notebooks/14_manuscript_figure_generation.ipynb`.
3. Confirm that `provenance/notebook_14/14_final_decision.json` reports `PASS_MANUSCRIPT_FIGURES_GENERATED_FROM_LOCKED_OUTPUTS`.
4. Visually inspect the six main and ten supplementary figures.
5. Commit the additions with a new commit such as `Add manuscript figure generation package`, then push with GitHub Desktop.

The package does not retrain models, alter protocols, re-rank candidates, or execute Quantum ESPRESSO.
