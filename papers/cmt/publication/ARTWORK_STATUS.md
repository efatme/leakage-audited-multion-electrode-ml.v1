# CMT artwork status

Figures 1–4 in `outputs/` are the authoritative core CMT artifacts generated from the frozen Stage A–F machine-readable outputs.

Figures 5–6 are supporting main-text figures added after the Stage F analysis freeze. They were rebuilt from the exact machine-readable source tables preserved by the earlier electrode workflow:

- Figure 5: `figures/source_data/Figure_5_applicability_domain_source.csv`
- Figure 6: `figures/source_data/Figure_6_sodium_candidates_source.csv` and `figures/source_data/Figure_6_dft_voltage_source.csv`

Adding Figures 5–6 does **not** change any Stage A–F model fit, useful-learnability classification, bootstrap result, or domain-robustness conclusion. Figure 5 is a supporting P1 applicability-domain diagnostic. Figure 6 is a supporting downstream DFT handoff and is not statistical validation of the ML analysis.

The repository also preserves the older six-figure combined-workflow set under the root `figures/` directory for provenance. Those legacy figures should not be substituted for the paper-specific CMT Figure 1–4 core analysis.

Any visually redesigned scientific figure used for submission must preserve the exact underlying numerical source data. Style-only concept images or AI-redrawn approximate charts should not replace quantitative repository figures unless they are rebuilt from the exact source tables and rechecked numerically.
