> **Legacy combined-workflow documentation.** These figure files document the earlier combined electrode workflow retained for provenance. The current CMT paper-specific figures and captions are under `papers/cmt/publication/outputs/`.

# Manuscript figure map

All figures are generated from locked repository outputs by `notebooks/14_manuscript_figure_generation.ipynb` and `software/manuscript_figures.py`. No model retraining, protocol selection, candidate re-ranking, or DFT execution occurs during figure generation.

| Figure | Short title | Primary source | Manuscript role | Claim boundary |
|---|---|---|---|---|
| Figure 1 | End-to-end workflow | Notebook 01 counts; accepted DFT voltage table | Methods overview | Workflow summary only |
| Figure 2 | Multi-ion dataset landscape | Notebook 01 family counts | Dataset description | Descriptive composition; no representativeness claim |
| Figure 3 | Leakage-induced optimism | Notebook 04 compact benchmark table | Central leakage result | P0 is a post-hoc full-feature baseline without the direct target; ratios do not represent deployable performance |
| Figure 4 | Physics-constraint value | Notebook 05 physical-consistency metrics | Multi-task learning result | Shows consistency-error reduction, not universal predictive improvement |
| Figure 5 | Applicability-domain error | Notebook 07 AD-stratified errors | Uncertainty and domain analysis | Some target/split cells have ratios at or below one |
| Figure 6 | Sodium triage and DFT spot-check | Notebook 08 candidates; Notebook 10 selection; accepted QE voltage | Case study | One fixed-geometry PBE spot-check; no experimental or broad validation claim |
| Figure S1 | Protocol feature counts | Notebook 02 protocol audit | Descriptor-policy detail | Counts are target specific |
| Figure S2 | Compiler validation | Notebook 03 validation summary | Rule-based compiler verification | Agreement is against the locked expert audit |
| Figure S3 | Clean domain-shift penalty | Notebook 04 benchmark table | Validation sensitivity | Ratios are relative to random splitting within P1 |
| Figure S4 | Predictive trade-off of constraints | Notebook 05 predictive metrics | Physics-constraint sensitivity | Ratios near one indicate limited average MAE change |
| Figure S5 | Conformal coverage | Notebook 07 compact UQ table | Calibration diagnostic | Observed coverage is workflow specific |
| Figure S6 | Out-of-domain fraction | Notebook 07 compact UQ table | Applicability-domain diagnostic | Flags are model/feature-space specific |
| Figure S7 | Sodium screening criteria | Notebook 08 funnel audit | Case-study selection | Individual criterion counts and the combined pass count are not a discovery claim |
| Figure S8 | Sodium rank robustness | Notebook 08 candidate table | Ranking sensitivity | Error bars show half-IQR around Monte Carlo median rank |
| Figure S9 | Charged-state k-point convergence | QE convergence trend | DFT convergence evidence | Charged-state energy trend only |
| Figure S10 | Voltage convergence | Accepted QE voltage table | DFT convergence evidence | Fixed geometry only; unconverged relaxations excluded |
