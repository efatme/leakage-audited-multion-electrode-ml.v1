> **Legacy combined-workflow documentation.** These figure files document the earlier combined electrode workflow retained for provenance. The current CMT paper-specific figures and captions are under `papers/cmt/publication/outputs/`.

# Manuscript figure captions

## Main figures

**Figure 1. Leakage-audited multi-ion insertion-electrode learning workflow.** The study begins with extraction and normalization of 3,354 insertion-electrode records spanning Li, Na, and K. A machine-readable battery-property dependency graph and rule-based leakage compiler applied to the expert-defined dependency graph define target-specific descriptor protocols. Models are then evaluated under random, framework-formula-grouped, coarse-chemistry-family-held-out, host-chemical-system-held-out, and working-ion-held-out splits. The workflow further includes physics-constrained multi-task learning, uncertainty and applicability-domain analysis, sodium candidate triage, and one convergence-audited fixed-geometry Quantum ESPRESSO spot-check.

**Figure 2. Chemical-family composition of the multi-ion dataset.** Stacked bars show the percentage of records assigned to the most prevalent coarse chemical families for Li (n = 2,774), Na (n = 416), and K (n = 164). Less frequent categories are combined as “Other families.” The plot describes the composition of the extracted computational dataset and does not imply equal or exhaustive coverage of insertion-electrode chemistry.

**Figure 3. Performance optimism associated with post-hoc feature leakage across validation regimes.** Each cell reports the ratio of the best-model mean absolute error (MAE) under the clean composition-only protocol P1 to the corresponding MAE under the full-feature post-hoc baseline P0. P0 excludes the direct target column but permits post-hoc computed-record descriptors that are unavailable at the intended composition-only prediction stage. Ratios greater than one indicate lower apparent error under the leaky baseline. The comparison is shown for random, framework-formula-grouped, host-chemical-system-held-out, coarse-chemistry-family-held-out, and working-ion-held-out validation.

**Figure 4. Reduction in physical inconsistency from soft physics constraints under the clean composition-only protocol.** Bars show the percentage reduction in mean energy-consistency MAE and stability-consistency MAE for soft-constrained direct predictions relative to unconstrained direct predictions. Reductions are summarized separately across the five validation regimes. This figure evaluates physical self-consistency and should not be interpreted as evidence that soft constraints universally improve every predictive target.

**Figure 5. Error stratification by applicability-domain status under the clean protocol.** Each cell reports the ratio of out-of-domain MAE to in-domain MAE for the same target and validation split. Values above one indicate larger errors among predictions flagged as out of domain. Elevated ratios are most pronounced for volume change and worst endpoint stability under several splits, whereas some cells are near or below one, demonstrating that the applicability-domain flag is informative but not uniformly monotonic for every target and split.

**Figure 6. Sodium candidate triage and original fixed-geometry DFT spot-check.** Screen-passing sodium candidates are plotted by computed gravimetric energy and worst endpoint stability; marker size represents the final triage score. The selected Na1-3CoPCO7 case is highlighted. The convergence-audited fixed-geometry PBE calculation gives an average voltage of 3.203421 V for the selected k-point meshes and 3.203062 V for the denser reference, an absolute difference of 0.000359 V. Additional endpoint-relaxation attempts did not satisfy the prescribed force-convergence criteria and are excluded from the voltage calculation.

## Supplementary figures

**Figure S1. Target-specific feature counts across descriptor protocols.** The heat map reports the number of features retained for each target under protocols P0-P4.

**Figure S2. Validation of the rule-based leakage compiler applied to the expert-defined dependency graph against the locked expert audit.** Leakage-class and protocol-allowance agreement are shown for each target. Across 2,430 audited target-feature rows, agreement was 100% and no false-safe classification was observed.

**Figure S3. Clean-protocol MAE relative to random splitting.** Each cell gives the ratio of P1 MAE under the indicated split to P1 MAE under random splitting for the same target. Values above one indicate degradation relative to random splitting.

**Figure S4. Predictive MAE trade-off of soft physics constraints.** Each cell reports soft-constrained direct MAE divided by unconstrained direct MAE under P1. Values near one indicate limited average predictive change while the consistency constraints are imposed.

**Figure S5. Conformal interval coverage under the clean protocol.** Observed coverage is reported across targets and validation splits using the locked uncertainty workflow.

**Figure S6. Percentage of predictions flagged out of domain.** The heat map reports the percentage of P1 predictions classified as out of domain for each target and split.

**Figure S7. Sodium case-study screening criteria.** Bars report the number of sodium records satisfying each individual criterion; the final bar gives the number satisfying all case-study criteria simultaneously.

**Figure S8. Rank robustness of the top sodium candidates.** Deterministic shortlist rank is compared with the Monte Carlo median rank. Vertical error bars represent one half of the interquartile range.

**Figure S9. Charged-state k-point energy convergence.** Absolute energy differences per atom are shown for the 4x5x3 to 5x6x4 and 5x6x4 to 6x7x4 charged-state refinements.

**Figure S10. Fixed-geometry voltage convergence.** The selected-mesh and denser-reference average voltages are shown. Their absolute difference is 0.000359 V.
