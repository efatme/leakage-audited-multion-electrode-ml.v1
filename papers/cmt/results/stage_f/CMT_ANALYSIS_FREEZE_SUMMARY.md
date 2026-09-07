# CMT analysis freeze summary

## Freeze status
Stages A-E all passed. Stage F performs no model fitting and freezes the analysis for publication outputs.

## Primary earliest useful stage (fixed ExtraTrees, framework-formula-grouped validation)

| Target | Earliest useful stage |
|---|---|
| average_voltage | P1 |
| capacity_grav | P1 |
| capacity_vol | P1 |
| energy_grav | P1 |
| energy_vol | P1 |
| max_delta_volume | none_through_P3 |
| stability_charge | P2 |
| stability_discharge | P3 |
| stability_worst | P2 |

## Sensitivity summary
- Exact ExtraTrees/Ridge earliest-stage agreement: 6/9 targets.
- Earliest-stage assignment unchanged across 5%, 10%, and 20% thresholds: 6/9 targets.
- Descriptive P1-to-P3 macro improvement in both MAE and RMSE across chemical-domain target comparisons: 26/27.
- The three stability targets require the strongest qualification because their exact stage assignments are estimator- and threshold-sensitive and weaken under coarse-chemistry-family-held-out validation.
- Maximum volume change remains a negative primary learnability result through P3 under the frozen criterion, despite positive P1-to-P3 error reductions in the framework-formula analysis.

## Manuscript control placement
- P0 permissive baseline: Supporting Information.
- P4 direct-target positive control: Supporting Information.
- Leakage-by-domain interaction: Supporting Information.
- Ridge estimator sensitivity: Supporting Information, with a short qualification in the main discussion.
- Physics-constrained multitask benchmark: repository-only unless specifically needed.

## Writing rule
Do not describe the stability-stage assignments as estimator-independent properties. Use wording such as 'under the fixed primary ExtraTrees estimator'.

## Gate provenance
- Stage A: PASS (cmt_stage_ab_v1)
- Stage B: PASS (cmt_stage_ab_v1)
- Stage C: PASS (cmt_stage_c_v1)
- Stage D: PASS (cmt_stage_d_v1)
- Stage E: PASS (cmt_stage_e_v1)
