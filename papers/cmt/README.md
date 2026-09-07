# CMT paper: stage-dependent learnability

This directory contains the redesigned analysis for the CMT paper. It does not alter or overwrite the original notebook outputs.

## Stage A + Stage B in one command

From the repository root:

```bash
python papers/cmt/analysis/run_stage_ab.py
```

The command performs two gated operations:

- **Stage A** validates and freezes the 3201-record modeling universe, all nine targets, the target-specific P1/P2/P3 feature lists, dependency/compiler provenance, and the exact five-fold framework-formula-grouped split manifest.
- **Stage B** runs the fixed ExtraTrees estimator across P1/P2/P3 for all nine targets using only the primary framework-formula-grouped validation. Complete out-of-fold predictions are saved for the later group-bootstrap learnability analysis.

Stage B is deliberately limited to the primary validation regime. Random validation and the coarse-chemistry-family, host-chemical-system, and working-ion stress tests are later robustness analyses and should not be run before the primary stage effect is established.

## Frozen estimator

Median imputation followed by `ExtraTreesRegressor` with 300 trees, `max_features=1.0`, `min_samples_leaf=1`, `bootstrap=False`, and `random_state=42`. `DummyRegressor(strategy="mean")` is evaluated on the identical folds.

CPU parallelism is an execution setting only. Folds are run serially, while each ExtraTrees fit uses the available cores up to a cap of 18. This mirrors the original Notebook 04 strategy and avoids nested parallelism.

## Interpretation boundary

`stage_b_earliest_stage_point_estimate.csv` is provisional. The final "earliest useful stage" requires the later framework-group-respecting bootstrap specified in the config. Stage B therefore does not make a final learnability claim.

## Resume behavior

If execution is interrupted, run the same command again. Completed target-stage checkpoints are reused only when their scientific signature matches the frozen feature list, split manifest, dataset identity, and estimator definition. Incomplete or stale checkpoints are ignored and recomputed.

## Stage E: final estimator sensitivity
After Stages B-D pass, run:

```bash
python papers/cmt/analysis/run_stage_e.py
```

Stage E performs only the secondary Ridge sensitivity on the frozen five-fold framework-formula split (9 targets x 3 stages x 5 folds = 135 fits) and applies the same 5000-replicate grouped bootstrap rule. It does not change the primary ExtraTrees result from Stage C and does not rerun domain or random validation.

## Final analysis freeze

Stages A–E have passed, and Stage F freezes the analysis without additional model fitting. The authoritative summary is:

`papers/cmt/results/stage_f/CMT_ANALYSIS_FREEZE_SUMMARY.md`

The current CMT scientific identity is **target-specific information provenance plus stage-dependent learnability**. P0/P4 leakage controls, the leakage-by-domain interaction, and the Ridge sensitivity are supporting analyses and do not define the paper's primary framing.

## Publication outputs

The authoritative CMT publication-output directory is:

`papers/cmt/publication/outputs/`

The root-level `figures/` directory belongs to the earlier combined electrode workflow and is retained only for provenance. Do not substitute those legacy figures for the frozen CMT stage-dependent-learnability figures without explicitly redesigning the manuscript and revalidating the claims.
