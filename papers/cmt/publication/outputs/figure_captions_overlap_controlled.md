# CMT figure captions - overlap-controlled draft

## Figure 1
Target-specific information provenance and stage eligibility. (a) Candidate descriptors are assigned separately for each target using the expert-defined dependency graph and rule-based compiler. P1 contains framework-composition and working-ion descriptors. P2 adds target-eligible descriptors obtained from DFT-relaxed structures. P3 adds target-eligible post-DFT energetic, stability, and electronic information while direct target duplicates and target-defining components remain excluded from the clean representations. The compiler reproduced the curated reference classification for all 2,430 feature-target pairs. (b) Number of eligible descriptors for each target at P1, P2, and P3.

## Figure 2
Stage-dependent learnability under framework-formula-grouped validation. DummyMean-relative MAE (a) and RMSE (b) skill are shown for the fixed ExtraTrees estimator at the composition (P1), relaxed-structure (P2), and legitimate post-DFT (P3) stages. Positive values indicate lower error than the fold-matched mean baseline. Black cell outlines mark the first stage satisfying the pre-specified useful-learnability rule: at least 10% improvement in both MAE and RMSE, with both 95% framework-group bootstrap intervals entirely above zero. Maximum volume change did not satisfy the full rule through P3.

## Figure 3
Incremental information value of later computational stages. Relative changes in MAE (a) and RMSE (b) are reported for P1 to P2, P2 to P3, and P1 to P3 under the fixed ExtraTrees framework-formula benchmark. Positive values indicate lower prediction error at the later stage. Points show observed relative improvement and horizontal bars show 95% paired framework-group bootstrap intervals. The dashed line marks zero change.

## Figure 4
Chemical-domain robustness of the net P1-to-P3 information gain. Relative MAE (a) and RMSE (b) reductions are shown for framework-formula-grouped validation and for the host-chemical-system, coarse-chemistry-family, and working-ion holdouts. Positive values indicate lower error at P3 than at P1. Framework-formula values come from the primary analysis, whereas the three chemical-domain panels use descriptive macro point estimates. These holdout results are robustness checks and are not independent earliest-stage classifications.
