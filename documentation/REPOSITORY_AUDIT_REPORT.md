# GitHub-readiness audit report

Audit date: 2026-09-08

## Changes made

- Removed local patch/completed-run ZIP archives from the repository tree.
- Removed Jupyter checkpoint folders and transient CMT runtime fit checkpoints.
- Added `.gitattributes` and expanded `.gitignore` for cross-platform and packaging hygiene.
- Corrected the stale Stage B progress record from 12/27 to the final PASS state of 27/27 blocks, consistent with `stage_b_gate.json`.
- Relabeled the mismatched charged-state force audit as a non-accepted calculation and documented that it is not part of the accepted voltage energy chain.
- Synchronized the shared repository path map with notebooks 01–14.
- Fixed the leakage-compiler unit-test import path and added `pytest` to the Conda environment specification.
- Marked the root figure documentation as legacy combined-workflow material and retained the current CMT publication outputs separately.

## Files intentionally retained despite size

Some historical machine-readable outputs are tens of megabytes because they preserve record-level predictions required for earlier analyses. No individual tracked file exceeds GitHub's 100 MB hard per-file limit in this cleaned tree.

The largest retained historical file is approximately 60 MB. GitHub may display a warning for files larger than 50 MB, but they remain below the hard limit. Do not duplicate them in new ZIP archives inside Git history.

## Scientific boundaries preserved

- `framework_uid` is a framework-formula grouping, not a crystallographic framework.
- `coarse_family` is a heuristic composition/anion grouping, not a structural family.
- P2 contains DFT-relaxed structural information and is not a pre-DFT representation.
- The dependency system is an expert-defined dependency graph plus a rule-based compiler.
- 2430/2430 is compiler/reference agreement, not independent-annotator agreement.
- Stability-stage assignments are estimator-sensitive and should be qualified as results under the fixed primary ExtraTrees estimator.

## Final validation checks

The cleaned tree passed these checks after cleanup:

- CMT Stage A gate: PASS
- CMT Stage B gate: PASS
- CMT Stage C gate: PASS
- CMT Stage D gate: PASS
- CMT Stage E gate: PASS
- CMT Stage F gate: PASS
- CMT publication-output verification: PASS
- DFT repository verification: PASS
- leakage-compiler unit tests: 4/4 PASS
- no ZIP archives remain inside the GitHub-ready repository tree
- no Jupyter checkpoint folders remain
- no CMT runtime `_checkpoints` folders remain
- no individual file exceeds 100 MB
