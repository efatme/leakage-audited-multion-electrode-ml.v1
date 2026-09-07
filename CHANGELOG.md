# Changelog

## 2026-09-08 — CMT repository redesign and GitHub cleanup

- Finalized the six-figure CMT publication map: core Figures 1–4 plus exact-data supporting applicability-domain and sodium/DFT handoff Figures 5–6.
- Added the frozen CMT Stage A–F analysis under `papers/cmt/`.
- Reframed the repository root as shared infrastructure plus paper-specific analyses instead of a single combined manuscript.
- Preserved the original 14-notebook electrode workflow and legacy figures for provenance.
- Added a Digital Discovery placeholder without claiming unperformed active-learning results.
- Removed local patch ZIPs, completed-run ZIPs, notebook checkpoints, and transient CMT fit checkpoints from the GitHub-ready tree.
- Added `.gitattributes` for cross-platform line-ending consistency and expanded `.gitignore` for packaging/runtime artifacts.
- Corrected the stale Stage B progress record to match the final PASS gate (27/27 target-stage blocks).
- Relabeled the historical charged-state force audit as evidence from a non-accepted calculation; it is not the force audit of the accepted charged SCF energy.
- Added beginner-oriented GitHub Desktop upload guidance and repository-scope documentation.
- Synchronized the shared repository-path helper with the configured notebook layout and fixed the leakage-compiler test import path.
