# CMT portability note

The executable CMT scripts and configuration files use repository-relative paths and are portable across Windows/Linux when run from the repository root.

Some frozen audit CSV files under `papers/cmt/results/stage_c/`, `stage_d/`, and `stage_e/` contain the absolute Windows paths recorded at the time of the original run. Those strings are execution-provenance evidence only; they are not active path dependencies and have intentionally not been rewritten after the fact.

The authoritative scientific gates and machine-readable result tables remain valid independently of the local checkout location.
