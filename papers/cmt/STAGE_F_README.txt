CMT Stage F - analysis freeze

Purpose:
- No model fitting.
- Consolidates validated Stages A-E.
- Creates the master result matrix, claims register, figure/table plan, and SHA256 manifests.

Windows / Jupyter command:
!python -u papers/cmt/analysis/run_stage_f.py

Expected final line includes:
Analysis status: FROZEN_FOR_PUBLICATION_OUTPUTS

Outputs:
papers/cmt/results/stage_f/
