# Repository scope and manuscript separation

This repository contains shared multi-ion insertion-electrode infrastructure plus manuscript-specific analyses. It is not a single-manuscript narrative.

## Current CMT analysis

The current CMT paper asks which target-specific information is legitimately available at each computational stage and when useful predictive skill first appears. Its authoritative outputs are under `papers/cmt/`.

The CMT analysis uses the terminology:

- **framework-formula-grouped validation** for grouping by `working ion | reduced framework formula`;
- **coarse-chemistry-family-held-out validation** for the heuristic composition/anion grouping;
- **host-chemical-system-held-out validation** for `host_chemsys_no_working_ion`;
- **working-ion-held-out validation** for leave-one-ion-out.

P2 is not pre-DFT: its structural descriptors come from DFT-relaxed structures.

The dependency system is an **expert-defined dependency graph plus rule-based compiler**. The compiler did not automatically discover leakage, and the 2430/2430 match is not independent-annotator agreement.

## Legacy combined-workflow material

The root `notebooks/`, `results/`, `figures/`, and related documentation preserve an earlier combined analysis containing leakage controls, physics-consistency analysis, uncertainty/applicability-domain analysis, sodium triage, and the DFT spot check.

Those artifacts remain for provenance. They should not be used to redefine the scientific identity of the current CMT paper.

## PLOS ONE separation

The submitted PLOS ONE work concerns cross-database transferability for two-dimensional materials and is scientifically separate from this electrode repository.

## Digital Discovery separation

The planned Digital Discovery work concerns decision-aware active learning under chemical-domain shift. It is not yet frozen and should not reuse CMT stage-dependent-learnability results as if they were active-learning evidence.
