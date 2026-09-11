# Multi-ion insertion-electrode machine-learning reproducibility repository

This repository preserves the shared extraction, descriptor-provenance, validation, uncertainty, screening, and DFT infrastructure for a multi-ion insertion-electrode machine-learning research program. It also contains a paper-specific, frozen analysis for the **Computational Materials Today (CMT)** study on **target-specific information provenance and stage-dependent learnability**.

The repository intentionally preserves earlier analysis history rather than rewriting it in place. Current paper-specific outputs are separated from legacy combined-workflow outputs so that provenance remains auditable.

## Archived CMT release

The frozen Computational Materials Today analysis is archived on Zenodo:

**DOI:** [10.5281/zenodo.22651122](https://doi.org/10.5281/zenodo.22651122)

GitHub release: `cmt-analysis-v1.0.1`

This archived release corresponds to the frozen Stage A–F CMT analysis and its publication outputs.

## Paper map

| Study | Scientific identity | Repository status |
|---|---|---|
| PLOS ONE | Cross-database transferability for 2D materials | Separate project/repository; not reproduced here |
| CMT | Target-specific descriptor eligibility and stage-dependent learnability for computed insertion electrodes | **Frozen analysis available under `papers/cmt/`** |
| Legacy electrode workflow | Leakage controls, physics consistency, UQ/AD, sodium triage, and DFT spot check | Preserved under the original notebooks/results/figures for provenance |

## CMT analysis status

The CMT analysis is frozen for publication outputs. The primary comparison uses the same fixed ExtraTrees estimator across the nested information stages:

- **P1:** composition representation;
- **P2:** composition plus target-eligible descriptors from DFT-relaxed structures;
- **P3:** legitimate target-specific post-DFT decision-support information.

The primary validation is **framework-formula-grouped validation**, where `framework_uid` means `working ion | reduced framework formula`. It is not a crystallographic framework identifier.

The CMT pipeline and frozen results are documented in:

- `papers/cmt/README.md`
- `papers/cmt/results/stage_f/CMT_ANALYSIS_FREEZE_SUMMARY.md`
- `papers/cmt/publication/outputs/`

## Repository structure

| Path | Contents |
|---|---|
| `papers/cmt/` | Current CMT stage-dependent-learnability analysis, gates, results, tables, and publication outputs |
| `notebooks/` | Fourteen original workflow notebooks retained for provenance |
| `data/processed/` | Processed and harmonized tables used by the workflow |
| `results/` | Historical audits, metrics, predictions, statistics, and validation outputs |
| `provenance/` | Decisions, manifests, hashes, and supplementary evidence |
| `configuration/` | Runtime, DFT, repository-layout, and legacy figure configuration |
| `software/` | Shared repository utilities, dependency-compiler code, and tests |
| `figures/` | **Legacy combined-workflow figure set**, retained for provenance; not the current CMT main-figure set |
| `documentation/` | Run order, data policy, legacy figure documentation, and GitHub/repository guidance |
| `dft_validation/` | Frozen Quantum ESPRESSO fixed-geometry spot-check package and provenance |

## Installation

A Conda environment specification is provided in `environment.yml`.

```bash
conda env create -f environment.yml
conda activate cmt-insertion-electrode-workflow
jupyter lab
```

Quantum ESPRESSO is not installed by `environment.yml`.

## CMT reproduction commands

Run commands from the repository root. The analysis is already frozen, so rerunning is necessary only for independent reproduction.

```bash
python papers/cmt/analysis/run_stage_ab.py
python papers/cmt/analysis/run_stage_c.py
python papers/cmt/analysis/run_stage_d.py
python papers/cmt/analysis/run_stage_e.py
python papers/cmt/analysis/run_stage_f.py
```

The final publication outputs can be verified without refitting models:

```bash
python papers/cmt/publication/verify_cmt_publication_outputs.py
```

## Legacy notebook workflow

The original 14-notebook sequence is retained because it records the development history of the electrode project. See `documentation/run_order.md`. The legacy figure set in `figures/` belongs to that earlier combined workflow and should not be confused with the current CMT publication outputs.

## DFT scope and known provenance boundary

The accepted fixed-geometry PBE voltage is reconstructed from the accepted selected and denser-reference SCF outputs. Endpoint relaxation attempts were excluded because they did not meet the prescribed force-convergence criterion.

A historical charged-state force audit came from a **different, non-accepted calculation** and has been explicitly relabeled under `dft_validation/audits/`. It must not be interpreted as a force audit of the accepted charged SCF energy.

Reconstruct and verify the frozen DFT result with:

```bash
python dft_validation/scripts/reconstruct_voltage.py
python dft_validation/scripts/verify_repository.py
```

## Data and repository policy

Raw Materials Project acquisition payloads, credentials, pseudopotential binaries, restart files, private caches, and local ZIP packaging artifacts are not tracked. Processed evidence required to inspect the analyses is included.

Large historical CSV outputs are retained because they support prior analyses. New packaging ZIPs, temporary model checkpoints, and local submission bundles are intentionally excluded from Git history.

## Citation and license

Citation metadata are provided in `CITATION.cff`. Original repository content is distributed under the BSD-3-Clause License. See `LICENSE` and the third-party notices retained under `dft_validation/`.

## Authors

- Md. Efatuzzaman Efat — corresponding author  
  Email: efatuzzaman@gmail.com
- Md. Saiful Islam,Phd
