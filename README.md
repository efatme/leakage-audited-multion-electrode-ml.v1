
# Leakage-audited multi-ion insertion-electrode learning

This repository contains the clean-room computational package for a
physics-constrained, leakage-audited machine-learning study of multi-ion
insertion-electrode materials. The workflow combines target-specific leakage
control, validation under increasingly difficult chemical-domain shifts,
physics-consistency analysis, uncertainty and applicability-domain diagnostics,
sodium-ion candidate triage, and one original Quantum ESPRESSO computational
spot check.

Repository:  
https://github.com/efatme/leakage-audited-multion-electrode-ml.v1

## Scientific scope

The repository supports the following study components:

- multi-ion insertion-electrode data extraction and harmonization;
- a machine-readable battery-property dependency graph;
- automatic target-specific leakage compilation;
- leakage-permissive and leakage-controlled descriptor protocols;
- random, grouped, and leave-ion-out validation;
- physics-constrained multi-task learning;
- uncertainty and applicability-domain analysis;
- sodium-ion candidate screening and provenance-controlled DFT preparation;
- publication figures generated only from locked outputs.

The study is a computational benchmarking and validation investigation. It does
not claim experimental validation, comprehensive materials discovery, or
general real-world deployment performance.

## Repository structure

| Path | Contents |
|---|---|
| `notebooks/` | Fourteen clean notebooks in the documented run order |
| `data/processed/` | Processed and harmonized tables used by the workflow |
| `results/` | Audits, metrics, predictions, statistics, and validation outputs |
| `provenance/` | Decisions, manifests, hashes, and supplementary evidence |
| `configuration/` | Runtime, DFT, repository-layout, and figure configuration |
| `software/` | Reusable leakage and manuscript-figure utilities |
| `figures/main/` | Six manuscript figures in PDF and 600-dpi PNG |
| `figures/supplementary/` | Ten supplementary figures in PDF and 600-dpi PNG |
| `figures/source_data/` | Figure-level source tables and hash records |
| `documentation/` | Run order, data policy, captions, and figure instructions |
| `dft_validation/` | Frozen Quantum ESPRESSO fixed-geometry spot-check package |

## Installation

A Conda environment specification is provided in `environment.yml`.

```bash
git clone https://github.com/efatme/leakage-audited-multion-electrode-ml.v1.git
cd leakage-audited-multion-electrode-ml.v1
conda env create -f environment.yml
conda activate cmt-insertion-electrode-workflow
jupyter lab
```

The environment includes the Python libraries used by the notebooks and figure
generator. Quantum ESPRESSO is not installed by `environment.yml`.

## Notebook execution

The authoritative sequence is listed in
[`documentation/run_order.md`](documentation/run_order.md). Run notebooks from
the repository root so that repository-relative paths resolve consistently.

Some upstream acquisition steps require authorized Materials Project access.
The raw sodium acquisition JSON is intentionally not redistributed. Its policy
is recorded in `configuration/runtime_data_policy.json`. The processed evidence
needed to inspect the published analyses is included.

## Reproducing the figures

From the repository root, run:

```bash
python software/manuscript_figures.py
```

Alternatively, run:

```text
notebooks/14_manuscript_figure_generation.ipynb
```

The generator reads locked repository outputs only. It does not fit models,
change splits, alter leakage protocols, re-rank candidates, or execute Quantum
ESPRESSO. Figure captions and manuscript placement guidance are available in:

- `documentation/manuscript_figure_captions.md`
- `documentation/manuscript_figure_map.md`
- `documentation/figure_generation.md`

## Quantum ESPRESSO spot check

The accepted fixed-geometry PBE result for the representative sodium-ion
candidate is:

- selected-mesh average voltage: **3.203420972 V**;
- denser-reference average voltage: **3.203061629 V**;
- absolute difference: **0.000359343 V**.

Reconstruct and verify the frozen result from the repository root:

```bash
python dft_validation/scripts/reconstruct_voltage.py
python dft_validation/scripts/verify_repository.py
```

Endpoint-relaxation attempts did not satisfy the prescribed force-convergence
criteria. Their energies were excluded from the reported voltage. No phonon,
diffusion, density-of-states, band-structure, experimental, or multi-candidate
DFT validation is claimed.

## Data availability and provenance

Selected processed tables, predictions, metrics, candidate structures, audit
records, figure source data, and DFT evidence are included. Raw Materials
Project acquisition payloads, credentials, pseudopotential binaries, restart
files, private caches, and unrelated historical projects are not redistributed.

See:

- `documentation/data_availability.md`
- `configuration/runtime_data_policy.json`
- `provenance/`
- `dft_validation/manifests/`

## Citation

Citation metadata are provided in `CITATION.cff`. After a versioned archival
release is deposited, add the release DOI to the citation metadata and
manuscript Code Availability statement.

## License

Original repository content is distributed under the BSD-3-Clause License.
See `LICENSE`. Third-party notices associated with the DFT evidence are retained
under `dft_validation/`.

## Authors

- Md. Efatuzzaman Efat — corresponding author  
  Email: efatuzzaman@gmail.com
- Md. Saiful Islam
