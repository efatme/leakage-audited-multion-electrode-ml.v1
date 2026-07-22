# Quantum ESPRESSO fixed-geometry voltage spot check

This repository contains the reproducibility evidence for a convergence-audited
fixed-geometry PBE average-voltage calculation for one representative
sodium-ion insertion-electrode candidate.

## Authors

1. Md. Efatuzzaman Efat — corresponding author  
   Email: efatuzzaman@gmail.com
2. Md. Saiful Islam

## Locked result

- Selected fixed-geometry average voltage: **3.203420972 V**
- Denser-reference average voltage: **3.203061629 V**
- Absolute voltage difference: **0.000359343 V**

The selected calculations used:

- charged structure: 5 × 6 × 4 k-point mesh;
- discharged structure: 3 × 3 × 2 k-point mesh;
- sodium metal: 12 × 12 × 12 k-point mesh;
- wavefunction and charge-density cutoffs: 120 and 960 Ry;
- Quantum ESPRESSO 7.5.

## Scope and claim boundary

This is an original computational spot check, not a fully relaxed validation.
Endpoint-relaxation attempts did not satisfy the prescribed force-convergence
criteria, and their energies were excluded from the voltage calculation.

No phonon, diffusion, density-of-states, band-structure, experimental, or
multi-candidate DFT validation is claimed in this repository.

## Reconstruct the voltage

From the repository root, run:

```bash
python3 scripts/reconstruct_voltage.py
```

The script reads the six accepted static QE outputs, reconstructs the selected
and denser-reference voltages, and checks them against the locked values.

## Verify the repository

From the repository root, run:

```bash
python3 scripts/verify_repository.py
```

This verifies the file-hash manifest, QE completion markers, accepted energy
chain, pseudopotential provenance records, repository metadata, and voltage
reconstruction.

## Pseudopotentials

UPF files are not redistributed. Exact filenames, source records, and SHA256
hashes are listed in:

`manifests/pseudopotential_provenance.csv`

The carbon, oxygen, and phosphorus files match archived SSSP 1.3.0 PBE
efficiency payloads exactly. The cobalt and sodium files match the recorded
GBRV source files exactly.

Download the recorded files and place them in a `pseudopotentials/` directory,
or adjust `pseudo_dir` in local working copies of the QE input files.

## Directory structure

- `calculations/`: public-safe QE inputs and completed static outputs;
- `scripts/`: public-layout reconstruction and verification scripts;
- `results/`: convergence and voltage evidence;
- `manifests/`: calculation, energy-chain, file, and pseudopotential records;
- `audits/`: force, relaxation, composition, provenance, and release audits;
- `environment/`: public-safe software and hardware environment record;
- `licenses/`: third-party license record retained for provenance.

## Citation

Citation metadata are provided in `CITATION.cff`.

## License

Original repository content is released under the BSD-3-Clause license. See
`LICENSE` and `THIRD_PARTY_NOTICES.md`.
