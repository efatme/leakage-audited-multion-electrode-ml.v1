# DFT audit notes

The accepted fixed-geometry voltage is defined by `dft_validation/manifests/accepted_energy_chain.csv`.

## Charged-state force-audit provenance

`charged_static_force_audit_nonaccepted_attempt.txt` is **not** the force audit of the accepted charged selected SCF calculation.

- accepted charged selected SCF energy: `-1436.42910286 Ry`;
- energy in the historical force audit: `-1436.49946735 Ry`.

Because those energies differ, the force-audit file is retained only as historical evidence of a different, excluded/non-accepted calculation. It must not be cited as force evidence for the accepted voltage energy chain.

The reported voltage uses the accepted static SCF outputs listed in `../manifests/accepted_energy_chain.csv`. Endpoint relaxation attempts were excluded from the reported voltage because they did not satisfy the prescribed force-convergence criterion.
