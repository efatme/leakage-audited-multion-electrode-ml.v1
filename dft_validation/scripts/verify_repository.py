#!/usr/bin/env python3
# SPDX-License-Identifier: BSD-3-Clause
"""Verify the integrity and scientific evidence of this public repository."""

from pathlib import Path
import csv
import hashlib
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def verify_manifest():
    manifest = ROOT / "MANIFEST.sha256"
    if not manifest.is_file():
        return False, ["MANIFEST.sha256 is missing"]

    expected = {}
    problems = []
    for number, line in enumerate(
        manifest.read_text(encoding="utf-8", errors="replace").splitlines(),
        start=1,
    ):
        if not line.strip():
            continue
        parts = line.split(None, 1)
        if len(parts) != 2:
            problems.append(f"Malformed manifest line {number}")
            continue
        digest, relative = parts
        expected[relative.strip()] = digest

    for relative, digest in expected.items():
        path = ROOT / relative
        if not path.is_file():
            problems.append(f"Missing manifest file: {relative}")
        elif sha256(path) != digest:
            problems.append(f"Hash mismatch: {relative}")

    actual = {
        path.relative_to(ROOT).as_posix()
        for path in ROOT.rglob("*")
        if path.is_file() and path.name != "MANIFEST.sha256"
    }
    unexpected = sorted(actual - set(expected))
    for relative in unexpected:
        problems.append(f"Unexpected file: {relative}")

    return not problems, problems


def verify_qe_outputs():
    outputs = sorted((ROOT / "calculations").rglob("pw.out"))
    problems = []
    if len(outputs) != 8:
        problems.append(f"Expected 8 QE outputs, found {len(outputs)}")

    for path in outputs:
        text = path.read_text(encoding="utf-8", errors="replace")
        relative = path.relative_to(ROOT).as_posix()
        if "JOB DONE" not in text:
            problems.append(f"Missing JOB DONE: {relative}")
        if "Error in routine" in text:
            problems.append(f"Fatal QE error marker: {relative}")
        if "Program PWSCF v.7.5" not in text:
            problems.append(f"QE 7.5 marker missing: {relative}")

    return not problems, problems


def verify_manifests():
    problems = []

    energy_path = ROOT / "manifests" / "accepted_energy_chain.csv"
    with energy_path.open("r", encoding="utf-8", newline="") as handle:
        energy_rows = list(csv.DictReader(handle))
    if len(energy_rows) != 6:
        problems.append(f"Expected 6 accepted energy records, found {len(energy_rows)}")
    for row in energy_rows:
        if row.get("valid_completed_static_output") != "YES":
            problems.append("An accepted energy record is not marked valid")

    pseudo_path = ROOT / "manifests" / "pseudopotential_provenance.csv"
    with pseudo_path.open("r", encoding="utf-8", newline="") as handle:
        pseudo_rows = list(csv.DictReader(handle))
    if len(pseudo_rows) != 5:
        problems.append(f"Expected 5 pseudopotential records, found {len(pseudo_rows)}")
    for row in pseudo_rows:
        if row.get("exact_sha256_match") != "YES":
            problems.append(f"Pseudopotential hash status failed: {row.get('filename', '')}")
        if row.get("upf_bundled_in_repository") != "NO":
            problems.append(f"Unexpected bundled-UPF status: {row.get('filename', '')}")

    bundled_upf = sorted(ROOT.rglob("*.UPF")) + sorted(ROOT.rglob("*.upf"))
    if bundled_upf:
        problems.append("UPF files are unexpectedly bundled")

    return not problems, problems


def verify_metadata():
    required = [
        ROOT / "README.md",
        ROOT / "LICENSE",
        ROOT / "CITATION.cff",
        ROOT / "THIRD_PARTY_NOTICES.md",
        ROOT / "audits" / "FINAL_PUBLIC_EXPORT_AUDIT.txt",
    ]
    problems = [
        f"Missing metadata file: {path.name}"
        for path in required
        if not path.is_file()
    ]
    return not problems, problems


def verify_voltage():
    completed = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "reconstruct_voltage.py")],
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    output = completed.stdout.strip()
    passed = completed.returncode == 0 and "VOLTAGE_RECONSTRUCTION=PASS" in output
    problems = [] if passed else ["Voltage reconstruction failed:\n" + output]
    return passed, problems


def main():
    checks = [
        ("manifest", verify_manifest),
        ("qe_outputs", verify_qe_outputs),
        ("scientific_manifests", verify_manifests),
        ("repository_metadata", verify_metadata),
        ("voltage", verify_voltage),
    ]

    all_problems = []
    for label, function in checks:
        passed, problems = function()
        print(f"{label}=" + ("PASS" if passed else "FAIL"))
        all_problems.extend(problems)

    if all_problems:
        print("\nPROBLEMS")
        for problem in all_problems:
            print(problem)
        print("REPOSITORY_VERIFICATION=FAIL")
        return 1

    print("REPOSITORY_VERIFICATION=PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
