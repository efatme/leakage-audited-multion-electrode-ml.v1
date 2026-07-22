#!/usr/bin/env python3
# SPDX-License-Identifier: BSD-3-Clause
"""Reconstruct the fixed-geometry voltages from accepted QE outputs."""

from pathlib import Path
from decimal import Decimal, getcontext
import re
import sys

getcontext().prec = 50
ROOT = Path(__file__).resolve().parents[1]
RY_TO_EV = Decimal("13.605693122994")
TOLERANCE_V = Decimal("1e-9")

CALCULATIONS = {
    "selected": {
        "charged": ROOT / "calculations/charged/selected_5x6x4/pw.out",
        "discharged": ROOT / "calculations/discharged/selected_3x3x2/pw.out",
        "na_metal": ROOT / "calculations/na_metal/selected_12x12x12/pw.out",
    },
    "reference": {
        "charged": ROOT / "calculations/charged/reference_6x7x4/pw.out",
        "discharged": ROOT / "calculations/discharged/reference_4x4x3/pw.out",
        "na_metal": ROOT / "calculations/na_metal/reference_16x16x16/pw.out",
    },
}

EXPECTED = {
    "selected": Decimal("3.203420972291968"),
    "reference": Decimal("3.203061628929602"),
    "difference": Decimal("0.000359343362367"),
}

ENERGY_PATTERN = re.compile(
    r"!\s+total energy\s+=\s+([+-]?[0-9.]+(?:[EeDd][+-]?[0-9]+)?)\s+Ry"
)


def final_energy(path):
    if not path.is_file():
        raise RuntimeError(f"Missing QE output: {path.relative_to(ROOT)}")

    text = path.read_text(encoding="utf-8", errors="replace")
    if "JOB DONE" not in text:
        raise RuntimeError(f"QE output lacks JOB DONE: {path.relative_to(ROOT)}")
    if "Error in routine" in text:
        raise RuntimeError(f"QE output contains a fatal error: {path.relative_to(ROOT)}")

    matches = ENERGY_PATTERN.findall(text)
    if not matches:
        raise RuntimeError(f"No total energy found: {path.relative_to(ROOT)}")

    return Decimal(matches[-1].replace("D", "E").replace("d", "e"))


def voltage(paths):
    charged = final_energy(paths["charged"])
    discharged = final_energy(paths["discharged"])
    na_metal = final_energy(paths["na_metal"])

    reaction_ev = (
        discharged / Decimal(4)
        - charged / Decimal(2)
        - na_metal
    ) * RY_TO_EV

    return -reaction_ev / Decimal(2)


def close_enough(actual, expected):
    return abs(actual - expected) <= TOLERANCE_V


def main():
    selected = voltage(CALCULATIONS["selected"])
    reference = voltage(CALCULATIONS["reference"])
    difference = abs(selected - reference)

    print(f"selected_voltage_V={selected}")
    print(f"reference_voltage_V={reference}")
    print(f"absolute_difference_V={difference}")

    passed = all([
        close_enough(selected, EXPECTED["selected"]),
        close_enough(reference, EXPECTED["reference"]),
        close_enough(difference, EXPECTED["difference"]),
    ])

    print("VOLTAGE_RECONSTRUCTION=" + ("PASS" if passed else "FAIL"))
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
