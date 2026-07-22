"""Dependency-aware leakage compiler for computed insertion-electrode properties.

The compiler applies machine-readable, target-specific physical and workflow rules.
It does not train predictive models.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict, Iterable, List

CAPACITY_TARGETS = {"capacity_grav", "capacity_vol"}
ENERGY_TARGETS = {"energy_grav", "energy_vol"}
STABILITY_TARGETS = {"stability_charge", "stability_discharge", "stability_worst"}
COMPOSITION_ORIGINS = {"composition_known_framework", "composition_known_working_ion"}


@dataclass(frozen=True)
class LeakageDecision:
    target: str
    feature: str
    feature_origin: str
    leakage_level: str
    leakage_reason: str
    rule_id: str
    dependency_path: str

    def to_dict(self) -> Dict[str, str]:
        return asdict(self)


def classify_feature(feature: str, feature_origin: str, target: str) -> LeakageDecision:
    """Classify a feature for one target using ordered physical/workflow rules."""
    if feature_origin == "domain_label_onehot":
        return LeakageDecision(target, feature, feature_origin, "L5",
            "Working-ion/domain labels encode validation-domain membership and must be controlled by split design.",
            "RULE_DOMAIN_LABEL", f"feature::{feature} -> concept::validation_domain -> target::{target}")

    if feature == target:
        return LeakageDecision(target, feature, feature_origin, "L1",
            "The feature is the exact target column.",
            "RULE_DIRECT_TARGET", f"feature::{feature} -> target::{target}")

    if feature_origin == "stored_electrode_target":
        if target == "average_voltage" and feature in ENERGY_TARGETS:
            return LeakageDecision(target, feature, feature_origin, "L2",
                "Stored energy is mathematically coupled to voltage and capacity.",
                "RULE_ENERGY_VOLTAGE_RELATION",
                f"feature::{feature} -> concept::energy_definition -> target::{target}")
        if target in CAPACITY_TARGETS and feature in (CAPACITY_TARGETS | ENERGY_TARGETS):
            return LeakageDecision(target, feature, feature_origin, "L2",
                "Capacity and energy variants are mathematically target-adjacent.",
                "RULE_CAPACITY_ENERGY_RELATION",
                f"feature::{feature} -> concept::capacity_energy_relation -> target::{target}")
        if target in ENERGY_TARGETS and feature in ({"average_voltage"} | CAPACITY_TARGETS | ENERGY_TARGETS):
            return LeakageDecision(target, feature, feature_origin, "L2",
                "Energy is defined by voltage and capacity, with unit-related energy/capacity variants.",
                "RULE_ENERGY_DEFINITION",
                f"feature::{feature} -> concept::energy_definition -> target::{target}")
        if target in STABILITY_TARGETS and feature in STABILITY_TARGETS:
            return LeakageDecision(target, feature, feature_origin, "L2",
                "Endpoint and worst-case stability values are definitionally linked.",
                "RULE_STABILITY_AGGREGATION",
                f"feature::{feature} -> concept::stability_aggregation -> target::{target}")
        return LeakageDecision(target, feature, feature_origin, "L4",
            "A different stored electrode property is available only as a post-hoc computed-record descriptor.",
            "RULE_OTHER_STORED_TARGET",
            f"feature::{feature} -> concept::stored_electrode_record -> target::{target}")

    if feature_origin == "stoichiometric_window":
        if target in (CAPACITY_TARGETS | ENERGY_TARGETS):
            return LeakageDecision(target, feature, feature_origin, "L3",
                "Working-ion insertion stoichiometry directly controls theoretical capacity and energy scale.",
                "RULE_STOICHIOMETRIC_WINDOW",
                f"feature::{feature} -> concept::working_ion_transfer -> target::{target}")
        return LeakageDecision(target, feature, feature_origin, "L0",
            "No direct target dependency is identified for this target.",
            "RULE_SAFE_DEFAULT", f"feature::{feature} -> target::{target}")

    if feature_origin == "stored_electrode_process_descriptor":
        if feature == "max_voltage_step" and target == "average_voltage":
            return LeakageDecision(target, feature, feature_origin, "L4",
                "A stored voltage-window descriptor is adjacent to the voltage target.",
                "RULE_VOLTAGE_WINDOW",
                f"feature::{feature} -> concept::voltage_window -> target::{target}")
        return LeakageDecision(target, feature, feature_origin, "L0",
            "No direct target dependency is identified for this target.",
            "RULE_SAFE_DEFAULT", f"feature::{feature} -> target::{target}")

    if feature_origin in {"composition_known_framework", "composition_known_working_ion", "other_numeric"}:
        return LeakageDecision(target, feature, feature_origin, "L0",
            "No direct target dependency is identified for this target.",
            "RULE_SAFE_DEFAULT", f"feature::{feature} -> target::{target}")

    if feature_origin == "post_dft_electronic_summary":
        return LeakageDecision(target, feature, feature_origin, "L4",
            "The electronic descriptor is generated in the same post-DFT record and is decision-support only.",
            "RULE_POST_DFT_ELECTRONIC",
            f"feature::{feature} -> concept::post_dft_record -> target::{target}")

    if feature_origin == "post_dft_provenance_flag":
        return LeakageDecision(target, feature, feature_origin, "L4",
            "The provenance flag is available only from the post-DFT record.",
            "RULE_POST_DFT_PROVENANCE",
            f"feature::{feature} -> concept::post_dft_record -> target::{target}")

    if feature_origin == "post_dft_energy_stability_summary":
        if "formation_energy" in feature:
            if target == "average_voltage":
                return LeakageDecision(target, feature, feature_origin, "L2",
                    "Endpoint formation energies can reconstruct insertion-voltage trends.",
                    "RULE_FORMATION_ENERGY_VOLTAGE",
                    f"feature::{feature} -> concept::insertion_energy_difference -> target::{target}")
            if target in ENERGY_TARGETS:
                return LeakageDecision(target, feature, feature_origin, "L2",
                    "Endpoint formation energies are target-defining or strongly target-adjacent for electrode energy.",
                    "RULE_FORMATION_ENERGY_ENERGY",
                    f"feature::{feature} -> concept::insertion_energy_difference -> target::{target}")
            if target in STABILITY_TARGETS:
                return LeakageDecision(target, feature, feature_origin, "L4",
                    "Formation energy is post-DFT adjacent to stability but is not the direct hull target.",
                    "RULE_FORMATION_ENERGY_STABILITY",
                    f"feature::{feature} -> concept::post_dft_thermodynamics -> target::{target}")
            return LeakageDecision(target, feature, feature_origin, "L4",
                "The thermodynamic summary is a post-DFT decision-support descriptor.",
                "RULE_POST_DFT_ENERGY_DEFAULT",
                f"feature::{feature} -> concept::post_dft_thermodynamics -> target::{target}")

        if ("energy_above_hull" in feature) or ("is_stable" in feature):
            if target in STABILITY_TARGETS:
                return LeakageDecision(target, feature, feature_origin, "L2",
                    "Endpoint hull/stability descriptors define or directly reconstruct stability targets.",
                    "RULE_HULL_STABILITY",
                    f"feature::{feature} -> concept::hull_stability -> target::{target}")
            return LeakageDecision(target, feature, feature_origin, "L4",
                "The hull/stability summary is a post-DFT decision-support descriptor for this target.",
                "RULE_HULL_OTHER",
                f"feature::{feature} -> concept::post_dft_thermodynamics -> target::{target}")

        raise ValueError(f"Unrecognized post-DFT energy/stability feature: {feature}")

    if feature_origin == "relaxed_structure_summary":
        if target == "max_delta_volume" and any(token in feature for token in ("volume", "density")):
            return LeakageDecision(target, feature, feature_origin, "L2",
                "Endpoint volume/density descriptors define or strongly encode the volume-change target.",
                "RULE_VOLUME_CHANGE",
                f"feature::{feature} -> concept::volume_change_definition -> target::{target}")
        return LeakageDecision(target, feature, feature_origin, "L0",
            "The relaxed-structure summary is not target-defining for this target.",
            "RULE_SAFE_STRUCTURE", f"feature::{feature} -> target::{target}")

    raise ValueError(
        f"No compiler rule for feature={feature!r}, origin={feature_origin!r}, target={target!r}"
    )


def protocol_allows(protocol: str, leakage_level: str, feature_origin: str) -> bool:
    """Apply deployment-stage and leakage-level policy for P0-P4."""
    if protocol == "P0":
        return leakage_level in {"L0", "L2", "L3", "L4"}
    if protocol == "P1":
        return leakage_level == "L0" and feature_origin in COMPOSITION_ORIGINS
    if protocol == "P2":
        return leakage_level == "L0" and feature_origin in (COMPOSITION_ORIGINS | {"relaxed_structure_summary"})
    if protocol == "P3":
        if feature_origin in COMPOSITION_ORIGINS:
            return leakage_level == "L0"
        if feature_origin == "relaxed_structure_summary":
            return leakage_level == "L0"
        if feature_origin == "stored_electrode_process_descriptor":
            return leakage_level in {"L0", "L4"}
        if feature_origin in {"post_dft_electronic_summary", "post_dft_energy_stability_summary"}:
            return leakage_level == "L4"
        return False
    if protocol == "P4":
        return leakage_level in {"L0", "L1", "L2", "L3", "L4"}
    raise ValueError(f"Unknown protocol: {protocol}")


def compile_target_feature_rows(feature_records: Iterable[Dict[str, str]], targets: Iterable[str]) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for target in targets:
        for record in feature_records:
            decision = classify_feature(record["feature"], record["feature_origin"], target)
            row: Dict[str, object] = decision.to_dict()
            for protocol in ("P0", "P1", "P2", "P3", "P4"):
                row[f"allowed_{protocol}"] = protocol_allows(protocol, decision.leakage_level, decision.feature_origin)
            rows.append(row)
    return rows
