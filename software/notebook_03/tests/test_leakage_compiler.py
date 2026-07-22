from electrode_audit.leakage_compiler import classify_feature, protocol_allows


def test_energy_definition_is_L2():
    d = classify_feature("average_voltage", "stored_electrode_target", "energy_grav")
    assert d.leakage_level == "L2"


def test_stoichiometry_is_L3_for_capacity():
    d = classify_feature("fracA_discharge", "stoichiometric_window", "capacity_grav")
    assert d.leakage_level == "L3"


def test_hull_is_L2_for_worst_stability():
    d = classify_feature("summary_worst_energy_above_hull", "post_dft_energy_stability_summary", "stability_worst")
    assert d.leakage_level == "L2"


def test_P2_excludes_volume_proxy_for_volume_change():
    d = classify_feature("charge_summary_volume", "relaxed_structure_summary", "max_delta_volume")
    assert d.leakage_level == "L2"
    assert not protocol_allows("P2", d.leakage_level, d.feature_origin)
