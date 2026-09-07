from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


TARGET_ORDER = [
    "average_voltage",
    "capacity_grav",
    "capacity_vol",
    "energy_grav",
    "energy_vol",
    "max_delta_volume",
    "stability_charge",
    "stability_discharge",
    "stability_worst",
]
STAGE_ORDER = ["P1", "P2", "P3"]
DOMAIN_ORDER = [
    "host-chemical-system-held-out validation",
    "coarse-chemistry-family-held-out validation",
    "working-ion-held-out validation",
]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def require(path: Path) -> Path:
    if not path.exists():
        raise FileNotFoundError(f"Required Stage F input is missing: {path}")
    return path


def read_gate(path: Path, expected_stage: str) -> dict:
    obj = json.loads(require(path).read_text(encoding="utf-8"))
    if obj.get("status") != "PASS" or obj.get("stage") != expected_stage:
        raise RuntimeError(f"Gate failed for Stage {expected_stage}: {path}")
    return obj


def pivot_stage_values(df: pd.DataFrame, value: str, prefix: str) -> pd.DataFrame:
    x = df.pivot(index="target", columns="stage", values=value).reindex(columns=STAGE_ORDER)
    x.columns = [f"{prefix}_{s}" for s in x.columns]
    return x.reset_index()


def clean_stage(x) -> str:
    if pd.isna(x):
        return ""
    return str(x)


def build_master(
    feature_manifest: pd.DataFrame,
    skill: pd.DataFrame,
    earliest: pd.DataFrame,
    threshold: pd.DataFrame,
    incr: pd.DataFrame,
    robustness: pd.DataFrame,
    transitions: pd.DataFrame,
    est_compare: pd.DataFrame,
) -> pd.DataFrame:
    master = pd.DataFrame({"target": TARGET_ORDER})

    feat = feature_manifest.pivot(index="target", columns="stage", values="n_features").reindex(columns=STAGE_ORDER)
    feat.columns = [f"n_features_{s}" for s in feat.columns]
    master = master.merge(feat.reset_index(), on="target", how="left")

    master = master.merge(
        earliest[["target", "earliest_useful_stage"]].rename(
            columns={"earliest_useful_stage": "primary_ExtraTrees_earliest_useful_stage_10pct"}
        ),
        on="target", how="left"
    )

    th = threshold.pivot(index="target", columns="threshold", values="earliest_useful_stage")
    for val, name in [(0.05, "earliest_stage_5pct"), (0.10, "earliest_stage_10pct"), (0.20, "earliest_stage_20pct")]:
        if val in th.columns:
            th = th.rename(columns={val: name})
    th = th.reset_index()
    master = master.merge(th, on="target", how="left")

    ec = est_compare[[
        "target",
        "secondary_Ridge_earliest_useful_stage",
        "same_earliest_stage",
    ]].rename(columns={"same_earliest_stage": "ExtraTrees_Ridge_same_earliest_stage"})
    master = master.merge(ec, on="target", how="left")

    for col, prefix in [
        ("et_mae", "ExtraTrees_MAE"),
        ("et_rmse", "ExtraTrees_RMSE"),
        ("mae_skill_vs_dummy", "MAE_skill"),
        ("mae_skill_ci_low", "MAE_skill_CI_low"),
        ("mae_skill_ci_high", "MAE_skill_CI_high"),
        ("rmse_skill_vs_dummy", "RMSE_skill"),
        ("rmse_skill_ci_low", "RMSE_skill_CI_low"),
        ("rmse_skill_ci_high", "RMSE_skill_CI_high"),
        ("useful_learnability_primary_rule", "useful_10pct"),
    ]:
        master = master.merge(pivot_stage_values(skill, col, prefix), on="target", how="left")

    for a, b, tag in [("P1", "P2", "P1_to_P2"), ("P2", "P3", "P2_to_P3"), ("P1", "P3", "P1_to_P3")]:
        z = incr[(incr["from_stage"] == a) & (incr["to_stage"] == b)][[
            "target",
            "relative_mae_improvement",
            "relative_mae_ci_low",
            "relative_mae_ci_high",
            "delta_mae_interval_direction",
            "relative_rmse_improvement",
            "relative_rmse_ci_low",
            "relative_rmse_ci_high",
            "delta_rmse_interval_direction",
        ]].copy()
        z = z.rename(columns={c: f"{tag}_{c}" for c in z.columns if c != "target"})
        master = master.merge(z, on="target", how="left")

    domain = robustness.pivot(index="target", columns="reporting_name", values="domain_point_earliest_stage_10pct")
    domain = domain.reindex(columns=DOMAIN_ORDER)
    domain.columns = [
        "domain_earliest_host_system",
        "domain_earliest_coarse_family",
        "domain_earliest_working_ion",
    ]
    master = master.merge(domain.reset_index(), on="target", how="left")

    # Reference-stage skills under each domain stress.
    for report, short in [
        ("host-chemical-system-held-out validation", "host_system"),
        ("coarse-chemistry-family-held-out validation", "coarse_family"),
        ("working-ion-held-out validation", "working_ion"),
    ]:
        z = robustness[robustness["reporting_name"] == report][[
            "target", "comparison_status", "reference_stage_for_skill_check",
            "reference_stage_macro_mae_skill", "reference_stage_macro_rmse_skill",
            "reference_stage_passes_10pct_macro_point",
        ]].copy()
        z = z.rename(columns={c: f"{short}_{c}" for c in z.columns if c != "target"})
        master = master.merge(z, on="target", how="left")

    # Net P1->P3 robustness summary.
    net = transitions[(transitions["from_stage"] == "P1") & (transitions["to_stage"] == "P3")].copy()
    for report, short in [
        ("host-chemical-system-held-out validation", "host_system"),
        ("coarse-chemistry-family-held-out validation", "coarse_family"),
        ("working-ion-held-out validation", "working_ion"),
    ]:
        z = net[net["reporting_name"] == report][[
            "target", "mean_delta_mae_positive_is_improvement",
            "mean_delta_rmse_positive_is_improvement",
            "fraction_folds_both_metrics_improve", "macro_both_metrics_improve",
        ]].copy()
        z = z.rename(columns={c: f"{short}_P1_to_P3_{c}" for c in z.columns if c != "target"})
        master = master.merge(z, on="target", how="left")

    # Concise interpretation class for manuscript planning.
    interp = []
    for _, r in master.iterrows():
        t = r["target"]
        if t in {"average_voltage", "capacity_grav", "capacity_vol", "energy_grav", "energy_vol"}:
            label = "composition-stage skill established; later-stage value is target-specific"
        elif t == "max_delta_volume":
            label = "useful skill not demonstrated through P3 in primary analysis; domain behavior is heterogeneous"
        else:
            label = "later-stage stability learning; estimator, threshold, and chemistry-shift sensitivity require qualification"
        interp.append(label)
    master["manuscript_interpretation_class"] = interp

    order = pd.Categorical(master["target"], TARGET_ORDER, ordered=True)
    return master.assign(_order=order).sort_values("_order").drop(columns="_order")


def build_claims(master: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame([
        {
            "claim_id": "CMT-01",
            "status": "PRIMARY",
            "placement": "Main text",
            "claim": "Descriptor eligibility is target-specific, and P1/P2/P3 form nested information stages with target-dependent feature counts.",
            "safe_wording": "Feature eligibility was defined separately for each target using the expert dependency graph and rule-based compiler.",
            "evidence": "Stage A feature manifest and dependency/compiler audits",
        },
        {
            "claim_id": "CMT-02",
            "status": "PRIMARY",
            "placement": "Main text",
            "claim": "Voltage, gravimetric/volumetric capacity, and gravimetric/volumetric energy already satisfy the frozen useful-skill criterion at P1 under fixed ExtraTrees.",
            "safe_wording": "Under framework-formula-grouped validation with the fixed ExtraTrees estimator, these five targets first demonstrated useful predictive skill at the composition stage.",
            "evidence": "Stage C earliest-stage and skill bootstrap outputs",
        },
        {
            "claim_id": "CMT-03",
            "status": "PRIMARY",
            "placement": "Main text",
            "claim": "The marginal value of P2 and P3 is target-specific rather than uniformly monotonic.",
            "safe_wording": "Relaxed-structure and legitimate post-DFT information provided unequal marginal gains across targets; some transitions improved error, whereas others provided little additional benefit or degraded one metric.",
            "evidence": "Stage C incremental information value bootstrap",
        },
        {
            "claim_id": "CMT-04",
            "status": "PRIMARY_CAUTIOUS",
            "placement": "Main text",
            "claim": "Useful predictive skill for maximum volume change was not demonstrated through P3 in the primary analysis.",
            "safe_wording": "The evaluated fixed ExtraTrees models did not satisfy the pre-specified useful-learnability criterion for maximum volume change through P3 under framework-formula-grouped validation.",
            "evidence": "Stage C earliest-stage and bootstrap outputs",
        },
        {
            "claim_id": "CMT-05",
            "status": "PRIMARY_QUALIFIED",
            "placement": "Main text + SI sensitivity",
            "claim": "The primary ExtraTrees earliest stages for charged, discharged, and worst stability are P2, P3, and P2, respectively, but these assignments are less robust than those for voltage/capacity/energy.",
            "safe_wording": "For the fixed primary ExtraTrees estimator, stability targets first satisfied the useful-skill criterion at later stages; the exact assignments were sensitive to estimator choice, threshold, and coarse chemistry-family shift.",
            "evidence": "Stages C, D, and E",
        },
        {
            "claim_id": "CMT-06",
            "status": "SUPPORTING",
            "placement": "Main text compact robustness panel; full detail in SI",
            "claim": "Net P1-to-P3 error reduction is broadly preserved under the three chemical-domain stress tests.",
            "safe_wording": "At the macro point-estimate level, P3 reduced both MAE and RMSE relative to P1 in 26 of 27 target-by-domain comparisons; this is descriptive robustness evidence rather than an inferential stage classification.",
            "evidence": "Stage D stage-transition summary",
        },
        {
            "claim_id": "CMT-07",
            "status": "SUPPORTING_CAUTIOUS",
            "placement": "Main text discussion + SI",
            "claim": "Coarse-chemistry-family-held-out validation is the strongest stress test for the stability targets and maximum volume change.",
            "safe_wording": "Under coarse-chemistry-family-held-out validation, the three stability targets did not satisfy the 10% macro point criterion through P3, and maximum volume change also remained below the criterion.",
            "evidence": "Stage D primary-stage robustness",
        },
        {
            "claim_id": "CMT-08",
            "status": "SENSITIVITY",
            "placement": "SI; mention in main discussion",
            "claim": "Ridge reproduces the exact earliest-stage classification for 6 of 9 targets; all disagreements are stability targets.",
            "safe_wording": "A fixed linear Ridge sensitivity analysis agreed with the primary ExtraTrees earliest-stage assignment for six targets; the three stability assignments were estimator-dependent.",
            "evidence": "Stage E estimator comparison",
        },
        {
            "claim_id": "CMT-09",
            "status": "SENSITIVITY",
            "placement": "SI",
            "claim": "The 5%, 10%, and 20% useful-skill thresholds leave the non-stability target assignments unchanged, while the stability assignments vary.",
            "safe_wording": "Threshold sensitivity affected the stability-stage assignments but not the five voltage/capacity/energy targets or the maximum-volume-change classification.",
            "evidence": "Stage C threshold sensitivity",
        },
        {
            "claim_id": "CMT-10",
            "status": "CONTROL",
            "placement": "SI",
            "claim": "P0 and P4 are controls, not the central scientific comparison.",
            "safe_wording": "P0 provides a permissive full-information baseline and P4 is an artificial direct-target positive-control stress test.",
            "evidence": "Frozen study design and existing control analyses",
        },
        {
            "claim_id": "CMT-11",
            "status": "CONTROL",
            "placement": "SI",
            "claim": "Leakage-by-domain interaction is supporting robustness evidence only.",
            "safe_wording": "The leakage-by-domain interaction is retained as a supporting analysis and does not define the CMT paper.",
            "evidence": "Existing Notebook 06 outputs and frozen study design",
        },
    ])


def build_figure_plan() -> pd.DataFrame:
    return pd.DataFrame([
        ["Figure 1", "Main", "Target-specific information provenance and stage eligibility", "Dependency framework; feature blocks; P1/P2/P3 computational stages; target-dependent feature counts", "Stage A"],
        ["Figure 2", "Main", "Stage-dependent learnability", "Nine targets x P1/P2/P3 heatmap of Dummy-relative MAE/RMSE skill; mark primary earliest useful stage", "Stage C"],
        ["Figure 3", "Main", "Incremental information value", "Paired P1->P2, P2->P3, P1->P3 relative MAE/RMSE changes with 95% group-bootstrap intervals", "Stage C"],
        ["Figure 4", "Main", "Chemical-domain robustness", "Compact target x validation heatmap for framework-formula primary result plus host-system, coarse-family and working-ion descriptive stage status", "Stages C-D"],
        ["Table 1", "Main", "Target-level summary", "P1/P2/P3 feature counts; primary earliest useful stage; concise robustness and estimator-sensitivity flags", "Stages A-E"],
        ["Table S1", "SI", "Full Stage C performance", "MAE/RMSE, DummyMean skill and bootstrap intervals for all 27 target-stage combinations", "Stage C"],
        ["Table S2", "SI", "Threshold sensitivity", "Earliest stage at 5%, 10%, 20% thresholds", "Stage C"],
        ["Table S3", "SI", "Domain robustness", "All domain macro metrics and fold-level transition summaries", "Stage D"],
        ["Table S4", "SI", "Estimator sensitivity", "Ridge performance, bootstrap skill and earliest-stage comparison", "Stage E"],
        ["Figure/Table S5", "SI", "Information controls", "P0/P4 controls and leakage-by-domain interaction with corrected wording", "Existing control outputs"],
    ], columns=["item", "placement", "title", "content", "source"])


def write_markdown_summary(path: Path, master: pd.DataFrame, claims: pd.DataFrame, gates: dict) -> None:
    n_same = int(master["ExtraTrees_Ridge_same_earliest_stage"].fillna(False).sum())
    stable_threshold = 0
    for _, r in master.iterrows():
        if clean_stage(r.get("earliest_stage_5pct")) == clean_stage(r.get("earliest_stage_10pct")) == clean_stage(r.get("earliest_stage_20pct")):
            stable_threshold += 1

    domain_transition_cols = [c for c in master.columns if c.endswith("P1_to_P3_macro_both_metrics_improve")]
    n_domain_pos = 0
    n_domain_total = 0
    for c in domain_transition_cols:
        vals = master[c].dropna()
        n_domain_pos += int(vals.astype(bool).sum())
        n_domain_total += int(len(vals))

    lines = [
        "# CMT analysis freeze summary",
        "",
        "## Freeze status",
        "Stages A-E all passed. Stage F performs no model fitting and freezes the analysis for publication outputs.",
        "",
        "## Primary earliest useful stage (fixed ExtraTrees, framework-formula-grouped validation)",
        "",
        "| Target | Earliest useful stage |",
        "|---|---|",
    ]
    for _, r in master.iterrows():
        lines.append(f"| {r['target']} | {r['primary_ExtraTrees_earliest_useful_stage_10pct']} |")
    lines += [
        "",
        "## Sensitivity summary",
        f"- Exact ExtraTrees/Ridge earliest-stage agreement: {n_same}/9 targets.",
        f"- Earliest-stage assignment unchanged across 5%, 10%, and 20% thresholds: {stable_threshold}/9 targets.",
        f"- Descriptive P1-to-P3 macro improvement in both MAE and RMSE across chemical-domain target comparisons: {n_domain_pos}/{n_domain_total}.",
        "- The three stability targets require the strongest qualification because their exact stage assignments are estimator- and threshold-sensitive and weaken under coarse-chemistry-family-held-out validation.",
        "- Maximum volume change remains a negative primary learnability result through P3 under the frozen criterion, despite positive P1-to-P3 error reductions in the framework-formula analysis.",
        "",
        "## Manuscript control placement",
        "- P0 permissive baseline: Supporting Information.",
        "- P4 direct-target positive control: Supporting Information.",
        "- Leakage-by-domain interaction: Supporting Information.",
        "- Ridge estimator sensitivity: Supporting Information, with a short qualification in the main discussion.",
        "- Physics-constrained multitask benchmark: repository-only unless specifically needed.",
        "",
        "## Writing rule",
        "Do not describe the stability-stage assignments as estimator-independent properties. Use wording such as 'under the fixed primary ExtraTrees estimator'.",
        "",
        "## Gate provenance",
    ]
    for s, g in gates.items():
        lines.append(f"- Stage {s}: {g.get('status')} ({g.get('analysis_id')})")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    repo = Path(__file__).resolve().parents[3]
    cmt = repo / "papers" / "cmt"
    results = cmt / "results"
    out = results / "stage_f"
    out.mkdir(parents=True, exist_ok=True)

    config = require(cmt / "config" / "cmt_stage_f_v1.json")

    print(f"Repository: {repo}")
    print(f"Config:     {config}")
    print(f"Output:     {out}")
    print("Stage F: no model fitting; consolidation and publication freeze only")

    gates = {
        "A": read_gate(results / "stage_ab" / "stage_a_gate.json", "A"),
        "B": read_gate(results / "stage_ab" / "stage_b_gate.json", "B"),
        "C": read_gate(results / "stage_c" / "stage_c_gate.json", "C"),
        "D": read_gate(results / "stage_d" / "stage_d_gate.json", "D"),
        "E": read_gate(results / "stage_e" / "stage_e_gate.json", "E"),
    }

    feature_manifest = pd.read_csv(require(results / "stage_ab" / "stage_a_feature_manifest.csv"))
    skill = pd.read_csv(require(results / "stage_c" / "stage_c_skill_bootstrap_summary.csv"))
    earliest = pd.read_csv(require(results / "stage_c" / "stage_c_earliest_useful_stage.csv"))
    threshold = pd.read_csv(require(results / "stage_c" / "stage_c_threshold_sensitivity.csv"))
    incr = pd.read_csv(require(results / "stage_c" / "stage_c_incremental_information_value_bootstrap.csv"))
    robustness = pd.read_csv(require(results / "stage_d" / "stage_d_primary_stage_robustness.csv"))
    transitions = pd.read_csv(require(results / "stage_d" / "stage_d_stage_transition_summary.csv"))
    est_compare = pd.read_csv(require(results / "stage_e" / "stage_e_estimator_earliest_stage_comparison.csv"))

    # Frozen cardinality gates.
    checks = []
    checks.append(("feature_manifest_27_rows", len(feature_manifest) == 27))
    checks.append(("skill_27_rows", len(skill) == 27))
    checks.append(("earliest_9_rows", len(earliest) == 9))
    checks.append(("threshold_27_rows", len(threshold) == 27))
    checks.append(("incremental_27_rows", len(incr) == 27))
    checks.append(("robustness_27_rows", len(robustness) == 27))
    checks.append(("transitions_81_rows", len(transitions) == 81))
    checks.append(("estimator_comparison_9_rows", len(est_compare) == 9))
    checks.append(("targets_match", set(earliest["target"]) == set(TARGET_ORDER)))
    checks.append(("stages_match", set(skill["stage"]) == set(STAGE_ORDER)))
    failed = [name for name, ok in checks if not ok]
    if failed:
        raise RuntimeError("Stage F preflight failed: " + ", ".join(failed))

    master = build_master(feature_manifest, skill, earliest, threshold, incr, robustness, transitions, est_compare)
    claims = build_claims(master)
    figplan = build_figure_plan()

    master_path = out / "stage_f_master_result_matrix.csv"
    claims_path = out / "stage_f_claims_register.csv"
    figplan_path = out / "stage_f_figure_table_plan.csv"
    summary_path = out / "CMT_ANALYSIS_FREEZE_SUMMARY.md"
    master.to_csv(master_path, index=False)
    claims.to_csv(claims_path, index=False)
    figplan.to_csv(figplan_path, index=False)
    write_markdown_summary(summary_path, master, claims, gates)

    # Record exact inputs used by Stage F.
    input_paths = [
        results / "stage_ab" / "stage_a_gate.json",
        results / "stage_ab" / "stage_b_gate.json",
        results / "stage_ab" / "stage_a_feature_manifest.csv",
        results / "stage_c" / "stage_c_gate.json",
        results / "stage_c" / "stage_c_skill_bootstrap_summary.csv",
        results / "stage_c" / "stage_c_earliest_useful_stage.csv",
        results / "stage_c" / "stage_c_threshold_sensitivity.csv",
        results / "stage_c" / "stage_c_incremental_information_value_bootstrap.csv",
        results / "stage_d" / "stage_d_gate.json",
        results / "stage_d" / "stage_d_primary_stage_robustness.csv",
        results / "stage_d" / "stage_d_stage_transition_summary.csv",
        results / "stage_e" / "stage_e_gate.json",
        results / "stage_e" / "stage_e_estimator_earliest_stage_comparison.csv",
        config,
    ]
    manifest = pd.DataFrame([
        {"relative_path": str(p.relative_to(repo)).replace("\\", "/"), "sha256": sha256(p), "bytes": p.stat().st_size}
        for p in input_paths
    ])
    manifest_path = out / "stage_f_input_manifest_sha256.csv"
    manifest.to_csv(manifest_path, index=False)

    output_paths = [master_path, claims_path, figplan_path, summary_path, manifest_path]
    out_hash = pd.DataFrame([
        {"relative_path": str(p.relative_to(repo)).replace("\\", "/"), "sha256": sha256(p), "bytes": p.stat().st_size}
        for p in output_paths
    ])
    out_hash_path = out / "stage_f_output_sha256.csv"
    out_hash.to_csv(out_hash_path, index=False)

    n_est_same = int(master["ExtraTrees_Ridge_same_earliest_stage"].fillna(False).sum())
    n_threshold_same = int((
        (master["earliest_stage_5pct"].astype(str) == master["earliest_stage_10pct"].astype(str)) &
        (master["earliest_stage_10pct"].astype(str) == master["earliest_stage_20pct"].astype(str))
    ).sum())
    net_cols = [c for c in master.columns if c.endswith("P1_to_P3_macro_both_metrics_improve")]
    n_net_pos = sum(int(master[c].fillna(False).astype(bool).sum()) for c in net_cols)
    n_net_total = sum(int(master[c].notna().sum()) for c in net_cols)

    gate = {
        "analysis_id": "cmt_stage_f_v1",
        "stage": "F",
        "status": "PASS",
        "model_fitting_performed": False,
        "analysis_status": "FROZEN_FOR_PUBLICATION_OUTPUTS",
        "n_targets": 9,
        "primary_estimator": "fixed ExtraTrees from Stage C",
        "primary_validation": "framework-formula-grouped validation",
        "ExtraTrees_Ridge_exact_earliest_stage_agreement": f"{n_est_same}/9",
        "threshold_5_10_20_exact_stage_stability": f"{n_threshold_same}/9",
        "domain_P1_to_P3_macro_both_metrics_improve": f"{n_net_pos}/{n_net_total}",
        "timestamp_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    (out / "stage_f_gate.json").write_text(json.dumps(gate, indent=2) + "\n", encoding="utf-8")

    print("Preflight: PASS")
    print(f"Stage F: PASS | targets=9 | ExtraTrees/Ridge agreement={n_est_same}/9 | threshold-stable={n_threshold_same}/9")
    print(f"Domain P1->P3 macro improvement in both MAE and RMSE: {n_net_pos}/{n_net_total}")
    print("Analysis status: FROZEN_FOR_PUBLICATION_OUTPUTS")
    print(f"Master result matrix: {master_path.name}")
    print(f"Claims register:       {claims_path.name}")
    print(f"Figure/table plan:     {figplan_path.name}")
    print(f"Freeze summary:        {summary_path.name}")


if __name__ == "__main__":
    main()
