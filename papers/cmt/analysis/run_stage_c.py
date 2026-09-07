#!/usr/bin/env python3
"""CMT Stage C: paired framework-formula bootstrap inference.

This script performs no model fitting. It consumes the frozen Stage B
framework-formula-grouped OOF predictions and computes:
  1) DummyMean-relative MAE/RMSE skill with group-bootstrap intervals;
  2) paired incremental information value for P1->P2, P2->P3, P1->P3;
  3) the final earliest useful information stage for each of nine targets;
  4) 5%, 10%, and 20% usefulness-threshold sensitivity.

Bootstrap unit: framework_uid, resampled with replacement separately within
frozen test folds. The same bootstrap multiplicities are used for all targets
and stages, preserving pairing for stage comparisons.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import yaml


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def locate_repo() -> Path:
    p = Path(__file__).resolve()
    for root in [p.parent, *p.parents]:
        if (root / "data").is_dir() and (root / "results").is_dir() and (root / "notebooks").is_dir():
            return root
    raise FileNotFoundError("Repository root not found.")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json(obj, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    tmp.replace(path)


def write_csv_atomic(df: pd.DataFrame, path: Path, **kwargs) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    df.to_csv(tmp, index=False, **kwargs)
    tmp.replace(path)


def add_check(rows, name, expected, observed, passed, detail=""):
    rows.append({
        "check": name,
        "expected": str(expected),
        "observed": str(observed),
        "pass": bool(passed),
        "detail": detail,
    })


def percentile_ci(values: np.ndarray, confidence_level: float) -> tuple[float, float]:
    alpha = 1.0 - float(confidence_level)
    q = [100.0 * alpha / 2.0, 100.0 * (1.0 - alpha / 2.0)]
    lo, hi = np.percentile(np.asarray(values, dtype=float), q)
    return float(lo), float(hi)


def direction_from_ci(lo: float, hi: float) -> str:
    if lo > 0.0:
        return "improvement_supported"
    if hi < 0.0:
        return "degradation_supported"
    return "interval_spans_zero"


def load_config(repo: Path, config_arg: str | None):
    config_path = Path(config_arg).resolve() if config_arg else repo / "papers/cmt/config/cmt_stage_c_v1.yaml"
    with config_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return config_path, cfg


def stage_c(repo: Path, config_path: Path, cfg: dict) -> dict:
    out_dir = repo / cfg["output_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)
    audit_rows = []

    input_paths = {k: repo / v for k, v in cfg["inputs"].items()}
    input_hash_rows = []
    for role, path in input_paths.items():
        exists = path.is_file()
        add_check(audit_rows, f"input_exists::{role}", True, exists, exists, str(path))
        if exists:
            input_hash_rows.append({
                "role": role,
                "relative_path": str(path.relative_to(repo)),
                "sha256": sha256_file(path),
                "size_bytes": int(path.stat().st_size),
            })
    input_hash_rows.append({
        "role": "stage_c_config",
        "relative_path": str(config_path.relative_to(repo)),
        "sha256": sha256_file(config_path),
        "size_bytes": int(config_path.stat().st_size),
    })

    if not all(r["pass"] for r in audit_rows):
        audit = pd.DataFrame(audit_rows)
        write_csv_atomic(audit, out_dir / "stage_c_input_audit.csv")
        write_json({"stage": "C", "status": "FAIL", "timestamp_utc": utc_now(), "reason": "missing_input"}, out_dir / "stage_c_gate.json")
        raise RuntimeError("Stage C blocked: required Stage B inputs are missing.")

    gate_b = json.loads(input_paths["stage_b_gate"].read_text(encoding="utf-8"))
    add_check(audit_rows, "stage_b_gate_status", "PASS", gate_b.get("status"), gate_b.get("status") == "PASS")
    add_check(audit_rows, "stage_b_scope", "primary_framework_formula_grouped_only", gate_b.get("scope"), gate_b.get("scope") == "primary_framework_formula_grouped_only")
    add_check(audit_rows, "stage_b_n_oof_predictions", 86427, gate_b.get("n_oof_predictions"), int(gate_b.get("n_oof_predictions", -1)) == 86427)
    add_check(audit_rows, "stage_b_n_extratrees_fits", 135, gate_b.get("n_extratrees_fits"), int(gate_b.get("n_extratrees_fits", -1)) == 135)

    with input_paths["stage_ab_config"].open("r", encoding="utf-8") as f:
        ab_cfg = yaml.safe_load(f)
    primary_threshold = float(cfg["useful_learnability_rule"]["primary_threshold"])
    ab_mae_threshold = float(ab_cfg["useful_learnability_rule"]["minimum_mae_skill_vs_dummy"])
    ab_rmse_threshold = float(ab_cfg["useful_learnability_rule"]["minimum_rmse_skill_vs_dummy"])
    add_check(audit_rows, "threshold_matches_stage_ab_mae", primary_threshold, ab_mae_threshold, np.isclose(primary_threshold, ab_mae_threshold))
    add_check(audit_rows, "threshold_matches_stage_ab_rmse", primary_threshold, ab_rmse_threshold, np.isclose(primary_threshold, ab_rmse_threshold))

    oof = pd.read_csv(input_paths["stage_b_oof_predictions"])
    summary_b = pd.read_csv(input_paths["stage_b_primary_summary"])
    split_manifest = pd.read_csv(input_paths["stage_a_split_manifest"])

    targets = list(cfg["targets"])
    stages = list(cfg["stages"])
    folds = sorted(oof["fold_id"].unique().tolist())
    expected_records = int(cfg["primary_validation"]["n_records"])
    expected_groups = int(cfg["primary_validation"]["n_framework_formula_groups"])
    expected_rows = expected_records * len(targets) * len(stages)

    required_cols = {
        "target", "stage", "fold_id", "record_index", "framework_uid", "y_true", "y_pred",
        "y_pred_dummy", "abs_error", "sq_error", "dummy_abs_error", "dummy_sq_error",
    }
    missing_cols = sorted(required_cols - set(oof.columns))
    add_check(audit_rows, "oof_required_columns", 0, len(missing_cols), len(missing_cols) == 0, ";".join(missing_cols))
    add_check(audit_rows, "oof_total_rows", expected_rows, len(oof), len(oof) == expected_rows)
    add_check(audit_rows, "n_targets", len(targets), oof["target"].nunique(), set(oof["target"]) == set(targets))
    add_check(audit_rows, "n_stages", len(stages), oof["stage"].nunique(), set(oof["stage"]) == set(stages))
    add_check(audit_rows, "n_folds", cfg["primary_validation"]["n_folds"], len(folds), len(folds) == int(cfg["primary_validation"]["n_folds"]))

    finite_ok = np.isfinite(oof[["y_true", "y_pred", "y_pred_dummy", "abs_error", "sq_error", "dummy_abs_error", "dummy_sq_error"]].to_numpy(dtype=float)).all()
    add_check(audit_rows, "all_numeric_inputs_finite", True, finite_ok, bool(finite_ok))

    coverage_ok = True
    duplicate_ok = True
    combo_counts = []
    for target in targets:
        for stage in stages:
            sub = oof[(oof["target"] == target) & (oof["stage"] == stage)]
            combo_counts.append(len(sub))
            coverage_ok &= (len(sub) == expected_records and sub["record_index"].nunique() == expected_records)
            duplicate_ok &= (not sub["record_index"].duplicated().any())
    add_check(audit_rows, "complete_record_coverage_each_target_stage", expected_records, min(combo_counts) if combo_counts else 0, coverage_ok)
    add_check(audit_rows, "no_duplicate_records_each_target_stage", True, duplicate_ok, duplicate_ok)

    # Metadata/fold identity must be identical across all target-stage blocks.
    base = oof[(oof["target"] == targets[0]) & (oof["stage"] == stages[0])][["record_index", "fold_id", "framework_uid"]].copy()
    base = base.sort_values("record_index").reset_index(drop=True)
    metadata_match = True
    y_match_across_stages = True
    for target in targets:
        y_reference = None
        for stage in stages:
            sub = oof[(oof["target"] == target) & (oof["stage"] == stage)].sort_values("record_index").reset_index(drop=True)
            metadata_match &= sub[["record_index", "fold_id", "framework_uid"]].equals(base)
            if y_reference is None:
                y_reference = sub[["record_index", "y_true"]].copy()
            else:
                y_match_across_stages &= sub["record_index"].equals(y_reference["record_index"]) and np.allclose(sub["y_true"].to_numpy(float), y_reference["y_true"].to_numpy(float), rtol=0.0, atol=0.0)
    add_check(audit_rows, "record_fold_group_metadata_identical_across_blocks", True, metadata_match, metadata_match)
    add_check(audit_rows, "target_truth_identical_across_stages", True, y_match_across_stages, y_match_across_stages)

    group_fold_counts = base.groupby("framework_uid")["fold_id"].nunique()
    one_fold_per_group = bool((group_fold_counts == 1).all())
    add_check(audit_rows, "framework_group_assigned_to_one_test_fold", True, one_fold_per_group, one_fold_per_group)
    add_check(audit_rows, "framework_group_count", expected_groups, base["framework_uid"].nunique(), base["framework_uid"].nunique() == expected_groups)
    add_check(audit_rows, "split_manifest_record_count", expected_records, len(split_manifest), len(split_manifest) == expected_records)

    # Recompute Stage B point estimates as a guard against reading the wrong OOF file.
    point_rows = []
    point_match = True
    for target in targets:
        for stage in stages:
            sub = oof[(oof["target"] == target) & (oof["stage"] == stage)]
            n = len(sub)
            et_mae = float(sub["abs_error"].sum() / n)
            et_rmse = float(np.sqrt(sub["sq_error"].sum() / n))
            dm_mae = float(sub["dummy_abs_error"].sum() / n)
            dm_rmse = float(np.sqrt(sub["dummy_sq_error"].sum() / n))
            mae_skill = float(1.0 - et_mae / dm_mae)
            rmse_skill = float(1.0 - et_rmse / dm_rmse)
            point_rows.append({
                "target": target, "stage": stage, "n_oof": n,
                "et_mae": et_mae, "et_rmse": et_rmse,
                "dummy_mae": dm_mae, "dummy_rmse": dm_rmse,
                "mae_skill_vs_dummy": mae_skill, "rmse_skill_vs_dummy": rmse_skill,
            })
            match = summary_b[(summary_b["target"] == target) & (summary_b["stage"] == stage)]
            if len(match) != 1:
                point_match = False
            else:
                r = match.iloc[0]
                point_match &= np.isclose(et_mae, float(r["pooled_et_mae"]), rtol=1e-12, atol=1e-12)
                point_match &= np.isclose(et_rmse, float(r["pooled_et_rmse"]), rtol=1e-12, atol=1e-12)
                point_match &= np.isclose(mae_skill, float(r["pooled_mae_skill_vs_dummy"]), rtol=1e-12, atol=1e-12)
                point_match &= np.isclose(rmse_skill, float(r["pooled_rmse_skill_vs_dummy"]), rtol=1e-12, atol=1e-12)
    add_check(audit_rows, "recomputed_points_match_stage_b_summary", True, point_match, point_match)

    audit = pd.DataFrame(audit_rows)
    write_csv_atomic(audit, out_dir / "stage_c_input_audit.csv")
    write_csv_atomic(pd.DataFrame(input_hash_rows).sort_values("role"), out_dir / "stage_c_input_sha256.csv")
    if not bool(audit["pass"].all()):
        write_json({
            "stage": "C", "status": "FAIL", "timestamp_utc": utc_now(),
            "n_checks": int(len(audit)), "n_failed_checks": int((~audit["pass"]).sum()),
            "failed_checks": audit.loc[~audit["pass"], "check"].tolist(),
        }, out_dir / "stage_c_gate.json")
        raise RuntimeError("Stage C input gate FAILED. See stage_c_input_audit.csv.")

    point = pd.DataFrame(point_rows)
    combo_order = [(t, s) for t in targets for s in stages]
    combo_index = {c: i for i, c in enumerate(combo_order)}
    n_combo = len(combo_order)

    # Aggregate record errors to framework-formula groups. This both preserves the
    # cluster resampling unit and makes 5000 paired replicates inexpensive.
    agg = (
        oof.groupby(["fold_id", "framework_uid", "target", "stage"], sort=True, observed=True)
        .agg(
            n_records=("record_index", "size"),
            et_abs_sum=("abs_error", "sum"),
            et_sq_sum=("sq_error", "sum"),
            dm_abs_sum=("dummy_abs_error", "sum"),
            dm_sq_sum=("dummy_sq_error", "sum"),
        )
        .reset_index()
    )
    agg["combo"] = list(zip(agg["target"], agg["stage"]))

    B = int(cfg["bootstrap"]["n_replicates"])
    conf = float(cfg["bootstrap"]["confidence_level"])
    seed = int(cfg["bootstrap"]["random_seed"])
    rng = np.random.default_rng(seed)

    total_n = np.zeros(B, dtype=float)
    et_abs_total = np.zeros((B, n_combo), dtype=float)
    et_sq_total = np.zeros((B, n_combo), dtype=float)
    dm_abs_total = np.zeros((B, n_combo), dtype=float)
    dm_sq_total = np.zeros((B, n_combo), dtype=float)
    fold_boot_rows = []

    for fold in folds:
        fold_base = base[base["fold_id"] == fold]
        groups = sorted(fold_base["framework_uid"].unique().tolist())
        n_groups = len(groups)
        # Group sizes are target/stage invariant; take them from the first combo.
        first_target, first_stage = combo_order[0]
        first = agg[(agg["fold_id"] == fold) & (agg["target"] == first_target) & (agg["stage"] == first_stage)].set_index("framework_uid").reindex(groups)
        if first.isna().any().any():
            raise RuntimeError(f"Missing group aggregate in fold {fold}.")
        group_n = first["n_records"].to_numpy(dtype=float)

        matrices = {}
        for metric in ["et_abs_sum", "et_sq_sum", "dm_abs_sum", "dm_sq_sum"]:
            mat = np.empty((n_groups, n_combo), dtype=float)
            for j, (target, stage) in enumerate(combo_order):
                x = agg[(agg["fold_id"] == fold) & (agg["target"] == target) & (agg["stage"] == stage)].set_index("framework_uid").reindex(groups)
                if x[metric].isna().any():
                    raise RuntimeError(f"Missing {metric} aggregate for {target} {stage} fold {fold}.")
                if not np.array_equal(x["n_records"].to_numpy(dtype=int), group_n.astype(int)):
                    raise RuntimeError(f"Group sizes differ across target/stage blocks in fold {fold}.")
                mat[:, j] = x[metric].to_numpy(dtype=float)
            matrices[metric] = mat

        # Multinomial counts are exactly equivalent to drawing n_groups groups
        # with replacement and are much faster than materializing record rows.
        counts = rng.multinomial(n_groups, np.full(n_groups, 1.0 / n_groups), size=B)
        total_n += counts @ group_n
        et_abs_total += counts @ matrices["et_abs_sum"]
        et_sq_total += counts @ matrices["et_sq_sum"]
        dm_abs_total += counts @ matrices["dm_abs_sum"]
        dm_sq_total += counts @ matrices["dm_sq_sum"]
        fold_boot_rows.append({
            "fold_id": int(fold),
            "n_records": int(len(fold_base)),
            "n_framework_groups": int(n_groups),
        })

    et_mae_boot = et_abs_total / total_n[:, None]
    et_rmse_boot = np.sqrt(et_sq_total / total_n[:, None])
    dm_mae_boot = dm_abs_total / total_n[:, None]
    dm_rmse_boot = np.sqrt(dm_sq_total / total_n[:, None])
    mae_skill_boot = 1.0 - et_mae_boot / dm_mae_boot
    rmse_skill_boot = 1.0 - et_rmse_boot / dm_rmse_boot

    # Primary skill inference.
    skill_rows = []
    skill_rep_parts = []
    for target, stage in combo_order:
        j = combo_index[(target, stage)]
        p = point[(point["target"] == target) & (point["stage"] == stage)].iloc[0]
        mae_lo, mae_hi = percentile_ci(mae_skill_boot[:, j], conf)
        rmse_lo, rmse_hi = percentile_ci(rmse_skill_boot[:, j], conf)
        pass_point = bool(p["mae_skill_vs_dummy"] >= primary_threshold and p["rmse_skill_vs_dummy"] >= primary_threshold)
        mae_positive = bool(mae_lo > 0.0)
        rmse_positive = bool(rmse_lo > 0.0)
        useful = bool(pass_point and mae_positive and rmse_positive)
        skill_rows.append({
            "target": target,
            "stage": stage,
            "n_oof": int(p["n_oof"]),
            "et_mae": float(p["et_mae"]),
            "et_rmse": float(p["et_rmse"]),
            "dummy_mae": float(p["dummy_mae"]),
            "dummy_rmse": float(p["dummy_rmse"]),
            "mae_skill_vs_dummy": float(p["mae_skill_vs_dummy"]),
            "mae_skill_ci_low": mae_lo,
            "mae_skill_ci_high": mae_hi,
            "rmse_skill_vs_dummy": float(p["rmse_skill_vs_dummy"]),
            "rmse_skill_ci_low": rmse_lo,
            "rmse_skill_ci_high": rmse_hi,
            "passes_primary_point_threshold": pass_point,
            "mae_skill_interval_entirely_above_zero": mae_positive,
            "rmse_skill_interval_entirely_above_zero": rmse_positive,
            "useful_learnability_primary_rule": useful,
        })
        skill_rep_parts.append(pd.DataFrame({
            "bootstrap_replicate": np.arange(1, B + 1, dtype=int),
            "target": target,
            "stage": stage,
            "mae_skill_vs_dummy": mae_skill_boot[:, j],
            "rmse_skill_vs_dummy": rmse_skill_boot[:, j],
        }))
    skill_summary = pd.DataFrame(skill_rows)
    write_csv_atomic(skill_summary, out_dir / "stage_c_skill_bootstrap_summary.csv")
    skill_reps = pd.concat(skill_rep_parts, ignore_index=True)
    write_csv_atomic(skill_reps, out_dir / "stage_c_skill_bootstrap_replicates.csv.gz", compression={"method": "gzip", "mtime": 0})

    # Paired incremental information value.
    transition_rows = []
    transition_rep_parts = []
    transitions = [tuple(x) for x in cfg["incremental_information_value"]["transitions"]]
    for target in targets:
        for a, b in transitions:
            ia, ib = combo_index[(target, a)], combo_index[(target, b)]
            pa = point[(point["target"] == target) & (point["stage"] == a)].iloc[0]
            pb = point[(point["target"] == target) & (point["stage"] == b)].iloc[0]

            d_mae = et_mae_boot[:, ia] - et_mae_boot[:, ib]
            r_mae = d_mae / et_mae_boot[:, ia]
            d_rmse = et_rmse_boot[:, ia] - et_rmse_boot[:, ib]
            r_rmse = d_rmse / et_rmse_boot[:, ia]
            d_mae_lo, d_mae_hi = percentile_ci(d_mae, conf)
            r_mae_lo, r_mae_hi = percentile_ci(r_mae, conf)
            d_rmse_lo, d_rmse_hi = percentile_ci(d_rmse, conf)
            r_rmse_lo, r_rmse_hi = percentile_ci(r_rmse, conf)

            point_d_mae = float(pa["et_mae"] - pb["et_mae"])
            point_r_mae = float(point_d_mae / pa["et_mae"])
            point_d_rmse = float(pa["et_rmse"] - pb["et_rmse"])
            point_r_rmse = float(point_d_rmse / pa["et_rmse"])
            transition_rows.append({
                "target": target,
                "from_stage": a,
                "to_stage": b,
                "delta_mae_positive_is_improvement": point_d_mae,
                "delta_mae_ci_low": d_mae_lo,
                "delta_mae_ci_high": d_mae_hi,
                "delta_mae_interval_direction": direction_from_ci(d_mae_lo, d_mae_hi),
                "relative_mae_improvement": point_r_mae,
                "relative_mae_ci_low": r_mae_lo,
                "relative_mae_ci_high": r_mae_hi,
                "delta_rmse_positive_is_improvement": point_d_rmse,
                "delta_rmse_ci_low": d_rmse_lo,
                "delta_rmse_ci_high": d_rmse_hi,
                "delta_rmse_interval_direction": direction_from_ci(d_rmse_lo, d_rmse_hi),
                "relative_rmse_improvement": point_r_rmse,
                "relative_rmse_ci_low": r_rmse_lo,
                "relative_rmse_ci_high": r_rmse_hi,
            })
            transition_rep_parts.append(pd.DataFrame({
                "bootstrap_replicate": np.arange(1, B + 1, dtype=int),
                "target": target,
                "from_stage": a,
                "to_stage": b,
                "delta_mae_positive_is_improvement": d_mae,
                "relative_mae_improvement": r_mae,
                "delta_rmse_positive_is_improvement": d_rmse,
                "relative_rmse_improvement": r_rmse,
            }))
    transition_summary = pd.DataFrame(transition_rows)
    write_csv_atomic(transition_summary, out_dir / "stage_c_incremental_information_value_bootstrap.csv")
    transition_reps = pd.concat(transition_rep_parts, ignore_index=True)
    write_csv_atomic(transition_reps, out_dir / "stage_c_incremental_information_value_replicates.csv.gz", compression={"method": "gzip", "mtime": 0})

    # Final earliest useful stage under the frozen primary rule.
    stage_rank = {s: i for i, s in enumerate(stages)}
    earliest_rows = []
    for target in targets:
        sub = skill_summary[skill_summary["target"] == target].copy()
        passing = sub[sub["useful_learnability_primary_rule"]]
        earliest = "none_through_P3" if passing.empty else sorted(passing["stage"].tolist(), key=lambda s: stage_rank[s])[0]
        earliest_rows.append({
            "target": target,
            "earliest_useful_stage": earliest,
            "criterion_mae_skill_threshold": primary_threshold,
            "criterion_rmse_skill_threshold": primary_threshold,
            "criterion_bootstrap_intervals_entirely_above_zero": True,
            "validation": cfg["primary_validation"]["reporting_name"],
            "estimator": "fixed ExtraTrees",
        })
    earliest_df = pd.DataFrame(earliest_rows)
    write_csv_atomic(earliest_df, out_dir / "stage_c_earliest_useful_stage.csv")

    # Threshold sensitivity is free once the bootstrap exists; the interval rule
    # remains fixed and only the point usefulness threshold changes.
    sens_rows = []
    for threshold in [float(x) for x in cfg["useful_learnability_rule"]["threshold_sensitivity"]]:
        for target in targets:
            sub = skill_summary[skill_summary["target"] == target].copy()
            sub["passes_threshold"] = (
                (sub["mae_skill_vs_dummy"] >= threshold)
                & (sub["rmse_skill_vs_dummy"] >= threshold)
                & sub["mae_skill_interval_entirely_above_zero"]
                & sub["rmse_skill_interval_entirely_above_zero"]
            )
            passing = sub[sub["passes_threshold"]]
            earliest = "none_through_P3" if passing.empty else sorted(passing["stage"].tolist(), key=lambda s: stage_rank[s])[0]
            sens_rows.append({
                "threshold": threshold,
                "target": target,
                "earliest_useful_stage": earliest,
            })
    sensitivity = pd.DataFrame(sens_rows)
    write_csv_atomic(sensitivity, out_dir / "stage_c_threshold_sensitivity.csv")

    bootstrap_meta = {
        "analysis_id": cfg["analysis_id"],
        "timestamp_utc": utc_now(),
        "n_replicates": B,
        "confidence_level": conf,
        "interval_method": cfg["bootstrap"]["interval_method"],
        "random_seed": seed,
        "resampling_unit": cfg["bootstrap"]["resampling_unit"],
        "resample_within_fold": bool(cfg["bootstrap"]["resample_within_fold"]),
        "paired_across_stages": bool(cfg["bootstrap"]["paired_across_stages"]),
        "paired_across_targets": bool(cfg["bootstrap"]["paired_across_targets"]),
        "folds": fold_boot_rows,
        "bootstrap_record_count_min": int(np.min(total_n)),
        "bootstrap_record_count_median": float(np.median(total_n)),
        "bootstrap_record_count_max": int(np.max(total_n)),
        "primary_useful_threshold": primary_threshold,
        "final_useful_rule": "both point skills >= threshold AND both 95% skill intervals entirely above zero",
        "model_refitting_performed": False,
    }
    write_json(bootstrap_meta, out_dir / "stage_c_bootstrap_metadata.json")

    env = {
        "timestamp_utc": utc_now(),
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "os_cpu_count": os.cpu_count(),
    }
    write_json(env, out_dir / "stage_c_environment.json")

    gate_c = {
        "stage": "C",
        "status": "PASS",
        "timestamp_utc": utc_now(),
        "analysis_id": cfg["analysis_id"],
        "config_sha256": sha256_file(config_path),
        "stage_b_oof_sha256": sha256_file(input_paths["stage_b_oof_predictions"]),
        "n_targets": len(targets),
        "n_stages": len(stages),
        "n_bootstrap_replicates": B,
        "primary_validation": cfg["primary_validation"]["reporting_name"],
        "n_final_earliest_stage_rows": len(earliest_df),
        "model_refitting_performed": False,
    }
    write_json(gate_c, out_dir / "stage_c_gate.json")

    output_files = [p for p in sorted(out_dir.iterdir()) if p.is_file() and p.name != "stage_c_output_sha256.csv"]
    output_manifest = pd.DataFrame([
        {
            "relative_path": str(p.relative_to(repo)),
            "sha256": sha256_file(p),
            "size_bytes": int(p.stat().st_size),
        }
        for p in output_files
    ])
    write_csv_atomic(output_manifest, out_dir / "stage_c_output_sha256.csv")
    return gate_c


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None, help="Path to cmt_stage_c_v1.yaml")
    args = parser.parse_args()

    repo = locate_repo()
    config_path, cfg = load_config(repo, args.config)
    print(f"Repository: {repo}")
    print(f"Config:     {config_path}")
    print(f"Input:      {repo / cfg['inputs']['stage_b_oof_predictions']}")
    print(f"Output:     {repo / cfg['output_dir']}")
    print("Stage C: no model fitting; paired framework-formula bootstrap only")

    gate = stage_c(repo, config_path, cfg)
    print(f"Stage C: {gate['status']} | bootstrap replicates={gate['n_bootstrap_replicates']} | targets={gate['n_targets']}")
    print("Final earliest-useful-stage table written to stage_c_earliest_useful_stage.csv")


if __name__ == "__main__":
    main()
