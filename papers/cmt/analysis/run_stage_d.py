#!/usr/bin/env python3
"""CMT Stage D: lean chemical-domain robustness analysis.

Default behavior is intentionally lean:
  1. verify Stage B and Stage C gates;
  2. verify that the five historical Notebook-04 target results are compatible
     with the frozen CMT estimator/splits and current Stage-B framework results;
  3. reuse those five target results only if every compatibility gate passes;
  4. fit only the four targets that Notebook 04 never benchmarked;
  5. combine both sources into one audited fold-level robustness table.

The scientific model is unchanged: median imputation + fixed 300-tree ExtraTrees,
with fold-matched DummyMean. No random validation is run here. The three Stage-D
stress tests are coarse chemistry family, host chemical system, and working ion.

Run from the repository root:
    python -u papers/cmt/analysis/run_stage_d.py

Optional full rerun (not required unless the reuse audit fails or for a later
uniform-environment reproduction pass):
    python -u papers/cmt/analysis/run_stage_d.py --force-full-rerun
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import scipy
import sklearn
import yaml
from scipy.stats import spearmanr
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline


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


def write_csv_atomic(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    df.to_csv(tmp, index=False)
    tmp.replace(path)


def add_check(rows, name, expected, observed, passed, detail=""):
    rows.append({
        "check": name,
        "expected": str(expected),
        "observed": str(observed),
        "pass": bool(passed),
        "detail": detail,
    })


def metrics(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred)) if len(y_true) > 1 else np.nan
    try:
        if np.allclose(y_true, y_true[0]) or np.allclose(y_pred, y_pred[0]):
            sp = np.nan
        else:
            sp = float(spearmanr(y_true, y_pred).statistic)
    except Exception:
        sp = np.nan
    return {"mae": mae, "rmse": rmse, "r2": r2, "spearman": sp}


def protocol_file(repo: Path, target: str, stage: str) -> Path:
    return repo / "data/processed/notebook_02/protocol_feature_lists" / f"02_protocol_{stage}_features_for_target_{target}.csv"


def scientific_signature(target, stage, split_name, feature_path, outer_path, valid_path, model_cfg):
    payload = {
        "target": target,
        "stage": stage,
        "split_name": split_name,
        "feature_sha256": sha256_file(feature_path),
        "outer_manifest_sha256": sha256_file(outer_path),
        "valid_universe_sha256": sha256_file(valid_path),
        "model": model_cfg,
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def load_yaml(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def stage_preflight(repo: Path, cfg_path: Path, cfg: dict, out_dir: Path, force_full_rerun: bool):
    checks = []
    required = {k: repo / v for k, v in cfg["inputs"].items() if k != "protocol_feature_dir"}
    for key, path in required.items():
        add_check(checks, f"input_exists::{key}", True, path.is_file(), path.is_file(), str(path))
    if not all(r["pass"] for r in checks):
        audit = pd.DataFrame(checks)
        write_csv_atomic(audit, out_dir / "stage_d_preflight_audit.csv")
        raise RuntimeError("Stage D preflight failed because required files are missing.")

    gate_b = json.loads(required["stage_b_gate"].read_text(encoding="utf-8"))
    gate_c = json.loads(required["stage_c_gate"].read_text(encoding="utf-8"))
    add_check(checks, "stage_b_gate", "PASS", gate_b.get("status"), gate_b.get("status") == "PASS")
    add_check(checks, "stage_c_gate", "PASS", gate_c.get("status"), gate_c.get("status") == "PASS")
    add_check(checks, "stage_b_targets", 9, gate_b.get("n_targets"), int(gate_b.get("n_targets", -1)) == 9)
    add_check(checks, "stage_c_targets", 9, gate_c.get("n_targets"), int(gate_c.get("n_targets", -1)) == 9)

    ab_cfg = load_yaml(required["stage_ab_config"])
    targets = list(ab_cfg["targets"].keys())
    stages = list(ab_cfg["stages"])
    valid = pd.read_csv(required["valid_record_universe"])
    split_plan = pd.read_csv(required["split_plan"])
    outer = pd.read_csv(required["outer_test_manifest"])
    old = pd.read_csv(required["historical_fold_results"])
    old_model = pd.read_csv(required["historical_model_audit"])
    current_summary = pd.read_csv(required["stage_b_primary_summary"])

    add_check(checks, "valid_records", 3201, len(valid), len(valid) == 3201)
    add_check(checks, "targets_from_stage_ab", 9, len(targets), len(targets) == 9)
    add_check(checks, "stages_from_stage_ab", "P1,P2,P3", ",".join(stages), stages == ["P1", "P2", "P3"])

    # All three robustness split plans must exactly match the frozen Notebook-05 split plan.
    expected_total_folds = 0
    for split_name, spec in cfg["robustness_validations"].items():
        plan = split_plan[split_plan["split_name"] == split_name].copy()
        out = outer[(outer["split_name"] == split_name) & (outer["set_role"] == "test")].copy()
        n_folds = int(plan["fold_id"].nunique())
        expected = int(spec["expected_folds"])
        expected_total_folds += expected
        add_check(checks, f"split_folds::{split_name}", expected, n_folds, n_folds == expected)
        outer_sizes = out.groupby("fold_id").size().rename("n_test_outer").reset_index()
        cmp = plan[["fold_id", "n_test"]].merge(outer_sizes, on="fold_id", how="left")
        same_sizes = bool((cmp["n_test"] == cmp["n_test_outer"]).all()) and len(cmp) == expected
        add_check(checks, f"split_test_sizes::{split_name}", True, same_sizes, same_sizes)

    reusable = list(cfg["historical_reuse"]["reusable_targets"])
    new_targets = list(cfg["historical_reuse"]["new_fit_targets"])
    add_check(checks, "reuse_plus_new_cover_all_targets", set(targets), set(reusable + new_targets), set(reusable + new_targets) == set(targets))
    add_check(checks, "reuse_and_new_disjoint", True, set(reusable).isdisjoint(new_targets), set(reusable).isdisjoint(new_targets))

    # Historical model definition audit.
    et_rows = old_model[old_model["model_name"] == "ExtraTrees"]
    model_repr = et_rows["model_repr"].iloc[0] if len(et_rows) == 1 else ""
    model_ok = len(et_rows) == 1 and "n_estimators=300" in model_repr and "random_state=42" in model_repr
    add_check(checks, "historical_model_core_config", True, model_ok, model_ok, model_repr)

    # Historical split sizes and feature counts must match the frozen plans for every reusable target/stage.
    hist_sub = old[
        old["target"].isin(reusable)
        & old["protocol"].isin(stages)
        & old["split_name"].isin(cfg["robustness_validations"].keys())
        & old["model_name"].isin(["ExtraTrees", "DummyMean"])
    ].copy()
    expected_hist_model_rows = len(reusable) * len(stages) * expected_total_folds * 2
    add_check(checks, "historical_domain_model_rows", expected_hist_model_rows, len(hist_sub), len(hist_sub) == expected_hist_model_rows)
    add_check(checks, "historical_status_all_ok", True, bool((hist_sub["status"] == "OK").all()), bool((hist_sub["status"] == "OK").all()))

    feature_count_ok = True
    for target in reusable:
        for stage in stages:
            n_expected = int(ab_cfg["targets"][target][stage])
            vals = hist_sub[(hist_sub["target"] == target) & (hist_sub["protocol"] == stage) & (hist_sub["model_name"] == "ExtraTrees")]["n_features"].unique().tolist()
            if vals != [n_expected]:
                feature_count_ok = False
    add_check(checks, "historical_feature_counts_match", True, feature_count_ok, feature_count_ok)

    size_ok = True
    for split_name in cfg["robustness_validations"].keys():
        plan = split_plan[split_plan["split_name"] == split_name][["fold_id", "n_train", "n_test"]].copy()
        hs = hist_sub[(hist_sub["split_name"] == split_name) & (hist_sub["model_name"] == "ExtraTrees")]
        hsizes = hs.groupby("fold_id")[["n_train", "n_test"]].first().reset_index()
        cmp = hsizes.merge(plan, on="fold_id", suffixes=("_hist", "_plan"), how="outer")
        if len(cmp) != len(plan) or not bool(((cmp["n_train_hist"] == cmp["n_train_plan"]) & (cmp["n_test_hist"] == cmp["n_test_plan"])).all()):
            size_ok = False
    add_check(checks, "historical_split_sizes_match", True, size_ok, size_ok)

    # Current Stage-B framework comparison is the decisive reuse compatibility test.
    old_framework = old[
        old["target"].isin(reusable)
        & old["protocol"].isin(stages)
        & (old["split_name"] == cfg["historical_reuse"]["framework_compatibility_split"])
        & (old["model_name"] == "ExtraTrees")
    ][["target", "protocol", "mae", "rmse"]].groupby(["target", "protocol"], as_index=False).mean()
    cur = current_summary[current_summary["target"].isin(reusable)][
        ["target", "stage", "fold_mean_et_mae", "fold_mean_et_rmse"]
    ].copy()
    comp = old_framework.merge(cur, left_on=["target", "protocol"], right_on=["target", "stage"], how="outer")
    comp["mae_relative_difference"] = (comp["fold_mean_et_mae"] - comp["mae"]).abs() / comp["mae"].abs().clip(lower=1e-15)
    comp["rmse_relative_difference"] = (comp["fold_mean_et_rmse"] - comp["rmse"]).abs() / comp["rmse"].abs().clip(lower=1e-15)
    tol = float(cfg["historical_reuse"]["maximum_relative_metric_difference"])
    comp["passes_tolerance"] = (comp["mae_relative_difference"] <= tol) & (comp["rmse_relative_difference"] <= tol)
    compatibility_ok = len(comp) == len(reusable) * len(stages) and bool(comp["passes_tolerance"].all())
    add_check(checks, "historical_framework_compatibility", True, compatibility_ok, compatibility_ok,
              f"max_mae_rel={comp['mae_relative_difference'].max():.8g}; max_rmse_rel={comp['rmse_relative_difference'].max():.8g}; tol={tol}")
    write_csv_atomic(comp, out_dir / "stage_d_historical_framework_compatibility.csv")

    reuse_ok = bool(cfg["historical_reuse"].get("enabled", True)) and model_ok and feature_count_ok and size_ok and compatibility_ok and not force_full_rerun
    run_targets = new_targets if reuse_ok else targets
    mode = "lean_historical_reuse_plus_four_new_targets" if reuse_ok else "full_nine_target_rerun"
    add_check(checks, "execution_mode_selected", "lean reuse if compatible", mode, True)

    audit = pd.DataFrame(checks)
    write_csv_atomic(audit, out_dir / "stage_d_preflight_audit.csv")
    if not bool(audit["pass"].all()):
        raise RuntimeError("Stage D preflight FAILED. See stage_d_preflight_audit.csv.")

    preflight = {
        "status": "PASS",
        "timestamp_utc": utc_now(),
        "execution_mode": mode,
        "historical_reuse_enabled": reuse_ok,
        "run_targets": run_targets,
        "reused_targets": reusable if reuse_ok else [],
        "expected_total_domain_folds_per_target_stage": expected_total_folds,
        "expected_new_extratrees_fits": len(run_targets) * len(stages) * expected_total_folds,
    }
    write_json(preflight, out_dir / "stage_d_preflight_gate.json")
    return ab_cfg, targets, stages, reusable, run_targets, old, valid, split_plan, outer, reuse_ok, preflight


def historical_rows(old, reusable, stages, cfg):
    domain_names = list(cfg["robustness_validations"].keys())
    sub = old[
        old["target"].isin(reusable)
        & old["protocol"].isin(stages)
        & old["split_name"].isin(domain_names)
        & old["model_name"].isin(["ExtraTrees", "DummyMean"])
    ].copy()
    keys = ["target", "protocol", "split_name", "fold_id", "heldout_group"]
    et = sub[sub["model_name"] == "ExtraTrees"].copy()
    dm = sub[sub["model_name"] == "DummyMean"].copy()
    keep_et = keys + ["n_train", "n_test", "n_features", "rmse", "mae", "r2", "spearman"]
    keep_dm = keys + ["rmse", "mae", "r2", "spearman"]
    m = et[keep_et].merge(dm[keep_dm], on=keys, how="inner", suffixes=("_et", "_dummy"))
    out = pd.DataFrame({
        "target": m["target"].astype(str),
        "stage": m["protocol"].astype(str),
        "split_name": m["split_name"].astype(str),
        "fold_id": m["fold_id"].astype(int),
        "heldout_group": m["heldout_group"].astype(str),
        "n_train": m["n_train"].astype(int),
        "n_test": m["n_test"].astype(int),
        "n_features": m["n_features"].astype(int),
        "et_mae": m["mae_et"].astype(float),
        "et_rmse": m["rmse_et"].astype(float),
        "et_r2": m["r2_et"].astype(float),
        "et_spearman": m["spearman_et"].astype(float),
        "dummy_mae": m["mae_dummy"].astype(float),
        "dummy_rmse": m["rmse_dummy"].astype(float),
        "dummy_r2": m["r2_dummy"].astype(float),
        "dummy_spearman": m["spearman_dummy"].astype(float),
        "source": "historical_notebook04_reused_after_compatibility_gate",
    })
    out["mae_skill_vs_dummy"] = 1.0 - out["et_mae"] / out["dummy_mae"]
    out["rmse_skill_vs_dummy"] = 1.0 - out["et_rmse"] / out["dummy_rmse"]
    return out


def fit_one_fold(X_all, y_all, train_pos, test_pos, model_cfg):
    X_train, X_test = X_all[train_pos], X_all[test_pos]
    y_train, y_test = y_all[train_pos], y_all[test_pos]
    p = model_cfg["params"]
    et = Pipeline([
        ("imputer", SimpleImputer(strategy=model_cfg["preprocessing"]["imputer"])),
        ("model", ExtraTreesRegressor(
            n_estimators=int(p["n_estimators"]),
            max_features=float(p["max_features"]),
            min_samples_leaf=int(p["min_samples_leaf"]),
            bootstrap=bool(p["bootstrap"]),
            random_state=int(p["random_state"]),
            n_jobs=int(model_cfg["execution"]["tree_n_jobs"]),
        )),
    ])
    dummy = DummyRegressor(strategy=model_cfg["baseline"]["strategy"])
    et.fit(X_train, y_train)
    dummy.fit(np.zeros((len(y_train), 1)), y_train)
    pred = et.predict(X_test)
    pred_dm = dummy.predict(np.zeros((len(y_test), 1)))
    return metrics(y_test, pred), metrics(y_test, pred_dm)


def run_new_fits(repo, cfg, ab_cfg, targets, stages, valid, outer, out_dir):
    master = pd.read_csv(repo / cfg["inputs"]["master_feature_table"])
    valid_ids = valid["record_index"].astype(str).tolist()
    master_idx = master.set_index("record_index", drop=False)
    master_valid = master_idx.loc[valid_ids].copy()
    id_to_pos = {rid: i for i, rid in enumerate(valid_ids)}
    outer_path = repo / cfg["inputs"]["outer_test_manifest"]
    valid_path = repo / cfg["inputs"]["valid_record_universe"]

    model_cfg = json.loads(json.dumps(cfg["model"]))
    jobs = cfg["model"]["execution"].get("tree_n_jobs", "auto")
    if str(jobs).lower() == "auto":
        jobs = min(int(cfg["model"]["execution"].get("tree_n_jobs_cap", 18)), max(1, joblib.cpu_count()))
    else:
        jobs = int(jobs)
    model_cfg["execution"]["tree_n_jobs"] = jobs
    print(f"execution: tree_n_jobs={jobs} | outer folds serial | checkpoint after every fold", flush=True)

    checkpoint_dir = out_dir / "_checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    completed_rows = []

    for target in targets:
        for stage in stages:
            fpath = protocol_file(repo, target, stage)
            features = pd.read_csv(fpath)["feature"].astype(str).tolist()
            X_all = master_valid[features].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
            y_all = pd.to_numeric(master_valid[target], errors="coerce").to_numpy(dtype=float)
            if not np.isfinite(y_all).all():
                raise ValueError(f"Non-finite target values for {target}")

            for split_name in cfg["execution_order"]:
                split_sub = outer[(outer["split_name"] == split_name) & (outer["set_role"] == "test")].copy()
                folds = sorted(split_sub["fold_id"].unique().tolist())
                sig = scientific_signature(target, stage, split_name, fpath, outer_path, valid_path, model_cfg)
                cpath = checkpoint_dir / f"{split_name}__{target}__{stage}.csv"
                if cpath.exists():
                    try:
                        cdf = pd.read_csv(cpath)
                        if "scientific_signature" not in cdf.columns or not set(cdf["scientific_signature"].astype(str)) <= {sig}:
                            cdf = pd.DataFrame()
                    except Exception:
                        cdf = pd.DataFrame()
                else:
                    cdf = pd.DataFrame()
                done = set(cdf["fold_id"].astype(int).tolist()) if len(cdf) else set()
                if len(done) == len(folds):
                    print(f"reuse checkpoint: {split_name:27s} {target:20s} {stage} | {len(folds)}/{len(folds)} folds", flush=True)
                    completed_rows.append(cdf)
                    continue

                rows = cdf.to_dict("records") if len(cdf) else []
                print(f"fit block: {split_name:27s} {target:20s} {stage} | {len(done)}/{len(folds)} already done", flush=True)
                for pos, fold_id in enumerate(folds, start=1):
                    if int(fold_id) in done:
                        continue
                    fs = split_sub[split_sub["fold_id"] == fold_id]
                    test_ids = fs["record_index"].astype(str).tolist()
                    heldout = str(fs["heldout_group"].iloc[0])
                    test_pos = np.asarray([id_to_pos[r] for r in test_ids], dtype=int)
                    mask = np.ones(len(valid_ids), dtype=bool)
                    mask[test_pos] = False
                    train_pos = np.where(mask)[0]
                    m_et, m_dm = fit_one_fold(X_all, y_all, train_pos, test_pos, model_cfg)
                    row = {
                        "target": target,
                        "stage": stage,
                        "split_name": split_name,
                        "fold_id": int(fold_id),
                        "heldout_group": heldout,
                        "n_train": int(len(train_pos)),
                        "n_test": int(len(test_pos)),
                        "n_features": int(len(features)),
                        **{f"et_{k}": v for k, v in m_et.items()},
                        **{f"dummy_{k}": v for k, v in m_dm.items()},
                        "source": "cmt_stage_d_new_fit",
                        "scientific_signature": sig,
                    }
                    row["mae_skill_vs_dummy"] = 1.0 - row["et_mae"] / row["dummy_mae"] if row["dummy_mae"] else np.nan
                    row["rmse_skill_vs_dummy"] = 1.0 - row["et_rmse"] / row["dummy_rmse"] if row["dummy_rmse"] else np.nan
                    rows.append(row)
                    cdf2 = pd.DataFrame(rows).sort_values("fold_id")
                    write_csv_atomic(cdf2, cpath)
                    print(f"  completed fold {pos:>2}/{len(folds)} | held out: {heldout}", flush=True)
                completed_rows.append(pd.read_csv(cpath))

    if not completed_rows:
        return pd.DataFrame()
    return pd.concat(completed_rows, ignore_index=True)


def summarize(fold_df, cfg, stage_c):
    threshold = float(cfg["point_skill_threshold"])
    reporting = {k: v["reporting_name"] for k, v in cfg["robustness_validations"].items()}
    fold_df = fold_df.copy()
    fold_df["reporting_name"] = fold_df["split_name"].map(reporting)

    summary_rows = []
    for (split_name, target, stage), sub in fold_df.groupby(["split_name", "target", "stage"], sort=False):
        row = {
            "split_name": split_name,
            "reporting_name": reporting[split_name],
            "target": target,
            "stage": stage,
            "n_folds": int(len(sub)),
            "n_test_records_sum": int(sub["n_test"].sum()),
            "macro_et_mae": float(sub["et_mae"].mean()),
            "macro_et_rmse": float(sub["et_rmse"].mean()),
            "median_et_mae": float(sub["et_mae"].median()),
            "median_et_rmse": float(sub["et_rmse"].median()),
            "macro_dummy_mae": float(sub["dummy_mae"].mean()),
            "macro_dummy_rmse": float(sub["dummy_rmse"].mean()),
            "fraction_folds_et_better_dummy_mae": float((sub["et_mae"] < sub["dummy_mae"]).mean()),
            "fraction_folds_et_better_dummy_rmse": float((sub["et_rmse"] < sub["dummy_rmse"]).mean()),
        }
        row["macro_mae_skill_vs_dummy"] = 1.0 - row["macro_et_mae"] / row["macro_dummy_mae"] if row["macro_dummy_mae"] else np.nan
        row["macro_rmse_skill_vs_dummy"] = 1.0 - row["macro_et_rmse"] / row["macro_dummy_rmse"] if row["macro_dummy_rmse"] else np.nan
        row["passes_10pct_macro_point_both"] = bool(row["macro_mae_skill_vs_dummy"] >= threshold and row["macro_rmse_skill_vs_dummy"] >= threshold)
        summary_rows.append(row)
    summary = pd.DataFrame(summary_rows).sort_values(["split_name", "target", "stage"])

    transitions = [("P1", "P2"), ("P2", "P3"), ("P1", "P3")]
    transition_rows = []
    for split_name in reporting:
        for target in sorted(fold_df["target"].unique()):
            base = fold_df[(fold_df["split_name"] == split_name) & (fold_df["target"] == target)].copy()
            for a, b in transitions:
                aa = base[base["stage"] == a][["fold_id", "heldout_group", "et_mae", "et_rmse"]].rename(columns={"et_mae": "mae_a", "et_rmse": "rmse_a"})
                bb = base[base["stage"] == b][["fold_id", "heldout_group", "et_mae", "et_rmse"]].rename(columns={"et_mae": "mae_b", "et_rmse": "rmse_b"})
                mm = aa.merge(bb, on=["fold_id", "heldout_group"], how="inner")
                mm["delta_mae"] = mm["mae_a"] - mm["mae_b"]
                mm["delta_rmse"] = mm["rmse_a"] - mm["rmse_b"]
                transition_rows.append({
                    "split_name": split_name,
                    "reporting_name": reporting[split_name],
                    "target": target,
                    "from_stage": a,
                    "to_stage": b,
                    "n_paired_folds": int(len(mm)),
                    "mean_delta_mae_positive_is_improvement": float(mm["delta_mae"].mean()),
                    "median_delta_mae": float(mm["delta_mae"].median()),
                    "fraction_folds_mae_improves": float((mm["delta_mae"] > 0).mean()),
                    "mean_delta_rmse_positive_is_improvement": float(mm["delta_rmse"].mean()),
                    "median_delta_rmse": float(mm["delta_rmse"].median()),
                    "fraction_folds_rmse_improves": float((mm["delta_rmse"] > 0).mean()),
                    "fraction_folds_both_metrics_improve": float(((mm["delta_mae"] > 0) & (mm["delta_rmse"] > 0)).mean()),
                    "macro_both_metrics_improve": bool(mm["delta_mae"].mean() > 0 and mm["delta_rmse"].mean() > 0),
                })
    transitions_df = pd.DataFrame(transition_rows).sort_values(["split_name", "target", "from_stage", "to_stage"])

    # Point-level descriptive comparison to the Stage-C primary stage. This does not redefine useful learnability.
    order = {"P1": 0, "P2": 1, "P3": 2}
    primary_map = dict(zip(stage_c["target"], stage_c["earliest_useful_stage"]))
    robustness_rows = []
    for split_name in reporting:
        for target in sorted(summary["target"].unique()):
            ss = summary[(summary["split_name"] == split_name) & (summary["target"] == target)].copy()
            passing = ss[ss["passes_10pct_macro_point_both"]]["stage"].tolist()
            point_stage = min(passing, key=lambda s: order[s]) if passing else "none_through_P3"
            primary = primary_map[target]
            if primary == "none_through_P3":
                status = "consistent_none_at_point_level" if point_stage == "none_through_P3" else "domain_point_skill_emerges_primary_remains_none"
                ref_stage = "P3"
            else:
                ref_stage = primary
                if point_stage == "none_through_P3":
                    status = "lost_under_domain_shift_at_point_level"
                elif point_stage == primary:
                    status = "same_stage_at_point_level"
                elif order[point_stage] > order[primary]:
                    status = "later_stage_required_at_point_level"
                else:
                    status = "earlier_stage_at_point_level"
            r = ss[ss["stage"] == ref_stage].iloc[0]
            robustness_rows.append({
                "target": target,
                "split_name": split_name,
                "reporting_name": reporting[split_name],
                "stage_c_primary_earliest_useful_stage": primary,
                "domain_point_earliest_stage_10pct": point_stage,
                "comparison_status": status,
                "reference_stage_for_skill_check": ref_stage,
                "reference_stage_macro_mae_skill": float(r["macro_mae_skill_vs_dummy"]),
                "reference_stage_macro_rmse_skill": float(r["macro_rmse_skill_vs_dummy"]),
                "reference_stage_passes_10pct_macro_point": bool(r["passes_10pct_macro_point_both"]),
                "interpretation_note": "descriptive domain robustness only; Stage-C bootstrap remains the primary earliest-useful-stage classification",
            })
    robustness = pd.DataFrame(robustness_rows).sort_values(["target", "split_name"])

    working = fold_df[fold_df["split_name"] == "leave_working_ion_out"].copy().sort_values(["target", "stage", "fold_id"])
    return fold_df, summary, transitions_df, robustness, working


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None)
    parser.add_argument("--force-full-rerun", action="store_true", help="Ignore historical reuse and refit all nine targets.")
    parser.add_argument("--audit-only", action="store_true", help="Run compatibility gates only; do not fit models.")
    args = parser.parse_args()

    repo = locate_repo()
    cfg_path = Path(args.config).resolve() if args.config else repo / "papers/cmt/config/cmt_stage_d_v1.yaml"
    cfg = load_yaml(cfg_path)
    out_dir = repo / "papers/cmt/results/stage_d"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Repository: {repo}", flush=True)
    print(f"Config:     {cfg_path}", flush=True)
    print(f"Output:     {out_dir}", flush=True)
    print("Stage D: supporting chemical-domain robustness only; no random validation", flush=True)

    ab_cfg, all_targets, stages, reusable, run_targets, old, valid, split_plan, outer, reuse_ok, preflight = stage_preflight(
        repo, cfg_path, cfg, out_dir, args.force_full_rerun
    )
    print(f"Preflight: PASS | mode={preflight['execution_mode']}", flush=True)
    print(f"Historical targets reused: {', '.join(preflight['reused_targets']) if reuse_ok else 'none'}", flush=True)
    print(f"Targets requiring new fits: {', '.join(run_targets)}", flush=True)
    print(f"Expected new ExtraTrees fits: {preflight['expected_new_extratrees_fits']}", flush=True)
    if args.audit_only:
        print("Audit-only requested; stopping before model fitting.", flush=True)
        return

    parts = []
    if reuse_ok:
        parts.append(historical_rows(old, reusable, stages, cfg))
    new_df = run_new_fits(repo, cfg, ab_cfg, run_targets, stages, valid, outer, out_dir)
    if len(new_df):
        parts.append(new_df)
    fold_df = pd.concat(parts, ignore_index=True)
    # Remove checkpoint-only signature from canonical output if present.
    if "scientific_signature" in fold_df.columns:
        fold_df = fold_df.drop(columns=["scientific_signature"])
    fold_df = fold_df.sort_values(["split_name", "target", "stage", "fold_id"]).reset_index(drop=True)

    stage_c = pd.read_csv(repo / cfg["inputs"]["stage_c_earliest_useful_stage"])
    fold_df, summary, transitions, robustness, working = summarize(fold_df, cfg, stage_c)
    write_csv_atomic(fold_df, out_dir / "stage_d_fold_metrics.csv")
    write_csv_atomic(summary, out_dir / "stage_d_domain_summary.csv")
    write_csv_atomic(transitions, out_dir / "stage_d_stage_transition_summary.csv")
    write_csv_atomic(robustness, out_dir / "stage_d_primary_stage_robustness.csv")
    write_csv_atomic(working, out_dir / "stage_d_working_ion_detail.csv")

    expected_folds = sum(int(v["expected_folds"]) for v in cfg["robustness_validations"].values())
    expected_rows = len(all_targets) * len(stages) * expected_folds
    checks = []
    add_check(checks, "canonical_fold_rows", expected_rows, len(fold_df), len(fold_df) == expected_rows)
    add_check(checks, "all_nine_targets", 9, fold_df["target"].nunique(), fold_df["target"].nunique() == 9)
    add_check(checks, "all_three_stages", 3, fold_df["stage"].nunique(), fold_df["stage"].nunique() == 3)
    add_check(checks, "all_three_domain_splits", 3, fold_df["split_name"].nunique(), fold_df["split_name"].nunique() == 3)
    finite_cols = ["et_mae", "et_rmse", "dummy_mae", "dummy_rmse"]
    finite_ok = bool(np.isfinite(fold_df[finite_cols].to_numpy(dtype=float)).all())
    add_check(checks, "finite_primary_metrics", True, finite_ok, finite_ok)
    dup = fold_df.duplicated(["target", "stage", "split_name", "fold_id"]).any()
    add_check(checks, "no_duplicate_target_stage_split_fold", False, bool(dup), not bool(dup))
    for split_name, spec in cfg["robustness_validations"].items():
        sub = fold_df[fold_df["split_name"] == split_name]
        counts = sub.groupby(["target", "stage"])["fold_id"].nunique()
        ok = len(counts) == len(all_targets) * len(stages) and bool((counts == int(spec["expected_folds"])).all())
        add_check(checks, f"complete_folds::{split_name}", True, ok, ok)
    audit = pd.DataFrame(checks)
    write_csv_atomic(audit, out_dir / "stage_d_execution_audit.csv")
    status = "PASS" if bool(audit["pass"].all()) else "FAIL"

    env = {
        "timestamp_utc": utc_now(),
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scipy": scipy.__version__,
        "scikit_learn": sklearn.__version__,
        "joblib": joblib.__version__,
        "pyyaml": yaml.__version__,
        "os_cpu_count": os.cpu_count(),
    }
    write_json(env, out_dir / "stage_d_environment.json")

    n_hist = int((fold_df["source"] == "historical_notebook04_reused_after_compatibility_gate").sum()) if "source" in fold_df.columns else 0
    n_new = int((fold_df["source"] == "cmt_stage_d_new_fit").sum()) if "source" in fold_df.columns else 0
    gate = {
        "stage": "D",
        "status": status,
        "timestamp_utc": utc_now(),
        "analysis_id": cfg["analysis_id"],
        "config_sha256": sha256_file(cfg_path),
        "scope": "coarse_chemistry_family_host_chemical_system_working_ion_robustness",
        "historical_reuse": reuse_ok,
        "n_targets": len(all_targets),
        "n_stages": len(stages),
        "n_domain_splits": len(cfg["robustness_validations"]),
        "n_canonical_fold_rows": len(fold_df),
        "n_historical_fold_rows_reused": n_hist,
        "n_new_extratrees_fits": n_new,
        "primary_earliest_useful_stage_remains_stage_c": True,
        "domain_point_stage_results_are_descriptive_supporting_evidence": True,
    }
    write_json(gate, out_dir / "stage_d_gate.json")

    output_files = [p for p in sorted(out_dir.iterdir()) if p.is_file()]
    manifest = pd.DataFrame([
        {"relative_path": str(p.relative_to(repo)), "sha256": sha256_file(p), "size_bytes": p.stat().st_size}
        for p in output_files
    ])
    write_csv_atomic(manifest, out_dir / "stage_d_output_sha256.csv")

    if status != "PASS":
        raise RuntimeError("Stage D execution gate FAILED. See stage_d_execution_audit.csv.")
    print(f"Stage D: PASS | canonical fold rows={len(fold_df)} | historical reused={n_hist} | new ExtraTrees fits={n_new}", flush=True)
    print("Stage-C bootstrap classification remains the primary earliest-useful-stage result.", flush=True)


if __name__ == "__main__":
    main()
