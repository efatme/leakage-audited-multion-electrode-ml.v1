#!/usr/bin/env python3
"""CMT Stage A + Stage B: frozen manifest and primary stage benchmark.

One-command workflow:
    python papers/cmt/analysis/run_stage_ab.py

Stage A validates and freezes the 3201-record universe, nine target-specific
P1/P2/P3 feature sets, dependency/compiler provenance, and the exact five-fold
framework-formula test manifest. Stage B runs only the primary fixed-estimator
P1->P2->P3 benchmark and stores complete out-of-fold predictions.
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
from joblib import Parallel, delayed
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

def bool_series_all_true(s: pd.Series) -> bool:
    if s.dtype == bool:
        return bool(s.all())
    return bool(s.astype(str).str.lower().isin(["true", "1", "yes"]).all())


def add_check(rows, name, expected, observed, passed, detail=""):
    rows.append({
        "check": name,
        "expected": str(expected),
        "observed": str(observed),
        "pass": bool(passed),
        "detail": detail,
    })


def load_config(repo: Path, config_arg: str | None):
    config_path = Path(config_arg).resolve() if config_arg else repo / "papers/cmt/config/cmt_stage_ab_v1.yaml"
    with config_path.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return config_path, config


def protocol_file(repo: Path, target: str, stage: str) -> Path:
    return repo / "data/processed/notebook_02/protocol_feature_lists" / f"02_protocol_{stage}_features_for_target_{target}.csv"


def stage_a(repo: Path, config_path: Path, cfg: dict, out_dir: Path) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    checks = []
    input_manifest = []

    required_input_keys = [
        "master_feature_table", "metadata_targets", "valid_record_universe", "split_plan",
        "outer_test_manifest", "protocol_integrity_audit", "target_specific_matrix",
        "dependency_nodes", "dependency_edges", "compiler_agreement", "historical_benchmark",
        "historical_model_audit", "historical_config",
    ]
    paths = {}
    for key in required_input_keys:
        path = repo / cfg["inputs"][key]
        paths[key] = path
        exists = path.is_file()
        add_check(checks, f"input_exists::{key}", True, exists, exists, str(path.relative_to(repo)) if exists else str(path))
        if exists:
            input_manifest.append({"role": key, "relative_path": str(path.relative_to(repo)), "sha256": sha256_file(path), "size_bytes": path.stat().st_size})

    if not all(r["pass"] for r in checks):
        audit = pd.DataFrame(checks)
        audit.to_csv(out_dir / "stage_a_audit.csv", index=False)
        gate = {"stage": "A", "status": "FAIL", "timestamp_utc": utc_now(), "reason": "missing_required_inputs"}
        write_json(gate, out_dir / "stage_a_gate.json")
        raise RuntimeError("Stage A failed: required input files are missing.")

    master = pd.read_csv(paths["master_feature_table"])
    meta = pd.read_csv(paths["metadata_targets"])
    valid = pd.read_csv(paths["valid_record_universe"])
    split_plan = pd.read_csv(paths["split_plan"])
    outer = pd.read_csv(paths["outer_test_manifest"])
    integrity = pd.read_csv(paths["protocol_integrity_audit"])
    target_matrix = pd.read_csv(paths["target_specific_matrix"])
    nodes = pd.read_csv(paths["dependency_nodes"])
    edges = pd.read_csv(paths["dependency_edges"])
    agreement = pd.read_csv(paths["compiler_agreement"])

    targets = list(cfg["targets"].keys())
    stages = list(cfg["stages"])
    expected = cfg["locked_dataset"]

    # Dataset identity and completeness.
    add_check(checks, "valid_record_universe_unique", len(valid), valid["record_index"].nunique(), valid["record_index"].nunique() == len(valid))
    add_check(checks, "n_records", expected["n_records"], len(valid), len(valid) == expected["n_records"])
    master_ids = set(master["record_index"].astype(str))
    valid_ids = valid["record_index"].astype(str).tolist()
    add_check(checks, "valid_records_present_in_master", len(valid_ids), sum(v in master_ids for v in valid_ids), all(v in master_ids for v in valid_ids))

    meta_sub = meta.set_index("record_index").loc[valid_ids].reset_index()
    for ion, n_expected in expected["ion_counts"].items():
        n_observed = int((meta_sub["working_ion"] == ion).sum())
        add_check(checks, f"ion_count::{ion}", n_expected, n_observed, n_observed == n_expected)
    add_check(checks, "framework_formula_groups", expected["n_framework_formula_groups"], meta_sub["framework_uid"].nunique(), meta_sub["framework_uid"].nunique() == expected["n_framework_formula_groups"])
    add_check(checks, "host_chemical_systems", expected["n_host_chemical_systems"], meta_sub["host_chemsys_no_working_ion"].nunique(), meta_sub["host_chemsys_no_working_ion"].nunique() == expected["n_host_chemical_systems"])
    add_check(checks, "coarse_chemistry_families", expected["n_coarse_chemistry_families"], meta_sub["coarse_family"].nunique(), meta_sub["coarse_family"].nunique() == expected["n_coarse_chemistry_families"])
    for target in targets:
        n_missing = int(meta_sub[target].isna().sum())
        add_check(checks, f"target_complete::{target}", 0, n_missing, n_missing == 0)

    # Provenance graph/compiler checks.
    pg = cfg["provenance_graph"]
    add_check(checks, "target_feature_matrix_rows", pg["n_feature_target_pairs"], len(target_matrix), len(target_matrix) == pg["n_feature_target_pairs"])
    add_check(checks, "dependency_graph_nodes", pg["n_nodes"], len(nodes), len(nodes) == pg["n_nodes"])
    add_check(checks, "dependency_graph_edges", pg["n_directed_relations"], len(edges), len(edges) == pg["n_directed_relations"])
    add_check(checks, "compiler_agreement_rows", pg["n_feature_target_pairs"], len(agreement), len(agreement) == pg["n_feature_target_pairs"])
    add_check(checks, "compiler_class_agreement_all", True, bool_series_all_true(agreement["class_agreement"]), bool_series_all_true(agreement["class_agreement"]))
    add_check(checks, "compiler_protocol_agreement_all", True, bool_series_all_true(agreement["all_protocols_agree"]), bool_series_all_true(agreement["all_protocols_agree"]))
    false_safe = agreement["false_safe"].astype(str).str.lower().isin(["true", "1", "yes"]).sum()
    add_check(checks, "compiler_false_safe_count", 0, int(false_safe), int(false_safe) == 0)

    # Feature-set checks and hashes.
    feature_rows = []
    feature_sets = {}
    master_cols = set(master.columns)
    for target in targets:
        feature_sets[target] = {}
        for stage in stages:
            p = protocol_file(repo, target, stage)
            exists = p.is_file()
            add_check(checks, f"feature_file_exists::{target}::{stage}", True, exists, exists)
            if not exists:
                continue
            df = pd.read_csv(p)
            feats = df["feature"].astype(str).tolist()
            feature_sets[target][stage] = feats
            n_expected = int(cfg["targets"][target][stage])
            add_check(checks, f"feature_count::{target}::{stage}", n_expected, len(feats), len(feats) == n_expected)
            add_check(checks, f"feature_unique::{target}::{stage}", len(feats), len(set(feats)), len(set(feats)) == len(feats))
            missing = sorted(set(feats) - master_cols)
            add_check(checks, f"features_present::{target}::{stage}", 0, len(missing), len(missing) == 0, ";".join(missing[:10]))
            direct_target_present = target in feats
            add_check(checks, f"direct_target_absent::{target}::{stage}", False, direct_target_present, not direct_target_present)
            h = sha256_file(p)
            input_manifest.append({"role": f"feature_list::{target}::{stage}", "relative_path": str(p.relative_to(repo)), "sha256": h, "size_bytes": p.stat().st_size})
            feature_rows.append({"target": target, "stage": stage, "n_features": len(feats), "sha256": h, "relative_path": str(p.relative_to(repo))})
        if all(s in feature_sets[target] for s in stages):
            p1, p2, p3 = map(lambda s: set(feature_sets[target][s]), stages)
            add_check(checks, f"nested_P1_in_P2::{target}", True, p1.issubset(p2), p1.issubset(p2))
            add_check(checks, f"nested_P2_in_P3::{target}", True, p2.issubset(p3), p2.issubset(p3))

    integ_sub = integrity[integrity["protocol"].isin(stages) & integrity["target"].isin(targets)].copy()
    add_check(checks, "protocol_integrity_rows_P1_P3", len(targets) * len(stages), len(integ_sub), len(integ_sub) == len(targets) * len(stages))
    integ_pass = bool((integ_sub["status"] == "PASS").all()) if len(integ_sub) else False
    add_check(checks, "protocol_integrity_all_pass", True, integ_pass, integ_pass)

    # Exact primary split frozen from Notebook 05 test-record manifest.
    primary = cfg["primary_validation"]
    split_name = primary["split_name"]
    primary_outer = outer[(outer["split_name"] == split_name) & (outer["set_role"] == "test")].copy()
    folds = sorted(primary_outer["fold_id"].unique().tolist())
    add_check(checks, "primary_n_folds", primary["n_folds"], len(folds), len(folds) == primary["n_folds"])
    all_test = []
    split_rows = []
    meta_index = meta_sub.set_index("record_index")
    universe = set(valid_ids)
    plan_sub = split_plan[split_plan["split_name"] == split_name].set_index("fold_id")
    for fold in folds:
        test_ids = primary_outer.loc[primary_outer["fold_id"] == fold, "record_index"].astype(str).tolist()
        train_ids = [r for r in valid_ids if r not in set(test_ids)]
        all_test.extend(test_ids)
        add_check(checks, f"split_test_unique::fold{fold}", len(test_ids), len(set(test_ids)), len(test_ids) == len(set(test_ids)))
        add_check(checks, f"split_test_in_universe::fold{fold}", len(test_ids), sum(r in universe for r in test_ids), all(r in universe for r in test_ids))
        train_groups = set(meta_index.loc[train_ids, primary["group_field"]].astype(str))
        test_groups = set(meta_index.loc[test_ids, primary["group_field"]].astype(str))
        overlap = train_groups & test_groups
        add_check(checks, f"split_group_overlap::fold{fold}", 0, len(overlap), len(overlap) == 0)
        if fold in plan_sub.index:
            add_check(checks, f"split_n_test::fold{fold}", int(plan_sub.loc[fold, "n_test"]), len(test_ids), int(plan_sub.loc[fold, "n_test"]) == len(test_ids))
            add_check(checks, f"split_n_train::fold{fold}", int(plan_sub.loc[fold, "n_train"]), len(train_ids), int(plan_sub.loc[fold, "n_train"]) == len(train_ids))
        for rid in test_ids:
            split_rows.append({
                "split_name": split_name,
                "fold_id": int(fold),
                "record_index": rid,
                "framework_uid": str(meta_index.loc[rid, "framework_uid"]),
                "working_ion": str(meta_index.loc[rid, "working_ion"]),
                "coarse_family": str(meta_index.loc[rid, "coarse_family"]),
                "host_chemsys_no_working_ion": str(meta_index.loc[rid, "host_chemsys_no_working_ion"]),
            })
    add_check(checks, "primary_each_record_tested_once", len(valid_ids), len(all_test), len(all_test) == len(valid_ids) and len(set(all_test)) == len(valid_ids) and set(all_test) == universe)

    # Historical estimator compatibility check; used as an audit, not as a source of new results.
    try:
        old_model_audit = pd.read_csv(paths["historical_model_audit"])
        et_repr = old_model_audit.loc[old_model_audit["model_name"] == "ExtraTrees", "model_repr"].iloc[0]
        expected_tokens = ["n_estimators=300", "random_state=42"]
        hist_ok = all(tok in et_repr for tok in expected_tokens)
        add_check(checks, "historical_ET_core_config_compatible", True, hist_ok, hist_ok, et_repr)
    except Exception as exc:
        add_check(checks, "historical_ET_core_config_compatible", True, False, False, repr(exc))

    config_hash = sha256_file(config_path)
    input_manifest.append({"role": "stage_ab_config", "relative_path": str(config_path.relative_to(repo)), "sha256": config_hash, "size_bytes": config_path.stat().st_size})
    audit = pd.DataFrame(checks)
    audit.to_csv(out_dir / "stage_a_audit.csv", index=False)
    pd.DataFrame(feature_rows).to_csv(out_dir / "stage_a_feature_manifest.csv", index=False)
    pd.DataFrame(split_rows).sort_values(["fold_id", "record_index"]).to_csv(out_dir / "stage_a_primary_split_manifest.csv", index=False)
    pd.DataFrame(input_manifest).sort_values(["role", "relative_path"]).to_csv(out_dir / "stage_a_input_sha256.csv", index=False)

    status = "PASS" if bool(audit["pass"].all()) else "FAIL"
    gate = {
        "stage": "A",
        "status": status,
        "timestamp_utc": utc_now(),
        "analysis_id": cfg["analysis_id"],
        "config_sha256": config_hash,
        "n_checks": int(len(audit)),
        "n_failed_checks": int((~audit["pass"]).sum()),
        "failed_checks": audit.loc[~audit["pass"], "check"].tolist(),
        "primary_validation": primary["reporting_name"],
        "n_records": len(valid_ids),
        "n_targets": len(targets),
        "stages": stages,
    }
    write_json(gate, out_dir / "stage_a_gate.json")
    if status != "PASS":
        raise RuntimeError("Stage A gate FAILED. See stage_a_audit.csv before running Stage B.")
    return gate


def metrics(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred)) if len(y_true) > 1 else np.nan
    try:
        if np.nanstd(y_true) == 0 or np.nanstd(y_pred) == 0:
            sp = np.nan
        else:
            sp = float(spearmanr(y_true, y_pred).statistic)
    except Exception:
        sp = np.nan
    return {"mae": mae, "rmse": rmse, "r2": r2, "spearman": sp}


def checkpoint_signature(target: str, stage: str, feature_path: Path, split_manifest_path: Path, cfg: dict) -> str:
    payload = {
        "target": target,
        "stage": stage,
        "feature_sha256": sha256_file(feature_path),
        "split_manifest_sha256": sha256_file(split_manifest_path),
        "locked_n_records": int(cfg["locked_dataset"]["n_records"]),
        "primary_validation": cfg["primary_validation"],
        "model_scientific_definition": {
            "primary_estimator": cfg["model"]["primary_estimator"],
            "preprocessing": cfg["model"]["preprocessing"],
            "params": cfg["model"]["params"],
            "baseline": cfg["model"]["baseline"],
        },
    }
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()

def run_fold_arrays(X_all, y_all, train_pos, test_pos, target, stage, fold, n_features, model_cfg):
    X_train = X_all[train_pos]
    X_test = X_all[test_pos]
    y_train = y_all[train_pos]
    y_test = y_all[test_pos]
    if np.isnan(y_train).any() or np.isnan(y_test).any():
        raise ValueError(f"Target missingness encountered for {target}, fold {fold}.")

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
    pred_dummy = dummy.predict(np.zeros((len(y_test), 1)))
    m_et = metrics(y_test, pred)
    m_dm = metrics(y_test, pred_dummy)
    fold_row = {
        "target": target, "stage": stage, "fold_id": int(fold),
        "n_train": len(train_pos), "n_test": len(test_pos), "n_features": int(n_features),
        **{f"et_{k}": v for k, v in m_et.items()},
        **{f"dummy_{k}": v for k, v in m_dm.items()},
    }
    fold_row["mae_skill_vs_dummy"] = 1.0 - fold_row["et_mae"] / fold_row["dummy_mae"] if fold_row["dummy_mae"] else np.nan
    fold_row["rmse_skill_vs_dummy"] = 1.0 - fold_row["et_rmse"] / fold_row["dummy_rmse"] if fold_row["dummy_rmse"] else np.nan
    return fold_row, np.asarray(test_pos, dtype=int), np.asarray(pred, dtype=float), np.asarray(pred_dummy, dtype=float)

def stage_b(repo: Path, config_path: Path, cfg: dict, out_dir: Path, force: bool = False) -> dict:
    gate_path = out_dir / "stage_a_gate.json"
    if not gate_path.exists():
        raise RuntimeError("Stage A gate is missing. Run Stage A first.")
    gate_a = json.loads(gate_path.read_text(encoding="utf-8"))
    if gate_a.get("status") != "PASS":
        raise RuntimeError("Stage A did not pass; Stage B is blocked.")
    current_hash = sha256_file(config_path)
    if current_hash != gate_a.get("config_sha256"):
        raise RuntimeError("Config changed after Stage A. Re-run Stage A before Stage B.")

    master = pd.read_csv(repo / cfg["inputs"]["master_feature_table"])
    valid = pd.read_csv(repo / cfg["inputs"]["valid_record_universe"])
    split_df = pd.read_csv(out_dir / "stage_a_primary_split_manifest.csv")
    valid_ids = valid["record_index"].astype(str).tolist()
    master_idx = master.set_index("record_index", drop=False)
    master_valid = master_idx.loc[valid_ids].copy()
    valid_ids_array = np.asarray(valid_ids, dtype=object)
    id_to_pos = {rid: i for i, rid in enumerate(valid_ids)}
    fold_positions = {}
    for fold in sorted(split_df["fold_id"].unique().tolist()):
        test_ids_fold = split_df.loc[split_df["fold_id"] == fold, "record_index"].astype(str).tolist()
        test_pos = np.asarray([id_to_pos[r] for r in test_ids_fold], dtype=int)
        is_test = np.zeros(len(valid_ids), dtype=bool)
        is_test[test_pos] = True
        train_pos = np.where(~is_test)[0]
        fold_positions[int(fold)] = (train_pos, test_pos)
    targets = list(cfg["targets"].keys())
    stages = list(cfg["stages"])
    folds = sorted(split_df["fold_id"].unique().tolist())
    checkpoint_dir = out_dir / "_checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    outer_jobs_limit = min(int(cfg["model"]["execution"]["outer_jobs_max"]), max(1, os.cpu_count() or 1))
    model_cfg_runtime = json.loads(json.dumps(cfg["model"]))
    tree_jobs_setting = cfg["model"]["execution"].get("tree_n_jobs", "auto")
    if str(tree_jobs_setting).lower() == "auto":
        tree_jobs = min(int(cfg["model"]["execution"].get("tree_n_jobs_cap", 18)), max(1, joblib.cpu_count()))
    else:
        tree_jobs = int(tree_jobs_setting)
    model_cfg_runtime["execution"]["tree_n_jobs"] = tree_jobs
    print(f"execution: tree_n_jobs={tree_jobs} outer_jobs_limit={outer_jobs_limit}")

    all_fold = []
    all_pred = []
    pending_blocks = []

    # Reuse only structurally complete checkpoints; collect everything else for batched fitting.
    split_manifest_path = out_dir / "stage_a_primary_split_manifest.csv"
    for target in targets:
        for stage in stages:
            stem = f"{target}__{stage}"
            mpath = checkpoint_dir / f"{stem}__fold_metrics.csv"
            ppath = checkpoint_dir / f"{stem}__oof_predictions.csv"
            fpath = protocol_file(repo, target, stage)
            expected_sig = checkpoint_signature(target, stage, fpath, split_manifest_path, cfg)
            checkpoint_ok = False
            if (not force) and mpath.exists() and ppath.exists():
                try:
                    mdf = pd.read_csv(mpath)
                    pdf = pd.read_csv(ppath)
                    sig_ok = (
                        "scientific_signature" in mdf.columns
                        and "scientific_signature" in pdf.columns
                        and set(mdf["scientific_signature"].astype(str)) == {expected_sig}
                        and set(pdf["scientific_signature"].astype(str)) == {expected_sig}
                    )
                    checkpoint_ok = (
                        sig_ok
                        and len(mdf) == len(folds)
                        and "record_index" in pdf.columns
                        and pdf["record_index"].nunique() == len(valid_ids)
                        and len(pdf) == len(valid_ids)
                    )
                except Exception:
                    checkpoint_ok = False
                if checkpoint_ok:
                    all_fold.append(mdf)
                    all_pred.append(pdf)
                    print(f"reuse checkpoint: {target} {stage}")
            if not checkpoint_ok:
                if (not force) and (mpath.exists() or ppath.exists()):
                    print(f"discard stale/incomplete checkpoint: {target} {stage}")
                pending_blocks.append((target, stage, mpath, ppath, fpath, expected_sig))

    batch_size = int(cfg["model"]["execution"].get("target_stage_batch_size", 2))
    for batch_start in range(0, len(pending_blocks), batch_size):
        batch = pending_blocks[batch_start:batch_start + batch_size]
        prepared = {}
        tasks = []
        for target, stage, mpath, ppath, fpath, expected_sig in batch:
            features = pd.read_csv(fpath)["feature"].astype(str).tolist()
            print(f"fit batch: {target:22s} {stage} | {len(features):3d} features | {len(folds)} folds")
            X_all = master_valid[features].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
            y_all = pd.to_numeric(master_valid[target], errors="coerce").to_numpy(dtype=float)
            prepared[(target, stage)] = {
                "X": X_all, "y": y_all, "features": features, "mpath": mpath, "ppath": ppath, "signature": expected_sig
            }
            for fold in folds:
                tasks.append((target, stage, int(fold), X_all, y_all, len(features)))

        batch_jobs = min(outer_jobs_limit, max(1, len(tasks)))
        print(f"execute batch: blocks={len(batch)} fold_fits={len(tasks)} jobs={batch_jobs}")
        if batch_jobs == 1:
            outputs = [
                run_fold_arrays(
                    X_all, y_all, fold_positions[fold][0], fold_positions[fold][1],
                    target, stage, fold, n_features, model_cfg_runtime
                )
                for target, stage, fold, X_all, y_all, n_features in tasks
            ]
        else:
            outputs = Parallel(n_jobs=batch_jobs, backend="loky", max_nbytes="1M")(
                delayed(run_fold_arrays)(
                    X_all, y_all, fold_positions[fold][0], fold_positions[fold][1],
                    target, stage, fold, n_features, model_cfg_runtime
                )
                for target, stage, fold, X_all, y_all, n_features in tasks
            )

        grouped = {}
        for out in outputs:
            fold_row = out[0]
            key = (fold_row["target"], fold_row["stage"])
            grouped.setdefault(key, []).append(out)

        for target, stage, mpath, ppath, fpath, expected_sig in batch:
            block_outputs = grouped[(target, stage)]
            y_all = prepared[(target, stage)]["y"]
            mdf = pd.DataFrame([x[0] for x in block_outputs]).sort_values("fold_id")
            mdf["scientific_signature"] = expected_sig
            pred_parts = []
            for fold_row, test_pos, pred, pred_dummy in block_outputs:
                fold = int(fold_row["fold_id"])
                y_test = y_all[test_pos]
                info = master_valid.iloc[test_pos]
                pred_parts.append(pd.DataFrame({
                    "target": target,
                    "stage": stage,
                    "fold_id": fold,
                    "record_index": valid_ids_array[test_pos],
                    "framework_uid": info["framework_uid"].astype(str).values,
                    "working_ion": info["working_ion"].astype(str).values,
                    "coarse_family": info["coarse_family"].astype(str).values,
                    "host_chemsys_no_working_ion": info["host_chemsys_no_working_ion"].astype(str).values,
                    "y_true": y_test,
                    "y_pred": pred,
                    "y_pred_dummy": pred_dummy,
                    "abs_error": np.abs(y_test - pred),
                    "sq_error": (y_test - pred) ** 2,
                    "dummy_abs_error": np.abs(y_test - pred_dummy),
                    "dummy_sq_error": (y_test - pred_dummy) ** 2,
                }))
            pdf = pd.concat(pred_parts, ignore_index=True).sort_values(["fold_id", "record_index"])
            pdf["scientific_signature"] = expected_sig
            write_csv_atomic(mdf, mpath)
            write_csv_atomic(pdf, ppath)
            all_fold.append(mdf)
            all_pred.append(pdf)

    fold_df = pd.concat(all_fold, ignore_index=True).sort_values(["target", "stage", "fold_id"])
    pred_df = pd.concat(all_pred, ignore_index=True).sort_values(["target", "stage", "fold_id", "record_index"])
    fold_df.to_csv(out_dir / "stage_b_fold_metrics.csv", index=False)
    pred_df.to_csv(out_dir / "stage_b_oof_predictions.csv", index=False)

    # Pooled OOF and fold-average summaries.
    summary_rows = []
    for (target, stage), sub in pred_df.groupby(["target", "stage"], sort=False):
        m_et = metrics(sub["y_true"], sub["y_pred"])
        m_dm = metrics(sub["y_true"], sub["y_pred_dummy"])
        fsub = fold_df[(fold_df["target"] == target) & (fold_df["stage"] == stage)]
        row = {
            "target": target, "stage": stage,
            "n_oof": len(sub), "n_features": int(fsub["n_features"].iloc[0]),
            **{f"pooled_et_{k}": v for k, v in m_et.items()},
            **{f"pooled_dummy_{k}": v for k, v in m_dm.items()},
            "fold_mean_et_mae": float(fsub["et_mae"].mean()),
            "fold_mean_et_rmse": float(fsub["et_rmse"].mean()),
            "fold_mean_et_r2": float(fsub["et_r2"].mean()),
            "fold_mean_dummy_mae": float(fsub["dummy_mae"].mean()),
            "fold_mean_dummy_rmse": float(fsub["dummy_rmse"].mean()),
        }
        row["pooled_mae_skill_vs_dummy"] = 1.0 - row["pooled_et_mae"] / row["pooled_dummy_mae"]
        row["pooled_rmse_skill_vs_dummy"] = 1.0 - row["pooled_et_rmse"] / row["pooled_dummy_rmse"]
        threshold = float(cfg["useful_learnability_rule"]["minimum_mae_skill_vs_dummy"])
        threshold_rmse = float(cfg["useful_learnability_rule"]["minimum_rmse_skill_vs_dummy"])
        row["passes_10pct_point_estimate"] = bool(row["pooled_mae_skill_vs_dummy"] >= threshold and row["pooled_rmse_skill_vs_dummy"] >= threshold_rmse)
        row["final_useful_label_pending_bootstrap"] = True
        summary_rows.append(row)
    summary = pd.DataFrame(summary_rows).sort_values(["target", "stage"])
    summary.to_csv(out_dir / "stage_b_primary_summary.csv", index=False)

    transitions = [("P1", "P2"), ("P2", "P3"), ("P1", "P3")]
    delta_rows = []
    for target in targets:
        sm = summary[summary["target"] == target].set_index("stage")
        for a, b in transitions:
            mae_a, mae_b = float(sm.loc[a, "pooled_et_mae"]), float(sm.loc[b, "pooled_et_mae"])
            rmse_a, rmse_b = float(sm.loc[a, "pooled_et_rmse"]), float(sm.loc[b, "pooled_et_rmse"])
            delta_rows.append({
                "target": target, "from_stage": a, "to_stage": b,
                "delta_mae_positive_is_improvement": mae_a - mae_b,
                "relative_mae_improvement": (mae_a - mae_b) / mae_a if mae_a else np.nan,
                "delta_rmse_positive_is_improvement": rmse_a - rmse_b,
                "relative_rmse_improvement": (rmse_a - rmse_b) / rmse_a if rmse_a else np.nan,
            })
    delta_df = pd.DataFrame(delta_rows)
    delta_df.to_csv(out_dir / "stage_b_incremental_information_value.csv", index=False)

    earliest_rows = []
    order = {s: i for i, s in enumerate(stages)}
    for target in targets:
        sub = summary[summary["target"] == target].copy()
        passing = sub[sub["passes_10pct_point_estimate"]].copy()
        earliest = None if passing.empty else sorted(passing["stage"].tolist(), key=lambda s: order[s])[0]
        earliest_rows.append({
            "target": target,
            "earliest_stage_point_estimate_only": earliest if earliest else "none_through_P3",
            "final_status": "PENDING_GROUP_BOOTSTRAP_CONFIRMATION",
        })
    earliest_df = pd.DataFrame(earliest_rows)
    earliest_df.to_csv(out_dir / "stage_b_earliest_stage_point_estimate.csv", index=False)

    # Historical five-target comparison using the same fold-mean estimand.
    old_path = repo / cfg["inputs"]["historical_benchmark"]
    hist_rows = []
    if old_path.exists():
        old = pd.read_csv(old_path)
        old = old[(old["model_name"] == "ExtraTrees") & (old["split_name"] == "framework_groupkfold") & (old["protocol"].isin(stages))]
        for _, r in summary.iterrows():
            match = old[(old["target"] == r["target"]) & (old["protocol"] == r["stage"])]
            if len(match) == 1:
                h = match.iloc[0]
                hist_rows.append({
                    "target": r["target"], "stage": r["stage"],
                    "old_fold_mean_mae": float(h["mae_mean"]), "new_fold_mean_mae": float(r["fold_mean_et_mae"]),
                    "mae_difference_new_minus_old": float(r["fold_mean_et_mae"] - h["mae_mean"]),
                    "old_fold_mean_rmse": float(h["rmse_mean"]), "new_fold_mean_rmse": float(r["fold_mean_et_rmse"]),
                    "rmse_difference_new_minus_old": float(r["fold_mean_et_rmse"] - h["rmse_mean"]),
                })
    pd.DataFrame(hist_rows).to_csv(out_dir / "stage_b_historical_compatibility.csv", index=False)

    # Structural execution gate.
    expected_metric_rows = len(targets) * len(stages) * len(folds)
    expected_pred_rows = len(targets) * len(stages) * int(cfg["locked_dataset"]["n_records"])
    structural_checks = []
    add_check(structural_checks, "fold_metric_rows", expected_metric_rows, len(fold_df), len(fold_df) == expected_metric_rows)
    add_check(structural_checks, "oof_prediction_rows", expected_pred_rows, len(pred_df), len(pred_df) == expected_pred_rows)
    finite_preds = bool(np.isfinite(pred_df[["y_true", "y_pred", "y_pred_dummy"]].to_numpy(dtype=float)).all())
    add_check(structural_checks, "finite_predictions", True, finite_preds, finite_preds)
    coverage_ok = True
    dup_ok = True
    for (target, stage), sub in pred_df.groupby(["target", "stage"]):
        coverage_ok &= sub["record_index"].nunique() == int(cfg["locked_dataset"]["n_records"])
        dup_ok &= not sub["record_index"].duplicated().any()
    add_check(structural_checks, "complete_oof_coverage_each_target_stage", True, coverage_ok, coverage_ok)
    add_check(structural_checks, "no_duplicate_oof_record_each_target_stage", True, dup_ok, dup_ok)
    execution_audit = pd.DataFrame(structural_checks)
    execution_audit.to_csv(out_dir / "stage_b_execution_audit.csv", index=False)
    status = "PASS" if bool(execution_audit["pass"].all()) else "FAIL"

    env = {
        "timestamp_utc": utc_now(), "python": sys.version, "platform": platform.platform(),
        "numpy": np.__version__, "pandas": pd.__version__, "scipy": scipy.__version__,
        "scikit_learn": sklearn.__version__, "joblib": joblib.__version__, "pyyaml": yaml.__version__,
        "os_cpu_count": os.cpu_count(), "outer_jobs_limit": outer_jobs_limit,
        "tree_n_jobs": tree_jobs,
    }
    write_json(env, out_dir / "stage_b_environment.json")
    gate_b = {
        "stage": "B", "status": status, "timestamp_utc": utc_now(), "analysis_id": cfg["analysis_id"],
        "config_sha256": current_hash, "primary_validation": cfg["primary_validation"]["reporting_name"],
        "n_targets": len(targets), "n_stages": len(stages), "n_folds": len(folds),
        "n_extratrees_fits": expected_metric_rows, "n_oof_predictions": len(pred_df),
        "scope": "primary_framework_formula_grouped_only",
        "earliest_useful_stage_status": "point_estimate_only_pending_group_bootstrap",
    }
    write_json(gate_b, out_dir / "stage_b_gate.json")

    output_files = [p for p in sorted(out_dir.iterdir()) if p.is_file()]
    output_manifest = pd.DataFrame([
        {"relative_path": str(p.relative_to(repo)), "sha256": sha256_file(p), "size_bytes": p.stat().st_size}
        for p in output_files
    ])
    output_manifest.to_csv(out_dir / "stage_ab_output_sha256.csv", index=False)
    if status != "PASS":
        raise RuntimeError("Stage B execution gate FAILED. See stage_b_execution_audit.csv.")
    return gate_b


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None, help="Path to cmt_stage_ab_v1.yaml")
    parser.add_argument("--stage", choices=["A", "B", "all"], default="all")
    parser.add_argument("--force", action="store_true", help="Ignore Stage B checkpoints and refit all primary models.")
    args = parser.parse_args()

    repo = locate_repo()
    config_path, cfg = load_config(repo, args.config)
    out_dir = repo / "papers/cmt/results/stage_ab"
    print(f"Repository: {repo}")
    print(f"Config:     {config_path}")
    print(f"Output:     {out_dir}")

    if args.stage in {"A", "all"}:
        gate_a = stage_a(repo, config_path, cfg, out_dir)
        print(f"Stage A: {gate_a['status']} | checks={gate_a['n_checks']} | failed={gate_a['n_failed_checks']}")
    if args.stage in {"B", "all"}:
        gate_b = stage_b(repo, config_path, cfg, out_dir, force=args.force)
        print(f"Stage B: {gate_b['status']} | ExtraTrees fits={gate_b['n_extratrees_fits']} | OOF predictions={gate_b['n_oof_predictions']}")
        print("Stage B intentionally stops before bootstrap-based final learnability classification.")


if __name__ == "__main__":
    main()
