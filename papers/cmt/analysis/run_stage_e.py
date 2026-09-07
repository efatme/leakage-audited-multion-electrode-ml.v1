#!/usr/bin/env python3
"""CMT Stage E: final Ridge estimator sensitivity before analysis freeze.

This is a deliberately small sensitivity analysis. It uses the same frozen
framework-formula-grouped folds and the same target-specific P1/P2/P3 feature
sets as the primary ExtraTrees analysis, but replaces ExtraTrees with the
pre-specified Ridge model (median imputation, standardization, alpha=1.0).

Stage E does NOT change the primary earliest-useful-stage result from Stage C.
It tests whether the qualitative information-stage conclusions are strongly
estimator-dependent. No random validation, no domain reruns, and no tuning.
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
import sklearn
import yaml
from scipy.stats import spearmanr
from sklearn.dummy import DummyRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


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
    rows.append({"check": name, "expected": str(expected), "observed": str(observed), "pass": bool(passed), "detail": detail})


def protocol_file(repo: Path, target: str, stage: str) -> Path:
    return repo / "data/processed/notebook_02/protocol_feature_lists" / f"02_protocol_{stage}_features_for_target_{target}.csv"


def load_feature_list(path: Path) -> list[str]:
    df = pd.read_csv(path)
    if "feature" in df.columns:
        vals = df["feature"]
    elif "feature_name" in df.columns:
        vals = df["feature_name"]
    else:
        vals = df.iloc[:, 0]
    return vals.dropna().astype(str).tolist()


def metric_dict(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mae = float(mean_absolute_error(y_true, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    r2 = float(r2_score(y_true, y_pred)) if len(y_true) > 1 else np.nan
    try:
        sp = np.nan if (np.nanstd(y_true) < 1e-14 or np.nanstd(y_pred) < 1e-14) else float(spearmanr(y_true, y_pred).statistic)
    except Exception:
        sp = np.nan
    return {"mae": mae, "rmse": rmse, "r2": r2, "spearman": sp}


def percentile_ci(values: np.ndarray, confidence_level: float) -> tuple[float, float]:
    alpha = 1.0 - float(confidence_level)
    lo, hi = np.percentile(np.asarray(values, dtype=float), [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(lo), float(hi)


def direction_from_ci(lo: float, hi: float) -> str:
    if lo > 0:
        return "improvement_supported"
    if hi < 0:
        return "degradation_supported"
    return "interval_spans_zero"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None)
    args = parser.parse_args()

    repo = locate_repo()
    config_path = Path(args.config).resolve() if args.config else repo / "papers/cmt/config/cmt_stage_e_v1.yaml"
    with config_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    out_dir = repo / cfg["output_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)
    checks = []

    paths = {k: repo / v for k, v in cfg["inputs"].items()}
    for role, path in paths.items():
        ok = path.is_file() or (role == "protocol_feature_dir" and path.is_dir())
        add_check(checks, f"input_exists::{role}", True, ok, ok, str(path))

    # Upstream gates must already be complete.
    for role in ["stage_b_gate", "stage_c_gate", "stage_d_gate"]:
        if paths[role].is_file():
            gate = json.loads(paths[role].read_text(encoding="utf-8"))
            add_check(checks, f"{role}_status", "PASS", gate.get("status"), gate.get("status") == "PASS")

    targets = list(cfg["targets"])
    stages = list(cfg["stages"])
    master = pd.read_csv(paths["master_feature_table"])
    valid = pd.read_csv(paths["valid_record_universe"])
    split = pd.read_csv(paths["stage_a_split_manifest"])
    extra_earliest = pd.read_csv(paths["stage_c_earliest_useful_stage"])

    add_check(checks, "valid_records", cfg["primary_validation"]["n_records"], len(valid), len(valid) == int(cfg["primary_validation"]["n_records"]))
    add_check(checks, "split_rows", cfg["primary_validation"]["n_records"], len(split), len(split) == int(cfg["primary_validation"]["n_records"]))
    add_check(checks, "split_folds", cfg["primary_validation"]["n_folds"], split["fold_id"].nunique(), split["fold_id"].nunique() == int(cfg["primary_validation"]["n_folds"]))
    add_check(checks, "stage_c_n_targets", len(targets), len(extra_earliest), len(extra_earliest) == len(targets))

    # Feature-list checks.
    feature_lists = {}
    for target in targets:
        for stage in stages:
            p = protocol_file(repo, target, stage)
            ok = p.is_file()
            add_check(checks, f"feature_file::{target}::{stage}", True, ok, ok, str(p))
            if ok:
                feats = load_feature_list(p)
                feature_lists[(target, stage)] = feats
                add_check(checks, f"feature_unique::{target}::{stage}", len(feats), len(set(feats)), len(feats) == len(set(feats)))
                missing = sorted(set(feats) - set(master.columns))
                add_check(checks, f"features_present::{target}::{stage}", 0, len(missing), len(missing) == 0, ";".join(missing[:10]))
        if all((target, s) in feature_lists for s in stages):
            p1, p2, p3 = [set(feature_lists[(target, s)]) for s in stages]
            add_check(checks, f"nested_P1_P2::{target}", True, p1.issubset(p2), p1.issubset(p2))
            add_check(checks, f"nested_P2_P3::{target}", True, p2.issubset(p3), p2.issubset(p3))

    audit = pd.DataFrame(checks)
    write_csv_atomic(audit, out_dir / "stage_e_preflight_audit.csv")
    if not bool(audit["pass"].all()):
        write_json({"stage":"E","status":"FAIL","timestamp_utc":utc_now(),"failed_checks":audit.loc[~audit["pass"],"check"].tolist()}, out_dir / "stage_e_gate.json")
        raise RuntimeError("Stage E preflight failed. See stage_e_preflight_audit.csv")

    print("Preflight: PASS")
    print("Stage E: 135 small Ridge fits only; no ExtraTrees/domain/random reruns")

    # Align master to frozen record order.
    ids = valid["record_index"].astype(str).tolist()
    master = master.copy()
    master["record_index"] = master["record_index"].astype(str)
    master_idx = master.set_index("record_index")
    missing_ids = [x for x in ids if x not in master_idx.index]
    if missing_ids:
        raise RuntimeError(f"Missing frozen records in master table: {missing_ids[:5]}")
    data = master_idx.loc[ids].copy()
    id_to_pos = {rid: i for i, rid in enumerate(ids)}
    split = split.copy()
    split["record_index"] = split["record_index"].astype(str)
    fold_by_id = dict(zip(split["record_index"], split["fold_id"].astype(int)))
    framework_by_id = dict(zip(split["record_index"], split["framework_uid"].astype(str)))
    folds = sorted(split["fold_id"].unique().tolist())

    fold_rows = []
    oof_rows = []
    alpha = float(cfg["secondary_estimator"]["params"]["alpha"])

    for target in targets:
        y_all = pd.to_numeric(data[target], errors="coerce").to_numpy(dtype=float)
        if np.isnan(y_all).any():
            raise RuntimeError(f"Target {target} contains missing values in frozen 3201-record universe.")
        for stage in stages:
            feats = feature_lists[(target, stage)]
            X_all = data[feats].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
            print(f"fit block: {target:22s} {stage} | features={len(feats)}")
            for fold in folds:
                test_ids = split.loc[split["fold_id"] == fold, "record_index"].astype(str).tolist()
                test_pos = np.array([id_to_pos[x] for x in test_ids], dtype=int)
                test_set = set(test_pos.tolist())
                train_pos = np.array([i for i in range(len(ids)) if i not in test_set], dtype=int)

                model = Pipeline([
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scale", StandardScaler()),
                    ("model", Ridge(alpha=alpha)),
                ])
                dummy = DummyRegressor(strategy="mean")
                model.fit(X_all[train_pos], y_all[train_pos])
                dummy.fit(np.zeros((len(train_pos),1)), y_all[train_pos])
                pred = model.predict(X_all[test_pos])
                pred_dm = dummy.predict(np.zeros((len(test_pos),1)))
                yt = y_all[test_pos]
                m = metric_dict(yt, pred)
                md = metric_dict(yt, pred_dm)
                row = {
                    "target": target, "stage": stage, "fold_id": int(fold),
                    "n_train": len(train_pos), "n_test": len(test_pos), "n_features": len(feats),
                    **{f"ridge_{k}":v for k,v in m.items()},
                    **{f"dummy_{k}":v for k,v in md.items()},
                }
                row["mae_skill_vs_dummy"] = 1.0 - row["ridge_mae"] / row["dummy_mae"]
                row["rmse_skill_vs_dummy"] = 1.0 - row["ridge_rmse"] / row["dummy_rmse"]
                fold_rows.append(row)

                for local_j, pos in enumerate(test_pos):
                    rid = ids[pos]
                    err = float(yt[local_j] - pred[local_j])
                    errd = float(yt[local_j] - pred_dm[local_j])
                    oof_rows.append({
                        "target": target, "stage": stage, "fold_id": int(fold),
                        "record_index": rid, "framework_uid": framework_by_id[rid],
                        "y_true": float(yt[local_j]), "y_pred": float(pred[local_j]), "y_pred_dummy": float(pred_dm[local_j]),
                        "abs_error": abs(err), "sq_error": err*err,
                        "dummy_abs_error": abs(errd), "dummy_sq_error": errd*errd,
                    })

    fold_df = pd.DataFrame(fold_rows)
    oof = pd.DataFrame(oof_rows)
    write_csv_atomic(fold_df, out_dir / "stage_e_ridge_fold_metrics.csv")
    write_csv_atomic(oof, out_dir / "stage_e_ridge_oof_predictions.csv")

    expected_oof = int(cfg["primary_validation"]["n_records"]) * len(targets) * len(stages)
    exec_checks=[]
    add_check(exec_checks,"ridge_fits",len(targets)*len(stages)*len(folds),len(fold_df),len(fold_df)==len(targets)*len(stages)*len(folds))
    add_check(exec_checks,"ridge_oof_rows",expected_oof,len(oof),len(oof)==expected_oof)
    add_check(exec_checks,"finite_predictions",True,np.isfinite(oof[["y_true","y_pred","y_pred_dummy"]].to_numpy(float)).all(),np.isfinite(oof[["y_true","y_pred","y_pred_dummy"]].to_numpy(float)).all())
    add_check(exec_checks,"no_duplicate_target_stage_record",False,oof.duplicated(["target","stage","record_index"]).any(),not oof.duplicated(["target","stage","record_index"]).any())
    exec_audit=pd.DataFrame(exec_checks)
    write_csv_atomic(exec_audit,out_dir/"stage_e_execution_audit.csv")
    if not bool(exec_audit["pass"].all()):
        raise RuntimeError("Stage E execution audit failed.")

    # Point summaries.
    point_rows=[]
    for target in targets:
        for stage in stages:
            sub=oof[(oof.target==target)&(oof.stage==stage)]
            n=len(sub)
            rmae=float(sub.abs_error.mean())
            rrmse=float(np.sqrt(sub.sq_error.mean()))
            dmae=float(sub.dummy_abs_error.mean())
            drmse=float(np.sqrt(sub.dummy_sq_error.mean()))
            point_rows.append({
                "target":target,"stage":stage,"n_oof":n,"ridge_mae":rmae,"ridge_rmse":rrmse,
                "dummy_mae":dmae,"dummy_rmse":drmse,
                "mae_skill_vs_dummy":1-rmae/dmae,"rmse_skill_vs_dummy":1-rrmse/drmse,
            })
    point=pd.DataFrame(point_rows)
    write_csv_atomic(point,out_dir/"stage_e_ridge_primary_summary.csv")

    # Same paired framework-formula bootstrap design as Stage C.
    combo_order=[(t,s) for t in targets for s in stages]
    combo_index={c:i for i,c in enumerate(combo_order)}
    agg=(oof.groupby(["fold_id","framework_uid","target","stage"],sort=True)
         .agg(n_records=("record_index","size"),ridge_abs_sum=("abs_error","sum"),ridge_sq_sum=("sq_error","sum"),dm_abs_sum=("dummy_abs_error","sum"),dm_sq_sum=("dummy_sq_error","sum"))
         .reset_index())
    B=int(cfg["bootstrap"]["n_replicates"]); conf=float(cfg["bootstrap"]["confidence_level"]); seed=int(cfg["bootstrap"]["random_seed"])
    rng=np.random.default_rng(seed)
    nc=len(combo_order)
    total_n=np.zeros(B); ra=np.zeros((B,nc)); rs=np.zeros((B,nc)); da=np.zeros((B,nc)); dsq=np.zeros((B,nc))
    base_meta=oof[(oof.target==targets[0])&(oof.stage==stages[0])][["record_index","fold_id","framework_uid"]]
    for fold in folds:
        groups=sorted(base_meta.loc[base_meta.fold_id==fold,"framework_uid"].unique().tolist())
        ng=len(groups)
        first=agg[(agg.fold_id==fold)&(agg.target==targets[0])&(agg.stage==stages[0])].set_index("framework_uid").reindex(groups)
        gn=first.n_records.to_numpy(float)
        mats={}
        for metric in ["ridge_abs_sum","ridge_sq_sum","dm_abs_sum","dm_sq_sum"]:
            mat=np.empty((ng,nc),float)
            for j,(t,s) in enumerate(combo_order):
                x=agg[(agg.fold_id==fold)&(agg.target==t)&(agg.stage==s)].set_index("framework_uid").reindex(groups)
                mat[:,j]=x[metric].to_numpy(float)
            mats[metric]=mat
        counts=rng.multinomial(ng,np.full(ng,1/ng),size=B)
        total_n += counts @ gn
        ra += counts @ mats["ridge_abs_sum"]
        rs += counts @ mats["ridge_sq_sum"]
        da += counts @ mats["dm_abs_sum"]
        dsq += counts @ mats["dm_sq_sum"]
    rmae=ra/total_n[:,None]; rrmse=np.sqrt(rs/total_n[:,None]); dmae=da/total_n[:,None]; drmse=np.sqrt(dsq/total_n[:,None])
    skill_mae=1-rmae/dmae; skill_rmse=1-rrmse/drmse

    threshold=float(cfg["useful_learnability_rule"]["primary_threshold"])
    skill_rows=[]
    for t,s in combo_order:
        j=combo_index[(t,s)]; p=point[(point.target==t)&(point.stage==s)].iloc[0]
        ml,mh=percentile_ci(skill_mae[:,j],conf); rl,rh=percentile_ci(skill_rmse[:,j],conf)
        pass_point=bool(p.mae_skill_vs_dummy>=threshold and p.rmse_skill_vs_dummy>=threshold)
        useful=bool(pass_point and ml>0 and rl>0)
        skill_rows.append({
            "target":t,"stage":s,
            "mae_skill_vs_dummy":float(p.mae_skill_vs_dummy),"mae_skill_ci_low":ml,"mae_skill_ci_high":mh,
            "rmse_skill_vs_dummy":float(p.rmse_skill_vs_dummy),"rmse_skill_ci_low":rl,"rmse_skill_ci_high":rh,
            "passes_10pct_point_both":pass_point,
            "mae_skill_interval_entirely_above_zero":bool(ml>0),
            "rmse_skill_interval_entirely_above_zero":bool(rl>0),
            "useful_learnability_same_rule":useful,
        })
    skill_df=pd.DataFrame(skill_rows)
    write_csv_atomic(skill_df,out_dir/"stage_e_ridge_skill_bootstrap_summary.csv")

    rank={s:i for i,s in enumerate(stages)}
    earliest=[]
    extra_map=dict(zip(extra_earliest.target,extra_earliest.earliest_useful_stage))
    for t in targets:
        sub=skill_df[(skill_df.target==t)&(skill_df.useful_learnability_same_rule)]
        ridge_stage="none_through_P3" if sub.empty else sorted(sub.stage.tolist(),key=lambda x:rank[x])[0]
        et_stage=str(extra_map[t])
        earliest.append({
            "target":t,"primary_ExtraTrees_earliest_useful_stage":et_stage,"secondary_Ridge_earliest_useful_stage":ridge_stage,
            "same_earliest_stage":ridge_stage==et_stage,
            "interpretation":"secondary estimator sensitivity only; Stage C ExtraTrees remains primary",
        })
    earliest_df=pd.DataFrame(earliest)
    write_csv_atomic(earliest_df,out_dir/"stage_e_estimator_earliest_stage_comparison.csv")

    # Paired information-stage transitions for Ridge.
    transitions=[tuple(x) for x in cfg["incremental_information_value"]["transitions"]]
    tr=[]
    for t in targets:
        for a,b in transitions:
            ia,ib=combo_index[(t,a)],combo_index[(t,b)]
            pa=point[(point.target==t)&(point.stage==a)].iloc[0]; pb=point[(point.target==t)&(point.stage==b)].iloc[0]
            dma=rmae[:,ia]-rmae[:,ib]; drm=rrmse[:,ia]-rrmse[:,ib]
            mal,mah=percentile_ci(dma,conf); rml,rmh=percentile_ci(drm,conf)
            tr.append({
                "target":t,"from_stage":a,"to_stage":b,
                "delta_mae_positive_is_improvement":float(pa.ridge_mae-pb.ridge_mae),"delta_mae_ci_low":mal,"delta_mae_ci_high":mah,"delta_mae_interval_direction":direction_from_ci(mal,mah),
                "delta_rmse_positive_is_improvement":float(pa.ridge_rmse-pb.ridge_rmse),"delta_rmse_ci_low":rml,"delta_rmse_ci_high":rmh,"delta_rmse_interval_direction":direction_from_ci(rml,rmh),
            })
    trdf=pd.DataFrame(tr)
    write_csv_atomic(trdf,out_dir/"stage_e_ridge_incremental_information_value.csv")

    summary={
        "stage":"E","status":"PASS","timestamp_utc":utc_now(),"analysis_id":cfg["analysis_id"],
        "config_sha256":sha256_file(config_path),"n_ridge_fits":int(len(fold_df)),"n_oof_predictions":int(len(oof)),
        "n_bootstrap_replicates":B,"primary_estimator_unchanged":"fixed ExtraTrees from Stage C",
        "secondary_estimator":"Ridge(alpha=1.0) with median imputation and standardization",
        "same_earliest_stage_count":int(earliest_df.same_earliest_stage.sum()),"n_targets":len(targets),
        "analysis_freeze_recommendation":"inspect Stage E results; no further model fitting is planned unless a serious inconsistency is found",
    }
    write_json(summary,out_dir/"stage_e_gate.json")
    env={"timestamp_utc":utc_now(),"python":sys.version,"platform":platform.platform(),"numpy":np.__version__,"pandas":pd.__version__,"sklearn":sklearn.__version__}
    write_json(env,out_dir/"stage_e_environment.json")
    outputs=[]
    for p in sorted(out_dir.iterdir()):
        if p.is_file() and p.name!="stage_e_output_sha256.csv":
            outputs.append({"relative_path":str(p.relative_to(repo)),"sha256":sha256_file(p),"size_bytes":int(p.stat().st_size)})
    write_csv_atomic(pd.DataFrame(outputs),out_dir/"stage_e_output_sha256.csv")

    print(f"Stage E: PASS | Ridge fits={len(fold_df)} | OOF predictions={len(oof)} | bootstrap={B}")
    print(f"Earliest-stage agreement with primary ExtraTrees: {int(earliest_df.same_earliest_stage.sum())}/{len(targets)} targets")
    print("Stage C ExtraTrees classification remains primary. Stage E is sensitivity only.")


if __name__ == "__main__":
    main()
