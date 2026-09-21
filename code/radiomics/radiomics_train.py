#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""影像组学建模：固定5折CV + 网格搜索(降维×模型×参数) + 全量训练后测试集评估

特征处理候选（全部在 Pipeline 内，避免泄漏）:
  - none   : 不降维（1316 特征全量）
  - lasso  : SelectFromModel(L1-LogisticRegression, C∈{0.01,0.1,1})
  - pca    : PCA(n_components∈{50,100,200})
  - kbest  : SelectKBest(ANOVA F, k∈{50,100,200})
模型: KNN / LogisticRegression / SVM(RBF,Linear) / RandomForest / LightGBM

流程:
  1. 固定5折CV(splits.json, RC_A 968) 网格搜索 → 按 mean CV AUC 选最优配置
  2. 最优配置在 index train(774) 全量训练 → 测试集 index test(194) 评估
  3. 附加: 外部 RC_B(82) 报告
输出: $RADIO_OUT/{grid_results.csv, best_config.json, metrics.json}

加固（2026-09-02 崩溃修复）:
  - BLAS/OMP 单线程: 防止 ProcessPoolExecutor×OpenBLAS 线程爆炸卡死整机
  - RADIO_NJOBS 默认 4（12核一半进程，每 worker 单线程）
  - 断点续传: 每完成一个 config 立即写 grid_results_partial.csv, 重启后自动跳过
  - flock 单实例锁: 禁止 train+grid 双开
"""
import json, os, sys
# ---- 线程限制：必须在 numpy/sklearn 导入之前设置 ----
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
import time
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif, SelectFromModel
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (roc_auc_score, accuracy_score, recall_score,
                             precision_score, f1_score, confusion_matrix, roc_curve)

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, "dltrain"))  # path adjusted for this repository's layout
from dltrain.common import set_seed, load_index, get_splits, external_test_ids

SEED = 42
set_seed(SEED)
OUT = os.environ.get("RADIO_OUT", "experiments/outputs/radiomics_grid")
os.makedirs(OUT, exist_ok=True)

# ---------------- 数据 ----------------
_feat_path = os.environ.get("RADIO_FEAT_CSV", "data_processed/radiomics_features.csv")
feat = pd.read_csv(_feat_path)
feat["id"] = feat["id"].astype(str)
df = feat.copy()  # 已含 id/label/split/dataset/pT + 1316 特征

META = ["id", "label", "split", "dataset", "pT"]
FEAT_COLS = [c for c in feat.columns if c not in META]
X_all = df[FEAT_COLS].apply(pd.to_numeric, errors="coerce").values
y_all = df["label"].values.astype(int)
X_all = np.nan_to_num(X_all, nan=0.0)
print(f"data: {X_all.shape} feat | label {np.bincount(y_all)} | cols {len(FEAT_COLS)}")

# ---------------- 候选定义 ----------------
def make_selector(name, param):
    if name == "none":
        return "passthrough", {}
    if name == "lasso":
        sel = SelectFromModel(
            LogisticRegression(penalty="l1", solver="liblinear", C=param["C"],
                               random_state=SEED, max_iter=5000))
        return sel, {}
    if name == "pca":
        return PCA(n_components=param["n"], random_state=SEED), {}
    if name == "kbest":
        return SelectKBest(f_classif, k=param["k"]), {}

MODELS = {
    "knn": lambda p: KNeighborsClassifier(n_neighbors=p["n"], weights=p["w"]),
    "lr":  lambda p: LogisticRegression(C=p["C"], max_iter=5000, random_state=SEED),
    "svm": lambda p: SVC(C=p["C"], kernel=p["kernel"], probability=True,
                         random_state=SEED, max_iter=5000),
    "rf":  lambda p: RandomForestClassifier(n_estimators=p["n"], max_depth=p["depth"],
                                            min_samples_leaf=p["leaf"],
                                            random_state=SEED, n_jobs=1),
    "lgb": lambda p: __import__("lightgbm").LGBMClassifier(
        n_estimators=p["n"], num_leaves=p["nl"], learning_rate=p["lr"],
        random_state=SEED, verbosity=-1, n_jobs=1),
}
MODEL_PARAMS = {
    "knn": [{"n": n, "w": w} for n in (3, 5, 7, 9) for w in ("uniform", "distance")],
    "lr":  [{"C": c} for c in (0.01, 0.1, 1, 10)],
    "svm": [{"C": c, "kernel": k} for c in (0.1, 1, 10) for k in ("linear", "rbf")],
    "rf":  [{"n": 200, "depth": d, "leaf": l} for d in (None, 10) for l in (1, 4)],
    "lgb": [{"n": n, "nl": nl, "lr": lr}
            for n in (100, 300) for nl in (16, 31) for lr in (0.05, 0.1)],
}
SELECTORS = {
    "none": [{}],
    "lasso": [{"C": c} for c in (0.01, 0.1, 1)],
    "pca": [{"n": n} for n in (50, 100, 200)],
    "kbest": [{"k": k} for k in (50, 100, 200)],
}

# ---------------- 固定5折 ----------------
splits = get_splits()  # [{train: [...774], val: [...194]}] x5, RC_A only
fold_ids = [[splits[f]["train"], splits[f]["val"]] for f in range(5)]
id2row = {pid: i for i, pid in enumerate(df["id"].values)}

def cv_eval(sel_name, sel_param, model_name, m_param):
    aucs, accs = [], []
    for tr_ids, va_ids in fold_ids:
        tr_i = [id2row[p] for p in tr_ids]
        va_i = [id2row[p] for p in va_ids]
        sel, _ = make_selector(sel_name, sel_param)
        clf = MODELS[model_name](m_param)
        pipe = Pipeline([("scale", StandardScaler()), ("sel", sel), ("clf", clf)])
        pipe.fit(X_all[tr_i], y_all[tr_i])
        p = pipe.predict_proba(X_all[va_i])[:, 1]
        aucs.append(roc_auc_score(y_all[va_i], p))
        accs.append(accuracy_score(y_all[va_i], (p >= 0.5).astype(int)))
    return np.mean(aucs), np.std(aucs), np.mean(accs)

# ---------------- 网格搜索（并行） ----------------
def job(args):
    sel_name, sel_param, model_name, m_param = args
    t0 = time.time()
    mean_auc, std_auc, mean_acc = cv_eval(sel_name, sel_param, model_name, m_param)
    return {"selector": sel_name, "sel_param": json.dumps(sel_param),
            "model": model_name, "model_param": json.dumps(m_param),
            "cv_auc_mean": round(mean_auc, 4), "cv_auc_std": round(std_auc, 4),
            "cv_acc_mean": round(mean_acc, 4), "time_s": round(time.time() - t0, 1)}

def main():
    # ---- 单实例锁：防止多个实例同时跑（train+grid 双开曾导致资源翻倍）----
    import fcntl
    lock_path = f"{OUT}/.lock"
    lock_fd = open(lock_path, "w")
    try:
        fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        print("ANOTHER INSTANCE RUNNING — abort", file=sys.stderr)
        sys.exit(1)

    tasks = []
    for sel_name, sel_ps in SELECTORS.items():
        for sp in sel_ps:
            for m_name, m_ps in MODEL_PARAMS.items():
                for mp in m_ps:
                    tasks.append((sel_name, sp, m_name, mp))
    print(f"total configs: {len(tasks)}  (x5 folds = {len(tasks)*5} fits)", flush=True)

    # ---- 断点续传：读取已有 partial，跳过已完成 config ----
    PARTIAL = f"{OUT}/grid_results_partial.csv"
    done = {}
    if os.path.exists(PARTIAL):
        prev = pd.read_csv(PARTIAL)
        for _, r in prev.iterrows():
            key = (r["selector"], r["sel_param"], r["model"], r["model_param"])
            done[key] = r
        print(f"resume: {len(done)}/{len(tasks)} already done", flush=True)

    results = []
    N_WORKERS = int(os.environ.get("RADIO_NJOBS", "4"))
    # 过滤已完成任务（断点续传核心：已完成的 config 不重跑）
    pending = []
    for t in tasks:
        sel_name, sel_param, model_name, m_param = t
        key = (sel_name, json.dumps(sel_param), model_name, json.dumps(m_param))
        if key in done:
            results.append(done[key])
        else:
            pending.append(t)
    print(f"pending: {len(pending)}", flush=True)
    if N_WORKERS > 1:
        from concurrent.futures import ProcessPoolExecutor
        with ProcessPoolExecutor(max_workers=N_WORKERS) as ex:
            for i, r in enumerate(ex.map(job, pending, chunksize=4)):
                results.append(r)
                pd.DataFrame([r]).to_csv(PARTIAL, mode="a", header=not os.path.exists(PARTIAL), index=False)
                if len(results) % 30 == 0:
                    print(f"  {len(results)}/{len(tasks)} done", flush=True)
    else:
        for i, t in enumerate(pending):
            r = job(t)
            results.append(r)
            pd.DataFrame([r]).to_csv(PARTIAL, mode="a", header=not os.path.exists(PARTIAL), index=False)
            if len(results) % 30 == 0:
                print(f"  {len(results)}/{len(tasks)} done", flush=True)

    # 合并历史 partial + 本次新增，去重后排序
    grid = pd.concat([pd.DataFrame(done.values()), pd.DataFrame(results)], ignore_index=True)
    grid = grid.drop_duplicates(
        subset=["selector", "sel_param", "model", "model_param"], keep="last")
    grid = grid.sort_values("cv_auc_mean", ascending=False).reset_index(drop=True)
    grid.to_csv(f"{OUT}/grid_results.csv", index=False)
    best = grid.iloc[0]
    print("\n=== TOP 10 (by CV AUC) ===")
    print(grid.head(10).to_string(index=False))

    # ---------------- 最优配置：全量训练 + 测试 ----------------
    sel_name, sel_param = best["selector"], json.loads(best["sel_param"])
    m_name, m_param = best["model"], json.loads(best["model_param"])
    sel, _ = make_selector(sel_name, sel_param)
    clf = MODELS[m_name](m_param)
    pipe = Pipeline([("scale", StandardScaler()), ("sel", sel), ("clf", clf)])

    tr_mask = df["split"].values == "train"
    te_mask = df["split"].values == "test"
    ts_mask = df["split"].values == "ts"
    pipe.fit(X_all[tr_mask], y_all[tr_mask])
    n_sel = len(sel.get_support(indices=True)) if hasattr(sel, "get_support") and sel_name != "none" else X_all.shape[1]
    if sel_name == "pca":
        n_sel = pipe.named_steps["sel"].n_components_

    def report(mask, name):
        p = pipe.predict_proba(X_all[mask])[:, 1]
        y = y_all[mask]
        yp = (p >= 0.5).astype(int)
        auc = roc_auc_score(y, p)
        tn, fp, fn, tp = confusion_matrix(y, yp).ravel()
        return {
            "set": name, "n": int(mask.sum()), "label1": int(y.sum()),
            "auc": round(auc, 4), "acc": round(accuracy_score(y, yp), 4),
            "sen": round(recall_score(y, yp), 4), "spe": round(tn / (tn + fp), 4),
            "ppv": round(precision_score(y, yp), 4),
            "f1": round(f1_score(y, yp), 4),
            "cm": f"TN{tn} FP{fp} FN{fn} TP{tp}",
        }

    metrics = {
        "best_config": {
            "selector": sel_name, "sel_param": sel_param,
            "model": m_name, "model_param": m_param,
            "n_selected_features": int(n_sel),
            "cv_auc_mean": float(best["cv_auc_mean"]),
            "cv_auc_std": float(best["cv_auc_std"]),
        },
        "train": report(tr_mask, "train"),
        "test": report(te_mask, "test"),
        "external": report(ts_mask, "external_ts"),
    }
    with open(f"{OUT}/best_config.json", "w") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    print("\n=== BEST ===")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))

    # 每降维方案最佳 AUC 汇总（对比）
    summ = grid.groupby("selector").agg(best_cv_auc=("cv_auc_mean", "max"),
                                        best_model=("model", lambda x: grid.loc[x.index[grid.loc[x].cv_auc_mean.idxmax()], "model"] if False else "see grid"))
    print("\n=== by selector best ===")
    for s in ("none", "lasso", "pca", "kbest"):
        g = grid[grid["selector"] == s]
        if len(g):
            b = g.iloc[0]
            print(f"  {s:6s} best CV AUC {b['cv_auc_mean']}±{b['cv_auc_std']}  ({b['model']} {b['model_param']})")

if __name__ == "__main__":
    main()
