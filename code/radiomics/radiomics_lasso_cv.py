#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""组学协议升级 -- L1 选择强度 C 与分类器超参联合 CV 网格搜索.

背景 (外部方法学意见, 2026-09-05): 前版协议将 L1 降维固定在 C=1 (无任何搜索), 而
DL 侧 18 骨干择优, 属不对等比较. 本实验将 L1 的 C (特征筛选强度) 加入与分类器
联合的 5 折 CV 网格; 每个 CV 折内重新拟合 L1 选择 (选择信息不泄漏进验证折);
仍不做其他降维手法 (用户指示).

协议 (manual/auto 两轮廓源各自独立):
  1) 网格 = L1 C ∈ {0.01,0.03,0.1,0.3,1.0,3.0} × 25 分类器配置 (与前版协议完全一致);
  2) 5 折 CV (seed 42): 折内 StandardScaler -> L1(liblinear,C) 选特征 -> 管线(标准化+分类器)
     拟合折内训练部分 -> 折内验证 AUC; 150 组合取均值最高者;
  3) 最优 (C*, clf*) 在 train774 全量重拟合 -> test194 + ext82 各评估一次;
  4) 10k bootstrap 95%CI / DeLong 配对 / 90%CI(TOST) / 校准 / DCA / 等权融合.
自检: sklearn AUC 恒等; DeLong 恒等守卫; 标签对齐; DL ext 标签对齐; 折内选择数记录.
输出: experiments/outputs/radiomics_lasso_cv/{summary.json, scores.npz,
      calibration.json, dca.json, cv_table_full.csv}
"""
import json, os, time, warnings
os.environ.setdefault("OPENBLAS_NUM_THREADS", "8")
os.environ.setdefault("OMP_NUM_THREADS", "8")
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, brier_score_loss
import lightgbm as lgb

warnings.filterwarnings("ignore")
ROOT = "/PATH/TO/rectal_project"
OUT = f"{ROOT}/experiments/outputs/radiomics_lasso_cv"
os.makedirs(OUT, exist_ok=True)
SEED = 42
META = ["id", "label", "split", "dataset", "pT"]
L1_GRID = (0.01, 0.03, 0.1, 0.3, 1.0, 3.0)


# ---------------- DeLong (复用已验证实现) ----------------
def compute_midrank(x):
    J = np.argsort(x); Z = x[J]; N = len(x)
    T = np.zeros(N); i = 0
    while i < N:
        j = i
        while j < N and Z[j] == Z[i]: j += 1
        T[i:j] = 0.5 * (i + j - 1) + 1.0
        i = j
    T2 = np.empty(N); T2[J] = T
    return T2


def delong_struct(y, p):
    m = int((y == 1).sum()); n = int((y == 0).sum())
    r_pos = compute_midrank(p[y == 1]); r_neg = compute_midrank(p[y == 0])
    r_all = compute_midrank(p)
    v10 = (r_all[y == 1] - r_pos) / n
    v01 = 1.0 - (r_all[y == 0] - r_neg) / m
    return v10, v01


def delong_test(y, p1, p2):
    v10_1, v01_1 = delong_struct(y, p1)
    v10_2, v01_2 = delong_struct(y, p2)
    auc1, auc2 = v10_1.mean(), v10_2.mean()
    m = int((y == 1).sum()); n = int((y == 0).sum())
    def s(v): return ((v - v.mean()) ** 2).sum()
    def c(a, b): return ((a - a.mean()) * (b - b.mean())).sum()
    S10 = np.array([[s(v10_1), c(v10_1, v10_2)], [c(v10_1, v10_2), s(v10_2)]]) / (m - 1)
    S01 = np.array([[s(v01_1), c(v01_1, v01_2)], [c(v01_1, v01_2), s(v01_2)]]) / (n - 1)
    S = S10 / m + S01 / n
    var_d = S[0, 0] + S[1, 1] - 2 * S[0, 1]
    if var_d <= 0:
        return float(auc1), float(auc2), 0.0, 1.0
    z = (auc1 - auc2) / np.sqrt(var_d)
    return float(auc1), float(auc2), float(z), float(2 * stats.norm.sf(abs(z)))


def auc_rank(y, p):
    r = compute_midrank(p)
    n1 = int((y == 1).sum()); n0 = len(y) - n1
    return (r[y == 1].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0)


def boot_ci_delta(y, p1, p2, n_boot=10000, seed=SEED, alpha=(2.5, 97.5)):
    rng = np.random.RandomState(seed)
    y = np.asarray(y); idx = np.arange(len(y)); d = np.empty(n_boot)
    for b in range(n_boot):
        s = rng.choice(idx, len(idx), replace=True)
        d[b] = auc_rank(y[s], p1[s]) - auc_rank(y[s], p2[s])
    return [float(np.percentile(d, alpha[0])), float(np.percentile(d, alpha[1]))]


def boot_ci_auc(y, p, n_boot=10000, seed=SEED):
    rng = np.random.RandomState(seed)
    y = np.asarray(y); idx = np.arange(len(y)); a = np.empty(n_boot)
    for b in range(n_boot):
        s = rng.choice(idx, len(idx), replace=True)
        a[b] = auc_rank(y[s], p[s])
    return [float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))]


# ---------------- 分类器网格 (与前版协议逐位一致) ----------------
def make_grids():
    grids = []
    for C in (0.01, 0.1, 1.0, 10.0):
        grids.append((f"LR C={C}", lambda C=C: LogisticRegression(C=C, max_iter=5000, random_state=SEED)))
    for C in (0.01, 0.1, 1.0, 10.0):
        grids.append((f"SVM-linear C={C}", lambda C=C: SVC(kernel="linear", C=C, probability=True, random_state=SEED)))
    for C in (0.1, 1.0, 10.0):
        for g in ("scale", 0.001, 0.01):
            grids.append((f"SVM-rbf C={C} gamma={g}", lambda C=C, g=g: SVC(kernel="rbf", C=C, gamma=g, probability=True, random_state=SEED)))
    for n in (500, 1000):
        for d in (None, 10):
            grids.append((f"RF n={n} depth={d}", lambda n=n, d=d: RandomForestClassifier(n_estimators=n, max_depth=d, random_state=SEED, n_jobs=4)))
    for lr in (0.05, 0.1):
        for nl in (15, 31):
            grids.append((f"LGBM n=300 lr={lr} leaves={nl}", lambda lr=lr, nl=nl: lgb.LGBMClassifier(n_estimators=300, learning_rate=lr, num_leaves=nl, random_state=SEED, verbosity=-1, n_jobs=4)))
    return grids


def fit_l1(Xtr_s, ytr, C):
    l1 = LogisticRegression(penalty="l1", solver="liblinear", C=C,
                            max_iter=5000, random_state=SEED)
    l1.fit(Xtr_s, ytr)
    return np.where(np.abs(l1.coef_[0]) > 1e-10)[0]


def run_source(df, feat_cols, source_name):
    print(f"===== source={source_name} n_cols={len(feat_cols)} =====", flush=True)
    t0 = time.time()
    ids = df["id"].astype(str).values
    y = df["label"].values.astype(int)
    sp = df["split"].values
    X = df[feat_cols].apply(pd.to_numeric, errors="coerce").values.astype(float)
    tr, te, ts = sp == "train", sp == "test", sp == "ts"

    med = np.nanmedian(X[tr], axis=0)
    med = np.where(np.isnan(med), 0.0, med)
    nan_r, nan_c = np.where(np.isnan(X))
    X[nan_r, nan_c] = med[nan_c]
    print(f"imputed {len(nan_r)} NaN cells with train medians", flush=True)

    tr_idx = np.where(tr)[0]
    skf = StratifiedKFold(5, shuffle=True, random_state=SEED)
    folds = list(skf.split(X[tr], y[tr]))
    grids = make_grids()

    # 折内 L1 选择缓存: 每 (fold, C) 只拟合一次, 供 25 个分类器共用
    sel_cache = {}
    for fi, (tri, vai) in enumerate(folds):
        scf = StandardScaler().fit(X[tr_idx[tri]])
        Xtr_s = scf.transform(X[tr_idx[tri]])
        for C in L1_GRID:
            sel_cache[(fi, C)] = fit_l1(Xtr_s, y[tr_idx[tri]], C)
        print(f"  fold{fi} L1 done (n_sel per C: "
              f"{[len(sel_cache[(fi, c)]) for c in L1_GRID]}) {time.time()-t0:.0f}s", flush=True)

    # 全训练集各 C 的选择数 (报告用)
    sc_full = StandardScaler().fit(X[tr])
    n_sel_full = {C: int(len(fit_l1(sc_full.transform(X[tr]), y[tr], C))) for C in L1_GRID}
    print(f"full-train n_selected per C: {n_sel_full}", flush=True)

    # 联合 CV
    cv_aucs = {}
    for fi, (tri, vai) in enumerate(folds):
        scf = StandardScaler().fit(X[tr_idx[tri]])
        Xtr_s = scf.transform(X[tr_idx[tri]]); Xva_s = scf.transform(X[tr_idx[vai]])
        for C in L1_GRID:
            sel = sel_cache[(fi, C)]
            if len(sel) == 0:
                continue
            A, B = Xtr_s[:, sel], Xva_s[:, sel]
            for name, ctor in grids:
                pipe = Pipeline([("scale", StandardScaler()), ("clf", ctor())])
                pipe.fit(A, y[tr_idx[tri]])
                p = pipe.predict_proba(B)[:, 1]
                cv_aucs.setdefault((name, C), []).append(roc_auc_score(y[tr_idx[vai]], p))
        print(f"  fold{fi} classifiers done {time.time()-t0:.0f}s", flush=True)

    rows = []
    for (name, C), aucs in cv_aucs.items():
        if len(aucs) < 5:
            continue
        rows.append({"model": name, "l1_C": C, "cv_auc_mean": round(float(np.mean(aucs)), 4),
                     "cv_auc_std": round(float(np.std(aucs)), 4),
                     "folds": [round(a, 4) for a in aucs]})
    rows.sort(key=lambda r: -r["cv_auc_mean"])
    pd.DataFrame(rows).to_csv(f"{OUT}/cv_table_full_{source_name}.csv", index=False)
    best = rows[0]
    print(f"BEST: {best['model']} @ L1 C={best['l1_C']} CV {best['cv_auc_mean']}±{best['cv_auc_std']}", flush=True)
    for r in rows[:5]:
        print("   top:", r["model"], "C=", r["l1_C"], r["cv_auc_mean"], flush=True)

    # 全量重训
    C_star = best["l1_C"]
    sel = fit_l1(sc_full.transform(X[tr]), y[tr], C_star)
    sel_cols = [feat_cols[i] for i in sel]
    Xs = X[:, sel]
    best_ctor = dict(grids)[best["model"]]
    pipe = Pipeline([("scale", StandardScaler()), ("clf", best_ctor())])
    pipe.fit(Xs[tr], y[tr])
    p_te = pipe.predict_proba(Xs[te])[:, 1]
    p_ts = pipe.predict_proba(Xs[ts])[:, 1]
    auc_te, auc_ts = roc_auc_score(y[te], p_te), roc_auc_score(y[ts], p_ts)
    print(f"test {auc_te:.4f}  ext {auc_ts:.4f}  ({time.time()-t0:.0f}s)", flush=True)

    return {
        "source": source_name, "l1_grid": list(L1_GRID),
        "n_l1_selected": len(sel_cols), "sel_cols": sel_cols,
        "n_selected_per_C_full_train": {str(k): v for k, v in n_sel_full.items()},
        "best": best,
        "top_cv_rows": rows[:10],
        "auc": {"test": round(float(auc_te), 4), "ext": round(float(auc_ts), 4)},
        "ci95": {"test": boot_ci_auc(y[te], p_te), "ext": boot_ci_auc(y[ts], p_ts)},
        "_y": y, "_te": te, "_ts": ts, "_p_te": p_te, "_p_ts": p_ts, "_ids": ids,
    }


def calibration(y, p, n_boot=1000, seed=SEED):
    p = np.clip(np.asarray(p, float), 1e-4, 1 - 1e-4)
    y = np.asarray(y)
    logit = np.log(p / (1 - p))
    lr2 = LogisticRegression(C=1e6, max_iter=1000).fit(logit.reshape(-1, 1), y)
    b = float(lr2.coef_[0][0]); a = float(lr2.intercept_[0])
    bins = np.clip((p * 10).astype(int), 0, 9)
    ece = 0.0; curve = []
    for k in range(10):
        m = bins == k
        if m.sum() == 0: continue
        mp_, er = p[m].mean(), y[m].mean()
        ece += m.sum() / len(y) * abs(mp_ - er)
        curve.append({"bin_center": round(float((k + 0.5) / 10), 2), "n": int(m.sum()),
                      "mean_p": round(float(mp_), 4), "event_rate": round(float(er), 4)})
    rng = np.random.RandomState(seed)
    idx = np.arange(len(y)); sl, ic = [], []
    for _ in range(n_boot):
        s = rng.choice(idx, len(idx), replace=True)
        if len(np.unique(y[s])) < 2: continue
        lg = logit[s]
        m = LogisticRegression(C=1e6, max_iter=1000).fit(lg.reshape(-1, 1), y[s])
        sl.append(float(m.coef_[0][0])); ic.append(float(m.intercept_[0]))
    return {"n": int(len(y)), "brier": round(float(brier_score_loss(y, p)), 4),
            "ece": round(float(ece), 4), "slope": round(b, 4), "intercept": round(a, 4),
            "slope_CI": [round(float(np.percentile(sl, 2.5)), 4), round(float(np.percentile(sl, 97.5)), 4)],
            "intercept_CI": [round(float(np.percentile(ic, 2.5)), 4), round(float(np.percentile(ic, 97.5)), 4)],
            "mean_p": round(float(p.mean()), 4), "prev": round(float(y.mean()), 4),
            "curve": curve}


def dca_nb(y, p, thresholds):
    y = np.asarray(y); n = len(y); prev = y.mean()
    out = []
    for t in thresholds:
        pred = p >= t
        tp = int((pred & (y == 1)).sum()); fp = int((pred & (y == 0)).sum())
        nb = tp / n - fp / n * (t / (1 - t))
        nb_all = prev - (1 - prev) * (t / (1 - t))
        out.append({"threshold": round(t, 2), "net_benefit": round(float(nb), 4),
                    "treat_all": round(float(nb_all), 4)})
    return out


def main():
    man = pd.read_csv(f"{ROOT}/data_processed/radiomics_features.csv")
    auto = pd.read_csv(f"{ROOT}/data_processed/radiomics_features_auto.csv")
    man["id"] = man["id"].astype(str); auto["id"] = auto["id"].astype(str)

    idx = pd.read_csv(f"{ROOT}/data_processed/index.csv")
    idx["id"] = idx["id"].astype(str)
    m = dict(zip(idx["id"], idx["label"]))
    lab_ok = all(m[str(i)] == int(l) for i, l in zip(man["id"], man["label"]))

    feat_cols = [c for c in man.columns if c not in META]
    feat_cols = [c for c in feat_cols if c in auto.columns]
    print(f"shared feature cols: {len(feat_cols)}", flush=True)

    rm = run_source(man, feat_cols, "manual")
    ra = run_source(auto, feat_cols, "auto")
    y = rm["_y"]; te, ts = rm["_te"], rm["_ts"]

    D = np.load(f"{ROOT}/experiments/outputs/delong/scores.npz")
    dl_y_ext = D["dl_man_ext_y"]
    ext_ids = rm["_ids"][ts]
    align_ok = bool(np.array_equal(dl_y_ext, y[ts]))
    print(f"DL ext label alignment: {align_ok}", flush=True)
    dl_man_p, dl_auto_p = D["dl_man_ext_p"], D["dl_aut_ext_p"]

    pairs = {}
    a1, a2, z, p_ = delong_test(y[te], rm["_p_te"], ra["_p_te"])
    pairs["rad_man_vs_rad_auto_test"] = {"auc1": round(a1, 4), "auc2": round(a2, 4),
        "delta": round(a1 - a2, 4), "p": round(p_, 6), "ci90": boot_ci_delta(y[te], rm["_p_te"], ra["_p_te"], alpha=(5, 95))}
    a1, a2, z, p_ = delong_test(y[ts], rm["_p_ts"], ra["_p_ts"])
    pairs["rad_man_vs_rad_auto_ext"] = {"auc1": round(a1, 4), "auc2": round(a2, 4),
        "delta": round(a1 - a2, 4), "p": round(p_, 6), "ci90": boot_ci_delta(y[ts], rm["_p_ts"], ra["_p_ts"], alpha=(5, 95))}
    a1, a2, z, p_ = delong_test(y[ts], rm["_p_ts"], dl_man_p)
    pairs["rad_man_vs_dl_man_ext"] = {"auc1": round(a1, 4), "auc2": round(a2, 4),
        "delta": round(a1 - a2, 4), "p": round(p_, 6), "ci95": boot_ci_delta(y[ts], rm["_p_ts"], dl_man_p)}
    a1, a2, z, p_ = delong_test(y[ts], ra["_p_ts"], dl_auto_p)
    pairs["rad_auto_vs_dl_auto_ext"] = {"auc1": round(a1, 4), "auc2": round(a2, 4),
        "delta": round(a1 - a2, 4), "p": round(p_, 6), "ci95": boot_ci_delta(y[ts], ra["_p_ts"], dl_auto_p)}
    print(json.dumps(pairs, indent=2), flush=True)

    _, _, _, p_guard = delong_test(y[ts], rm["_p_ts"], rm["_p_ts"].copy())

    calib = {
        "rad_manual_test": calibration(y[te], rm["_p_te"]),
        "rad_manual_ext": calibration(y[ts], rm["_p_ts"]),
        "rad_auto_test": calibration(y[te], ra["_p_te"]),
        "rad_auto_ext": calibration(y[ts], ra["_p_ts"]),
        "dl_manual_ext": calibration(y[ts], dl_man_p),
        "dl_auto_ext": calibration(y[ts], dl_auto_p),
    }
    th = np.arange(0.20, 0.701, 0.01)
    dca = {
        "ext": {
            "rad_manual": dca_nb(y[ts], rm["_p_ts"], th),
            "rad_auto": dca_nb(y[ts], ra["_p_ts"], th),
            "dl_manual": dca_nb(y[ts], dl_man_p, th),
            "dl_auto": dca_nb(y[ts], dl_auto_p, th),
        },
        "test": {
            "rad_manual": dca_nb(y[te], rm["_p_te"], th),
            "rad_auto": dca_nb(y[te], ra["_p_te"], th),
        },
    }

    fusion = {}
    f_man = (rm["_p_ts"] + dl_man_p) / 2
    f_auto = (ra["_p_ts"] + dl_auto_p) / 2
    fusion["manual_ext"] = {"auc": round(float(roc_auc_score(y[ts], f_man)), 4),
                            "ci95": boot_ci_auc(y[ts], f_man, n_boot=5000)}
    fusion["auto_ext"] = {"auc": round(float(roc_auc_score(y[ts], f_auto)), 4),
                          "ci95": boot_ci_auc(y[ts], f_auto, n_boot=5000)}
    print("fusion:", fusion, flush=True)

    summary = {
        "note": "Radiomics protocol with L1-strength hyperparameter search: "
                "L1 selection C in joint grid with classifier grid, per-CV-fold L1 refit "
                "(no selection leakage); user-directed fix of the earlier fixed-C=1 asymmetry "
                "critique; no other dimensionality-reduction methods",
        "selfcheck": {
            "sklearn_agree_manual_ext": bool(abs(auc_rank(y[ts], rm["_p_ts"]) - roc_auc_score(y[ts], rm["_p_ts"])) < 1e-10),
            "delong_identity_guard_p": round(float(p_guard), 4),
            "labels_match_index": bool(lab_ok),
            "dl_ext_labels_aligned": align_ok,
            "best_manual": rm["best"], "best_auto": ra["best"],
            "n_selected_manual": rm["n_l1_selected"], "n_selected_auto": ra["n_l1_selected"],
        },
        "manual": {k: v for k, v in rm.items() if not k.startswith("_")},
        "auto": {k: v for k, v in ra.items() if not k.startswith("_")},
        "paired_tests": pairs,
        "fusion_equal_weight": fusion,
    }
    json.dump(summary, open(f"{OUT}/summary.json", "w"), indent=2)
    json.dump(calib, open(f"{OUT}/calibration.json", "w"), indent=2)
    json.dump(dca, open(f"{OUT}/dca.json", "w"), indent=2)
    np.savez(f"{OUT}/scores.npz",
             y=y, te=te, ts=ts,
             man_p_te=rm["_p_te"], man_p_ts=rm["_p_ts"],
             auto_p_te=ra["_p_te"], auto_p_ts=ra["_p_ts"],
             dl_man_ext_p=dl_man_p, dl_auto_ext_p=dl_auto_p,
             ids=rm["_ids"])
    print("SAVED", OUT, flush=True)
    print(json.dumps(summary["selfcheck"], indent=2))


if __name__ == "__main__":
    main()
