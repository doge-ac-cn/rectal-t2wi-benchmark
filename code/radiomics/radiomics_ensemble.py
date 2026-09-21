#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""组学协议升级 -- 单 L1(LASSO) 降维 + 五家族嵌套交叉验证软投票集成.
(用户指示, 回应审稿意见 "L1+单分类器范式过时"; 替换 lasso_cv 的单最优分类器协议)

协议 (manual / auto 两个轮廓源各自独立执行, 完全对称):
  外层: 训练集 774 上 5 折分层 CV (seed 42) -> 集成性能的无偏估计
    内层: 每个外层训练子集内 4 折分层 CV (seed 43) -> 每家族超参选择
      - 每个内折: 标准化(内折训练子集) -> 每个 L1 C 重做 L1 选择 (无选择泄漏)
      - 每个 (family, config, C) 取内折均值 AUC; 每家族取最优 (config, C)
    外层折: 最优配置在外层训练子集重训 (L1 在该子集重选) ->
      5 家族概率均值(软投票) -> 外层验证折 AUC + 逐家族 AUC
  最终模型: 全训练集内层 5 折 (seed 42) 选择 -> 每家族最优在全训练集重训 ->
    软投票 -> 内部 test194 + 外部 ts82 各评一次
家族: LR / SVM(linear+rbf) / RF / DecisionTree / LightGBM (28 配置)
L1: 唯一降维步骤, 强度 C 在内层 CV 中与家族配置联合选择, 每次分裂重选.

输出: experiments/outputs/radiomics_ensemble/
  summary.json, scores.npz (lasso_cv 兼容 schema + 训练折 OOF),
  calibration.json, dca.json, ensemble_spec.json, cv_final_<src>.csv
"""
import os, json, time, warnings
os.environ.setdefault("OPENBLAS_NUM_THREADS", "8")
os.environ.setdefault("OMP_NUM_THREADS", "8")
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, brier_score_loss
import lightgbm as lgb

ROOT = "/PATH/TO/rectal_project"
OUT = f"{ROOT}/experiments/outputs/radiomics_ensemble"
os.makedirs(OUT, exist_ok=True)
SEED = 42
META = ["id", "label", "split", "dataset", "pT"]
L1_GRID = (0.01, 0.03, 0.1, 0.3, 1.0, 3.0)


# ---------------- DeLong (逐位复用已验证实现) ----------------
from scipy import stats as _stats

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


def auc_rank(y, p):
    r = compute_midrank(p)
    n1 = int((y == 1).sum()); n0 = len(y) - n1
    return (r[y == 1].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0)


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
    return float(auc1), float(auc2), float(z), float(2 * _stats.norm.sf(abs(z)))


def boot_ci_auc(y, p, n_boot=10000, seed=SEED):
    rng = np.random.RandomState(seed); idx = np.arange(len(y))
    y = np.asarray(y); a = np.empty(n_boot)
    for b in range(n_boot):
        s = rng.choice(idx, len(idx), replace=True)
        a[b] = auc_rank(y[s], p[s])
    return [round(float(np.percentile(a, 2.5)), 4), round(float(np.percentile(a, 97.5)), 4)]


def boot_ci_delta(y, p1, p2, n_boot=10000, seed=SEED, alpha=(2.5, 97.5)):
    rng = np.random.RandomState(seed); idx = np.arange(len(y))
    y = np.asarray(y); a = np.empty(n_boot)
    for b in range(n_boot):
        s = rng.choice(idx, len(idx), replace=True)
        a[b] = auc_rank(y[s], p1[s]) - auc_rank(y[s], p2[s])
    return [round(float(np.percentile(a, alpha[0])), 4), round(float(np.percentile(a, alpha[1])), 4)]


# ---------------- 家族网格 ----------------
def make_families():
    lr = [(f"LR C={c}", lambda c=c: LogisticRegression(C=c, max_iter=5000, random_state=SEED))
          for c in (0.01, 0.1, 1.0, 10.0)]
    svm = []
    for c in (0.01, 1.0, 10.0):
        svm.append((f"SVM-linear C={c}", lambda c=c: SVC(kernel="linear", C=c, probability=True, random_state=SEED)))
    for c in (0.1, 1.0, 10.0):
        for g in ("scale", 0.001, 0.01):
            svm.append((f"SVM-rbf C={c} gamma={g}", lambda c=c, g=g: SVC(kernel="rbf", C=c, gamma=g, probability=True, random_state=SEED)))
    rf = [(f"RF n=500 depth={d}", lambda d=d: RandomForestClassifier(n_estimators=500, max_depth=d, random_state=SEED, n_jobs=4))
          for d in (None, 10)]
    dt = [(f"DT depth={d} msl={m}", lambda d=d, m=m: DecisionTreeClassifier(max_depth=d, min_samples_leaf=m, random_state=SEED))
          for d in (3, 5, None) for m in (10, 30)]
    lg = [(f"LGBM n=300 lr={r} leaves={n}", lambda r=r, n=n: lgb.LGBMClassifier(
            n_estimators=300, learning_rate=r, num_leaves=n, random_state=SEED, verbosity=-1, n_jobs=4))
          for r in (0.05, 0.1) for n in (15, 31)]
    return {"LR": lr, "SVM": svm, "RF": rf, "DT": dt, "LGBM": lg}


def fit_l1(Xtr_s, ytr, C):
    l1 = LogisticRegression(penalty="l1", solver="liblinear", C=C, max_iter=5000, random_state=SEED)
    l1.fit(Xtr_s, ytr)
    return np.where(np.abs(l1.coef_[0]) > 1e-10)[0]


def select_best_per_family(X, y, fams, all_ctors, n_folds, seed):
    """内层 CV: 每 (内折, C) 重做 L1 选择, 28 配置拟合, 每家族取最优 (config, C)."""
    skf = StratifiedKFold(n_folds, shuffle=True, random_state=seed)
    folds = list(skf.split(X, y))
    acc = {}
    for tri, vai in folds:
        sc = StandardScaler().fit(X[tri])
        Xtr_s, Xva_s = sc.transform(X[tri]), sc.transform(X[vai])
        for C in L1_GRID:
            sel = fit_l1(Xtr_s, y[tri], C)
            if len(sel) == 0:
                continue
            A, B = Xtr_s[:, sel], Xva_s[:, sel]
            for fam, grids in fams.items():
                for name, _ in grids:
                    pipe = Pipeline([("scale", StandardScaler()), ("clf", all_ctors[name]())])
                    pipe.fit(A, y[tri])
                    p = pipe.predict_proba(B)[:, 1]
                    acc.setdefault((fam, name, C), []).append(roc_auc_score(y[vai], p))
    rows = []
    for (fam, name, C), a in acc.items():
        if len(a) < n_folds:
            continue
        rows.append({"family": fam, "model": name, "l1_C": C,
                     "cv_auc_mean": round(float(np.mean(a)), 4),
                     "cv_auc_std": round(float(np.std(a)), 4)})
    cvtab = pd.DataFrame(rows).sort_values("cv_auc_mean", ascending=False).reset_index(drop=True)
    best = {}
    for fam in fams:
        r = cvtab[cvtab.family == fam].iloc[0]
        best[fam] = (r["model"], float(r["l1_C"]), float(r["cv_auc_mean"]))
    return best, cvtab


def fit_members(best, Xtr, ytr, all_ctors, feat_cols):
    sc_full = StandardScaler().fit(Xtr)
    members = []
    for fam, (name, C, cvm) in best.items():
        sel = fit_l1(sc_full.transform(Xtr), ytr, C)
        if len(sel) == 0:
            sel = np.arange(Xtr.shape[1])
        pipe = Pipeline([("scale", StandardScaler()), ("clf", all_ctors[name]())])
        pipe.fit(Xtr[:, sel], ytr)
        members.append({"family": fam, "model": name, "l1_C": C, "inner_cv_auc": round(cvm, 4),
                        "sel": sel, "pipe": pipe})
    return members


def predict_members(members, X):
    ps = [m["pipe"].predict_proba(X[:, m["sel"]])[:, 1] for m in members]
    return np.mean(ps, axis=0), ps


def outer_cv_ensemble(Xtr, ytr, fams, all_ctors):
    skf = StratifiedKFold(5, shuffle=True, random_state=SEED)
    folds = list(skf.split(Xtr, ytr))
    ens_aucs, fam_outer = [], {f: [] for f in fams}
    oof = np.full(len(ytr), np.nan)
    for tri, vai in folds:
        best, _ = select_best_per_family(Xtr[tri], ytr[tri], fams, all_ctors, 4, seed=43)
        members = fit_members(best, Xtr[tri], ytr[tri], all_ctors, None)
        pe, ps_ = predict_members(members, Xtr[vai])
        ens_aucs.append(roc_auc_score(ytr[vai], pe)); oof[vai] = pe
        for m, pp in zip(members, ps_):
            fam_outer[m["family"]].append(round(float(roc_auc_score(ytr[vai], pp)), 4))
    return ens_aucs, fam_outer, oof, folds


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
        m = LogisticRegression(C=1e6, max_iter=1000).fit(logit[s].reshape(-1, 1), y[s])
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


def run_source(df, feat_cols, source_name, fams, all_ctors):
    print(f"===== source={source_name} n_cols={len(feat_cols)} =====", flush=True)
    t0 = time.time()
    ids = df["id"].astype(str).values
    y = df["label"].values.astype(int)
    sp = df["split"].values
    X = df[feat_cols].apply(pd.to_numeric, errors="coerce").values.astype(float)
    tr, te, ts = sp == "train", sp == "test", sp == "ts"
    med = np.nanmedian(X[tr], axis=0); med = np.where(np.isnan(med), 0.0, med)
    nan_r, nan_c = np.where(np.isnan(X)); X[nan_r, nan_c] = med[nan_c]
    print(f"imputed {len(nan_r)} NaN cells with train medians", flush=True)

    Xtr, ytr = X[tr], y[tr]
    ens_aucs, fam_outer, oof_tr, folds = outer_cv_ensemble(Xtr, ytr, fams, all_ctors)
    print(f"ENSEMBLE outer CV: {np.mean(ens_aucs):.4f}+-{np.std(ens_aucs):.4f} "
          f"folds={[round(a,4) for a in ens_aucs]} ({time.time()-t0:.0f}s)", flush=True)

    best_final, cvtab_final = select_best_per_family(Xtr, ytr, fams, all_ctors, 5, seed=SEED)
    cvtab_final.to_csv(f"{OUT}/cv_final_{source_name}.csv", index=False)
    members = fit_members(best_final, Xtr, ytr, all_ctors, feat_cols)
    p_te, ps_te = predict_members(members, X[te])
    p_ts, ps_ts = predict_members(members, X[ts])
    auc_te, auc_ts = roc_auc_score(y[te], p_te), roc_auc_score(y[ts], p_ts)
    print(f"ENSEMBLE test {auc_te:.4f}  ext {auc_ts:.4f}  ({time.time()-t0:.0f}s)", flush=True)

    spec = {"source": source_name,
            "ensemble_outer_cv_auc_mean": round(float(np.mean(ens_aucs)), 4),
            "ensemble_outer_cv_auc_std": round(float(np.std(ens_aucs)), 4),
            "ensemble_outer_cv_folds": [round(a, 4) for a in ens_aucs],
            "per_family_outer_cv_auc": fam_outer,
            "final_members": [
                {"family": m["family"], "model": m["model"], "l1_C": m["l1_C"],
                 "inner_cv_auc": m["inner_cv_auc"], "n_selected": int(len(m["sel"])),
                 "sel_cols": [feat_cols[i] for i in m["sel"]],
                 "auc_test": round(float(roc_auc_score(y[te], qte)), 4),
                 "auc_ext": round(float(roc_auc_score(y[ts], qts)), 4)}
                for m, qte, qts in zip(members, ps_te, ps_ts)],
            "union_selected": sorted({feat_cols[i] for m in members for i in m["sel"]}),
            "l1_grid": list(L1_GRID)}
    return {"y": y, "te": te, "ts": ts, "ids": ids,
            "p_te": p_te, "p_ts": p_ts, "oof_tr": oof_tr, "tr": tr,
            "auc": {"test": round(float(auc_te), 4), "ext": round(float(auc_ts), 4)},
            "ci95": {"test": boot_ci_auc(y[te], p_te), "ext": boot_ci_auc(y[ts], p_ts)},
            "spec": spec}


def main():
    man = pd.read_csv(f"{ROOT}/data_processed/radiomics_features.csv")
    auto = pd.read_csv(f"{ROOT}/data_processed/radiomics_features_auto.csv")
    man["id"] = man["id"].astype(str); auto["id"] = auto["id"].astype(str)
    idx = pd.read_csv(f"{ROOT}/data_processed/index.csv"); idx["id"] = idx["id"].astype(str)
    m = dict(zip(idx["id"], idx["label"]))
    lab_ok = all(m[str(i)] == int(l) for i, l in zip(man["id"], man["label"]))

    feat_cols = [c for c in man.columns if c not in META]
    feat_cols = [c for c in feat_cols if c in auto.columns]
    print(f"shared feature cols: {len(feat_cols)}", flush=True)

    fams = make_families()
    all_ctors = {n: c for fam in fams.values() for n, c in fam}

    rm = run_source(man, feat_cols, "manual", fams, all_ctors)
    ra = run_source(auto, feat_cols, "auto", fams, all_ctors)
    y, te, ts = rm["y"], rm["te"], rm["ts"]

    D = np.load(f"{ROOT}/experiments/outputs/delong/scores.npz")
    dl_y_ext = D["dl_man_ext_y"]
    align_ok = bool(np.array_equal(dl_y_ext, y[ts]))
    dl_man_p, dl_auto_p = D["dl_man_ext_p"], D["dl_aut_ext_p"]

    S61 = np.load(f"{ROOT}/experiments/outputs/radiomics_lasso_cv/scores.npz", allow_pickle=True)

    pairs = {}
    for key, (a_, b_, mask_) in {
        "rad_ens_man_vs_auto_test": (rm["p_te"], ra["p_te"], te),
        "rad_ens_man_vs_auto_ext": (rm["p_ts"], ra["p_ts"], ts),
        "rad_ens_man_vs_dl_man_ext": (rm["p_ts"], dl_man_p, ts),
        "rad_ens_auto_vs_dl_auto_ext": (ra["p_ts"], dl_auto_p, ts),
        "rad_ens_vs_single_man_ext": (rm["p_ts"], S61["man_p_ts"], ts),
        "rad_ens_vs_single_auto_ext": (ra["p_ts"], S61["auto_p_ts"], ts),
        "rad_ens_vs_single_man_test": (rm["p_te"], S61["man_p_te"], te),
    }.items():
        a1, a2, z, pv = delong_test(y[mask_], a_, b_)
        pairs[key] = {"auc1": round(a1, 4), "auc2": round(a2, 4), "delta": round(a1 - a2, 4),
                      "p": round(pv, 6),
                      "ci95" if "ext" in key and "dl" in key or "single" in key else "ci90":
                          boot_ci_delta(y[mask_], a_, b_, alpha=(2.5, 97.5) if ("dl" in key or "single" in key) else (5, 95))}
    _, _, _, p_guard = delong_test(y[ts], rm["p_ts"], rm["p_ts"].copy())
    print(json.dumps(pairs, indent=2), flush=True)

    calib = {"rad_manual_test": calibration(y[te], rm["p_te"]),
             "rad_manual_ext": calibration(y[ts], rm["p_ts"]),
             "rad_auto_test": calibration(y[te], ra["p_te"]),
             "rad_auto_ext": calibration(y[ts], ra["p_ts"]),
             "dl_manual_ext": calibration(y[ts], dl_man_p),
             "dl_auto_ext": calibration(y[ts], dl_auto_p)}
    th = np.arange(0.20, 0.701, 0.01)
    dca = {"ext": {"rad_manual": dca_nb(y[ts], rm["p_ts"], th),
                   "rad_auto": dca_nb(y[ts], ra["p_ts"], th),
                   "dl_manual": dca_nb(y[ts], dl_man_p, th),
                   "dl_auto": dca_nb(y[ts], dl_auto_p, th)},
           "test": {"rad_manual": dca_nb(y[te], rm["p_te"], th),
                    "rad_auto": dca_nb(y[te], ra["p_te"], th)}}
    f_man = (rm["p_ts"] + dl_man_p) / 2
    f_auto = (ra["p_ts"] + dl_auto_p) / 2
    fusion = {"manual_ext": {"auc": round(float(roc_auc_score(y[ts], f_man)), 4),
                             "ci95": boot_ci_auc(y[ts], f_man, n_boot=5000)},
              "auto_ext": {"auc": round(float(roc_auc_score(y[ts], f_auto)), 4),
                           "ci95": boot_ci_auc(y[ts], f_auto, n_boot=5000)}}
    print("fusion:", fusion, flush=True)

    summary = {
        "note": "Radiomics protocol upgrade (user-directed): single L1(LASSO) selection stage "
                "(strength C tuned by inner CV, refit within every split - no selection leakage) + "
                "nested-CV soft-voting ensemble of {LR, SVM, RF, DecisionTree, LightGBM}; "
                "outer 5-fold CV on training set = unbiased ensemble estimate; no other "
                "dimensionality-reduction methods",
        "selfcheck": {
            "sklearn_agree_manual_ext": bool(abs(auc_rank(y[ts], rm["p_ts"]) - roc_auc_score(y[ts], rm["p_ts"])) < 1e-10),
            "delong_identity_guard_p": round(float(p_guard), 4),
            "labels_match_index": bool(lab_ok),
            "dl_ext_labels_aligned": align_ok,
            "reference": {"manual": {"test": 0.8762, "ext": 0.8915},
                                 "auto": {"test": 0.7477, "ext": 0.7892}}},
        "ensemble": {"manual": rm["spec"], "auto": ra["spec"]},
        "auc": {"manual": rm["auc"], "auto": ra["auc"]},
        "ci95": {"manual": rm["ci95"], "auto": ra["ci95"]},
        "paired_tests": pairs,
        "fusion_equal_weight": fusion,
    }
    json.dump(summary, open(f"{OUT}/summary.json", "w"), indent=2)
    json.dump(calib, open(f"{OUT}/calibration.json", "w"), indent=2)
    json.dump(dca, open(f"{OUT}/dca.json", "w"), indent=2)
    json.dump({"manual": rm["spec"], "auto": ra["spec"]}, open(f"{OUT}/ensemble_spec.json", "w"), indent=2)
    np.savez(f"{OUT}/scores.npz",
             y=y, te=te, ts=ts, tr=rm["tr"],
             man_p_te=rm["p_te"], man_p_ts=rm["p_ts"],
             auto_p_te=ra["p_te"], auto_p_ts=ra["p_ts"],
             dl_man_ext_p=dl_man_p, dl_auto_ext_p=dl_auto_p,
             man_oof_tr=rm["oof_tr"], auto_oof_tr=ra["oof_tr"],
             ids=rm["ids"])
    print("SAVED", OUT, flush=True)
    print(json.dumps(summary["selfcheck"], indent=2))


if __name__ == "__main__":
    main()
