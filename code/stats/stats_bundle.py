#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""统计报告完善包 (模拟审稿 R1.1/R2.2/R3.2/R3.3 + R2 建议项特征族).

A. ConvNeXt DL (manual+auto) test-194 5折 ensemble 推理 (补 Table 4 缺失的 DL test 行; 缓存 npz)
B. 二分类操作指标: 4 链 × {test194, ext82} × {full, T2vT3 子集}; thr=0.5 与 Youden 双口径;
   Youden 阈值优先取 DL OOF (训练折内), 否则评测集内 (标注 optimistic); 另报 per-stage 敏感性
C. TOST 双界值敏感性 (0.05 / 0.10), 4 个自动化对比, 10k bootstrap
D. 外部集可检出差异定量化: MDD_80% = (z_.975+z_.80)*SE(bootstrap) 每对比
E. 特征族 x 厂商/层厚: 标准化均差 d (GE vs Siemens; GE厚 vs GE薄), 按特征族汇总
输出: experiments/outputs/stats_bundle/{summary.json, per_stage_metrics.csv,
      tost_dual_margin.json, mdd.json, feature_family_shift.csv, dl_test_preds.npz}
"""
import os, json
os.environ.setdefault("OPENBLAS_NUM_THREADS", "8")
os.environ.setdefault("OMP_NUM_THREADS", "8")
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import roc_auc_score, roc_curve

ROOT = "/PATH/TO/rectal_project"
OUTD = f"{ROOT}/experiments/outputs/stats_bundle"
os.makedirs(OUTD, exist_ok=True)
SEED = 42
rng = np.random.RandomState(SEED)


# ---------- DeLong (同源已验证实现) ----------
def compute_midrank(x):
    J = np.argsort(x); Z = x[J]; N = len(x)
    T = np.zeros(N); i = 0
    while i < N:
        j = i
        while j < N and Z[j] == Z[i]:
            j += 1
        T[i:j] = 0.5 * (i + j - 1)
        i = j
    T2 = np.empty(N); T2[J] = T + 1
    return T2


def delong_test(y, p1, p2):
    y = np.asarray(y).astype(int)
    m = int((y == 1).sum()); n = int((y == 0).sum())

    def struct(pos, neg):
        v = np.zeros(len(pos)); w = np.zeros(len(neg))
        for i in range(len(pos)):
            for j in range(len(neg)):
                s = pos[i] - neg[j]
                if s > 0:
                    v[i] += 1
                elif s < 0:
                    w[j] += 1
        return v / len(neg), w / len(pos)

    v10_1, v01_1 = struct(p1[y == 1], p1[y == 0])
    v10_2, v01_2 = struct(p2[y == 1], p2[y == 0])
    s10 = np.cov(np.vstack([v10_1, v10_2]))
    s01 = np.cov(np.vstack([v01_1, v01_2]))
    s = s10 / m + s01 / n
    d = roc_auc_score(y, p1) - roc_auc_score(y, p2)
    var = float(np.dot(np.dot([1.0, -1.0], s), [1.0, -1.0]))
    from scipy.stats import norm
    z = abs(d) / np.sqrt(var) if var > 0 else np.inf
    return 2 * norm.sf(z), d


def boot_delta(y, p1, p2, n_boot=10000, seed=SEED):
    r = np.random.RandomState(seed); y = np.asarray(y); d = []
    for _ in range(n_boot):
        idx = r.randint(0, len(y), len(y))
        if len(set(y[idx])) < 2:
            continue
        d.append(roc_auc_score(y[idx], p1[idx]) - roc_auc_score(y[idx], p2[idx]))
    return float(np.std(d, ddof=1)), float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))


# ---------- A. ConvNeXt test 推理 (缓存) ----------
def dl_test_inference():
    cache = f"{OUTD}/dl_test_preds.npz"
    if os.path.exists(cache):
        z = np.load(cache, allow_pickle=True)
        return z["p_man"], z["p_aut"], z["y"], z["ids"]
    import sys, torch
    sys.path.insert(0, f"{ROOT}/code")
    from dltrain.common import set_seed, Crop2D25Dataset, DATA
    from train_2d25 import make_model
    set_seed(SEED)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    idx = pd.read_csv(f"{ROOT}/data_processed/index.csv")
    idx["id"] = idx["id"].astype(str)
    te_ids = idx[idx["split"] == "test"]["id"].tolist()
    y = idx[idx["split"] == "test"]["label"].values
    res = {}
    for arm, d in (("man", "convnext_base_d25_roi"), ("aut", "convnext_base_d25roi_auto")):
        variant = "d25_roi" if arm == "man" else "d25_roi_auto"
        ids_ok = [i for i in te_ids if os.path.exists(f"{DATA}/crops/{variant}/{i}.npz")]
        ds = Crop2D25Dataset(ids_ok, variant, labels=np.zeros(len(ids_ok)), aug=False)
        dl = torch.utils.data.DataLoader(ds, batch_size=32, num_workers=2)
        pf = []
        for f in range(5):
            m = make_model("convnext_base").to(device)
            m.load_state_dict(torch.load(f"{ROOT}/experiments/outputs/{d}/fold{f}_best.pt",
                                         map_location=device))
            m.eval()
            p = []
            with torch.no_grad():
                for x, _ in dl:
                    p.append(torch.sigmoid(m(x.to(device))).squeeze(1).cpu().numpy())
            pf.append(np.concatenate(p))
            del m
        assert ids_ok == te_ids, f"missing test crops in {arm}: {set(te_ids)-set(ids_ok)}"
        res[f"p_{arm}"] = np.mean(pf, 0)
        print(f"convnext {arm} test ens AUC {roc_auc_score(y, res[f'p_{arm}']):.4f}", flush=True)
    np.savez(cache, p_man=res["p_man"], p_aut=res["p_aut"], y=y,
             ids=np.array(te_ids, dtype=object))
    return res["p_man"], res["p_aut"], y, np.array(te_ids, dtype=object)


# ---------- B. 操作指标 ----------
def binary_metrics(y, p, thr):
    pred = (p >= thr).astype(int)
    y = np.asarray(y).astype(int)
    tp = int(((pred == 1) & (y == 1)).sum()); fn = int(((pred == 0) & (y == 1)).sum())
    tn = int(((pred == 0) & (y == 0)).sum()); fp = int(((pred == 1) & (y == 0)).sum())
    return {"acc": round((tp + tn) / max(1, len(y)), 3),
            "sen": round(tp / max(1, tp + fn), 3), "spe": round(tn / max(1, tn + fp), 3),
            "thr": round(float(thr), 4)}


def youden_thr(y, p):
    fpr, tpr, thr = roc_curve(y, p)
    j = tpr - fpr
    return float(thr[int(np.argmax(j))])


def main():
    # ---- A ----
    dl_man_te, dl_aut_te, y_te_arr, _ = dl_test_inference()

    # ---- 装配概率表 ----
    idx = pd.read_csv(f"{ROOT}/data_processed/index.csv")
    idx["id"] = idx["id"].astype(str)
    z = np.load(f"{ROOT}/experiments/outputs/radiomics_ensemble/scores.npz", allow_pickle=True)
    ids_all = z["ids"].astype(str)
    te_mask = z["te"].astype(bool); ts_mask = z["ts"].astype(bool)
    y_all = z["y"].astype(int)
    pT = dict(zip(idx["id"], idx["pT"].astype(int)))

    dl_ext_man = z["dl_man_ext_p"]; dl_ext_aut = z["dl_auto_ext_p"]
    chains_te = {
        "radiomics_manual": (y_all[te_mask], z["man_p_te"]),
        "radiomics_auto": (y_all[te_mask], z["auto_p_te"]),
        "dl_manual": (y_te_arr, dl_man_te),
        "dl_auto": (y_te_arr, dl_aut_te)}
    chains_ext = {
        "radiomics_manual": (y_all[ts_mask], z["man_p_ts"]),
        "radiomics_auto": (y_all[ts_mask], z["auto_p_ts"]),
        "dl_manual": (y_all[ts_mask], dl_ext_man),
        "dl_auto": (y_all[ts_mask], dl_ext_aut)}
    # dl test y 必须与 scores 的 test 子集同序 (都来自 index split=test 行序)
    assert (y_te_arr == y_all[te_mask]).all(), "DL test label order mismatch vs radiomics scores"

    oof_path = f"{ROOT}/experiments/outputs/recal/oof_probs.npz"
    oof = np.load(oof_path, allow_pickle=True) if os.path.exists(oof_path) else None
    oof_ens_path = f"{ROOT}/experiments/outputs/recal/oof_probs_ens.npz"
    oof_ens = np.load(oof_ens_path, allow_pickle=True) if os.path.exists(oof_ens_path) else None

    rows = []
    for setname, chains in (("test194", chains_te), ("ext82", chains_ext)):
        ids_set = ids_all[te_mask] if setname == "test194" else ids_all[ts_mask]
        for cname, (y, p) in chains.items():
            if p is None:
                continue
            pt = np.array([pT[i] for i in ids_set])
            sub = pt >= 2  # T2vT3
            # 阈值: 0.5 固定 + Youden(OOF 优先) + Youden(评测集)
            thrs = {"fixed0.5": 0.5}
            yt = youden_thr(y, p)
            src = "evalset(optimistic)"
            if oof is not None:
                key = cname  # oof_probs.npz 键: radiomics_manual / radiomics_auto / dl_manual / dl_auto
                src_oof = oof_ens if (oof_ens is not None and key.startswith("radiomics")) else oof
                if key in src_oof:
                    cand = src_oof[key]
                    if np.isfinite(np.asarray(cand, float)).all():
                        yt_oof = youden_thr(src_oof["y"].astype(int), cand)
                        thrs["youden_oof"] = yt_oof
                        src = "trainOOF"
            thrs["youden_eval"] = yt
            for sub_name, mask in (("full", np.ones(len(y), bool)), ("T2vT3", sub)):
                ys, ps, pts = y[mask], p[mask], pt[mask]
                row = {"set": setname, "subset": sub_name, "chain": cname, "n": int(len(ys)),
                       "n_pos": int(ys.sum()), "auc": round(float(roc_auc_score(ys, ps)), 4)}
                for tn_, t in thrs.items():
                    m = binary_metrics(ys, ps, t)
                    row[f"acc_{tn_}"] = m["acc"]; row[f"sen_{tn_}"] = m["sen"]; row[f"spe_{tn_}"] = m["spe"]
                # per-stage 敏感性 (fixed 0.5 与主 Youden)
                for stage in (1, 2, 3):
                    ms = pts == stage
                    if ms.sum():
                        row[f"stage{stage}_n"] = int(ms.sum())
                        row[f"stage{stage}_sen0.5"] = round(float((ps[ms] < 0.5).mean()), 3)
                        row[f"stage{stage}_senY"] = round(float((ps[ms] < yt).mean()), 3)
                row["youden_src"] = src
                rows.append(row)
    pd.DataFrame(rows).to_csv(f"{OUTD}/per_stage_metrics.csv", index=False)
    print(f"per-stage rows: {len(rows)}")

    # ---- C. TOST 双界值 ----
    tost = {}
    for margin in (0.05, 0.10):
        res_m = {}
        for setname, (y, pm, pa) in {
            "test_rad": (y_all[te_mask], z["man_p_te"], z["auto_p_te"]),
            "ext_rad": (y_all[ts_mask], z["man_p_ts"], z["auto_p_ts"]),
            "test_dl": (y_te_arr, dl_man_te, dl_aut_te),
            "ext_dl": (y_all[ts_mask], dl_ext_man, dl_ext_aut)}.items():
            r = np.random.RandomState(SEED); deltas = []
            for _ in range(10000):
                i = r.randint(0, len(y), len(y))
                deltas.append(roc_auc_score(y[i], pm[i]) - roc_auc_score(y[i], pa[i]))
            lo, hi = np.percentile(deltas, [5, 95])
            point = roc_auc_score(y, pm) - roc_auc_score(y, pa)
            res_m[setname] = {"delta": round(float(point), 4),
                              "ci90": [round(float(lo), 4), round(float(hi), 4)],
                              "equivalent_at_margin": bool(lo > -margin and hi < margin)}
        tost[f"margin_{margin}"] = res_m
    json.dump(tost, open(f"{OUTD}/tost_dual_margin.json", "w"), indent=1)

    # ---- D. MDD ----
    mdd = {}
    for setname, (y, pm, pa) in {
        "test_rad": (y_all[te_mask], z["man_p_te"], z["auto_p_te"]),
        "ext_rad": (y_all[ts_mask], z["man_p_ts"], z["auto_p_ts"]),
        "test_dl": (y_te_arr, dl_man_te, dl_aut_te),
        "ext_dl": (y_all[ts_mask], dl_ext_man, dl_ext_aut)}.items():
        se, lo, hi = boot_delta(y, pm, pa)
        mdd[setname] = {"se_delta": round(se, 4),
                        "mdd_80power": round(2.80 * se, 4),
                        "ci95": [round(lo, 4), round(hi, 4)]}
    json.dump(mdd, open(f"{OUTD}/mdd.json", "w"), indent=1)
    print("MDD ext:", {k: round(v["mdd_80power"], 3) for k, v in mdd.items()})

    # ---- E. 特征族 x 厂商/层厚 ----
    feat = pd.read_csv(f"{ROOT}/data_processed/radiomics_features.csv")
    feat["id"] = feat["id"].astype(str)
    vm = pd.read_csv(f"{ROOT}/data_processed/vendor_map.csv")
    vm["id"] = vm["id"].astype(str)
    meta = feat[["id"]].merge(vm[["id", "vendor", "spacing_z"]], on="id", how="inner")
    feat = feat.set_index("id").loc[meta["id"]].reset_index()
    fcols = [c for c in feat.columns if c not in ("id", "label", "split", "dataset", "pT")]
    X = feat[fcols].apply(pd.to_numeric, errors="coerce").values
    v = meta["vendor"].str.startswith("GE").values
    sz = meta["spacing_z"].values

    def smd(m1, m2):
        x1, x2 = X[m1], X[m2]
        mu1, mu2 = np.nanmean(x1, 0), np.nanmean(x2, 0)
        s1, s2 = np.nanstd(x1, 0, ddof=1), np.nanstd(x2, 0, ddof=1)
        return (mu1 - mu2) / np.maximum(np.sqrt((s1**2 + s2**2) / 2), 1e-9)

    d_vendor = np.abs(smd(v, ~v))
    thick = v & (sz > 3.2); thin = v & (sz <= 2.0)
    d_thick = np.abs(smd(thick, thin))

    def fam(c):
        parts = c.split("_")
        filt = parts[0] if not parts[0].startswith(("log", "wavelet", "square", "gradient", "lbp")) else "_".join(parts[:1])
        cls = parts[1] if len(parts) > 1 else "?"
        return f"{filt}|{cls}"

    fams = [fam(c) for c in fcols]
    fdf = pd.DataFrame({"feature": fcols, "family": fams,
                        "d_vendor": d_vendor, "d_thickness": d_thick})
    agg = fdf.groupby("family").agg(
        n=("feature", "size"),
        d_vendor_med=("d_vendor", "median"), d_vendor_p90=("d_vendor", lambda s: float(np.percentile(s, 90))),
        d_thick_med=("d_thickness", "median"), d_thick_p90=("d_thickness", lambda s: float(np.percentile(s, 90))),
        frac_vendor_gt08=("d_vendor", lambda s: float((s > 0.8).mean())),
        frac_thick_gt08=("d_thickness", lambda s: float((s > 0.8).mean()))).reset_index()
    agg = agg.sort_values("d_thick_med", ascending=False)
    agg.to_csv(f"{OUTD}/feature_family_shift.csv", index=False)
    print("top thickness-sensitive families:")
    print(agg.head(8).round(3).to_string())

    json.dump({"note": "stats bundle", "n_rows_perstage": len(rows)}, open(f"{OUTD}/summary.json", "w"))
    print("SAVED", OUTD)


if __name__ == "__main__":
    main()
