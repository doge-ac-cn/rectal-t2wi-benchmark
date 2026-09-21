#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Representation-family MDD (landing spot for main-text Section 3.7 claim).

Protocol = mdd_power_1000.py verbatim (vectorized mid-rank bootstrap,
B=10000, seed 42, MDD80 = 2.8016 x SE), applied to the two primary
2.5D-versus-best-3D ENSEMBLE contrasts on the external cohort (n=82):
  C1: ResNet50 2.5D tumor-ROI ensemble vs ImageNet-inflated 3D ensemble
  C2: ConvNeXt-B 2.5D tumor-ROI ensemble vs ImageNet-inflated 3D ensemble

Anchors (asserted before bootstrap): ensemble AUCs 0.8754 / 0.8706 / 0.806.
Output: experiments/outputs/mdd_repr/mdd_repr.json
"""
import os, json
import numpy as np
from scipy.stats import rankdata, norm
from sklearn.metrics import roc_auc_score

ROOT = "/PATH/TO/rectal_project"
OUT = f"{ROOT}/experiments/outputs/mdd_repr"
os.makedirs(OUT, exist_ok=True)

def ens(path):
    d = np.load(path, allow_pickle=True)
    y = d["ext_labels"].astype(int)
    p = np.mean([d[f"f{k}"].astype(float) for k in range(5)], axis=0)
    return y, p

y1, p_r50 = ens(f"{ROOT}/experiments/outputs/resnet50_d25_roi/ext_preds.npz")
y2, p_cnx = ens(f"{ROOT}/experiments/outputs/convnext_base_d25_roi/ext_preds.npz")
y3, p_d3 = ens(f"{ROOT}/experiments/outputs/resnet50d3_pt_std/ext_preds_5fold.npz")

# label-identity guards across arms
assert np.array_equal(y1, y2) and np.array_equal(y1, y3), "external label misalignment"
assert len(y1) == 82, len(y1)

a_r50 = roc_auc_score(y1, p_r50)
a_cnx = roc_auc_score(y1, p_cnx)
a_d3 = roc_auc_score(y1, p_d3)
print(f"ensemble AUC anchors: r50={a_r50:.4f} cnx={a_cnx:.4f} d3inf={a_d3:.4f}")
for got, want in ((a_r50, 0.8754), (a_cnx, 0.8706), (a_d3, 0.8056)):
    assert abs(got - want) < 5e-4, f"anchor drift: {got} vs {want}"

def auc_vec(yb, pb):
    r = rankdata(pb, axis=0)
    m = yb.sum(axis=0).astype(float)
    n = yb.shape[0] - m
    posmask = (yb == 1).astype(float)
    return ((r * posmask).sum(axis=0) - m * (m + 1) / 2) / (m * n)

rng = np.random.default_rng(42)
B = 10000
n = len(y1)
idx = rng.integers(0, n, size=(B, n))
yb = y1[idx].T

out = {}
for name, (pa, pb) in {
    "r50_25d_vs_d3inf": (p_r50, p_d3),
    "cnx_25d_vs_d3inf": (p_cnx, p_d3),
}.items():
    d = auc_vec(yb, pa[idx].T) - auc_vec(yb, pb[idx].T)
    se = float(d.std(ddof=1))
    mdd = 2.8016 * se
    point = float(roc_auc_score(y1, pa) - roc_auc_score(y1, pb))
    lo, hi = np.percentile(d, [5, 95])
    out[name] = {"n": n, "point_delta": round(point, 4), "se_boot": round(se, 4),
                 "mdd80": round(mdd, 4), "power_at_delta0.05": round(float(norm.cdf(0.05 / se - 1.959964)), 3),
                 "ci90_boot": [round(float(lo), 4), round(float(hi), 4)]}
    print(name, json.dumps(out[name]), flush=True)

with open(f"{OUT}/mdd_repr.json", "w") as f:
    json.dump(out, f, indent=2)
print("SAVED mdd_repr.json")
