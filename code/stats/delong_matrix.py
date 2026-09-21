#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""DeLong 配对检验扩展矩阵 (在配对 delong 基础上扩展)
1. DeLong 实现自检 (AUC 与 sklearn 逐位一致 + 恒等守卫 + 零假设均匀性模拟)
2. 主对比的 paired bootstrap 95% CI (DL/组学 掩膜来源, 读 delong/scores.npz)
3. 2.5D vs 3D 家族 两两配对 DeLong (RC_B n=82, 5折 ensemble 概率)
   + Holm 校正; 附 convnext_base vs 3D-scratch 的逐折配对 p
输出: experiments/outputs/delong_matrix/summary.json
"""
import os
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
import json
import numpy as np
from scipy.stats import norm, kstest
from sklearn.metrics import roc_auc_score

ROOT = '/PATH/TO/rectal_project'
OUT = f'{ROOT}/experiments/outputs/delong_matrix'
os.makedirs(OUT, exist_ok=True)
RNG = np.random.default_rng(42)


# ---------- DeLong (Sun & Xu 2014, 同 delong.py) ----------
def _midrank(x):
    J = np.argsort(x, kind='mergesort')
    Z = x[J]; N = len(x); T = np.zeros(N); i = 0
    while i < N:
        j = i
        while j < N and Z[j] == Z[i]:
            j += 1
        T[i:j] = 0.5 * (i + j - 1)
        i = j
    T2 = np.empty(N); T2[J] = T + 1
    return T2


def delong_components(y, scores):
    """多模型共用标签的结构分量: 返回 aucs, v10, v01"""
    y = np.asarray(y).astype(int)
    pos = y == 1; neg = y == 0
    n1, n0 = int(pos.sum()), int(neg.sum())
    P = np.vstack(scores)
    r = np.apply_along_axis(_midrank, 1, P)
    rp = np.apply_along_axis(_midrank, 1, P[:, pos])
    rn = np.apply_along_axis(_midrank, 1, P[:, neg])
    aucs = (r[:, pos].sum(1) - n1 * (n1 + 1) / 2.0) / (n1 * n0)
    v10 = (r[:, pos] - rp) / n0
    v01 = (r[:, neg] - rn) / n1
    return aucs, v10, v01, n1, n0


def delong_test(y, p1, p2):
    aucs, v10, v01, n1, n0 = delong_components(y, [p1, p2])
    S = np.cov(v10) / n1 + np.cov(v01) / n0
    diff = aucs[0] - aucs[1]
    var = S[0, 0] + S[1, 1] - 2 * S[0, 1]
    if var <= 0:
        return float(aucs[0]), float(aucs[1]), float('inf') * np.sign(diff), 0.0 if diff != 0 else 1.0
    z = diff / np.sqrt(var)
    return float(aucs[0]), float(aucs[1]), float(z), float(2 * norm.sf(abs(z)))


# ---------- 自检 ----------
def selftest():
    rep = {}
    # 1) AUC 与 sklearn 逐位一致 (含大量并列分)
    for trial, (n1, n0) in enumerate([(43, 39), (81, 113), (3, 7)]):
        y = np.r_[np.ones(n1, int), np.zeros(n0, int)]
        rng = np.random.default_rng(trial)
        p = np.round(rng.normal(size=len(y)), 1)  # 粗化->大量tie
        a, _, _, _ = delong_test(y, p, p + 1e-9)
        rep[f'auc_match_trial{trial}'] = bool(abs(a - roc_auc_score(y, p)) < 1e-10)
    # 2) 恒等守卫: 同分输入 -> p=1
    y = np.r_[np.ones(20, int), np.zeros(20, int)]
    p = np.random.default_rng(0).normal(size=40)
    _, _, _, p_id = delong_test(y, p, p.copy())
    rep['identical_guard_p_eq_1'] = bool(p_id == 1.0)
    # 3) 零假设均匀性: 两独立随机模型, p 应 ~U(0,1)
    ps = []
    rng = np.random.default_rng(7)
    for _ in range(300):
        y = (rng.random(60) < 0.5).astype(int)
        ps.append(delong_test(y, rng.normal(size=60), rng.normal(size=60))[3])
    rep['null_ks_pvalue'] = float(kstest(ps, 'uniform').pvalue)
    rep['null_pass'] = bool(rep['null_ks_pvalue'] > 0.01)
    return rep


# ---------- paired bootstrap CI ----------
def boot_ci(y, p1, p2, n_boot=10000):
    y = np.asarray(y); p1 = np.asarray(p1); p2 = np.asarray(p2)
    n = len(y); diffs = np.empty(n_boot)
    for b in range(n_boot):
        idx = RNG.integers(0, n, n)
        diffs[b] = roc_auc_score(y[idx], p1[idx]) - roc_auc_score(y[idx], p2[idx])
    return [round(float(np.percentile(diffs, 2.5)), 4),
            round(float(np.percentile(diffs, 97.5)), 4)]


def holm(pvals):
    """Holm-Bonferroni: 返回校正后 p"""
    m = len(pvals); order = np.argsort(pvals); out = np.empty(m)
    mx = 0.0
    for rank, i in enumerate(order):
        adj = (m - rank) * pvals[i]
        mx = max(mx, adj)
        out[i] = min(1.0, mx)
    return [round(float(x), 5) for x in out]


# ---------- 模型分数加载 ----------
def ens(d):
    z = np.load(d)
    labs = z['ext_labels']
    return labs, np.mean([z[f'f{k}'] for k in range(5)], axis=0)


def ens_folds(d):
    """fold 级分数列表 [(f_k prob, labels)]"""
    z = np.load(d)
    return [z[f'f{k}'] for k in range(5)], z['ext_labels']


MODELS = {
    'd25_convnext_base(man)': f'{ROOT}/experiments/outputs/convnext_base_d25_roi/ext_preds.npz',
    'd25_resnet50(man)':      f'{ROOT}/experiments/outputs/resnet50_d25_roi/ext_preds.npz',
    'd3_scratch_m0_48':       f'{ROOT}/experiments/outputs/resnet50d3_scratch_std/ext_preds_5fold.npz',
    'd3_imagenet2d3d':        f'{ROOT}/experiments/outputs/resnet50d3_pt_std/ext_preds_5fold.npz',
    'd3_medicalnet':          f'{ROOT}/experiments/outputs/medicalnet_d3_m0_48/ext_preds.npz',
    'd3_swinunetr_ssl':       f'{ROOT}/experiments/outputs/swinunetr_ssl_d3_m0_48/ext_preds.npz',
    'd3_m10_64':              f'{ROOT}/experiments/outputs/resnet50d3_m10_64/ext_preds_fold{{k}}.npz',
}

scores = {}
labels = {}
for name, path in MODELS.items():
    if '{k}' in path:
        fs, labs = [], None
        for k in range(5):
            z = np.load(path.format(k=k))
            fs.append(z['f' + path.format(k=k).split('fold')[1][0]]
                      if False else z[[x for x in z.files if x != 'ext_labels'][0]])
            labs = z['ext_labels']
        scores[name] = np.mean(fs, axis=0); labels[name] = labs
    else:
        labels[name], scores[name] = ens(path)

# 标签一致性
lab0 = labels['d25_convnext_base(man)']
assert all(np.array_equal(lab0, labels[k]) for k in labels), 'ext labels 不一致!'
y = lab0
assert len(y) == 82 and y.sum() == 43

rep = {'selftest': selftest(), 'note': 'ensemble=5折平均概率; AUC(delong) 可与 per-fold 均值报告略有差异'}

# ---------- 主对比 bootstrap CI (读 delong/scores.npz) ----------
S = np.load(f'{ROOT}/experiments/outputs/delong/scores.npz')
prim = []
for tag, (ya, pa, pb) in {
    'DL man vs auto (ext)':        ('dl_man_ext_y', 'dl_man_ext_p', 'dl_aut_ext_p'),
    'Radio man vs auto (ext)':     ('man_ext_y', 'man_ext_p', 'aut_ext_p'),
    'Radio man vs auto (test194)': ('man_test_y', 'man_test_p', 'aut_test_p'),
}.items():
    yy, a, b = S[ya], S[pa], S[pb]
    z = delong_test(yy, a, b)
    prim.append({'comparison': tag, 'n': int(len(yy)),
                 'auc_a': round(z[0], 4), 'auc_b': round(z[1], 4),
                 'delta': round(z[0] - z[1], 4), 'z': round(z[2], 3),
                 'p_delong': round(z[3], 5),
                 'boot95CI_delta': boot_ci(yy, a, b)})
rep['primary_mask_source'] = prim

# ---------- 2.5D vs 3D 家族两两矩阵 (ext, Holm 校正) ----------
names = list(MODELS.keys())
pairs = []
for i in range(len(names)):
    for j in range(i + 1, len(names)):
        a1, a2, z, p = delong_test(y, scores[names[i]], scores[names[j]])
        pairs.append({'a': names[i], 'b': names[j],
                      'auc_a': round(a1, 4), 'auc_b': round(a2, 4),
                      'delta': round(a1 - a2, 4), 'z': round(z, 3),
                      'p_raw': round(p, 5)})
raw = [x['p_raw'] for x in pairs]
adj = holm(raw)
for x, a_ in zip(pairs, adj):
    x['p_holm'] = a_
rep['family_25d_vs_3d'] = pairs
rep['model_ens_auc'] = {k: round(float(roc_auc_score(y, v)), 4) for k, v in scores.items()}

# ---------- 逐折配对稳健性: convnext_base vs 3D scratch ----------
fp, fl = ens_folds(MODELS['d25_convnext_base(man)'])
fs, _ = ens_folds(MODELS['d3_scratch_m0_48'])
fold_ps = []
for k in range(5):
    fold_ps.append(round(delong_test(fl, fp[k], fs[k])[3], 4))
rep['per_fold_p_d25convnext_vs_d3scratch'] = fold_ps

json.dump(rep, open(f'{OUT}/summary.json', 'w'), indent=1, ensure_ascii=False)

print('selftest:', rep['selftest'])
print('\n=== 主对比 (掩膜来源) ===')
for t in prim:
    print(f"{t['comparison']}: {t['auc_a']} vs {t['auc_b']} Δ={t['delta']} "
          f"z={t['z']} p={t['p_delong']} boot95%CI={t['boot95CI_delta']}")
print('\n=== 家族矩阵 (ext n=82, ensemble) ===')
print('ens AUC:', rep['model_ens_auc'])
for x in pairs:
    print(f"{x['a']} vs {x['b']}: {x['delta']:+.4f} p_raw={x['p_raw']} p_holm={x['p_holm']}")
print('\n逐折配对 p (d25convnext vs d3scratch):', fold_ps)
print('\nsaved', f'{OUT}/summary.json')
