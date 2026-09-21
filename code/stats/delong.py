"""DeLong test for paired correlated ROC curves (verified implementation,
extracted verbatim from the analysis codebase used for the manuscript).
Includes the fast algorithm of Sun & Xu 2014.
"""
import numpy as np
from scipy.stats import norm


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
def delong_test(y, p1, p2):
    """return (auc1, auc2, z, p_two_sided)"""
    y = np.asarray(y).astype(int)
    p1 = np.asarray(p1, float); p2 = np.asarray(p2, float)
    pos = y == 1; neg = y == 0
    n1 = int(pos.sum()); n2 = int(neg.sum())
    P = np.vstack([p1, p2])
    r = np.apply_along_axis(_midrank, 1, P)
    rp = np.apply_along_axis(_midrank, 1, P[:, pos])
    rn = np.apply_along_axis(_midrank, 1, P[:, neg])
    aucs = (r[:, pos].sum(1) - n1 * (n1 + 1) / 2.0) / (n1 * n2)
    v10 = (r[:, pos] - rp) / n2      # placement vs negatives, normalized to [0,1]
    v01 = (r[:, neg] - rn) / n1      # placement vs positives, normalized to [0,1]
    S = np.cov(v10) / n1 + np.cov(v01) / n2
    diff = aucs[0] - aucs[1]
    var = S[0, 0] + S[1, 1] - 2 * S[0, 1]
    if var <= 0:
        return aucs[0], aucs[1], float('inf') * np.sign(diff), 0.0 if diff != 0 else 1.0
    z = diff / np.sqrt(var)
    p = 2 * norm.sf(abs(z))
    return float(aucs[0]), float(aucs[1]), float(z), float(p)


# ---------- 组学分数 ----------
