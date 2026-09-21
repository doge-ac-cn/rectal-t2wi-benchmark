#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""rad_ens: 嵌套 CV 集成组学模型的共享重建助手.

下游脚本 (扰动鲁棒性 / 和声化 / 厂商留出 / Dice 分层 分析) 用
ensemble_spec.json 里的 (family, model, l1_C, sel_cols) 重建最终集成:
  members = rad_ens.build_members(spec_source, X, y, feat_cols)
  p = rad_ens.predict_ensemble(members, X)
L1 在给定数据上按 spec 的 C 重选 (与 radiomics_ensemble.py final 步骤逐位一致, 只要 X/y/特征列同源).
"""
import os
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
import lightgbm as lgb

ROOT = "/PATH/TO/rectal_project"
SPEC = f"{ROOT}/experiments/outputs/radiomics_ensemble/ensemble_spec.json"
SEED = 42


def _ctors():
    return {
        "LR": {f"LR C={c}": lambda c=c: LogisticRegression(C=c, max_iter=5000, random_state=SEED)
               for c in (0.01, 0.1, 1.0, 10.0)},
        "SVM": {**{f"SVM-linear C={c}": lambda c=c: SVC(kernel="linear", C=c, probability=True, random_state=SEED)
                   for c in (0.01, 1.0, 10.0)},
                **{f"SVM-rbf C={c} gamma={g}": lambda c=c, g=g: SVC(kernel="rbf", C=c, gamma=g, probability=True, random_state=SEED)
                   for c in (0.1, 1.0, 10.0) for g in ("scale", 0.001, 0.01)}},
        "RF": {f"RF n=500 depth={d}": lambda d=d: RandomForestClassifier(n_estimators=500, max_depth=d, random_state=SEED, n_jobs=4)
               for d in (None, 10)},
        "DT": {f"DT depth={d} msl={m}": lambda d=d, m=m: DecisionTreeClassifier(max_depth=d, min_samples_leaf=m, random_state=SEED)
               for d in (3, 5, None) for m in (10, 30)},
        "LGBM": {f"LGBM n=300 lr={r} leaves={n}": lambda r=r, n=n: lgb.LGBMClassifier(
                n_estimators=300, learning_rate=r, num_leaves=n, random_state=SEED, verbosity=-1, n_jobs=4)
                for r in (0.05, 0.1) for n in (15, 31)},
    }


def load_spec(source):
    import json
    sp = json.load(open(SPEC))
    return sp[source]


def fit_l1(Xtr_s, ytr, C):
    l1 = LogisticRegression(penalty="l1", solver="liblinear", C=C, max_iter=5000, random_state=SEED)
    l1.fit(Xtr_s, ytr)
    return np.where(np.abs(l1.coef_[0]) > 1e-10)[0]


def build_members(spec, X, y, feat_cols):
    """按 spec 的 sel_cols/C/model 在给定 (X, y) 上重建集成成员.
    与 radiomics_ensemble.py final 步骤逐位一致 (同一数据时, L1 确定性重选结果相同);
    对不同数据源 (扰动/和声化/厂商臂) 由调用方决定语义:
    本函数只负责 '以 spec 记录的特征列 + 配置' 重训."""
    ctors = _ctors()
    col_ix = {c: i for i, c in enumerate(feat_cols)}
    members = []
    for m in spec["final_members"]:
        missing = [c for c in m["sel_cols"] if c not in col_ix]
        if missing:
            raise KeyError(f"feature cols missing in this data source: {len(missing)} e.g. {missing[:3]}")
        sel = np.array([col_ix[c] for c in m["sel_cols"]])
        pipe = Pipeline([("scale", StandardScaler()), ("clf", ctors[m["family"]][m["model"]]())])
        pipe.fit(X[:, sel], y)
        members.append({"family": m["family"], "model": m["model"], "sel": sel, "pipe": pipe})
    return members


def predict_ensemble(members, X):
    ps = [m["pipe"].predict_proba(X[:, m["sel"]])[:, 1] for m in members]
    return np.mean(ps, axis=0)
