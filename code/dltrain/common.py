#!/usr/bin/env python
"""Shared utilities for DL classification experiments.
- Fixed 5-fold splits from splits.json
- Crop dataset loaders (2D/2.5D and 3D)
- Deterministic training loop with AMP, early stopping, per-fold best model
- Metrics: AUC / ACC / SEN / SPE
"""
import os, json, random, glob
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score, accuracy_score, confusion_matrix

SEED = 42
DATA = '/PATH/TO/rectal_project/data_processed'
SPLITS = '/PATH/TO/nnunet_raw/Dataset001_RectalCancer/splits.json'
EXTERNAL_IDS = None  # set from index (split == ts)


def set_seed(seed=SEED):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_index():
    return pd.read_csv(f'{DATA}/index.csv')


def get_splits():
    return json.load(open(SPLITS))


def external_test_ids(idx):
    return idx[idx['split'].astype(str) == 'ts']['id'].tolist()


class Crop2D25Dataset(Dataset):
    """2D (single slice) or 2.5D (3 slices) crops from npz."""
    def __init__(self, ids, variant, labels=None, aug=False, seed=SEED):
        self.ids = list(ids)
        self.variant = variant
        self.labels = labels if labels is not None else np.zeros(len(ids))
        self.aug = aug
        self.rng = np.random.RandomState(seed)

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i):
        pid = self.ids[i]
        a = np.load(f'{DATA}/crops/{self.variant}/{pid}.npz')['arr_0']
        if a.ndim == 2:
            a = np.stack([a] * 3, 0)          # replicate to 3ch (ImageNet stem)
        else:                                  # 3x224x224 already
            a = a.astype(np.float32)
        if self.aug:
            a = self._augment(a)
        return torch.from_numpy(a.copy()), float(self.labels[i])

    def _augment(self, a):
        k = self.rng.randint(0, 4)
        if k: a = np.rot90(a, k, axes=(1, 2))
        if self.rng.rand() < 0.5: a = a[:, :, ::-1].copy()
        if self.rng.rand() < 0.3: a = a[:, ::-1, :].copy()
        a = a * float(self.rng.uniform(0.95, 1.05)) + float(self.rng.uniform(-0.05, 0.05))
        return a


class Crop3DDataset(Dataset):
    def __init__(self, ids, variant, labels=None, aug=False, seed=SEED):
        self.ids = list(ids); self.variant = variant
        self.labels = labels if labels is not None else np.zeros(len(ids))
        self.aug = aug; self.rng = np.random.RandomState(seed)

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i):
        pid = self.ids[i]
        a = np.load(f'{DATA}/crops/{self.variant}/{pid}.npz')['arr_0'].astype(np.float32)
        a = a[np.newaxis, ...]                # 1xNxNxN
        if self.aug:
            k = self.rng.randint(0, 4)
            if k: a = np.rot90(a, k, axes=(1, 2))
            if self.rng.rand() < 0.5: a = a[:, :, :, ::-1].copy()
            if self.rng.rand() < 0.5: a = a[:, :, ::-1, :].copy()
            a = a * float(self.rng.uniform(0.95, 1.05))
        return torch.from_numpy(a.copy()), float(self.labels[i])


def metrics(y, p, thr=0.5):
    auc = roc_auc_score(y, p) if len(set(y)) > 1 else float('nan')
    pred = (p >= thr).astype(int)
    acc = accuracy_score(y, pred)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    sen = tp / (tp + fn) if (tp + fn) else float('nan')
    spe = tn / (tn + fp) if (tn + fp) else float('nan')
    return {'auc': auc, 'acc': acc, 'sen': sen, 'spe': spe}


def train_loop(model, train_ds, val_ds, device, epochs=100, batch=32, lr=3e-4,
               wd=1e-4, pos_weight=None, log_every=20, patience=15, tag='model', seed=SEED,
               clip=None):
    """clip: 可选梯度裁剪 (global norm). 默认 None = 不裁剪 (与既有全部臂行为一致)."""
    set_seed(seed)
    train_loader = DataLoader(train_ds, batch_size=batch, shuffle=True, num_workers=4,
                              persistent_workers=True, pin_memory=True, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=batch, shuffle=False, num_workers=4,
                            persistent_workers=True, pin_memory=True)
    if pos_weight is None:
        labels = np.array([train_ds.labels[i] for i in range(len(train_ds))])
        pw = (labels == 0).sum() / max(1, (labels == 1).sum())
        pos_weight = torch.tensor([pw], device=device)
    crit = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    scaler = torch.amp.GradScaler('cuda', enabled=True)
    best_auc, best_state, bad = -1, None, 0
    hist = []
    for ep in range(epochs):
        model.train()
        losses = []
        for x, y in train_loader:
            x, y = x.to(device), y.to(device).float()
            opt.zero_grad()
            with torch.amp.autocast('cuda', enabled=True):
                out = model(x).squeeze(1)
                loss = crit(out, y)
            scaler.scale(loss).backward()
            if clip is not None:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
            scaler.step(opt); scaler.update()
            losses.append(loss.item())
        sched.step()
        # validation
        model.eval()
        yv, pv = [], []
        with torch.no_grad():
            for x, y in val_loader:
                x = x.to(device)
                with torch.amp.autocast('cuda', enabled=True):
                    out = model(x).squeeze(1)
                yv.extend(y.tolist()); pv.extend(torch.sigmoid(out).cpu().numpy().tolist())
        m = metrics(np.array(yv), np.array(pv))
        hist.append({'epoch': ep, 'loss': float(np.mean(losses)), **{k: float(v) for k, v in m.items()}})
        if ep % log_every == 0:
            print(f'  ep{ep} loss={np.mean(losses):.4f} valAUC={m["auc"]:.4f} acc={m["acc"]:.4f}', flush=True)
        if m['auc'] > best_auc:
            best_auc = m['auc']; best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if bad >= patience:
                print(f'  early stop at ep{ep} bestAUC={best_auc:.4f}')
                break
    return best_auc, best_state, hist
