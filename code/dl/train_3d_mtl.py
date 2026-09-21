#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""3D 多任务联合 — 3D U-Net 分割+分类 (验证 "3D 失败主因是监督强度" 假设).

对照基线 (3D-ResNet50 纯分类, 从零训练, 内 0.7801±0.051 / 外 0.7826±0.016).
假设: 3D 纯分类失败机制 = 大容量 3D 编码器 + 无预训练 + 仅体积级弱监督(BCE); 分割稠密监督补足梯度.
实现 (MONAI ResidualUnit, 显式暴露 bottleneck, 避免新版 UNet hook 脆弱):
  - 编码器: 4x ResidualUnit stride=2: 1->16(24³) ->32(12³) ->64(6³) ->128(3³)
  - 分类头: bottleneck(3³,256) GAP -> FC (BCE)   [监督 = 体积级标签]
  - 解码器(轻量): 3³->6³->12³->24³->48³ 上采样+conv, 输出 1 通道 (DiceCE) [监督 = d3_m0_48_mask]
  - 总损失 L = l_cls + lambda_s * l_seg
  - 5 折 CV + RC_B 外部 (外部只评估分类头)
用法:
  python code/train_3d_mtl.py --lambda_s 0.5 --folds 0 1 2 3 4
"""
import argparse, os, sys, json, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
from dltrain.common import (set_seed, load_index, get_splits, external_test_ids,
                            metrics, DATA)
SEED = 42


def dice_ce_loss(seg_logits, mask):
    ce = F.binary_cross_entropy_with_logits(seg_logits, mask.float())
    prob = torch.sigmoid(seg_logits)
    inter = (prob * mask).sum(dim=(2, 3, 4))
    union = prob.sum(dim=(2, 3, 4)) + mask.sum(dim=(2, 3, 4)) + 1e-6
    dice = 1 - (2 * inter / union).mean()
    return ce + dice


class MTL3DDataset(Dataset):
    """返回 (图像, 掩膜, 标签); 图像 d3_m0_48, 掩膜 d3_m0_48_mask (几何对齐)."""

    def __init__(self, ids, variant='d3_m0_48', mask_variant='d3_m0_48_mask',
                 labels=None, aug=False, seed=SEED):
        self.ids = list(ids)
        self.variant = variant
        self.mask_variant = mask_variant
        self.labels = labels if labels is not None else np.zeros(len(ids))
        self.aug = aug
        self.rng = np.random.RandomState(seed)

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i):
        pid = self.ids[i]
        a = np.load(f'{DATA}/crops/{self.variant}/{pid}.npz')['arr_0'].astype(np.float32)
        m = np.load(f'{DATA}/crops/{self.mask_variant}/{pid}.npz')['arr_0'].astype(np.float32)
        a = a[np.newaxis, ...]                  # 1x48x48x48
        m = m[np.newaxis, ...]
        if self.aug:
            k = self.rng.randint(0, 4)
            if k:
                a = np.rot90(a, k, axes=(2, 3)); m = np.rot90(m, k, axes=(2, 3))
            if self.rng.rand() < 0.5:
                a = a[:, :, :, ::-1].copy(); m = m[:, :, :, ::-1].copy()
            if self.rng.rand() < 0.5:
                a = a[:, :, ::-1, :].copy(); m = m[:, :, ::-1, :].copy()
            a = a * float(self.rng.uniform(0.95, 1.05))
        return torch.from_numpy(a.copy()), torch.from_numpy(m.copy()), float(self.labels[i])


class SmallUNet3D(nn.Module):
    """显式 3D 编码器-解码器 (MONAI ResidualUnit). 前向返回 (cls, seg).

    cls_only=True: 去掉解码器与分割头, 纯分类 (同容量对照, 隔离监督强度变量).
    """

    def __init__(self, in_channels=1, n_cls=1, lambda_s=0.5, base=16, cls_only=False):
        super().__init__()
        from monai.networks.blocks import ResidualUnit
        self.lambda_s = lambda_s
        self.cls_only = cls_only
        ch = [base, base * 2, base * 4, base * 8]   # 16,32,64,128
        # encode path (每次 stride=2: 48->24->12->6->3)
        self.enc = nn.ModuleList()
        cin = in_channels
        for c in ch:
            self.enc.append(ResidualUnit(3, cin, c, strides=2, subunits=2))
            cin = c
        # bottleneck
        self.bottom = ResidualUnit(3, ch[-1], ch[-1] * 2, strides=1, subunits=2)  # 128->256 @3³
        # 分类头 (bottleneck GAP)
        self.fc = nn.Linear(ch[-1] * 2, n_cls)
        if not cls_only:
            # decode path (轻量, 上采样后 conv, 不含 skip)
            self.dec = nn.ModuleList()
            cd = ch[-1] * 2
            for c in reversed(ch):
                self.dec.append(nn.Conv3d(cd, c, kernel_size=3, padding=1))
                cd = c
            self.seg_out = nn.Conv3d(ch[0], 1, kernel_size=1)
        else:
            self.dec = nn.ModuleList()
            self.seg_out = None

    def forward(self, x):
        # encode
        for blk in self.enc:
            x = blk(x)
        b = self.bottom(x)                         # 256 @3³
        cls = self.fc(b.mean(dim=(2, 3, 4)))       # B,1
        if self.cls_only:
            return cls, None
        # decode: b -> 48³
        s = b
        for i, conv in enumerate(self.dec):
            s = F.interpolate(s, scale_factor=2, mode='trilinear', align_corners=False)
            s = conv(s)
            s = F.relu(s)
        seg = self.seg_out(s)                      # B,1,48,48,48
        return cls, seg


def train_mtl3d(model, train_ds, val_ds, device, epochs=100, batch=8, lr=3e-4,
                wd=1e-4, log_every=10, patience=20, tag='mtl3d'):
    set_seed()
    train_loader = DataLoader(train_ds, batch_size=batch, shuffle=True, num_workers=4,
                              persistent_workers=True, pin_memory=True, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=batch, shuffle=False, num_workers=4,
                            persistent_workers=True, pin_memory=True)
    labels = np.array([train_ds.labels[i] for i in range(len(train_ds))])
    pw = (labels == 0).sum() / max(1, (labels == 1).sum())
    crit_cls = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pw], device=device))
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    scaler = torch.amp.GradScaler('cuda', enabled=True)
    best_auc, best_state, bad = -1, None, 0
    hist = []
    for ep in range(epochs):
        model.train()
        losses = []
        for x, msk, y in train_loader:
            x, msk, y = x.to(device), msk.to(device), y.to(device).float()
            opt.zero_grad()
            with torch.amp.autocast('cuda', enabled=True):
                cls, seg = model(x)
                l_cls = crit_cls(cls.squeeze(1), y)
                if seg is not None:
                    l_seg = dice_ce_loss(seg, msk)
                    loss = l_cls + model.lambda_s * l_seg
                else:
                    loss = l_cls          # cls_only: 纯分类
            scaler.scale(loss).backward()
            scaler.step(opt); scaler.update()
            losses.append(loss.item())
        sched.step()
        model.eval()
        yv, pv = [], []
        with torch.no_grad():
            for x, msk, y in val_loader:
                x = x.to(device)
                with torch.amp.autocast('cuda', enabled=True):
                    cls, _ = model(x)
                yv.extend(y.tolist()); pv.extend(torch.sigmoid(cls.squeeze(1)).cpu().numpy().tolist())
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--lambda_s', type=float, default=0.5)
    ap.add_argument('--folds', nargs='+', type=int, default=[0, 1, 2, 3, 4])
    ap.add_argument('--epochs', type=int, default=100)
    ap.add_argument('--batch', type=int, default=8)
    ap.add_argument('--lr', type=float, default=3e-4)
    ap.add_argument('--tag', default=None)
    ap.add_argument('--cls_only', action='store_true',
                    help='纯分类对照 (去掉分割头与 seg loss, 同容量)')
    args = ap.parse_args()

    set_seed()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    idx = load_index()
    idx['id'] = idx['id'].astype(str)
    splits = get_splits()
    ext_ids = external_test_ids(idx)
    ext_labels = idx.set_index('id').loc[ext_ids]['label'].values
    tag = args.tag or f'unet3d_mtl_ls{args.lambda_s}'
    if args.cls_only:
        tag = 'unet3d_clsonly'
    outdir = f'/PATH/TO/rectal_project/experiments/outputs/{tag}'
    os.makedirs(outdir, exist_ok=True)

    def _exists(ids, variant):
        return [i for i in ids if os.path.exists(f'{DATA}/crops/{variant}/{i}.npz')]

    fold_results = []
    ext_preds = {}
    for fold in args.folds:
        print(f'===== fold {fold} (lambda_s={args.lambda_s}) =====', flush=True)
        tr_ids = _exists(splits[fold]['train'], 'd3_m0_48')
        va_ids = _exists(splits[fold]['val'], 'd3_m0_48')
        tr_ids = [i for i in tr_ids if os.path.exists(f'{DATA}/crops/d3_m0_48_mask/{i}.npz')]
        va_ids = [i for i in va_ids if os.path.exists(f'{DATA}/crops/d3_m0_48_mask/{i}.npz')]
        label_map = idx.set_index('id')['label']
        train_ds = MTL3DDataset(tr_ids, labels=label_map.loc[tr_ids].values, aug=True)
        val_ds = MTL3DDataset(va_ids, labels=label_map.loc[va_ids].values, aug=False)
        model = SmallUNet3D(lambda_s=args.lambda_s).to(device)
        if args.cls_only:
            model = SmallUNet3D(lambda_s=0.0, cls_only=True).to(device)
        t0 = time.time()
        best_auc, best_state, hist = train_mtl3d(model, train_ds, val_ds, device,
                                                 epochs=args.epochs, batch=args.batch,
                                                 lr=args.lr, tag=tag)
        torch.save(best_state, f'{outdir}/fold{fold}_best.pt')
        with open(f'{outdir}/fold{fold}_hist.json', 'w') as fp:
            json.dump(hist, fp)
        model.load_state_dict(best_state)
        model.eval()
        ext_ids_f = _exists(ext_ids, 'd3_m0_48')
        ext_ids_f = [i for i in ext_ids_f if os.path.exists(f'{DATA}/crops/d3_m0_48_mask/{i}.npz')]
        ext_labels_f = idx.set_index('id').loc[ext_ids_f]['label'].values
        ext_ds = MTL3DDataset(ext_ids_f, labels=ext_labels_f, aug=False)
        pe = []
        with torch.no_grad():
            for x, _, _ in DataLoader(ext_ds, batch_size=8, num_workers=4):
                x = x.to(device)
                with torch.amp.autocast('cuda', enabled=True):
                    cls, _ = model(x)
                pe.extend(torch.sigmoid(cls.squeeze(1)).cpu().numpy().tolist())
        ext_preds[fold] = pe
        ext_m = metrics(ext_labels_f, np.array(pe))
        fold_results.append({'fold': fold, 'val_auc': best_auc, 'time_s': time.time() - t0, 'ext': ext_m})
        print(f'fold{fold} valAUC={best_auc:.4f} extAUC={ext_m["auc"]:.4f} ({time.time()-t0:.0f}s)', flush=True)

    val_aucs = [r['val_auc'] for r in fold_results]
    ext_aucs = [r['ext']['auc'] for r in fold_results]
    summary = {
        'tag': tag, 'lambda_s': args.lambda_s, 'folds': args.folds,
        'val_auc_mean_std': [float(np.mean(val_aucs)), float(np.std(val_aucs))],
        'ext_auc_mean_std': [float(np.mean(ext_aucs)), float(np.std(ext_aucs))],
        'per_fold': fold_results,
    }
    with open(f'{outdir}/summary.json', 'w') as fp:
        json.dump(summary, fp, indent=1)
    np.savez(f'{outdir}/ext_preds.npz', **{f'f{k}': np.array(v) for k, v in ext_preds.items()},
             ext_labels=np.array(ext_labels_f))
    print('SUMMARY', json.dumps(summary, indent=1))


if __name__ == '__main__':
    main()
