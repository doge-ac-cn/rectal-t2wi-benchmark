#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""2.5D 多任务联合(分割+分类) MTL 基线.

设计 (预注册):
  - 共享 ResNet50 编码器 (d25_roi 三通道, ImageNet 预训练)
  - 分类头: GAP+FC (BCE)
  - 分割头: layer3+layer4 特征上采样 → 3 通道 224x224 (DiceCE), 监督 d25_roi_mask 三层
  - 总损失 L = lambda_s * L_seg + L_cls
  - 5 折 CV (splits.json) + RC_B 外部; 外部只用分类头 (不用外部掩膜监督)
对照基线 (d25_roi, 内 0.8314±0.024 / 外 0.8515±0.037).
用法:
  python code/train_2d25_mtl.py --lambda_s 0.5 --folds 0 1 2 3 4
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
    """Dice + BCE 组合 (对 3 通道掩膜平均)"""
    ce = F.binary_cross_entropy_with_logits(seg_logits, mask.float())
    prob = torch.sigmoid(seg_logits)
    inter = (prob * mask).sum(dim=(2, 3))
    union = prob.sum(dim=(2, 3)) + mask.sum(dim=(2, 3)) + 1e-6
    dice = 1 - (2 * inter / union).mean()
    return ce + dice


class MTLDataset(Dataset):
    """返回 (图像, 掩膜, 标签); 数据增强同步作用于图像与掩膜."""

    def __init__(self, ids, variant='d25_roi', mask_variant='d25_roi_mask', labels=None, aug=False, seed=SEED):
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
        if a.ndim == 2:
            a = np.stack([a] * 3, 0)
        if self.aug:
            k = self.rng.randint(0, 4)
            if k:
                a = np.rot90(a, k, axes=(1, 2))
                m = np.rot90(m, k, axes=(1, 2))
            if self.rng.rand() < 0.5:
                a = a[:, :, ::-1].copy(); m = m[:, :, ::-1].copy()
            if self.rng.rand() < 0.3:
                a = a[:, ::-1, :].copy(); m = m[:, ::-1, :].copy()
            a = a * float(self.rng.uniform(0.95, 1.05)) + float(self.rng.uniform(-0.05, 0.05))
        return torch.from_numpy(a.copy()), torch.from_numpy(m.copy()), float(self.labels[i])


class MTLResNet50(nn.Module):
    """共享 ResNet50 编码器 + 分类头(BCE) + 分割头(DiceCE)."""

    def __init__(self, n_cls=1, lambda_s=0.5):
        super().__init__()
        import torchvision.models as tv
        m = tv.resnet50(weights='IMAGENET1K_V1')
        self.stem = nn.Sequential(m.conv1, m.bn1, m.relu, m.maxpool)
        self.layer1 = m.layer1
        self.layer2 = m.layer2
        self.layer3 = m.layer3
        self.layer4 = m.layer4
        self.avgpool = m.avgpool
        self.fc = nn.Linear(2048, n_cls)
        # 轻量分割 decoder: layer4(7x7) + layer3(14x14) -> 224x224, 3 通道(3 层掩膜)
        self.seg_reduce = nn.Conv2d(2048, 1024, 1)
        self.seg_conv1 = nn.Conv2d(1024, 512, 1)
        self.seg_conv2 = nn.Conv2d(512, 512, 3, padding=1)
        self.seg_out = nn.Conv2d(512, 3, 1)
        self.lambda_s = lambda_s

    def forward(self, x):
        x = self.stem(x)
        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)
        cls = self.fc(self.avgpool(x4).flatten(1))
        s = F.interpolate(self.seg_reduce(x4), size=(14, 14), mode='bilinear', align_corners=False) + x3
        s = self.seg_conv1(s)
        s = F.interpolate(s, scale_factor=16, mode='bilinear', align_corners=False)
        s = self.seg_conv2(s)
        seg = self.seg_out(s)
        return cls, seg


def train_mtl(model, train_ds, val_ds, device, epochs=100, batch=32, lr=3e-4,
              wd=1e-4, log_every=20, patience=20, tag='mtl'):
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
                l_seg = dice_ce_loss(seg, msk)
                loss = l_cls + model.lambda_s * l_seg
            scaler.scale(loss).backward()
            scaler.step(opt); scaler.update()
            losses.append(loss.item())
        sched.step()
        # validation (主指标: 分类 AUC)
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
    ap.add_argument('--batch', type=int, default=32)
    ap.add_argument('--lr', type=float, default=3e-4)
    ap.add_argument('--tag', default=None)
    args = ap.parse_args()

    set_seed()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    idx = load_index()
    idx['id'] = idx['id'].astype(str)
    splits = get_splits()
    ext_ids = external_test_ids(idx)
    ext_labels = idx.set_index('id').loc[ext_ids]['label'].values
    tag = args.tag or f'resnet50_mtl_ls{args.lambda_s}'
    outdir = f'/PATH/TO/rectal_project/experiments/outputs/{tag}'
    os.makedirs(outdir, exist_ok=True)

    fold_results = []
    ext_preds = {}
    for fold in args.folds:
        print(f'===== fold {fold} (lambda_s={args.lambda_s}) =====', flush=True)
        tr_ids = [i for i in splits[fold]['train'] if os.path.exists(f'{DATA}/crops/d25_roi/{i}.npz')]
        va_ids = [i for i in splits[fold]['val'] if os.path.exists(f'{DATA}/crops/d25_roi/{i}.npz')]
        label_map = idx.set_index('id')['label']
        train_ds = MTLDataset(tr_ids, labels=label_map.loc[tr_ids].values, aug=True)
        val_ds = MTLDataset(va_ids, labels=label_map.loc[va_ids].values, aug=False)
        model = MTLResNet50(lambda_s=args.lambda_s).to(device)
        t0 = time.time()
        best_auc, best_state, hist = train_mtl(model, train_ds, val_ds, device,
                                               epochs=args.epochs, batch=args.batch,
                                               lr=args.lr, tag=tag)
        torch.save(best_state, f'{outdir}/fold{fold}_best.pt')
        with open(f'{outdir}/fold{fold}_hist.json', 'w') as fp:
            json.dump(hist, fp)
        # external eval: 只用分类头
        model.load_state_dict(best_state)
        model.eval()
        ext_ids_f = [i for i in ext_ids if os.path.exists(f'{DATA}/crops/d25_roi/{i}.npz')]
        ext_labels_f = idx.set_index('id').loc[ext_ids_f]['label'].values
        ext_ds = MTLDataset(ext_ids_f, labels=ext_labels_f, aug=False)
        pe = []
        with torch.no_grad():
            for x, _, _ in DataLoader(ext_ds, batch_size=32, num_workers=4):
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
