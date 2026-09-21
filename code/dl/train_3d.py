#!/usr/bin/env python
"""Train 3D classifier. Uses MONAI ResNet (spatial_dims=3).
Usage:
  python code/train_3d.py --variant d3_m0_48 --model resnet50 --batch 16
  python code/train_3d.py --variant d3_m10_48 --model medicalnet --weights path.pth
"""
import argparse, os, sys, json, time
import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
from dltrain.common import (set_seed, load_index, get_splits, external_test_ids,
                            Crop3DDataset, metrics, train_loop, DATA)


def make_model_3d(name, n_class=1, pretrained_path=None, n_input=1):
    if name == 'plainconvunet':   # Triad PlainConvEncoder + GAP 分类头
        from dynamic_network_architectures.building_blocks.plain_conv_encoder import PlainConvEncoder
        class PlainCls(nn.Module):
            def __init__(self):
                super().__init__()
                self.encoder = PlainConvEncoder(
                    input_channels=n_input, n_stages=6,
                    features_per_stage=[32, 64, 128, 256, 320, 320],
                    conv_op=nn.Conv3d, kernel_sizes=[(3, 3, 3)] * 6,
                    strides=[(1, 1, 1)] + [(2, 2, 2)] * 5,
                    n_conv_per_stage=[2] * 6, conv_bias=True,
                    norm_op=nn.InstanceNorm3d,
                    norm_op_kwargs={"eps": 1e-5, "affine": True},
                    nonlin=nn.LeakyReLU, nonlin_kwargs={"inplace": True})
                self.head = nn.Linear(320, n_class)
            def forward(self, x):
                return self.head(self.encoder(x).mean(dim=(2, 3, 4)))
        m = PlainCls()
        if pretrained_path and os.path.exists(pretrained_path):
            sd = torch.load(pretrained_path, map_location='cpu')
            if isinstance(sd, dict) and 'state_dict' in sd:
                sd = sd['state_dict']
            sd = {k.replace('module.', ''): v for k, v in sd.items()}
            if all(k.startswith('encoder.') for k in sd):     # encoder-only checkpoint 带 encoder. 壳
                sd = {k[len('encoder.'):]: v for k, v in sd.items()}
            missing, unexpected = m.encoder.load_state_dict(sd, strict=False)
            print(f'plainconvunet triad: missing={len(missing)} unexpected={len(unexpected)}')
            assert len(missing) == 0 and len(unexpected) == 0, \
                f'missing={missing[:5]} unexpected={unexpected[:5]}'
        return m
    if name == 'swinunetr':   # SwinUNETR SSL encoder + GAP 分类头
        from monai.networks.nets import SwinUNETR
        class SwinCls(nn.Module):
            def __init__(self):
                super().__init__()
                self.net = SwinUNETR(in_channels=n_input, out_channels=2,
                                     feature_size=48, spatial_dims=3)
                with torch.no_grad():   # 探测 bottleneck 通道数, 不硬编码
                    f = self.net.swinViT(torch.zeros(1, n_input, 48, 48, 48))[-1]
                    f = f.mean(dim=(2, 3, 4))
                self.head = nn.Linear(f.shape[1], n_class)
            def forward(self, x):
                hs = self.net.swinViT(x)          # 多尺度 [48³,96³,...,bottleneck]
                return self.head(hs[-1].mean(dim=(2, 3, 4)))
        m = SwinCls()
        if pretrained_path and os.path.exists(pretrained_path):
            sd = torch.load(pretrained_path, map_location='cpu')
            if isinstance(sd, dict) and 'state_dict' in sd:
                sd = sd['state_dict']
            sd = {k.replace('module.', ''): v for k, v in sd.items()}   # DataParallel 前缀
            # SSL 预训练任务头(旋转/对比)不属于分割 encoder, 剥离
            sd = {k: v for k, v in sd.items()
                  if not k.startswith(('rotation_head', 'contrastive_head'))}
            # SSL checkpoint 以 encoder 裸前缀存储 -> 补 swinViT. 前缀
            # 旧版 MONAI MLP 命名 mlp.fc1/fc2 -> 现版 mlp.linear1/linear2
            sd = {f'swinViT.' + k.replace('mlp.fc1.', 'mlp.linear1.').replace('mlp.fc2.', 'mlp.linear2.'): v
                  for k, v in sd.items()}
            missing, unexpected = m.net.load_state_dict(sd, strict=False)
            enc_missing = [k for k in missing if k.startswith('swinViT')]
            print(f'swinunetr ssl: encoder keys missing={len(enc_missing)} '
                  f'(decoder randomly init={len(unexpected)}, 分类时不经 decoder)')
            assert len(enc_missing) == 0, f'encoder keys missing: {enc_missing[:5]}'
        return m
    from monai.networks.nets import ResNet, resnet50 as monai_resnet50
    layers = {'resnet10': [1, 1, 1, 1], 'resnet18': [2, 2, 2, 2],
              'resnet34': [3, 4, 6, 3], 'resnet50': [3, 4, 6, 3]}[name]
    # 标准 planes(与 torchvision/MedicalNet 布局一致); MONAI Bottleneck 内部已做 ×4 expansion,
    # resnet50 传 [256,...] 会得到 4 倍宽 737M 网络(容量控制检验已证伪, 2026-09-04 修正)
    inplanes = {'resnet10': [64, 128, 256, 512], 'resnet18': [64, 128, 256, 512],
                'resnet34': [64, 128, 256, 512], 'resnet50': [64, 128, 256, 512]}[name]
    m = ResNet(block='bottleneck' if name == 'resnet50' else 'basic',
               layers=layers, block_inplanes=inplanes, spatial_dims=3,
               n_input_channels=n_input, num_classes=n_class)
    if pretrained_path and os.path.exists(pretrained_path):
        sd = torch.load(pretrained_path, map_location='cpu')
        if isinstance(sd, dict) and 'state_dict' in sd:      # MedicalNet 官方权重: {'state_dict': {...}}
            sd = sd['state_dict']
        sd = {k.replace('module.', ''): v for k, v in sd.items()}   # DataParallel 前缀
        sd = {k: v for k, v in sd.items() if 'fc' not in k and 'classifier' not in k}
        missing, unexpected = m.load_state_dict(sd, strict=False)
        # MONAI downsample conv 无 bias -> 忽略权重里多余的 *.downsample.0.bias
        unexpected = [k for k in unexpected if not k.endswith('downsample.0.bias')]
        print('loaded pretrained; missing:', len(missing), 'unexpected(non-fc/bias):', len(unexpected), unexpected[:4])
        assert len(unexpected) == 0, f'unexpected keys remain: {unexpected}'
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--variant', required=True)
    ap.add_argument('--model', default='resnet50')
    ap.add_argument('--weights', default=None)
    ap.add_argument('--folds', nargs='+', type=int, default=[0, 1, 2, 3, 4])
    ap.add_argument('--epochs', type=int, default=100)
    ap.add_argument('--batch', type=int, default=16)
    ap.add_argument('--lr', type=float, default=3e-4)
    ap.add_argument('--tag', default=None)
    ap.add_argument('--seed', type=int, default=42)
    args = ap.parse_args()

    set_seed(args.seed)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    idx = load_index()
    idx['id'] = idx['id'].astype(str)   # splits.json ids are str
    splits = get_splits()
    ext_ids = external_test_ids(idx)
    ext_labels = idx.set_index('id').loc[ext_ids]['label'].values
    tag = args.tag or f'{args.model}_{args.variant}'
    outdir = f'/PATH/TO/rectal_project/experiments/outputs/{tag}'
    os.makedirs(outdir, exist_ok=True)

    fold_results = []
    ext_preds = {}
    for fold in args.folds:
        print(f'===== fold {fold} =====', flush=True)
        tr_ids = splits[fold]['train']; va_ids = splits[fold]['val']
        label_map = idx.set_index('id')['label']
        train_ds = Crop3DDataset(tr_ids, args.variant, labels=label_map.loc[tr_ids].values, aug=True, seed=args.seed)
        val_ds = Crop3DDataset(va_ids, args.variant, labels=label_map.loc[va_ids].values, aug=False, seed=args.seed)
        model = make_model_3d(args.model, pretrained_path=args.weights).to(device)
        t0 = time.time()
        best_auc, best_state, hist = train_loop(
            model, train_ds, val_ds, device, epochs=args.epochs, batch=args.batch, lr=args.lr,
            patience=20, tag=tag, seed=args.seed)
        torch.save(best_state, f'{outdir}/fold{fold}_best.pt')
        with open(f'{outdir}/fold{fold}_hist.json', 'w') as fp:
            json.dump(hist, fp)
        model.load_state_dict(best_state); model.eval()
        ext_ds = Crop3DDataset(ext_ids, args.variant, labels=ext_labels, aug=False)
        pe = []
        with torch.no_grad():
            for x, _ in torch.utils.data.DataLoader(ext_ds, batch_size=8, num_workers=4):
                x = x.to(device)
                with torch.amp.autocast('cuda', enabled=True):
                    pe.extend(torch.sigmoid(model(x).squeeze(1)).cpu().numpy().tolist())
        ext_preds[fold] = pe
        ext_m = metrics(ext_labels, np.array(pe))
        fold_results.append({'fold': fold, 'val_auc': best_auc, 'time_s': time.time() - t0, 'ext': ext_m})
        print(f'fold{fold} valAUC={best_auc:.4f} extAUC={ext_m["auc"]:.4f} ({time.time()-t0:.0f}s)', flush=True)

    val_aucs = [r['val_auc'] for r in fold_results]
    ext_aucs = [r['ext']['auc'] for r in fold_results]
    summary = {
        'tag': tag, 'variant': args.variant, 'model': args.model, 'folds': args.folds, 'seed': args.seed,
        'val_auc_mean_std': [float(np.mean(val_aucs)), float(np.std(val_aucs))],
        'ext_auc_mean_std': [float(np.mean(ext_aucs)), float(np.std(ext_aucs))],
        'per_fold': fold_results,
    }
    with open(f'{outdir}/summary.json', 'w') as fp:
        json.dump(summary, fp, indent=1)
    np.savez(f'{outdir}/ext_preds.npz', **{f'f{k}': np.array(v) for k, v in ext_preds.items()},
             ext_labels=np.array(ext_labels))
    print('SUMMARY', json.dumps(summary, indent=1))


if __name__ == '__main__':
    main()
