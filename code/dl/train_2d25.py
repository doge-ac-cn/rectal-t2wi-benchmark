#!/usr/bin/env python
"""Train 2D/2.5D classifier (backbone comparison series).
Usage:
  python code/train_2d25.py --variant d2_roi --model resnet50 --folds 0 1 2 3 4
  python code/train_2d25.py --variant d25_rect --model efficientnet_b0 --folds 0
"""
import argparse, os, sys, json, time
import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
from dltrain.common import (set_seed, load_index, get_splits, external_test_ids,
                            Crop2D25Dataset, metrics, train_loop, DATA)

MODELS = {
    # ResNet 规模梯度: 18 11.2M → 152 58.1M (注意无 resnet151, 标准为 152)
    'resnet18': ('tv', 'resnet18'),
    'resnet34': ('tv', 'resnet34'),
    'resnet50': ('tv', 'resnet50'),
    'resnet101': ('tv', 'resnet101'),
    'resnet152': ('tv', 'resnet152'),
    # EfficientNet 规模梯度: b0 5.3M → b7 63.8M
    'efficientnet_b0': ('tv', 'efficientnet_b0'),
    'efficientnet_b1': ('tv', 'efficientnet_b1'),
    'efficientnet_b2': ('tv', 'efficientnet_b2'),
    'efficientnet_b3': ('tv', 'efficientnet_b3'),
    'efficientnet_b4': ('tv', 'efficientnet_b4'),
    'efficientnet_b5': ('tv', 'efficientnet_b5'),
    'efficientnet_b6': ('tv', 'efficientnet_b6'),
    'efficientnet_b7': ('tv', 'efficientnet_b7'),
    # ConvNeXt: tiny 28.6M → base 88.6M → large 196.2M
    'convnext_t': ('tv', 'convnext_tiny'),
    'convnext_base': ('tv', 'convnext_base'),
    'convnext_large': ('tv', 'convnext_large'),
    # Swin: tiny 28.3M
    'swin_t': ('tv', 'swin_t'),
    # ViT: base 86.6M / large 303.3M (vit_b16 即 base, 别名避免混淆)
    'vit_b16': ('tv', 'vit_b_16'),
    'vit_base': ('tv', 'vit_b_16'),
    'vit_large': ('tv', 'vit_l_16'),
    'resnet50_timm': ('timm', 'resnet50'),
    'convnext_t_timm': ('timm', 'convnext_tiny'),
    'swin_t_timm': ('timm', 'swin_tiny_patch4_window7_224'),
    'vit_b16_timm': ('timm', 'vit_base_patch16_224'),
}


def make_model(name, n_class=1, pretrained=True, in_channels=3):
    kind, arch = MODELS[name]
    if kind == 'tv':
        import torchvision.models as tv
        m = getattr(tv, arch)(weights='IMAGENET1K_V1' if pretrained else None)
        if hasattr(m, 'fc'):
            m.fc = nn.Linear(m.fc.in_features, n_class)
        elif hasattr(m, 'head'):
            m.head = nn.Linear(m.head.in_features, n_class)
        elif hasattr(m, 'heads'):
            m.heads.head = nn.Linear(m.heads.head.in_features, n_class)
        elif hasattr(m, 'classifier'):
            # torchvision EfficientNet / ConvNeXt: classifier 是 Sequential, 末层为 Linear
            # 需保留 LayerNorm/Flatten/Dropout 等前序层，仅替换末层 Linear
            if isinstance(m.classifier, nn.Sequential) and len(m.classifier) > 0:
                last = m.classifier[-1]
                m.classifier[-1] = nn.Linear(last.in_features, n_class)
            else:
                m.classifier = nn.Linear(m.classifier.in_features, n_class)
        else:
            raise ValueError(f'unknown head for {arch}')
    else:
        import timm
        m = timm.create_model(arch, pretrained=pretrained, num_classes=n_class)
    # stacking-depth variants (d5/d7 slices): extend conv1; keep ImageNet weights on the
    # first 3 channels (identical to the d25 arm), mean-replicate them onto extra channels.
    if in_channels != 3 and hasattr(m, 'conv1'):
        old = m.conv1
        new_conv = nn.Conv2d(in_channels, old.out_channels, kernel_size=old.kernel_size,
                             stride=old.stride, padding=old.padding,
                             dilation=old.dilation, groups=old.groups, bias=(old.bias is not None))
        with torch.no_grad():
            if in_channels > 3:
                new_conv.weight[:, :3] = old.weight
                new_conv.weight[:, 3:] = old.weight.mean(1, keepdim=True).repeat(1, in_channels - 3, 1, 1)
            elif in_channels < 3:
                # shrink: channel-average the ImageNet RGB weights (grayscale convention)
                new_conv.weight[:, :] = old.weight.mean(1, keepdim=True)
        m.conv1 = new_conv
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--variant', required=True)
    ap.add_argument('--ext_variant', default=None,
                    help='外部测试用变体 (如 cg_d25_roi); 默认与 --variant 相同')
    ap.add_argument('--model', default='resnet50')
    ap.add_argument('--folds', nargs='+', type=int, default=[0, 1, 2, 3, 4])
    ap.add_argument('--epochs', type=int, default=100)
    ap.add_argument('--batch', type=int, default=32)
    ap.add_argument('--lr', type=float, default=3e-4)
    ap.add_argument('--tag', default=None)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--splits', default=None,
                    help='自定义折划分 JSON ([{"train":[ids],"val":[ids]},...]); 默认标准 splits.json')
    ap.add_argument('--target_file', default=None,
                    help='目标评估集 pid 列表文件(每行一个 id); 默认 RC_B ts (ext)')
    args = ap.parse_args()

    set_seed(args.seed)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    idx = load_index()
    idx['id'] = idx['id'].astype(str)   # splits.json ids are str
    if args.splits:
        splits = json.load(open(args.splits))
    else:
        splits = get_splits()
    if args.target_file:
        tgt_ids = [l.strip() for l in open(args.target_file) if l.strip()]
    else:
        tgt_ids = external_test_ids(idx)
    tgt_labels = idx.set_index('id').loc[tgt_ids]['label'].values
    ext_ids = tgt_ids
    ext_labels = tgt_labels

    # 过滤无 crops 文件的病例(one thin-slice case has an empty harmonized mask -> no hist crops; consistent with the 1,049-case cohort)
    def _exists(ids, variant):
        return [i for i in ids if os.path.exists(f'{DATA}/crops/{variant}/{i}.npz')]

    tag = args.tag or f'{args.model}_{args.variant}'
    outdir = f'/PATH/TO/rectal_project/experiments/outputs/{tag}'
    os.makedirs(outdir, exist_ok=True)

    fold_results = []
    ext_preds = {}
    for fold in args.folds:
        print(f'===== fold {fold} =====', flush=True)
        tr_ids = [i for i in splits[fold]['train'] if os.path.exists(f'{DATA}/crops/{args.variant}/{i}.npz')]
        va_ids = [i for i in splits[fold]['val'] if os.path.exists(f'{DATA}/crops/{args.variant}/{i}.npz')]
        label_map = idx.set_index('id')['label']
        train_ds = Crop2D25Dataset(tr_ids, args.variant, labels=label_map.loc[tr_ids].values, aug=True, seed=args.seed)
        val_ds = Crop2D25Dataset(va_ids, args.variant, labels=label_map.loc[va_ids].values, aug=False, seed=args.seed)
        _a = np.load(f'{DATA}/crops/{args.variant}/{tr_ids[0]}.npz')['arr_0']
        in_ch = _a.shape[0] if _a.ndim == 3 else 3
        model = make_model(args.model, in_channels=in_ch).to(device)
        t0 = time.time()
        best_auc, best_state, hist = train_loop(
            model, train_ds, val_ds, device, epochs=args.epochs, batch=args.batch, lr=args.lr,
            patience=20, tag=tag, seed=args.seed)
        torch.save(best_state, f'{outdir}/fold{fold}_best.pt')
        with open(f'{outdir}/fold{fold}_hist.json', 'w') as fp:
            json.dump(hist, fp)
        # external eval
        model.load_state_dict(best_state)
        model.eval()
        # external eval (过滤缺失 crops); 若指定 ext_variant 则外部用转换后变体
        ext_variant = args.ext_variant or args.variant
        ext_ids_f = [i for i in ext_ids if os.path.exists(f'{DATA}/crops/{ext_variant}/{i}.npz')]
        ext_labels_f = idx.set_index('id').loc[ext_ids_f]['label'].values
        ext_ds = Crop2D25Dataset(ext_ids_f, ext_variant, labels=ext_labels_f, aug=False)
        pe = []
        with torch.no_grad():
            for x, _ in torch.utils.data.DataLoader(ext_ds, batch_size=32, num_workers=4):
                x = x.to(device)
                with torch.amp.autocast('cuda', enabled=True):
                    pe.extend(torch.sigmoid(model(x).squeeze(1)).cpu().numpy().tolist())
        ext_preds[fold] = pe
        ext_m = metrics(ext_labels_f, np.array(pe))
        fold_results.append({'fold': fold, 'val_auc': best_auc, 'time_s': time.time() - t0, 'ext': ext_m})
        print(f'fold{fold} valAUC={best_auc:.4f} extAUC={ext_m["auc"]:.4f} ({time.time()-t0:.0f}s)', flush=True)

    # summary
    val_aucs = [r['val_auc'] for r in fold_results]
    ext_aucs = [r['ext']['auc'] for r in fold_results]
    summary = {
        'tag': tag, 'variant': args.variant, 'ext_variant': args.ext_variant or args.variant,
        'model': args.model, 'folds': args.folds, 'seed': args.seed,
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
