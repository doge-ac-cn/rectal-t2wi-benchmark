#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""分割指标补全 -- Dice + HD95 + ASSD 逐例 (模拟审稿 R1.5/R2.4).

配对: 手动掩膜 (nnUNet_raw labelsTr/Ts) vs nnU-Net 自动掩膜 (data_processed/auto_masks/{rca,rcb}).
几何: canonical 后以手动掩膜 zooms 为物理采样; 形状不一致时自动掩膜最近邻重采样到手动网格
      (与 Dice 评估脚本同约定).
表面距离: 表面体素 = 掩膜 - binary_erosion; 对称表面距离集合 -> HD95 (95 分位) 与 ASSD (均值).
输出: experiments/outputs/hd95/{hd95_per_patient.csv, hd95_summary.json}
"""
import os, glob, time, json, multiprocessing as mp
import argparse
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np
import nibabel as nib
import pandas as pd
from scipy import ndimage

BASE = '/PATH/TO/nnunet_raw/Dataset001_RectalCancer'
PROJ = '/PATH/TO/rectal_project'
AUTO = {'RC_A_val': f'{PROJ}/data_processed/auto_masks/rca', 'RC_B': f'{PROJ}/data_processed/auto_masks/rcb'}
OUTD = f'{PROJ}/experiments/outputs/hd95'
NPROC = 8


def load_mask_canon(path):
    im = nib.as_closest_canonical(nib.load(path))
    return (im.get_fdata() > 0.5).astype(np.uint8), np.array(im.header.get_zooms()[:3], float)


def surface(m):
    er = ndimage.binary_erosion(m.astype(bool), structure=ndimage.generate_binary_structure(3, 1))
    return m.astype(bool) & ~er


def process_case(args):
    pid, dataset, split = args
    try:
        mp_ = glob.glob(f'{BASE}/labelsTr/{pid}.nii.gz') or glob.glob(f'{BASE}/labelsTs/{pid}.nii.gz')
        ap_ = glob.glob(f'{AUTO[dataset]}/{pid}.nii.gz')
        if not mp_ or not ap_:
            return {'id': pid, 'dataset': dataset, 'error': 'mask missing'}
        m, sp = load_mask_canon(mp_[0])
        a, _ = load_mask_canon(ap_[0])
        if m.shape != a.shape:
            a = ndimage.zoom(a.astype(np.float32), np.array(m.shape) / np.array(a.shape), order=0) > 0.5
            a = a.astype(np.uint8)
        inter = int((m & a).sum())
        dice = 2 * inter / max(1, int(m.sum() + a.sum()))
        sm, sa = surface(m), surface(a)
        if sm.sum() == 0 or sa.sum() == 0:
            return {'id': pid, 'dataset': dataset, 'dice': round(dice, 4), 'error': 'empty surface'}
        d_a = ndimage.distance_transform_edt(~sa, sampling=sp)[sm]
        d_m = ndimage.distance_transform_edt(~sm, sampling=sp)[sa]
        dists = np.r_[d_a, d_m]
        return {'id': pid, 'dataset': dataset, 'split': split,
                'dice': round(dice, 4),
                'hd95_mm': round(float(np.percentile(dists, 95)), 3),
                'assd_mm': round(float(dists.mean()), 3),
                'max_sd_mm': round(float(dists.max()), 3)}
    except Exception as e:
        return {'id': pid, 'dataset': dataset, 'error': str(e)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--auto-root-rca', default=None,
                    help='指向 auto_masks_1000ep/rca; 默认旧 auto_masks/rca')
    ap.add_argument('--auto-root-rcb', default=None,
                    help='指向 auto_masks_1000ep/rcb; 默认旧 auto_masks/rcb')
    ap.add_argument('--tag', default='',
                    help='输出子目录后缀, 如 hd95_1000ep')
    ap.add_argument('--dice-csv', default=None,
                    help='与当前掩码链同源的逐病例 Dice csv (列: id,dice), 用于 crosscheck')
    args = ap.parse_args()
    global AUTO, OUTD
    AUTO = {'RC_A_val': args.auto_root_rca or AUTO['RC_A_val'],
            'RC_B': args.auto_root_rcb or AUTO['RC_B']}
    if args.tag:
        OUTD = f"{PROJ}/experiments/outputs/hd95_{args.tag}"
    idx = pd.read_csv(f'{PROJ}/data_processed/index.csv')
    idx['id'] = idx['id'].astype(str)
    tasks = []
    for _, r in idx.iterrows():
        ds = 'RC_A_val' if r['dataset'] == 'RC_A' else 'RC_B'
        tasks.append((str(r['id']), ds, str(r['split'])))
    t0 = time.time()
    with mp.Pool(NPROC) as pool:
        res = pool.map(process_case, tasks)
    df = pd.DataFrame(res)
    errs = df[df['error'].notna()] if 'error' in df else df.iloc[0:0]
    ok = df[df['error'].isna()] if 'error' in df else df
    os.makedirs(OUTD, exist_ok=True)
    df.to_csv(f'{OUTD}/hd95_per_patient.csv', index=False)

    # 与 Dice 评估结果交叉验证 (失败时显式记录原因, 不再静默置 null)
    CX_ERR = None
    try:
        old = pd.read_csv(args.dice_csv) if args.dice_csv else pd.read_csv(f'{PROJ}/data_processed/dice_per_patient.csv')
        old['id'] = old['id'].astype(str)
        mm = ok.merge(old[['id', 'dice']], on='id', suffixes=('', '_diceval'))
        agree = float((mm['dice'] - mm['dice_diceval']).abs().max())
        if agree > 1e-3:
            print(f'[hd95][WARN] crosscheck dice mismatch: maxabsdice={agree} — different mask chain?')
    except FileNotFoundError as e:
        agree, CX_ERR = None, f'dice per-case csv not found ({e.filename}); pass --dice-csv to enable'
    except Exception as e:
        agree, CX_ERR = None, f'crosscheck failed: {type(e).__name__}: {e}'

    summary = {}
    for ds, g in ok.groupby('dataset'):
        summary[ds] = {
            'n': int(len(g)),
            'dice_mean': round(float(g['dice'].mean()), 4),
            'hd95_median': round(float(g['hd95_mm'].median()), 3),
            'hd95_mean': round(float(g['hd95_mm'].mean()), 3),
            'hd95_p75': round(float(g['hd95_mm'].quantile(0.75)), 3),
            'assd_median': round(float(g['assd_mm'].median()), 3),
            'frac_hd95_gt5mm': round(float((g['hd95_mm'] > 5).mean()), 4),
            'frac_hd95_gt10mm': round(float((g['hd95_mm'] > 10).mean()), 4)}
    summary['_crosscheck_maxabsdice_vs_diceval'] = agree
    if CX_ERR:
        summary['_crosscheck_error'] = CX_ERR
    summary['_n_errors'] = int(len(errs))
    json.dump(summary, open(f'{OUTD}/hd95_summary.json', 'w'), indent=1)
    print(json.dumps(summary, indent=1))
    if len(errs):
        print(errs.head(8).to_dict('records'))
    print(f'time {time.time()-t0:.0f}s -> {OUTD}')


if __name__ == '__main__':
    main()
