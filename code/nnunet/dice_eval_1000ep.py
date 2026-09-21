#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Dice 评估 (参数化版, 复用 dice_eval 逻辑).
用法: python dice_eval.py --auto data_processed/auto_masks_1000ep --out experiments/outputs/nnunet_1000ep/dice_summary.json
"""
import os, json, glob, argparse
import numpy as np
import nibabel as nib

RAW = '/PATH/TO/nnunet_raw/Dataset001_RectalCancer'
ROOT = '/PATH/TO/rectal_project'


def load_mask(p):
    m = nib.as_closest_canonical(nib.load(p)).get_fdata()
    return (m > 0).astype(np.uint8)


def dice(a, b):
    inter = (a & b).sum()
    return 2 * inter / (a.sum() + b.sum() + 1e-6)


def eval_group(pids, manual_fn, auto_dir, name):
    dices, per_case, errs = [], [], []
    for pid in pids:
        mp = manual_fn(pid)
        ap = os.path.join(auto_dir, f'{pid}.nii.gz')
        if not os.path.exists(ap):
            errs.append(f'{pid}: auto missing')
            continue
        try:
            m = load_mask(mp)
            a = load_mask(ap)
            if m.shape != a.shape:
                errs.append(f'{pid}: shape {m.shape} vs {a.shape}')
                continue
            d = dice(m, a)
            dices.append(d)
            rel_ve = (int(a.sum()) - int(m.sum())) / (int(m.sum()) + 1e-6)
            is_empty = int(a.sum()) == 0  # empty prediction ⇔ rel_vol_err = -1
            per_case.append((str(pid), float(d), float(rel_ve), is_empty))
        except Exception as e:
            errs.append(f'{pid}: {e}')
    dices = np.array(dices)
    res = {'group': name, 'n': int(len(dices)),
           'dice_mean': float(dices.mean()) if len(dices) else None,
           'dice_std': float(dices.std()) if len(dices) else None,
           'dice_median': float(np.median(dices)) if len(dices) else None,
           'dice_ge75': float((dices >= 0.75).mean()) if len(dices) else None,
           'errors': errs[:10], 'n_errors': len(errs)}
    return res, per_case


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--auto', required=True)
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    AUTO = os.path.join(ROOT, a.auto)
    import pandas as pd
    idx = pd.read_csv(f'{ROOT}/data_processed/index.csv')
    idx['id'] = idx['id'].astype(str)
    splits = json.load(open(f'{RAW}/splits.json'))
    val_set = set()
    for f in splits:
        val_set.update(f['val'])
    rca_val_pids = sorted(p for p in val_set if os.path.exists(os.path.join(AUTO, 'rca', f'{p}.nii.gz')))
    rcb_pids = idx[idx['split'] == 'ts']['id'].tolist()
    results, per_case_rows = [], []
    for pids, mfn, adir, name in [
        (rca_val_pids, lambda p: f'{RAW}/labelsTr/{p}.nii.gz', f'{AUTO}/rca', 'RC_A_val'),
        (rcb_pids, lambda p: f'{RAW}/rcb_masks_pref/{p}_mask.nii.gz', f'{AUTO}/rcb', 'RC_B'),
    ]:
        res, per_case = eval_group(pids, mfn, adir, name)
        results.append(res)
        for pid, d, rve, is_empty in per_case:
            per_case_rows.append({'dataset': name, 'pid': pid, 'dice': d, 'rel_vol_err': rve, 'is_empty': is_empty})
    os.makedirs(os.path.dirname(os.path.join(ROOT, a.out)), exist_ok=True)
    with open(os.path.join(ROOT, a.out), 'w') as f:
        json.dump({'summary': results, 'per_case': per_case_rows}, f, indent=1)
    print(json.dumps(results, indent=1))
    # CSV 副本
    import csv
    csvp = os.path.join(ROOT, a.out.replace('.json', '_per_patient.csv'))
    with open(csvp, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['dataset', 'pid', 'dice', 'rel_vol_err', 'is_empty'])
        w.writeheader()
        for r in per_case_rows:
            w.writerow(r)
    print('SAVED', a.out)


if __name__ == '__main__':
    main()
