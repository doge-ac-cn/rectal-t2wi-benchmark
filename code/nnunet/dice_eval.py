#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Dice 质量评估: nnU-Net auto masks vs manual masks.

- RC_A: labelsTr/{pid}.nii.gz (manual) vs auto_masks/rca/{pid}.nii.gz (predicted on val fold)
- RC_B: rcb_masks_pref/{pid}_mask.nii.gz (manual) vs auto_masks/rcb/{pid}.nii.gz
- 计算体素 Dice (按各自原始网格, 已同空间)
用法: python code/dice_eval.py
输出: experiments/outputs/dice.json
"""
import os, json, glob
import numpy as np
import nibabel as nib

RAW = '/PATH/TO/nnunet_raw/Dataset001_RectalCancer'
ROOT = '/PATH/TO/rectal_project'
AUTO = f'{ROOT}/data_processed/auto_masks'
OUT = f'{ROOT}/experiments/outputs/dice.json'


def load_mask(p):
    m = nib.as_closest_canonical(nib.load(p)).get_fdata()
    return (m > 0).astype(np.uint8)


def dice(a, b):
    inter = (a & b).sum()
    return 2 * inter / (a.sum() + b.sum() + 1e-6)


def eval_group(pids, manual_fn, auto_dir, name):
    dices = []
    per_case = []
    errs = []
    for pid in pids:
        mp = manual_fn(pid)
        ap = os.path.join(auto_dir, f'{pid}.nii.gz')
        if not os.path.exists(ap):
            errs.append(f'{pid}: auto missing')
            continue
        try:
            m = load_mask(mp)
            a = load_mask(ap)
            # 若形状不同(网格差异)跳过并记录
            if m.shape != a.shape:
                errs.append(f'{pid}: shape {m.shape} vs {a.shape}')
                continue
            d = dice(m, a)
            dices.append(d)
            rel_ve = (int(a.sum()) - int(m.sum())) / (int(m.sum()) + 1e-6)
            per_case.append((str(pid), float(d), float(rel_ve)))
        except Exception as e:
            errs.append(f'{pid}: {e}')
    dices = np.array(dices)
    res = {'group': name, 'n': int(len(dices)), 'dice_mean': float(dices.mean()) if len(dices) else None,
           'dice_std': float(dices.std()) if len(dices) else None,
           'dice_median': float(np.median(dices)) if len(dices) else None,
           'dice_ge75': float((dices >= 0.75).mean()) if len(dices) else None,
           'errors': errs[:10], 'n_errors': len(errs)}
    return res, per_case


def main():
    import pandas as pd
    idx = pd.read_csv(f'{ROOT}/data_processed/index.csv')
    idx['id'] = idx['id'].astype(str)
    rca_val_pids = []
    # RC_A val: 逐折 val 并集 (predict 用的正是这些)
    import json as _json
    splits = _json.load(open(f'{RAW}/splits.json'))
    val_set = set()
    for f in splits:
        val_set.update(f['val'])
    rca_val_pids = sorted(p for p in val_set if os.path.exists(
        os.path.join(AUTO, 'rca', f'{p}.nii.gz')))
    rcb_pids = idx[idx['split'] == 'ts']['id'].tolist()

    results = []
    per_case_rows = []
    for pids, mfn, adir, name in [
        (rca_val_pids, lambda p: f'{RAW}/labelsTr/{p}.nii.gz', f'{AUTO}/rca', 'RC_A_val'),
        (rcb_pids, lambda p: f'{RAW}/rcb_masks_pref/{p}_mask.nii.gz', f'{AUTO}/rcb', 'RC_B'),
    ]:
        res, per_case = eval_group(pids, mfn, adir, name)
        results.append(res)
        for pid, d, rve in per_case:
            per_case_rows.append({'dataset': name, 'pid': pid, 'dice': d, 'rel_vol_err': rve})
    for r in results:
        print(json.dumps(r, ensure_ascii=False))
    with open(OUT, 'w') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    pd.DataFrame(per_case_rows).to_csv(
        f'{ROOT}/experiments/outputs/dice_per_patient.csv', index=False)
    print('saved', OUT)


if __name__ == '__main__':
    main()
