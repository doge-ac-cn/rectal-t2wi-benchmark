#!/usr/bin/env python
"""Step3: QC validation of generated crops + montage visualization."""
import os, glob, json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT = '/PATH/TO/rectal_project/data_processed'
QC = '/PATH/TO/rectal_project/experiments/outputs/geometry'

def main():
    os.makedirs(QC, exist_ok=True)
    idx = pd.read_csv(f'{OUT}/index.csv')
    print('index rows:', len(idx))
    variants = sorted([os.path.basename(p) for p in glob.glob(f'{OUT}/crops/*')])
    print('variants:', variants)
    # 1) load all d2_roi and compute stats (min/max/mean/std, zeros frac)
    stats = {}
    for v in ['d2_roi', 'd2_rect', 'd25_roi', 'd25_rect', 'd3_m0_48', 'd3_m10_48']:
        files = sorted(glob.glob(f'{OUT}/crops/{v}/*.npz'))
        vals = []
        for f in files[:200]:
            a = np.load(f)['arr_0']
            vals.append({'shape': list(a.shape), 'min': float(a.min()), 'max': float(a.max()),
                        'mean': float(a.mean()), 'std': float(a.std()), 'zeros': float((a == 0).mean())})
        stats[v] = vals
    with open(f'{QC}/crop_stats.json', 'w') as fp:
        json.dump({k: v for k, v in stats.items()}, fp)
    for v, vals in stats.items():
        print(f'{v}: n={len(vals)} shape={vals[0]["shape"]} mean={np.mean([x["mean"] for x in vals]):.3f} '
              f'std={np.mean([x["std"] for x in vals]):.3f} zeros%={np.mean([x["zeros"] for x in vals])*100:.1f}')
    # 2) montage: 8 cases x key variants
    pids = idx['id'].tolist()
    rng = np.random.RandomState(0)
    sample = rng.choice(pids, 8, replace=False)
    fig, axes = plt.subplots(8, 5, figsize=(15, 20))
    for i, pid in enumerate(sample):
        row = idx[idx['id'] == pid].iloc[0]
        d2 = np.load(f"{OUT}/crops/d2_roi/{pid}.npz")['arr_0']
        d2r = np.load(f"{OUT}/crops/d2_rect/{pid}.npz")['arr_0']
        d25 = np.load(f"{OUT}/crops/d25_roi/{pid}.npz")['arr_0']
        d3 = np.load(f"{OUT}/crops/d3_m0_48/{pid}.npz")['arr_0']
        d3e = np.load(f"{OUT}/crops/d3_m10_48/{pid}.npz")['arr_0']
        sl = d3.shape[0] // 2
        axes[i, 0].imshow(d2, cmap='gray'); axes[i, 0].set_title(f'{pid} label={row["label"]}')
        axes[i, 1].imshow(d2r, cmap='gray'); axes[i, 1].set_title('d2_rect')
        axes[i, 2].imshow(d25[1], cmap='gray'); axes[i, 2].set_title('d25_roi mid')
        axes[i, 3].imshow(d3[sl], cmap='gray'); axes[i, 3].set_title('d3_m0_48 mid')
        axes[i, 4].imshow(d3e[sl], cmap='gray'); axes[i, 4].set_title('d3_m10_48 mid')
        for ax in axes[i]:
            ax.axis('off')
    plt.tight_layout()
    plt.savefig(f'{QC}/montage.png', dpi=100)
    print('saved', f'{QC}/montage.png')

if __name__ == '__main__':
    main()
