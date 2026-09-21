#!/usr/bin/env python
"""Step1: dataset-wide geometry statistics (physical space)."""
import nibabel as nib
import numpy as np
import glob, json, time
from collections import Counter
from scipy import ndimage

BASE = '/PATH/TO/nnunet_raw/Dataset001_RectalCancer'
OUT = '/PATH/TO/rectal_project/experiments/outputs/geometry'

def process(img_path, mask_path):
    img = nib.load(img_path)
    msk = nib.load(mask_path)
    arr = img.get_fdata()
    m = (msk.get_fdata() > 0).astype(np.uint8)
    sp = np.array(img.header.get_zooms()[:3], dtype=float)
    # tumor volume
    vol_mm3 = m.sum() * np.prod(sp)
    # bbox in voxels
    coords = np.argwhere(m)
    if len(coords) == 0:
        return None
    mn = coords.min(0); mx = coords.max(0)
    bbox_vox = mx - mn + 1
    bbox_mm = bbox_vox * sp
    # max slice (axial) by tumor area
    areas = m.sum(axis=(0, 1))
    max_slice = int(areas.argmax())
    max_area_vox = areas[max_slice]
    max_area_mm2 = max_area_vox * sp[0] * sp[1]
    # number of slices with tumor
    n_slices = int((areas > 0).sum())
    # connectivity: number of connected components
    lbl, ncc = ndimage.label(m)
    return {
        'id': img_path.split('/')[-1].replace('_0000.nii.gz', ''),
        'shape': list(arr.shape), 'spacing': [float(x) for x in sp],
        'vol_mm3': float(vol_mm3), 'bbox_mm': [float(x) for x in bbox_mm],
        'bbox_vox': [int(x) for x in bbox_vox],
        'max_slice': int(max_slice), 'max_area_mm2': float(max_area_mm2),
        'n_tumor_slices': int(n_slices), 'n_cc': int(ncc),
        'voxel_range': [float(arr.min()), float(arr.max())],
    }

def main():
    os.makedirs(OUT, exist_ok=True)
    rows = []
    for split in ['imagesTr', 'imagesTs']:
        files = sorted(glob.glob(f'{BASE}/{split}/*.nii.gz'))
        for f in files:
            m = f.replace('imagesTr', 'labelsTr').replace('imagesTs', 'labelsTs').replace('_0000', '')
            if not glob.glob(m):
                print('MISSING mask', m); continue
            r = process(f, m)
            if r: rows.append(r)
    # save
    with open(f'{OUT}/geometry_stats.json', 'w') as fp:
        json.dump(rows, fp, indent=1)
    # summary
    sp_all = Counter()
    for r in rows: sp_all[tuple(round(x,4) for x in r['spacing'])] += 1
    vols = np.array([r['vol_mm3'] for r in rows])
    bboxes = np.array([r['bbox_mm'] for r in rows])
    print('n cases:', len(rows))
    print('spacings:', sp_all.most_common(12))
    print('tumor vol mm3: median %.0f, p5 %.0f, p95 %.0f' % (np.median(vols), np.percentile(vols,5), np.percentile(vols,95)))
    print('bbox mm X: med %.1f p5 %.1f p95 %.1f' % (np.median(bboxes[:,0]), np.percentile(bboxes[:,0],5), np.percentile(bboxes[:,0],95)))
    print('bbox mm Y: med %.1f p5 %.1f p95 %.1f' % (np.median(bboxes[:,1]), np.percentile(bboxes[:,1],5), np.percentile(bboxes[:,1],95)))
    print('bbox mm Z: med %.1f p5 %.1f p95 %.1f' % (np.median(bboxes[:,2]), np.percentile(bboxes[:,2],5), np.percentile(bboxes[:,2],95)))
    print('max_slice area mm2: med %.0f p95 %.0f' % (np.median([r['max_area_mm2'] for r in rows]), np.percentile([r['max_area_mm2'] for r in rows],95)))
    print('n_tumor_slices: med %d p5 %d p95 %d' % (np.median([r['n_tumor_slices'] for r in rows]), np.percentile([r['n_tumor_slices'] for r in rows],5), np.percentile([r['n_tumor_slices'] for r in rows],95)))
    # RC_A vs RC_B
    a = [r for r in rows if 'imagesTr' in str(r['id']) or True]
    print('sample rows:', rows[0])

if __name__ == '__main__':
    import os
    main()
