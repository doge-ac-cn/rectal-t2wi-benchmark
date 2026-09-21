#!/usr/bin/env python
"""Step2 (parallel): dataset generation — unified resampling + crop datasets.

Array convention: NIfTI canonical (RAS) -> arr[x, y, z]; slice axis = z (axis 2).
Crops:
  d2_roi / d2_rect      : 2D max-area slice -> 224x224
  d25_roi / d25_rect    : max slice +-1 (3ch) -> 3x224x224
  d3_m{E}_N             : 3D fit-to-cube, dilation E mm (0/5/10), grid N^3 (32/48/64)
Intensity: per-volume z-score after 0.5-99.5 pct clip.
"""
import os, glob, time, json, multiprocessing as mp
import numpy as np
import nibabel as nib
import pandas as pd
from scipy import ndimage
from skimage.transform import resize

BASE = '/PATH/TO/nnunet_raw/Dataset001_RectalCancer'
STAGE = f'{BASE}/staging_table.xlsx'
OUT = '/PATH/TO/rectal_project/data_processed'
TARGET = (0.7188, 0.7188, 1.5)
RECT_MM = 64.0
MARGIN_PX = 1
N2D = 224
GRID3D = (32, 48, 64)
DILATIONS = (0, 5, 10)


def resample(arr, m, sp, target):
    if np.allclose(sp, target):
        return arr, m
    factors = np.array(sp, float) / np.array(target, float)
    arr2 = ndimage.zoom(arr, factors, order=3, prefilter=True)
    m2 = ndimage.zoom(m.astype(np.float32), factors, order=0)
    return arr2.astype(np.float32), (m2 > 0.5).astype(np.uint8)


def load_resampled(img_path, mask_path):
    img = nib.as_closest_canonical(nib.load(img_path))
    msk = nib.as_closest_canonical(nib.load(mask_path))
    arr = img.get_fdata(dtype=np.float32)
    m = (msk.get_fdata() > 0).astype(np.uint8)
    sp = np.array(img.header.get_zooms()[:3], float)
    if arr.shape != m.shape:
        m = ndimage.zoom(m.astype(np.float32), np.array(m.shape) / np.array(arr.shape), order=0) > 0.5
        m = m.astype(np.uint8)
    arr, m = resample(arr, m, sp, TARGET)
    mask_valid = arr > 0
    lo, hi = np.percentile(arr[mask_valid], [0.5, 99.5]) if mask_valid.any() else (0, 1)
    arr_n = np.clip(arr, lo, hi)
    mu, sd = arr_n[mask_valid].mean(), arr_n[mask_valid].std()
    arr_n = (arr_n - mu) / (sd + 1e-8)
    arr_n[~mask_valid] = 0
    return arr_n.astype(np.float32), m


def dilate_mask(m, dil_mm):
    """按物理mm外扩: 欧氏距离变换(采样=体素物理尺寸), dist<=mm 即外扩. O(n)快速, 物理精确."""
    if dil_mm == 0:
        return m
    dist = ndimage.distance_transform_edt(m == 0, sampling=TARGET)
    return (dist <= dil_mm).astype(np.uint8)


def crop2d_slice(slc, bbox, rect=False, out=N2D):
    side = int(round(RECT_MM / TARGET[0]))
    if rect:
        cx, cy = (bbox[0] + bbox[1]) // 2, (bbox[2] + bbox[3]) // 2
        x0, x1 = max(0, cx - side // 2), min(slc.shape[0], cx + (side - side // 2))
        y0, y1 = max(0, cy - side // 2), min(slc.shape[1], cy + (side - side // 2))
    else:
        x0, x1 = max(0, bbox[0] - MARGIN_PX), min(slc.shape[0], bbox[1] + MARGIN_PX)
        y0, y1 = max(0, bbox[2] - MARGIN_PX), min(slc.shape[1], bbox[3] + MARGIN_PX)
    crop = slc[x0:x1, y0:y1]
    if crop.size == 0:
        return None
    return resize(crop, (out, out), order=1, anti_aliasing=False, preserve_range=True).astype(np.float32)


def crop3d(arr, mk, N):
    coords = np.argwhere(mk)
    if len(coords) == 0:
        return None
    mn = coords.min(0); mx = coords.max(0)
    side = int(max(mx - mn + 1))
    side = max(side, 8)
    center = (mn + mx) // 2
    half = side // 2
    x0, x1 = center[0] - half, center[0] + (side - half)
    y0, y1 = center[1] - half, center[1] + (side - half)
    z0, z1 = center[2] - half, center[2] + (side - half)
    pad_x0 = max(0, -x0); pad_x1 = max(0, x1 - arr.shape[0])
    pad_y0 = max(0, -y0); pad_y1 = max(0, y1 - arr.shape[1])
    pad_z0 = max(0, -z0); pad_z1 = max(0, z1 - arr.shape[2])
    x0, x1 = max(0, x0), min(arr.shape[0], x1)
    y0, y1 = max(0, y0), min(arr.shape[1], y1)
    z0, z1 = max(0, z0), min(arr.shape[2], z1)
    crop = np.zeros((side, side, side), np.float32)
    crop[pad_x0:side - pad_x1, pad_y0:side - pad_y1, pad_z0:side - pad_z1] = arr[x0:x1, y0:y1, z0:z1]
    return resize(crop, (N, N, N), order=1, anti_aliasing=False, preserve_range=True).astype(np.float32)


def process_case(args):
    pid, label, split, dataset, pT, imgp, maskp = args
    # 断点续传: 13个变体全部存在则跳过
    variant_names = (['d2_roi', 'd2_rect', 'd25_roi', 'd25_rect'] +
                     [f'd3_m{d}_{n}' for d in DILATIONS for n in GRID3D])
    if all(os.path.exists(f'{OUT}/crops/{v}/{pid}.npz') for v in variant_names):
        return {'id': pid, 'skipped': True, 'label': int(label), 'split': str(split),
                'dataset': str(dataset), 'pT': str(pT)}
    try:
        arr, m = load_resampled(imgp, maskp)
        if int(m.sum()) == 0:
            return {'id': pid, 'error': 'empty mask'}
        areas = m.sum(axis=(0, 1))
        smax = int(areas.argmax())
        coords = np.argwhere(m[:, :, smax] > 0)
        bbox2d = (int(coords[:, 0].min()), int(coords[:, 0].max()) + 1,
                  int(coords[:, 1].min()), int(coords[:, 1].max()) + 1)
        variants = {}
        for name, rect, nch in [('d2_roi', False, 1), ('d2_rect', True, 1),
                                ('d25_roi', False, 3), ('d25_rect', True, 3)]:
            if nch == 1:
                c = crop2d_slice(arr[:, :, smax], bbox2d, rect=rect)
                if c is None: return {'id': pid, 'error': f'{name} crop failed'}
                variants[name] = c
            else:
                chans = []
                for s in range(smax - 1, smax + 2):
                    if s < 0 or s >= arr.shape[2]:
                        chans.append(np.zeros((N2D, N2D), np.float32))
                    else:
                        c = crop2d_slice(arr[:, :, s], bbox2d, rect=rect)
                        if c is None: return {'id': pid, 'error': f'{name} crop failed'}
                        chans.append(c)
                variants[name] = np.stack(chans, 0)
        masks = {d: (m if d == 0 else dilate_mask(m, d)) for d in DILATIONS}
        for dil in DILATIONS:
            for N in GRID3D:
                c = crop3d(arr, masks[dil], N)
                if c is None: return {'id': pid, 'error': f'd3_m{dil}_{N} crop failed'}
                variants[f'd3_m{dil}_{N}'] = c
        rec = {'id': pid, 'label': int(label), 'split': str(split), 'dataset': str(dataset),
               'pT': str(pT), 'max_slice_z': int(smax), 'bbox2d_xy': list(bbox2d),
               'tumor_vol_mm3': float(m.sum() * np.prod(TARGET)),
               'error': None}
        for name, c in variants.items():
            d = f'{OUT}/crops/{name}'
            os.makedirs(d, exist_ok=True)
            np.savez_compressed(f'{d}/{pid}.npz', c)
            rec[f'{name}_shape'] = list(c.shape)
        return rec
    except Exception as e:
        return {'id': pid, 'error': str(e)}


def main():
    os.makedirs(OUT, exist_ok=True)
    st = pd.read_excel(STAGE)
    st = st[st['eligible'] == 1].copy() if 'eligible' in st.columns else st
    tasks = []
    for _, r in st.iterrows():
        pid = str(r['ID'])
        imgp = glob.glob(f'{BASE}/imagesTr/{pid}_*.nii.gz') or glob.glob(f'{BASE}/imagesTs/{pid}_*.nii.gz')
        maskp = glob.glob(f'{BASE}/labelsTr/{pid}.nii.gz') or glob.glob(f'{BASE}/labelsTs/{pid}.nii.gz')
        if not imgp or not maskp:
            continue
        tasks.append((pid, r['label'], r['split'], r.get('dataset', ''), r.get('pT_Stage', ''), imgp[0], maskp[0]))
    print('tasks:', len(tasks))
    nproc = min(16, mp.cpu_count())
    t0 = time.time()
    results = []
    with mp.Pool(nproc) as pool:
        for i, res in enumerate(pool.imap_unordered(process_case, tasks, chunksize=8)):
            results.append(res)
            if (i + 1) % 50 == 0:
                print(f'{i+1}/{len(tasks)} done, {time.time()-t0:.0f}s', flush=True)
    errs = [r for r in results if r.get('error')]
    ok = [r for r in results if not r.get('error')]
    skipped = [r for r in results if r.get('skipped')]
    print('OK', len(ok), 'skipped', len(skipped), 'errors', len(errs))
    if errs:
        with open(f'{OUT}/qc_errors.json', 'w') as fp:
            json.dump(errs, fp, indent=1)
        print('errors sample:', errs[:5])
    idx = pd.DataFrame(ok + skipped)
    idx.to_csv(f'{OUT}/index.csv', index=False)
    print('index rows:', len(idx))
    print('label:'); print(idx['label'].value_counts().sort_index())
    print('split:'); print(idx['split'].value_counts())
    print('total time %.0fs' % (time.time() - t0))


if __name__ == '__main__':
    main()
