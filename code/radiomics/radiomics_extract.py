#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""手工掩膜影像组学特征提取（对齐论文口径）
- 重采样: 0.7188 x 0.7188 x 1.5 mm3, 三次B样条(图像) / nearest(掩膜)
- 图像: 全图 z-score 归一化（重采样后）
- 特征: Original + LoG(sigma 1-5mm) + Wavelet(8) 共14个filter，PyRadiomics 标准107特征
- 断点续传: 输出 features_radiomics/{pid}.csv, 已存在则跳过
用法:
  python radiomics_extract.py --pids pids.txt --out features_radiomics --procs 6
"""
import argparse, os, sys, csv, time
import numpy as np
import SimpleITK as sitk

from radiomics import featureextractor

RAW = "/PATH/TO/nnunet_raw/Dataset001_RectalCancer"
IMG_RCA = f"{RAW}/imagesTr"       # {id}_0000.nii.gz
MSK_RCA = f"{RAW}/labelsTr"       # {id}.nii.gz
IMG_RCB = f"{RAW}/imagesTs"       # {id}_0000.nii.gz
MSK_RCB = f"{RAW}/rcb_masks_pref" # {id}_mask.nii.gz

SPACING = [0.7188, 0.7188, 1.5]
LOG_SIGMAS = [1.0, 2.0, 3.0, 4.0, 5.0]


def make_extractor():
    params = {
        "imageType": {
            "Original": {},
            "LoG": {"sigma": LOG_SIGMAS},
            "Wavelet": {},
        },
        "featureClass": {
            "firstorder": [],
            "glcm": [],
            "glrlm": [],
            "glszm": [],
            "gldm": [],
            "ngtdm": [],
            "shape": [],
        },
        "setting": {
            "resampledPixelSpacing": SPACING,
            "interpolator": "sitkBSpline",
            "padDistance": 5,
            "normalize": False,       # 手动 z-score（可复现）
            "binWidth": 25,
            "label": 1,
            "additionalInfo": False,
        },
    }
    return featureextractor.RadiomicsFeatureExtractor(params)


def find_paths(pid):
    """返回 (img_path, msk_path)"""
    img = f"{IMG_RCA}/{pid}_0000.nii.gz"
    msk = f"{MSK_RCA}/{pid}.nii.gz"
    if os.path.exists(img) and os.path.exists(msk):
        return img, msk
    img = f"{IMG_RCB}/{pid}_0000.nii.gz"
    msk = f"{MSK_RCB}/{pid}_mask.nii.gz"
    if os.path.exists(img) and os.path.exists(msk):
        return img, msk
    return None, None


def zscore_image(img_sitk):
    """全图 z-score 归一化（在重采样之后进行）"""
    arr = sitk.GetArrayFromImage(img_sitk).astype(np.float64)
    mu, sd = arr.mean(), arr.std()
    if sd < 1e-8:
        sd = 1.0
    arr = (arr - mu) / sd
    out = sitk.GetImageFromArray(arr)
    out.CopyInformation(img_sitk)
    return out


def extract_one(pid, extractor, outdir):
    out_csv = os.path.join(outdir, f"{pid}.csv")
    if os.path.exists(out_csv):
        return "skip", None
    img_p, msk_p = find_paths(pid)
    if img_p is None:
        return "nopath", None
    t0 = time.time()
    img = sitk.ReadImage(img_p)
    msk = sitk.ReadImage(msk_p)
    # 重采样（图像 BSpline，掩膜 Nearest）由 pyradiomics 内部执行；
    # 但 z-score 需要在重采样后、特征提取前，因此这里手动先重采样
    rf = sitk.ResampleImageFilter()
    rf.SetReferenceImage(img)
    rf.SetOutputSpacing(SPACING)
    rf.SetSize([int(round(sz * sp / ns)) for sz, sp, ns in zip(
        img.GetSize(), img.GetSpacing(), SPACING)])
    # 用 interpolator 设置
    rf.SetInterpolator(sitk.sitkBSpline)
    img_rs = rf.Execute(img)
    img_rs = zscore_image(img_rs)

    # 掩膜重采样（nearest），对齐同一网格
    rf2 = sitk.ResampleImageFilter()
    rf2.SetReferenceImage(img_rs)
    rf2.SetInterpolator(sitk.sitkNearestNeighbor)
    msk_rs = rf2.Execute(msk)
    # 掩膜二值化 label=1
    msk_arr = sitk.GetArrayFromImage(msk_rs)
    msk_arr = (msk_arr >= 1).astype(np.uint8)
    msk_rs = sitk.GetImageFromArray(msk_arr)
    msk_rs.CopyInformation(img_rs)

    # 组学提取（禁用内部重采样，避免二次重采样）
    try:
        res = extractor.execute(img_rs, msk_rs, label=1)
    except Exception as e:
        return "error", str(e)
    feat = {k: float(v) if v is not None else float("nan")
            for k, v in res.items() if k.startswith(("original_", "log-", "wavelet-"))}
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["pid"] + list(feat.keys()))
        w.writerow([pid] + list(feat.values()))
    return "ok", time.time() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pids", required=True)
    ap.add_argument("--out", default="features_radiomics")
    ap.add_argument("--procs", type=int, default=6)
    ap.add_argument("--limit", type=int, default=0, help="调试用：只处理前N例")
    args = ap.parse_args()

    pids = [l.strip() for l in open(args.pids) if l.strip()]
    if args.limit > 0:
        pids = pids[:args.limit]
    os.makedirs(args.out, exist_ok=True)

    # 顺序执行；并行在外部用 --procs 分片调用（每片一个进程）
    extractor = make_extractor()

    stats = {"ok": 0, "skip": 0, "nopath": 0, "error": 0}
    errs = []
    t_all = time.time()
    for pid in pids:
        st, dt = extract_one(pid, extractor, args.out)
        stats[st] = stats.get(st, 0) + 1
        if st == "error":
            errs.append((pid, dt))
        if stats["ok"] % 20 == 0 and stats["ok"] > 0:
            print(f"  ok={stats['ok']} skip={stats['skip']} err={stats['error']} "
                  f"elapsed={time.time()-t_all:.0f}s", flush=True)
    print(f"DONE total={len(pids)} ok={stats['ok']} skip={stats['skip']} "
          f"nopath={stats['nopath']} err={stats['error']} elapsed={time.time()-t_all:.0f}s")
    for pid, e in errs[:10]:
        print(f"  ERR {pid}: {e}")


if __name__ == "__main__":
    main()
