#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""用 nnU-Net 自动掩膜提取影像组学特征 (全自动管线)
- 与手工管线完全相同的提取口径: 重采样 0.7188x0.7188x1.5mm3(BSpline/Nearest),
  全图 z-score, binWidth=25, Original+LoG+Wavelet 1316 特征
- 掩膜来源: data_processed/auto_masks/rca|rcb/{pid}.nii.gz
- 输出: features_radiomics_auto/{pid}.csv (断点续传, 已存在则跳过)
用法:
  python code/radiomics_auto.py --pids pids.txt --out features_radiomics_auto --procs 6
"""
import argparse, os, sys, csv, time
import numpy as np
import SimpleITK as sitk

from radiomics import featureextractor

RAW = "/PATH/TO/nnunet_raw/Dataset001_RectalCancer"
AUTO = "/PATH/TO/rectal_project/data_processed/auto_masks"
IMG_RCA = f"{RAW}/imagesTr"   # {id}_0000.nii.gz
IMG_RCB = f"{RAW}/imagesTs"   # {id}_0000.nii.gz
MSK_AUTO_RCA = f"{AUTO}/rca"  # {id}.nii.gz
MSK_AUTO_RCB = f"{AUTO}/rcb"  # {id}.nii.gz

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
            "normalize": False,
            "binWidth": 25,
            "label": 1,
            "additionalInfo": False,
        },
    }
    return featureextractor.RadiomicsFeatureExtractor(params)


def find_paths(pid):
    img = f"{IMG_RCA}/{pid}_0000.nii.gz"
    msk = f"{MSK_AUTO_RCA}/{pid}.nii.gz"
    if os.path.exists(img) and os.path.exists(msk):
        return img, msk
    img = f"{IMG_RCB}/{pid}_0000.nii.gz"
    msk = f"{MSK_AUTO_RCB}/{pid}.nii.gz"
    if os.path.exists(img) and os.path.exists(msk):
        return img, msk
    return None, None


def zscore_image(img_sitk):
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
    rf = sitk.ResampleImageFilter()
    rf.SetReferenceImage(img)
    rf.SetOutputSpacing(SPACING)
    rf.SetSize([int(round(sz * sp / ns)) for sz, sp, ns in zip(
        img.GetSize(), img.GetSpacing(), SPACING)])
    rf.SetInterpolator(sitk.sitkBSpline)
    img_rs = rf.Execute(img)
    img_rs = zscore_image(img_rs)

    rf2 = sitk.ResampleImageFilter()
    rf2.SetReferenceImage(img_rs)
    rf2.SetInterpolator(sitk.sitkNearestNeighbor)
    msk_rs = rf2.Execute(msk)
    msk_arr = sitk.GetArrayFromImage(msk_rs)
    msk_arr = (msk_arr >= 1).astype(np.uint8)
    msk_rs = sitk.GetImageFromArray(msk_arr)
    msk_rs.CopyInformation(img_rs)

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
    ap.add_argument("--out", default="features_radiomics_auto")
    ap.add_argument("--procs", type=int, default=6)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--auto-root", default=AUTO,
                    help="掩膜根目录 (含 rca/ rcb/ 子目录); auto1000 版指向 auto_masks_1000ep")
    args = ap.parse_args()

    # 掩膜根目录参数化 (模块级常量在此重设, find_paths 运行时读取)
    global MSK_AUTO_RCA, MSK_AUTO_RCB
    MSK_AUTO_RCA = f"{args.auto_root}/rca"
    MSK_AUTO_RCB = f"{args.auto_root}/rcb"

    pids = [l.strip() for l in open(args.pids) if l.strip()]
    if args.limit > 0:
        pids = pids[:args.limit]
    os.makedirs(args.out, exist_ok=True)

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
