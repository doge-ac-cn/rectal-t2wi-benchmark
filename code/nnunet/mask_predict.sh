#!/usr/bin/env bash
# nnU-Net 预测自动掩膜
# 前置: nnU-Net 5折训练完成 (nnUNet_results/Dataset001_RectalCancer)
# RC_A: fold k 模型只预测 fold k 的 val 子集 (与分类实验同划分 -> 无泄漏)
# RC_B: 5折模型集成 (nnUNetv2_predict -f 0 1 2 3 4 自动平均概率)
# 用法: setsid nohup bash code/mask_predict.sh > logs/mask_predict.log 2>&1 &
set -e
set -o pipefail
export nnUNet_raw="/PATH/TO/nnunet_raw"
export nnUNet_preprocessed="/PATH/TO/dataset/nnUNet_preprocessed"
export nnUNet_results="/PATH/TO/dataset/nnUNet_results"
cd /PATH/TO/rectal_project
AUTO=data_processed/auto_masks
BASE="$nnUNet_raw/Dataset001_RectalCancer"
mkdir -p "$AUTO/rca" "$AUTO/rcb"

echo "===== $(date '+%H:%M') predict RC_A val per fold ====="
for fold in 0 1 2 3 4; do
  tmp="$AUTO/tmp_val_fold${fold}"
  rm -rf "$tmp"; mkdir -p "$tmp"
  python - "$fold" "$tmp" <<'EOF'
import json, os, sys
fold, tmp = sys.argv[1], sys.argv[2]
base='/PATH/TO/nnunet_raw/Dataset001_RectalCancer'
sp = json.load(open(f'{base}/splits.json'))[int(fold)]
n = 0
for pid in sp['val']:
    src = f'{base}/imagesTr/{pid}_0000.nii.gz'
    dst = f'{tmp}/{pid}_0000.nii.gz'
    if not os.path.exists(dst):
        os.symlink(src, dst)
        n += 1
print(f'fold {fold}: symlinked {n} val images')
EOF
  echo "--- $(date '+%H:%M') predict fold $fold val ---"
  nnUNetv2_predict -i "$tmp" -o "$AUTO/rca" -d 1 -c 3d_fullres -tr nnUNetTrainer_250epochs -f "$fold" 2>&1 | tail -5
  rm -rf "$tmp"
done

echo "===== $(date '+%H:%M') predict RC_B (5-fold ensemble) ====="
nnUNetv2_predict -i "$BASE/imagesTs" -o "$AUTO/rcb" -d 1 -c 3d_fullres -tr nnUNetTrainer_250epochs -f 0 1 2 3 4 2>&1 | tail -5

echo "===== $(date '+%H:%M') verify ====="
echo "RC_A masks: $(ls "$AUTO/rca"/*.nii.gz 2>/dev/null | wc -l) / 968"
echo "RC_B masks: $(ls "$AUTO/rcb"/*.nii.gz 2>/dev/null | wc -l) / 82"
echo "ALL MASK PREDICT DONE $(date '+%Y-%m-%d %H:%M')"
