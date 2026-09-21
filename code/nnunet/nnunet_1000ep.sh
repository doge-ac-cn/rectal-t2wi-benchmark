#!/usr/bin/env bash
# nnU-Net 默认训练器 (1000 epochs) 补全 + 推理 + Dice 评估
# 背景: 当前 auto mask 来自 nnUNetTrainer_250epochs; 默认 1000 轮当时只跑到 fold0≈84/fold2≈97 epoch.
#       本链续训默认 trainer 5 折 (自动从 checkpoint_latest 续), 推理到独立目录 (不覆盖 250ep 产物).
# 用法: setsid nohup bash code/nnunet_1000ep.sh > logs/nnunet1000.log 2>&1 < /dev/null &
set -e
set -o pipefail
export nnUNet_raw="/PATH/TO/nnunet_raw"
export nnUNet_preprocessed="/PATH/TO/dataset/nnUNet_preprocessed"
export nnUNet_results="/PATH/TO/dataset/nnUNet_results"
export ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS=6
export OMP_NUM_THREADS=6
cd /PATH/TO/rectal_project
AUTO=data_processed/auto_masks_1000ep
BASE="$nnUNet_raw/Dataset001_RectalCancer"
mkdir -p "$AUTO/rca" "$AUTO/rcb"

for fold in 0 1 2 3 4; do
  echo "===== train fold $fold (default trainer, 1000 epochs, auto-resume) $(date +%H:%M:%S) ====="
  nnUNetv2_train 1 3d_fullres "$fold" -device cuda > logs/train_fold${fold}.log 2>&1
  echo "fold $fold DONE $(date +%H:%M:%S)"
done

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
  nnUNetv2_predict -i "$tmp" -o "$AUTO/rca" -d 1 -c 3d_fullres -f "$fold" 2>&1 | tail -3
  rm -rf "$tmp"
done

echo "===== $(date '+%H:%M') predict RC_B (5-fold ensemble) ====="
nnUNetv2_predict -i "$BASE/imagesTs" -o "$AUTO/rcb" -d 1 -c 3d_fullres -f 0 1 2 3 4 2>&1 | tail -3
echo "RC_A masks: $(ls "$AUTO/rca"/*.nii.gz 2>/dev/null | wc -l) / 968"
echo "RC_B masks: $(ls "$AUTO/rcb"/*.nii.gz 2>/dev/null | wc -l) / 82"

echo "===== $(date '+%H:%M') Dice eval ====="
python code/dice_eval.py --auto data_processed/auto_masks_1000ep --out experiments/outputs/nnunet_1000ep/dice_summary.json
echo "ALL DONE $(date)"
