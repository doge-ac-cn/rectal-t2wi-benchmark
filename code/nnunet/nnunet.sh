#!/usr/bin/env bash
# nnU-Net 自动分割 (Dataset001_RectalCancer, T2WI -> tumor mask)
# v2 (2026-09-02 10:00):
#   - 去掉坏掉的 plans_from_paper 行 (module path 调用失败, 由 plan_and_preprocess 完成 planning)
#   - 复用项目 5 折 splits.json -> splits_final.json (与分类实验同划分, 防泄漏)
#   - fold 逐个训练 + 独立日志 + pipefail (fold 失败即停)
# 用法: setsid nohup bash code/nnunet.sh > logs/run.log 2>&1 &
set -e
set -o pipefail
export nnUNet_raw="/PATH/TO/nnunet_raw"
export nnUNet_preprocessed="/PATH/TO/dataset/nnUNet_preprocessed"
export nnUNet_results="/PATH/TO/dataset/nnUNet_results"
mkdir -p "$nnUNet_preprocessed" "$nnUNet_results"
cd /PATH/TO/rectal_project

# 限制 CPU 线程: 12 核只用一半, 避免干扰其他任务
export ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS=6
export OMP_NUM_THREADS=6

echo "===== $(date '+%Y-%m-%d %H:%M') plan_and_preprocess ====="
nnUNetv2_plan_and_preprocess -d 1 -c 3d_fullres --verify_dataset_integrity 2>&1 | tee logs/preprocess.log
echo "preprocess done $(date '+%H:%M')"

echo "===== $(date '+%H:%M') copy splits_final ====="
cp "$nnUNet_raw/Dataset001_RectalCancer/splits.json" "$nnUNet_preprocessed/Dataset001_RectalCancer/splits_final.json"
NFOLDS=$(python -c "import json;print(len(json.load(open('$nnUNet_preprocessed/Dataset001_RectalCancer/splits_final.json'))))")
echo "splits_final.json copied, folds=$NFOLDS"

for fold in 0 1 2 3 4; do
  echo "===== $(date '+%H:%M') train fold $fold ====="
  nnUNetv2_train 1 3d_fullres "$fold" -device cuda 2>&1 | tee "logs/train_fold${fold}.log"
  echo "fold $fold DONE $(date '+%H:%M')"
done
echo "ALL TRAIN DONE $(date '+%Y-%m-%d %H:%M')"
