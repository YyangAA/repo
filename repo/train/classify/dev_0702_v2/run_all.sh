#!/bin/bash
# run_all.sh - 一键运行 v2 优化版训练流程
# 使用方法: cd /mnt/tidal-alsh-share2/dataset/askfollow/usr/yangxin/keyan/repo-dev_v4/repo && bash train/classify/dev_0702_v2/run_all.sh

set -e

VENV="/mnt/tidal-alsh-share2/dataset/askfollow/usr/yangxin/keyan/repo-dev_v4/repo/venv310/bin/activate"
SCRIPT_DIR="train/classify/dev_0702_v2"

echo "=========================================="
echo "  v2 Training Pipeline - Step by Step"
echo "=========================================="

source $VENV

echo ""
echo "[Step 1/4] LASSO Feature Selection (1a_lasso_v3.py)..."
python ${SCRIPT_DIR}/1a_lasso_v3.py

echo ""
echo "[Step 2/4] Cross-Region Feature Enhancement (1b_add_cross_features_v8.py)..."
python ${SCRIPT_DIR}/1b_add_cross_features_v8.py

echo ""
echo "[Step 3/4] Second-round LASSO (2_lasso_v8.py)..."
python ${SCRIPT_DIR}/2_lasso_v8.py

echo ""
echo "[Step 4/4] SVM Training (3_train_svm_v8.py)..."
python ${SCRIPT_DIR}/3_train_svm_v8.py

echo ""
echo "=========================================="
echo "  v2 Training Complete!"
echo "  Models saved to: ./checkpoint/results_v8.9_0702_v2"
echo ""
echo "  To run inference:"
echo "  python infer/classify/SVM_RBF_inference_pipeline_v8.py \\"
echo "    --model_dir ./checkpoint/results_v8.9_0702_v2 \\"
echo "    --output ./data/inference_results_v8.9_0702_v2.csv \\"
echo "    --raw_features ./data/inference_results_v8.9_0702_final_raw_features.csv"
echo "=========================================="
