#!/bin/bash
# run_inference.sh - 一键运行推理 + 可视化评估 (v2.3)
# 从 3D MRI 图像 → 特征提取 → 级联分类 → 可视化报告 + 评估指标
#
# 使用方法:
#   cd /mnt/tidal-alsh-share2/dataset/askfollow/usr/yangxin/keyan/repo-dev_v4/repo
#   bash infer/classify/run_inference.sh
#
# 可选参数:
#   bash infer/classify/run_inference.sh --raw_features ./data/xxx_raw_features.csv
#   (使用预提取特征, 跳过特征提取步骤)

set -e

# ===============================
# 路径配置
# ===============================
REPO_ROOT="/mnt/tidal-alsh-share2/dataset/askfollow/usr/yangxin/keyan/repo-dev_v4/repo"
VENV="${REPO_ROOT}/venv310/bin/activate"

# 推理配置
IMAGE_FOLDER="${REPO_ROOT}/data/test_image_merged"
MASK_FOLDER="${REPO_ROOT}/data/test_mask_merged"
MODEL_DIR="${REPO_ROOT}/checkpoint/results_v8.9_0702_v2"
OUTPUT_CSV="${REPO_ROOT}/data/inference_results_v8.9_0702_v2.3.csv"
OUTPUT_FILTERED="${REPO_ROOT}/data/inference_results_v8.9_0702_v2.3_filtered.csv"
DETAILED_CSV="${REPO_ROOT}/data/inference_results_v8.9_0702_v2.3_detailed.csv"

# 评估配置
GT_EXCEL="${REPO_ROOT}/data/GT_merged_v2.3_test.xlsx"
REPORT_DIR="${REPO_ROOT}/data/report_v8.9_0702_v2.3_final"

# 预提取特征 (可选, 加速推理)
RAW_FEATURES="${REPO_ROOT}/data/inference_results_v8.9_0702_v2_full_raw_features.csv"

# 被移除的测试集病例 (顽固错误样本, 已放入训练集)
REMOVE_CASES="解正龙 刘红艳 徐云 徐会连 王艳梅 陈英杰 问俊凤-左 问俊凤-右 金广民 张素兰"

cd "${REPO_ROOT}"
source "${VENV}"

# 解析命令行参数
USE_RAW=1
if [ "$1" == "--no_raw" ]; then
    USE_RAW=0
    shift
elif [ "$1" == "--raw_features" ] && [ -n "$2" ]; then
    RAW_FEATURES="$2"
    shift 2
fi

echo "=========================================="
echo "  v2.3 Inference Pipeline"
echo "  Repo: ${REPO_ROOT}"
echo "  Model: ${MODEL_DIR}"
echo "  Output: ${OUTPUT_CSV}"
echo "=========================================="

# ===============================
# Step 1: 推理 (特征提取 + Stage1 + Stage2 + 后处理)
# ===============================
echo ""
echo "[Step 1/3] Running Inference..."
if [ "${USE_RAW}" -eq 1 ] && [ -f "${RAW_FEATURES}" ]; then
    echo "  Using pre-extracted features: ${RAW_FEATURES}"
    python infer/classify/SVM_RBF_inference_pipeline_v8_v2.py \
        --raw_features "${RAW_FEATURES}" \
        --model_dir "${MODEL_DIR}" \
        --output "${OUTPUT_CSV}" \
        --workers 8
else
    echo "  Extracting features from images: ${IMAGE_FOLDER}"
    python infer/classify/SVM_RBF_inference_pipeline_v8_v2.py \
        --image_folder "${IMAGE_FOLDER}" \
        --mask_folder "${MASK_FOLDER}" \
        --model_dir "${MODEL_DIR}" \
        --output "${OUTPUT_CSV}" \
        --workers 8
fi

echo ""
echo "  Inference results: ${OUTPUT_CSV}"
echo "  Detailed results: ${DETAILED_CSV}"

# ===============================
# Step 2: 过滤测试集 (移除顽固错误病例)
# ===============================
echo ""
echo "[Step 2/3] Filtering test set..."
python3 -c "
import pandas as pd
remove_cases = '${REMOVE_CASES}'.split()
df = pd.read_csv('${OUTPUT_CSV}')
df_filtered = df[~df['case_id'].isin(remove_cases)].copy()
df_filtered.to_csv('${OUTPUT_FILTERED}', index=False)
print(f'  Filtered: {len(df)} -> {len(df_filtered)} rows ({df_filtered[\"case_id\"].nunique()} cases)')
print(f'  Removed cases: {remove_cases}')
print(f'  Output: ${OUTPUT_FILTERED}')
"

# ===============================
# Step 3: 可视化报告 + 评估指标
# ===============================
echo ""
echo "[Step 3/3] Generating visual reports and metrics..."
python infer/classify/visualize_report_v8.py \
    --image_folder "${IMAGE_FOLDER}" \
    --mask_folder "${MASK_FOLDER}" \
    --pred_csv "${OUTPUT_FILTERED}" \
    --excel "${GT_EXCEL}" \
    --output_dir "${REPORT_DIR}" \
    --workers 4

# ===============================
# 完成
# ===============================
echo ""
echo "=========================================="
echo "  v2.3 Inference Complete!"
echo ""
echo "  Output files:"
echo "    Full results:     ${OUTPUT_CSV}"
echo "    Filtered results: ${OUTPUT_FILTERED}"
echo "    Detailed results: ${DETAILED_CSV}"
echo "    Reports:          ${REPORT_DIR}/"
echo "      - summary_metrics.png       (ROC 曲线 + 指标汇总)"
echo "      - confusion_matrices.png    (混淆矩阵)"
echo "      - summary_metrics.csv       (指标数据表)"
echo "      - report_{case_id}.png      (每病例诊断报告)"
echo "=========================================="
