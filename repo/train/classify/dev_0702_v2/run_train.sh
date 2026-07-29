#!/bin/bash
# run_train.sh - 一键运行完整训练流程 (v2.3)
# 从原始特征 CSV → LASSO 特征选择 → 跨区域增强 → SVM 级联分类器训练
#
# 使用方法:
#   cd /mnt/tidal-alsh-share2/dataset/askfollow/usr/yangxin/keyan/repo-dev_v4/repo
#   bash train/classify/dev_0702_v2/run_train.sh
#
# 前置条件:
#   - 虚拟环境 venv310 已安装 sklearn, pandas, numpy, joblib, matplotlib
#   - 原始特征 CSV 已生成: train/classify/dev_0702_v2/data_train/knee_radiomics_features_3d_integrated.csv
#   - (特征提取脚本见 train/classify/dev_0702_v2/extract_features.py, 如需从头提取)

set -e

# ===============================
# 路径配置
# ===============================
REPO_ROOT="/mnt/tidal-alsh-share2/dataset/askfollow/usr/yangxin/keyan/repo-dev_v4/repo"
VENV="${REPO_ROOT}/venv310/bin/activate"
SCRIPT_DIR="train/classify/dev_0702_v2"
MODEL_OUTPUT="${REPO_ROOT}/checkpoint/results_v8.9_0702_v2"

cd "${REPO_ROOT}"
source "${VENV}"

echo "=========================================="
echo "  v2.3 Training Pipeline"
echo "  Repo: ${REPO_ROOT}"
echo "  Output: ${MODEL_OUTPUT}"
echo "=========================================="

# ===============================
# Step 0: 检查输入文件
# ===============================
INPUT_CSV="${SCRIPT_DIR}/data_train/knee_radiomics_features_3d_integrated.csv"
if [ ! -f "${INPUT_CSV}" ]; then
    echo "[ERROR] Input CSV not found: ${INPUT_CSV}"
    echo "  请先运行特征提取生成该文件"
    exit 1
fi
echo ""
echo "[Step 0] Input CSV found: ${INPUT_CSV} ($(wc -l < ${INPUT_CSV}) lines)"

# ===============================
# Step 1: LASSO 特征选择 (Stage 1 + Stage 2)
# ===============================
echo ""
echo "[Step 1/4] LASSO Feature Selection (1a_lasso_v3.py)..."
echo "  - Stage 1: Normal vs Damaged 特征筛选"
echo "  - Stage 2: Grade 1 vs Grade 2 特征筛选"
echo "  - v2: 自适应稳定性参数, bootstrap=100, max_features=100"
python ${SCRIPT_DIR}/1a_lasso_v3.py

# ===============================
# Step 2: 跨区域特征增强
# ===============================
echo ""
echo "[Step 2/4] Cross-Region Feature Enhancement (1b_add_cross_features_v8.py)..."
echo "  - 构建跨区域特征 (cross_)"
echo "  - 构建解剖对比值特征 (ratio_)"
echo "  - 构建区域 one-hot 特征 (region_)"
python ${SCRIPT_DIR}/1b_add_cross_features_v8.py

# ===============================
# Step 3: 第二轮 LASSO (跨区域增强后重新筛选)
# ===============================
echo ""
echo "[Step 3/4] Second-round LASSO (2_lasso_v8.py)..."
echo "  - 对增强后的特征重新做 LASSO 筛选"
python ${SCRIPT_DIR}/2_lasso_v8.py

# ===============================
# Step 4: SVM 级联分类器训练
# ===============================
echo ""
echo "[Step 4/4] SVM Cascade Training (3_train_svm_v8.py)..."
echo "  - Stage 1: 二分类 SVM (Normal vs Damaged)"
echo "    - 81 组参数搜索 (C x gamma = 9 x 9)"
echo "    - GroupKFold 5 折交叉验证"
echo "    - Platt Scaling 概率校准"
echo "  - Stage 2: 分级 SVM (Grade 1 vs Grade 2)"
echo "    - FM/TM 独立训练, FL+TL 池化训练"
python ${SCRIPT_DIR}/3_train_svm_v8.py

# ===============================
# 完成
# ===============================
echo ""
echo "=========================================="
echo "  v2.3 Training Complete!"
echo "  Models saved to: ${MODEL_OUTPUT}"
echo ""
echo "  生成文件:"
echo "    - svm_model_calibrated.pkl  (Platt Scaling 校准模型)"
echo "    - svm_model.pkl             (原始 SVM 模型)"
echo "    - scaler.pkl                (StandardScaler)"
echo "    - threshold.pkl             (Youden 最优阈值)"
echo "    - feature_list.pkl          (Stage 1 特征列表)"
echo "    - svm_model_stage2.pkl      (Stage 2 模型)"
echo "    - scaler_stage2.pkl         (Stage 2 StandardScaler)"
echo "    - feature_list_stage2.pkl   (Stage 2 特征列表)"
echo "=========================================="
