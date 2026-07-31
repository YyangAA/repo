#!/bin/bash
# =============================================================================
# pipeline.sh - 端到端一体化推理流程
#   分割 (nnUNet)  →  3D 重构  →  级联分类推理 (v8_v2)  →  可视化诊断报告 (v8)
#
# 说明:
#   - 前半段 (分割) 运行在 nnUNet repo (conda 环境 knee_yx)，产出 image_3d / mask_3d
#   - 后半段 (分类 + 可视化) 复用 run_inference.sh 的新版脚本 (venv310 环境)，
#     直接读取上一步的分割结果与分类模型进行推理，最终生成诊断报告
#
# 使用方法:
#   bash pipeline.sh
# =============================================================================

set -e

# ========== 指定使用 GPU ==========
export CUDA_VISIBLE_DEVICES=1

# ===============================
# 路径 / 环境配置
# ===============================
# --- 分割 repo (nnUNet) ---
SEG_REPO="/mnt/sda/yx/knee/nnUNet/repo"
SEG_CONDA_ENV="knee_yx"
NNUNET_RAW="/mnt/sda/yx/knee/nnUNet/nnUNet_raw/Dataset9999_Knee2D"

# --- 分类 repo (级联分类, 新版流程) ---
CLS_REPO="/mnt/tidal-alsh-share2/dataset/askfollow/usr/yangxin/keyan/repo-dev_v4/repo"
CLS_VENV="${CLS_REPO}/venv310/bin/activate"

# --- 分割结果 (作为分类推理的输入) ---
IMAGE_3D="${SEG_REPO}/data/image_3d"
MASK_3D="${SEG_REPO}/data/mask_3d"

# --- 分类模型 (新版 v8.9) ---
MODEL_DIR="${CLS_REPO}/checkpoint/results_v8.9_0702_v2"

# --- 分类推理输出 ---
OUTPUT_CSV="${SEG_REPO}/data/inference_results.csv"

# --- 可视化诊断报告 ---
# GT_EXCEL 可选: 传入含真实标签的 Excel 可展示 预测 vs 真实 对比, 文件不存在时只展示预测
GT_EXCEL="/mnt/sda/yx/knee/5t/data_unzipped/第一批5T.xlsx"
REPORT_DIR="${SEG_REPO}/data/report"

# 初始化 conda (使 conda activate 在非交互 shell 中可用)
source "$(conda info --base)/etc/profile.d/conda.sh"

# =============================================================================
# 阶段一: 分割 (nnUNet)  ——  DICOM → npy → nii → 2D 分割 → 3D 重构
# =============================================================================
echo "=========================================="
echo "  [Stage 1/3] Segmentation (nnUNet)"
echo "  Repo: ${SEG_REPO}"
echo "=========================================="

cd "${SEG_REPO}"
conda activate "${SEG_CONDA_ENV}"

# 清理历史产物
rm -rf ./data/inference_output
rm -rf ./data/vis_results_old
rm -rf ./data/inference_results.csv
rm -rf ./data/npy
rm -rf "${NNUNET_RAW}"
rm -rf ./data/image_3d
rm -rf ./data/mask_3d
rm -rf ./data/report

# DICOM → npy → nii
python ./infer/segmentation/1_dcm2npy.py
python ./infer/segmentation/2_npy2nii.py

# nnUNet 2D 分割推理
nnUNetv2_predict \
    -i "${NNUNET_RAW}/imagesTr" \
    -o ./data/inference_output \
    -d 260426 -c 2d -f 0 -tr nnUNetTrainer_FreezeEncoder

# 可选: 打印分割结果预测
# python ./infer/segmentation/4_vis.py

# 2D 切片重构为 3D (生成 image_3d / mask_3d)
python ./infer/segmentation/5__nii23D.py

echo "  分割完成, 3D 结果: ${IMAGE_3D} , ${MASK_3D}"

# =============================================================================
# 阶段二: 级联分类推理 (v8_v2)  ——  特征提取 + Stage1 + Stage2 + 后处理
# =============================================================================
echo ""
echo "=========================================="
echo "  [Stage 2/3] Classification Inference (v8_v2)"
echo "  Repo:   ${CLS_REPO}"
echo "  Model:  ${MODEL_DIR}"
echo "  Output: ${OUTPUT_CSV}"
echo "=========================================="

# 切换到分类 repo 与其 venv310 环境
conda deactivate
cd "${CLS_REPO}"
source "${CLS_VENV}"

python infer/classify/SVM_RBF_inference_pipeline_v8_v2.py \
    --image_folder "${IMAGE_3D}" \
    --mask_folder "${MASK_3D}" \
    --model_dir "${MODEL_DIR}" \
    --output "${OUTPUT_CSV}" \
    --workers 8

if [ ! -f "${OUTPUT_CSV}" ]; then
    echo "Error: 分类推理失败, 未生成结果文件 ${OUTPUT_CSV}"
    exit 1
fi
echo "  分类推理完成: ${OUTPUT_CSV}"

# =============================================================================
# 阶段三: 综合可视化诊断报告 (v8)
#   分割结果图 + 损伤热力图 + 预测结果 (+ 真实标签对比, 当提供 GT_EXCEL 时)
# =============================================================================
echo ""
echo "=========================================="
echo "  [Stage 3/3] Visualization Report (v8)"
echo "  Output: ${REPORT_DIR}"
echo "=========================================="

VIS_ARGS=(
    --image_folder "${IMAGE_3D}"
    --mask_folder "${MASK_3D}"
    --pred_csv "${OUTPUT_CSV}"
    --output_dir "${REPORT_DIR}"
    --workers 4
)
# GT Excel 存在时追加对比评估
if [ -f "${GT_EXCEL}" ]; then
    echo "  使用真实标签进行对比评估: ${GT_EXCEL}"
    VIS_ARGS+=(--excel "${GT_EXCEL}")
else
    echo "  未找到 GT Excel, 仅展示预测结果 (${GT_EXCEL})"
fi

python infer/classify/visualize_report_v8.py "${VIS_ARGS[@]}"

# =============================================================================
# 完成
# =============================================================================
echo ""
echo "=========================================="
echo "  Pipeline Complete!"
echo "    分割结果:   ${IMAGE_3D} , ${MASK_3D}"
echo "    分类结果:   ${OUTPUT_CSV}"
echo "    诊断报告:   ${REPORT_DIR}/"
echo "      - summary_metrics.png     (ROC 曲线 + 指标汇总)"
echo "      - confusion_matrices.png  (混淆矩阵)"
echo "      - summary_metrics.csv     (指标数据表)"
echo "      - report_{case_id}.png    (每病例诊断报告)"
echo "=========================================="
