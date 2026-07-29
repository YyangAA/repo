"""
utils.py
========
评估流程的公共工具函数，供 run_inference.py / evaluate.py / visualize.py 复用。

包含：
- 路径与环境配置常量（可通过环境变量或命令行覆盖）
- nnU-Net 数据集元信息读取（dataset.json / splits_final.json）
- nii.gz 读取（基于 SimpleITK，与 nnU-Net 官方 IO 保持一致）
- 标签、颜色映射等辅助信息

作者说明：本文件不修改任何现有训练工程，仅在 evaluation/ 目录内独立使用。
"""

import os
import json
from typing import Dict, List, Tuple

import numpy as np
import SimpleITK as sitk


# ============================================================================
# 1. 默认路径配置（repo-dev_v4 工程根）
#    这些路径可以通过命令行参数覆盖；此处给出默认值以保证「开箱即用 / 可复现」。
# ============================================================================
# evaluation/ 目录的绝对路径
EVAL_DIR = os.path.dirname(os.path.abspath(__file__))

# 工程根：.../repo-dev_v4
# EVAL_DIR = .../repo-dev_v4/repo/infer/segmentation/evaluation
REPO_DEV_ROOT = os.path.abspath(os.path.join(EVAL_DIR, "..", "..", "..", ".."))

# nnU-Net 三大目录（默认指向 repo-dev_v4 下的标准结构）
DEFAULT_NNUNET_RAW = os.path.join(REPO_DEV_ROOT, "nnUNet_raw")
DEFAULT_NNUNET_PREPROCESSED = os.path.join(REPO_DEV_ROOT, "nnUNet_preprocessed")
DEFAULT_NNUNET_RESULTS = os.path.join(REPO_DEV_ROOT, "nnUNet_results")

# 数据集 / 模型标识
DATASET_NAME = "Dataset260426_Knee2D"
TRAINER = "nnUNetTrainer_FreezeEncoder"
PLANS = "nnUNetPlans"
CONFIGURATION = "2d"
DEFAULT_FOLD = 0
DEFAULT_CHECKPOINT = "checkpoint_best.pth"  # 可选 checkpoint_final.pth

# 输出目录（全部集中在 evaluation/ 下，不污染现有工程）
OUTPUTS_DIR = os.path.join(EVAL_DIR, "outputs")
PREDICTIONS_DIR = os.path.join(OUTPUTS_DIR, "predictions")
METRICS_DIR = os.path.join(EVAL_DIR, "metrics")
FIGURES_DIR = os.path.join(EVAL_DIR, "figures")
LOGS_DIR = os.path.join(EVAL_DIR, "logs")


# ============================================================================
# 2. 标签信息
# ============================================================================
# 与 dataset.json 保持一致
LABELS: Dict[str, int] = {
    "background": 0,
    "Femoral_Medial": 1,
    "Femoral_Lateral": 2,
    "Tibial_Medial": 3,
    "Tibial_Lateral": 4,
}
# 前景类别 id 列表（评估针对前景）
FOREGROUND_IDS: List[int] = [1, 2, 3, 4]
# id -> 名称
ID_TO_NAME: Dict[int, str] = {v: k for k, v in LABELS.items()}

# 可视化颜色（RGB, 0-1）——每个前景类别一个固定颜色，便于论文图一致
CLASS_COLORS: Dict[int, Tuple[float, float, float]] = {
    1: (1.0, 0.0, 0.0),   # Femoral_Medial  红
    2: (0.0, 1.0, 0.0),   # Femoral_Lateral 绿
    3: (0.0, 0.5, 1.0),   # Tibial_Medial   蓝
    4: (1.0, 1.0, 0.0),   # Tibial_Lateral  黄
}


# ============================================================================
# 3. 目录创建
# ============================================================================
def ensure_dirs(*dirs: str) -> None:
    """创建目录（若不存在）。"""
    for d in dirs:
        if d:
            os.makedirs(d, exist_ok=True)


def ensure_default_output_dirs() -> None:
    ensure_dirs(OUTPUTS_DIR, PREDICTIONS_DIR, METRICS_DIR, FIGURES_DIR, LOGS_DIR)


# ============================================================================
# 4. nnU-Net 元信息读取
# ============================================================================
def get_model_folder(nnunet_results: str) -> str:
    """返回训练结果里的 <trainer>__<plans>__<config> 目录。"""
    return os.path.join(
        nnunet_results,
        DATASET_NAME,
        f"{TRAINER}__{PLANS}__{CONFIGURATION}",
    )


def load_dataset_json(nnunet_raw: str) -> dict:
    path = os.path.join(nnunet_raw, DATASET_NAME, "dataset.json")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_splits(nnunet_preprocessed: str) -> List[dict]:
    """读取 splits_final.json（5 折交叉验证划分）。"""
    path = os.path.join(nnunet_preprocessed, DATASET_NAME, "splits_final.json")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def get_val_case_ids(nnunet_preprocessed: str, fold: int = DEFAULT_FOLD) -> List[str]:
    """返回指定 fold 的验证集 case id 列表（即 held-out 评估集）。"""
    splits = load_splits(nnunet_preprocessed)
    if fold < 0 or fold >= len(splits):
        raise ValueError(f"fold={fold} 超出 splits 范围 (0..{len(splits)-1})")
    return list(splits[fold]["val"])


# ============================================================================
# 5. 文件名 / 路径工具
#    imagesTr 中输入文件为 <case>_0000.nii.gz（channel 0）
#    labelsTr / gt_segmentations / predictions 中为 <case>.nii.gz
# ============================================================================
FILE_ENDING = ".nii.gz"
CHANNEL_SUFFIX = "_0000"  # 单模态，仅 channel 0


def image_path(nnunet_raw: str, case_id: str, subdir: str = "imagesTr") -> str:
    return os.path.join(nnunet_raw, DATASET_NAME, subdir, f"{case_id}{CHANNEL_SUFFIX}{FILE_ENDING}")


def label_path(nnunet_raw: str, case_id: str, subdir: str = "labelsTr") -> str:
    return os.path.join(nnunet_raw, DATASET_NAME, subdir, f"{case_id}{FILE_ENDING}")


def gt_seg_path(nnunet_preprocessed: str, case_id: str) -> str:
    return os.path.join(nnunet_preprocessed, DATASET_NAME, "gt_segmentations", f"{case_id}{FILE_ENDING}")


def prediction_path(predictions_dir: str, case_id: str) -> str:
    return os.path.join(predictions_dir, f"{case_id}{FILE_ENDING}")


# ============================================================================
# 6. nii.gz 读取（SimpleITK，与 nnU-Net SimpleITKIO 对齐）
# ============================================================================
def read_nii_array(path: str) -> Tuple[np.ndarray, Tuple[float, ...]]:
    """
    读取 nii.gz，返回 (numpy array, spacing)。
    SimpleITK 读出的 numpy array 轴顺序为 (z, y, x)，spacing 顺序为 (x, y, z)。
    对本 2D 数据集，z=1。
    """
    img = sitk.ReadImage(path)
    arr = sitk.GetArrayFromImage(img)  # (z, y, x)
    spacing = img.GetSpacing()          # (x, y, z)
    return arr, spacing


def to_2d(arr: np.ndarray) -> np.ndarray:
    """
    将 nnU-Net 2D 数据（存储为 (1, H, W) 或 (H, W, 1)）压成 (H, W)。
    """
    a = np.asarray(arr)
    a = np.squeeze(a)
    if a.ndim != 2:
        # 若仍非 2D，取第一个非平凡切片，保证鲁棒
        a = a.reshape(a.shape[-2], a.shape[-1]) if a.ndim >= 2 else a
    return a


def get_2d_spacing(spacing: Tuple[float, ...]) -> Tuple[float, float]:
    """
    从 SimpleITK spacing (x, y, z) 得到 2D 面内 spacing (y, x) 对应像素物理尺寸。
    本数据集 spacing=[1,1]，故为 (1.0, 1.0)。返回 (row_spacing, col_spacing)。
    """
    sp = list(spacing)
    if len(sp) >= 2:
        # SimpleITK (x, y, z) -> array (row=y, col=x)
        return (float(sp[1]), float(sp[0]))
    return (1.0, 1.0)
