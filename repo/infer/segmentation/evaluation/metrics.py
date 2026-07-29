"""
metrics.py
==========
医学图像分割评估指标（逐类别 + 前景整体）。

实现指标
--------
Overlap / Region:
  - Dice (DSC), IoU (Jaccard), Precision, Recall(Sensitivity), Specificity, Accuracy
Boundary / Distance（物理单位，基于 spacing）:
  - HD (Hausdorff), HD95 (95th percentile Hausdorff), ASD (Average Surface Distance)

边界距离实现
------------
优先使用纯 scipy 实现（scipy.ndimage.distance_transform_edt），无需 medpy，
从而在仅有 scipy/numpy 的环境即可运行；若安装了 medpy 也可切换（默认 scipy）。

边界情况处理（分割论文通用约定）
--------------------------------
- GT 与 Pred 均为空：重叠类指标 NaN（该类别不参与统计），距离类 NaN。
- 仅一方为空：重叠类指标 = 0；距离类无法定义，记 NaN（不使用对角线代替，避免污染均值）。
- 两者均非空：正常计算。
"""

from typing import Dict, List, Tuple

import numpy as np
from scipy import ndimage


OVERLAP_METRICS = ["Dice", "IoU", "Precision", "Recall", "Specificity", "Accuracy"]
BOUNDARY_METRICS = ["HD", "HD95", "ASD"]
ALL_METRICS = OVERLAP_METRICS + BOUNDARY_METRICS


# ---------------------------------------------------------------------------
# 混淆矩阵与重叠指标
# ---------------------------------------------------------------------------
def _confusion(pred: np.ndarray, gt: np.ndarray) -> Tuple[int, int, int, int]:
    pred = pred.astype(bool)
    gt = gt.astype(bool)
    tp = int(np.logical_and(pred, gt).sum())
    fp = int(np.logical_and(pred, np.logical_not(gt)).sum())
    tn = int(np.logical_and(np.logical_not(pred), np.logical_not(gt)).sum())
    fn = int(np.logical_and(np.logical_not(pred), gt).sum())
    return tp, fp, tn, fn


def _safe_div(a: float, b: float) -> float:
    return float(a) / float(b) if b != 0 else float("nan")


def binary_overlap_metrics(pred: np.ndarray, gt: np.ndarray) -> Dict[str, float]:
    tp, fp, tn, fn = _confusion(pred, gt)
    p_sum = pred.sum()
    g_sum = gt.sum()

    result: Dict[str, float] = {}
    if g_sum == 0 and p_sum == 0:
        for m in OVERLAP_METRICS:
            result[m] = float("nan")
        result["TP"], result["FP"], result["TN"], result["FN"] = tp, fp, tn, fn
        return result

    result.update({
        "Dice": _safe_div(2 * tp, 2 * tp + fp + fn),
        "IoU": _safe_div(tp, tp + fp + fn),
        "Precision": _safe_div(tp, tp + fp),
        "Recall": _safe_div(tp, tp + fn),
        "Specificity": _safe_div(tn, tn + fp),
        "Accuracy": _safe_div(tp + tn, tp + tn + fp + fn),
        "TP": tp, "FP": fp, "TN": tn, "FN": fn,
    })
    return result


# ---------------------------------------------------------------------------
# 边界距离（纯 scipy 实现）
# ---------------------------------------------------------------------------
def _surface_border(mask: np.ndarray) -> np.ndarray:
    """提取二值掩码的边界像素（前景与其腐蚀结果之差）。"""
    mask = mask.astype(bool)
    if mask.sum() == 0:
        return np.zeros_like(mask)
    # 结构元素：面连通（2D 十字），与 medpy 默认一致
    struct = ndimage.generate_binary_structure(mask.ndim, 1)
    eroded = ndimage.binary_erosion(mask, structure=struct, border_value=0)
    return mask & (~eroded)


def _surface_distances(pred: np.ndarray, gt: np.ndarray,
                       spacing: Tuple[float, float]) -> np.ndarray:
    """
    返回 pred 边界到 gt 边界的对称表面距离集合（两个方向拼接），物理单位。
    """
    pred_border = _surface_border(pred)
    gt_border = _surface_border(gt)

    # 到 gt 边界的距离场：对 gt_border 取反做 EDT
    dt_to_gt = ndimage.distance_transform_edt(~gt_border, sampling=spacing)
    dt_to_pred = ndimage.distance_transform_edt(~pred_border, sampling=spacing)

    sds_pred_to_gt = dt_to_gt[pred_border]
    sds_gt_to_pred = dt_to_pred[gt_border]
    return np.concatenate([sds_pred_to_gt, sds_gt_to_pred])


def binary_boundary_metrics(pred: np.ndarray, gt: np.ndarray,
                            spacing: Tuple[float, float]) -> Dict[str, float]:
    result: Dict[str, float] = {m: float("nan") for m in BOUNDARY_METRICS}
    pred = pred.astype(bool)
    gt = gt.astype(bool)
    if pred.sum() == 0 or gt.sum() == 0:
        return result
    try:
        sds = _surface_distances(pred, gt, spacing)
        if sds.size == 0:
            return result
        result["HD"] = float(np.max(sds))
        result["HD95"] = float(np.percentile(sds, 95))
        result["ASD"] = float(np.mean(sds))
    except Exception:
        pass
    return result


# ---------------------------------------------------------------------------
# 样本级评估与汇总
# ---------------------------------------------------------------------------
def evaluate_case(pred_arr: np.ndarray, gt_arr: np.ndarray,
                  class_ids: List[int],
                  spacing: Tuple[float, float]) -> Dict[int, Dict[str, float]]:
    per_class: Dict[int, Dict[str, float]] = {}
    for c in class_ids:
        pred_c = (pred_arr == c)
        gt_c = (gt_arr == c)
        m = binary_overlap_metrics(pred_c, gt_c)
        m.update(binary_boundary_metrics(pred_c, gt_c, spacing))
        per_class[c] = m
    return per_class


def foreground_mean(per_class: Dict[int, Dict[str, float]],
                    class_ids: List[int]) -> Dict[str, float]:
    agg: Dict[str, float] = {}
    for metric in ALL_METRICS:
        vals = np.array([per_class[c].get(metric, float("nan")) for c in class_ids], dtype=float)
        agg[metric] = float("nan") if np.all(np.isnan(vals)) else float(np.nanmean(vals))
    return agg


def summarize(values: List[float]) -> Dict[str, float]:
    arr = np.array(values, dtype=float)
    valid = arr[~np.isnan(arr)]
    if valid.size == 0:
        return {"Mean": float("nan"), "Std": float("nan"), "Median": float("nan"),
                "Min": float("nan"), "Max": float("nan"), "N": 0}
    return {
        "Mean": float(np.mean(valid)),
        "Std": float(np.std(valid, ddof=1)) if valid.size > 1 else 0.0,
        "Median": float(np.median(valid)),
        "Min": float(np.min(valid)),
        "Max": float(np.max(valid)),
        "N": int(valid.size),
    }
