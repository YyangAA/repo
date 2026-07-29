#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
visualize_report_v8.py
v8.4+ 综合可视化诊断报告 (v2.1 增强版)：生成包含 分割结果、损伤预测、真实标签对比、损伤热力图 的完整展示图。
当提供了 Excel 真实标签时，还会生成汇总评估报告（AUC / ROC 曲线 / 各项分类指标）。

基于 visualize_report_v5.py 适配 v8.4 推理结果格式。

用法:
    python infer/classify/visualize_report_v8.py \
        --image_folder  "./data/image_3d" \
        --mask_folder   "./data/mask_3d" \
        --pred_csv      "./data/inference_results_v8.csv" \
        --output_dir    "./data/report_v8.4" \
        --excel         "/path/to/ground_truth.xlsx"   (可选: 有真实标签时传入)
        --case_id       "张三"                         (可选: 不传则批量处理全部)
        --rotate_90                                    (可选: 顺时针旋转90度)

输出:
    每个病例生成一张报告图 report_{case_id}.png，包含：
    - Row 0: 代表性切片的分割叠加图（4 区域不同颜色）
    - Row 1: 损伤概率热力图（绿=正常, 红=损伤）
    - Row 2: 诊断结果文本面板（预测 vs 真实标签对比表，含 Stage2 Otsu 阈值）
    当有真实标签时额外生成:
    - summary_metrics.png: ROC 曲线 + 指标汇总表
    - confusion_matrices.png: 二分类 + 三分类混淆矩阵
    - summary_metrics.csv: 指标数据表
"""

import os
import sys
import argparse
import re
import numpy as np
import SimpleITK as sitk
from concurrent.futures import ProcessPoolExecutor, as_completed
from pypinyin import lazy_pinyin, Style

# ===============================
# 字体配置（必须在导入pyplot之前）
# ===============================
import matplotlib
matplotlib.use('Agg')  # 服务器无GUI环境

# 强制刷新字体缓存并加载系统字体
import matplotlib.font_manager as fm

# 清除缓存并重新构建字体列表
fm._load_fontmanager(try_read_cache=False)

# 手动添加文泉驿字体路径（确保能找到）
font_paths = [
    '/usr/share/fonts/truetype/wqy/wqy-microhei.ttc',
    '/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc',
]
for fp in font_paths:
    if os.path.exists(fp):
        fm.fontManager.addfont(fp)
        print(f"[Font] Loaded: {fp}")

# 现在导入pyplot
import matplotlib.pyplot as plt

# 设置中文字体（使用文泉驿）
plt.rcParams['font.family'] = ['sans-serif']
plt.rcParams['font.sans-serif'] = ['WenQuanYi Zen Hei', 'WenQuanYi Micro Hei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# 验证字体可用性
print(f"[Font] Available fonts: {plt.rcParams['font.sans-serif']}")
available_fonts = [f.name for f in fm.fontManager.ttflist]
print(f"[Font] WenQuanYi available: {'WenQuanYi Micro Hei' in available_fonts}")

import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Patch
import matplotlib.patheffects as path_effects
import pandas as pd
from scipy import ndimage  # 用于图像旋转
from sklearn.metrics import (
    roc_auc_score, roc_curve, auc,
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix
)

# ===============================
# 全局配置
# ===============================

REGION_NAMES = {
    1: "Femur_Medial",
    2: "Femur_Lateral",
    3: "Tibia_Medial",
    4: "Tibia_Lateral"
}

REGION_CN = {
    "Femur_Medial":  "Femur Med",
    "Femur_Lateral": "Femur Lat",
    "Tibia_Medial":  "Tibia Med",
    "Tibia_Lateral": "Tibia Lat"
}

# 区域缩写 (用于在图像上直接标注, 保持简短避免遮挡)
REGION_ABBR = {
    "Femur_Medial":  "FM",
    "Femur_Lateral": "FL",
    "Tibia_Medial":  "TM",
    "Tibia_Lateral": "TL",
}


def case_id_to_pinyin(case_id):
    """把 case_id 中的中文转成纯拼音 (无声调数字), 非中文字符原样保留。
    例: '问俊凤-左' -> 'wenjunfeng-zuo', '王艳梅' -> 'wangyanmei'
    目的: 规避 matplotlib 中文字体方框问题, 标题与文件名统一用拼音。"""
    if not case_id:
        return str(case_id)
    parts = lazy_pinyin(str(case_id), style=Style.NORMAL, errors='default')
    return ''.join(parts)


# 分割区域配色 (RGB + Alpha)
REGION_COLORS = {
    1: (0.18, 0.80, 0.44, 0.55),   # 绿色 - Femur_Medial
    2: (0.20, 0.60, 1.00, 0.55),   # 蓝色 - Femur_Lateral
    3: (1.00, 0.60, 0.00, 0.55),   # 橙色 - Tibia_Medial
    4: (0.74, 0.20, 0.90, 0.55),   # 紫色 - Tibia_Lateral
}

# 损伤概率热力图 colormap: 绿→黄→红
DAMAGE_CMAP = LinearSegmentedColormap.from_list(
    "damage", ["#27ae60", "#f1c40f", "#e74c3c"], N=256
)


# ===============================
# 1. 数据加载
# ===============================

def load_ground_truth(excel_path):
    """从 Excel 读取每个病例 4 个区域的真实损伤分级"""
    if excel_path is None or not os.path.exists(excel_path):
        return {}

    region_columns = {
        "股骨内侧": "Femur_Medial",
        "股骨外侧": "Femur_Lateral",
        "胫骨内侧": "Tibia_Medial",
        "胫骨外侧": "Tibia_Lateral"
    }

    df = pd.read_excel(excel_path)
    df["name"] = df["患者姓名"].astype(str).str.replace("_knee", "", regex=False).str.strip()

    grade_dict = {}
    for _, row in df.iterrows():
        cid = row["name"]
        grade_dict[cid] = {}
        for col_ch, col_en in region_columns.items():
            if col_ch in df.columns:
                grade_dict[cid][col_en] = int(row[col_ch]) if pd.notna(row[col_ch]) else -1
            else:
                grade_dict[cid][col_en] = -1
    return grade_dict


def load_predictions(csv_path):
    """读取 v8 推理输出的 CSV（级联二分类: Stage1 Normal/Damaged + Stage2 Grade1/2）
    v2.1: 支持 confidence_score 和 stage1_gmm_method 列"""
    if csv_path is None or not os.path.exists(csv_path):
        return {}

    df = pd.read_csv(csv_path)
    pred_dict = {}
    for _, row in df.iterrows():
        cid = str(row["case_id"])
        region = row["region"]
        pred = int(row.get("predicted_label", row.get("predicted_grade", 0)))
        prob = float(row.get("probability_damage", 0.0))
        threshold = float(row.get("threshold_used", 0.5))
        grade = int(row["predicted_grade"]) if "predicted_grade" in row.index else pred
        prob_grade2 = float(row.get("probability_grade2", 0.0))
        grade_reason = str(row.get("grade_reason", ""))
        # v2.1: 新增列
        confidence = float(row.get("confidence_score", -1.0))
        gmm_method = str(row.get("stage1_gmm_method", ""))

        if cid not in pred_dict:
            pred_dict[cid] = {}
        pred_dict[cid][region] = {
            "pred":         pred,          # 0=Normal, 1=Damaged (Stage 1 SVM)
            "prob":         prob,          # Stage 1 损伤概率
            "threshold":    threshold,     # 使用的最优阈值 (GMM adaptive / Youden)
            "grade":        grade,         # 0/1/2 三级分类（级联结果）
            "prob_grade2":  prob_grade2,   # Stage 2 Grade 2 概率
            "grade_reason": grade_reason,  # v8 新增: 分级原因
            "confidence":   confidence,    # v2.1 新增: 置信度评分 [0, 1]
            "gmm_method":   gmm_method,    # v2.1 新增: Stage1 GMM 方法
        }

    return pred_dict


# ===============================
# 2. 切片选择
# ===============================

def select_representative_slices(mask_array, num_slices=3):
    """从 3D mask 中选择软骨面积最大的代表性切片"""
    depth = mask_array.shape[0]
    if depth == 0:
        return []

    scores = [(z, np.sum(mask_array[z] > 0)) for z in range(depth)]
    scores.sort(key=lambda x: x[1], reverse=True)
    valid = [(z, s) for z, s in scores if s > 0]

    if not valid:
        return [depth // 2]

    if len(valid) <= num_slices:
        return sorted([z for z, _ in valid])

    # 从排名靠前的候选中均匀采样，避免选到相邻切片
    candidates = sorted([z for z, _ in valid[:num_slices * 3]])
    step = max(1, len(candidates) // num_slices)
    selected = candidates[::step][:num_slices]
    return sorted(selected)


# ===============================
# 3. 图像旋转函数
# ===============================

def rotate_slice_90_clockwise(img_slice, mask_slice):
    """
    将切片顺时针旋转90度
    scipy.ndimage.rotate 使用逆时针角度，所以用 -90 实现顺时针
    """
    img_rotated = ndimage.rotate(img_slice, angle=-90, reshape=False, order=1)
    mask_rotated = ndimage.rotate(mask_slice, angle=-90, reshape=False, order=0)
    return img_rotated, mask_rotated


# ===============================
# 4. 绘制分割叠加图
# ===============================

# def render_segmentation_overlay(ax, img_slice, mask_slice, title=""):
#     """原图 + 彩色分割叠加"""
#     ax.imshow(img_slice, cmap='gray', aspect='equal')

#     overlay = np.zeros((*mask_slice.shape, 4), dtype=np.float32)
#     for label_id, color in REGION_COLORS.items():
#         region_mask = (mask_slice == label_id)
#         if np.any(region_mask):
#             overlay[region_mask] = color

#     ax.imshow(overlay, aspect='equal')
#     ax.set_title(title, fontsize=9, fontweight='bold', pad=4)
#     ax.axis('off')

def render_segmentation_overlay(ax, img_slice, mask_slice, title="", alpha=0.3):
    """原图 + 彩色分割叠加（不在图上标注文字）"""
    ax.imshow(img_slice, cmap='gray', aspect='equal')

    overlay = np.zeros((*mask_slice.shape, 4), dtype=np.float32)
    for label_id, color in REGION_COLORS.items():
        region_mask = (mask_slice == label_id)
        if np.any(region_mask):
            if len(color) == 3:
                r, g, b = color
            else:
                r, g, b = color[:3]
            overlay[region_mask] = [r, g, b, alpha]

    ax.imshow(overlay, aspect='equal')
    ax.set_title(title, fontsize=11, fontweight='bold', pad=4)
    ax.axis('off')

# ===============================
# 5. 绘制损伤热力图
# ===============================

def render_damage_heatmap(ax, img_slice, mask_slice, pred_case, title="", gt_case=None):
    """根据各区域损伤概率生成热力图叠加（不在图上标注文字）"""
    ax.imshow(img_slice, cmap='gray', aspect='equal')

    GRADE_MIN_HEAT = {0: 0.0, 1: 0.50, 2: 0.80}

    heat = np.full(mask_slice.shape, np.nan, dtype=np.float32)
    for label_id, region_name in REGION_NAMES.items():
        region_mask = (mask_slice == label_id)
        if not np.any(region_mask):
            continue
        prob = 0.0
        grade = 0
        if pred_case and region_name in pred_case:
            prob = pred_case[region_name]["prob"]
            grade = pred_case[region_name].get("grade", pred_case[region_name].get("pred", 0))
        display_val = max(prob, GRADE_MIN_HEAT.get(grade, 0.0))
        heat[region_mask] = display_val

    masked_heat = np.ma.masked_where(np.isnan(heat), heat)
    im = ax.imshow(masked_heat, cmap=DAMAGE_CMAP, vmin=0, vmax=1, alpha=0.7, aspect='equal')

    ax.set_title(title, fontsize=11, fontweight='bold', pad=4)
    ax.axis('off')
    return im


# ===============================
# 6. 诊断结果面板
# ===============================

# Grade 颜色映射
GRADE_COLORS = {
    0: '#27ae60',   # 绿色 - Normal
    1: '#e67e22',   # 橙色 - Grade 1 (轻度损伤)
    2: '#e74c3c',   # 红色 - Grade 2 (严重损伤, 面积显著减小)
}

GRADE_LABELS = {
    0: "Normal",
    1: "G1 Mild",
    2: "G2 Severe",
}


def render_diagnosis_panel(ax, case_id, pred_case, gt_case):
    """精简诊断面板：只保留预测结果和损伤概率，字体放大
    - 去掉病人姓名标题
    - 图例不在此面板绘制（改为图级图例，放在右侧 colorbar 上方）
    """
    ax.axis('off')

    col_x = [0.02, 0.22, 0.42, 0.62, 0.85]
    headers = ["Region", "Pred Grade", "P(Damage)", "GT Grade", "Match"]

    y = 0.92
    for i, h in enumerate(headers):
        ax.text(col_x[i], y, h, fontsize=11, fontweight='bold', va='top',
                transform=ax.transAxes, color='#2c3e50')

    ax.plot([0.0, 1.0], [y - 0.04, y - 0.04], color='#bdc3c7', linewidth=1.0,
            transform=ax.transAxes)

    y -= 0.14

    for label_id, region_name in REGION_NAMES.items():
        display_name = REGION_CN.get(region_name, region_name)
        color_rgb = REGION_COLORS[label_id][:3]

        ax.text(col_x[0], y, display_name, fontsize=11, va='top', color=color_rgb,
                fontweight='bold', transform=ax.transAxes)

        if pred_case and region_name in pred_case:
            pred_info = pred_case[region_name]
            grade = pred_info.get("grade", pred_info["pred"])
            prob = pred_info["prob"]

            grade_str = GRADE_LABELS.get(grade, f"Grade {grade}")
            grade_color = GRADE_COLORS.get(grade, '#95a5a6')
            prob_str = f"{prob:.1%}"
        else:
            grade_str, grade_color, prob_str = "N/A", '#95a5a6', "-"
            grade = -1

        ax.text(col_x[1], y, grade_str, fontsize=11, va='top', color=grade_color,
                fontweight='bold', transform=ax.transAxes)
        ax.text(col_x[2], y, prob_str, fontsize=11, va='top',
                transform=ax.transAxes)

        if gt_case and region_name in gt_case:
            gt = gt_case[region_name]
            if gt >= 0:
                gt_str = GRADE_LABELS.get(gt, f"Grade {gt}")
                gt_color = GRADE_COLORS.get(gt, '#95a5a6')
            else:
                gt_str, gt_color = "N/A", '#95a5a6'
                gt = -1
        else:
            gt_str, gt_color = "N/A", '#95a5a6'
            gt = -1

        ax.text(col_x[3], y, gt_str, fontsize=11, va='top', color=gt_color,
                fontweight='bold', transform=ax.transAxes)

        if gt >= 0 and grade >= 0:
            if grade == gt:
                match_str, match_color = "OK", '#27ae60'
            elif (grade > 0) == (gt > 0):
                match_str, match_color = "~", '#e67e22'
            else:
                match_str, match_color = "MISS", '#e74c3c'
        else:
            match_str, match_color = "-", '#95a5a6'

        ax.text(col_x[4], y, match_str, fontsize=11, va='top', color=match_color,
                fontweight='bold', transform=ax.transAxes)

        y -= 0.18

    # 图例不在此绘制，改由 add_report_legend() 在图级坐标上画到右侧 colorbar 上方


def add_report_legend(fig):
    """在图级坐标右上角（Damage Prob colorbar 上方的空白处）绘制竖排图例"""
    legend_items = [
        Patch(facecolor=REGION_COLORS[lid][:3],
              label=f"{lid}: {REGION_CN[rn]}")
        for lid, rn in REGION_NAMES.items()
    ]
    legend_items.append(Patch(facecolor='none', edgecolor='none', label=''))
    for g, lbl in GRADE_LABELS.items():
        legend_items.append(Patch(facecolor=GRADE_COLORS[g], label=lbl))

    fig.legend(handles=legend_items, loc='upper left', fontsize=9,
               ncol=1, frameon=True, fancybox=True, handlelength=1.2,
               borderpad=0.8, labelspacing=0.55,
               bbox_to_anchor=(0.905, 0.93))


# ===============================
# 7. 汇总评估指标
# ===============================

def _fuzzy_find_gt(case_id, gt_dict, gt_key_map):
    """模糊匹配 gt_dict 的 key"""
    lookup_keys = [
        case_id,
        case_id.split('_')[0],
        case_id.replace("_knee", ""),
        case_id.split('-')[0],          # 支持 "张林侠-左" -> "张林侠"
        case_id.split('-')[0].split('_')[0],  # 多级分割
    ]
    for lk in lookup_keys:
        if lk in gt_key_map:
            return gt_dict[gt_key_map[lk]]
    return None


def compute_metrics_per_region(pred_dict, gt_dict):
    """
    汇总所有病例，按区域计算分类评估指标。

    计算两个维度的指标：
    1. 二分类: Normal(0) vs Damaged(1+2) — AUC, Acc, Sens, Spec, F1
    2. 三分类: Grade 0/1/2 精确匹配准确率

    返回: {region_name: metrics_dict or None}
    """
    # 建立 gt_dict 模糊匹配映射
    gt_key_map = {}
    for k in gt_dict:
        gt_key_map[k] = k
        gt_key_map[k.replace("_knee", "").strip()] = k

    region_results = {}

    for region_name in REGION_NAMES.values():
        y_true_binary, y_prob, y_pred_binary = [], [], []
        y_true_grade, y_pred_grade = [], []

        for case_id, pred_case in pred_dict.items():
            if region_name not in pred_case:
                continue

            gt_case = _fuzzy_find_gt(case_id, gt_dict, gt_key_map)
            if gt_case is None or region_name not in gt_case:
                continue

            gt_grade = gt_case[region_name]
            if gt_grade < 0:
                continue

            pred_info = pred_case[region_name]
            gt_binary = 1 if gt_grade > 0 else 0
            pred_grade = pred_info.get("grade", pred_info["pred"])

            y_true_binary.append(gt_binary)
            y_prob.append(pred_info["prob"])
            y_pred_binary.append(pred_info["pred"])
            y_true_grade.append(gt_grade)
            y_pred_grade.append(pred_grade)

        if len(y_true_binary) < 2:
            region_results[region_name] = None
            continue

        y_true_binary = np.array(y_true_binary)
        y_prob = np.array(y_prob)
        y_pred_binary = np.array(y_pred_binary)
        y_true_grade = np.array(y_true_grade)
        y_pred_grade = np.array(y_pred_grade)

        metrics = {
            'n':     len(y_true_binary),
            'n_pos': int(y_true_binary.sum()),
            'n_neg': int((1 - y_true_binary).sum()),
        }

        # ---- 二分类指标 (Normal vs Damaged) ----
        if len(np.unique(y_true_binary)) >= 2:
            fpr, tpr, _ = roc_curve(y_true_binary, y_prob)
            metrics['auc'] = auc(fpr, tpr)
            metrics['fpr'] = fpr
            metrics['tpr'] = tpr
        else:
            metrics['auc'] = float('nan')
            metrics['fpr'] = None
            metrics['tpr'] = None

        metrics['accuracy']  = accuracy_score(y_true_binary, y_pred_binary)
        metrics['precision'] = precision_score(y_true_binary, y_pred_binary, zero_division=0)
        metrics['recall']    = recall_score(y_true_binary, y_pred_binary, zero_division=0)
        metrics['f1']        = f1_score(y_true_binary, y_pred_binary, zero_division=0)

        cm = confusion_matrix(y_true_binary, y_pred_binary, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()
        metrics['specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        metrics['tp'] = int(tp)
        metrics['tn'] = int(tn)
        metrics['fp'] = int(fp)
        metrics['fn'] = int(fn)

        # ---- 二分类混淆矩阵 ----
        metrics['cm_binary'] = cm

        # ---- 三分类指标 (Grade 0/1/2) ----
        metrics['grade_accuracy'] = float(np.mean(y_true_grade == y_pred_grade))
        metrics['cm_grade'] = confusion_matrix(y_true_grade, y_pred_grade, labels=[0, 1, 2])

        # Grade 分布统计
        for g in [0, 1, 2]:
            metrics[f'gt_grade{g}'] = int((y_true_grade == g).sum())
            metrics[f'pred_grade{g}'] = int((y_pred_grade == g).sum())
            metrics[f'grade{g}_correct'] = int(
                ((y_true_grade == g) & (y_pred_grade == g)).sum()
            )

        region_results[region_name] = metrics

    return region_results


def _draw_confusion_matrix(ax, cm, labels, title, metric_val=None, metric_name="AUC", cmap='Blues'):
    """
    在给定的 axes 上绘制混淆矩阵热力图。
    """
    n_classes = len(labels)
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap, aspect='equal')

    # 在每个单元格里写数字
    thresh = cm.max() / 2.0
    for i in range(n_classes):
        for j in range(n_classes):
            val = cm[i, j]
            color = 'white' if val > thresh else 'black'
            ax.text(j, i, str(val), ha='center', va='center',
                    fontsize=14 if n_classes <= 2 else 12,
                    fontweight='bold', color=color)

    ax.set_xticks(range(n_classes))
    ax.set_yticks(range(n_classes))
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel('Predicted', fontsize=10)
    ax.set_ylabel('True', fontsize=10)

    # 标题 + 指标值
    if metric_val is not None and not np.isnan(metric_val):
        full_title = f"{title}\n{metric_name} = {metric_val:.3f}"
    else:
        full_title = title
    ax.set_title(full_title, fontsize=11, fontweight='bold', pad=8)

    return im



def render_summary_report(region_results, output_dir):
    """
    生成汇总评估报告图 (summary_metrics.png) + CSV (summary_metrics.csv)。

    布局:
      Row 0: ROC 曲线 (1 个大图, 4 条线)
      Row 1: 二分类指标表格 (跨4列, 文字表格)
    """
    region_list = list(REGION_NAMES.values())

    fig = plt.figure(figsize=(16, 12), facecolor='white')
    gs = gridspec.GridSpec(2, 1, height_ratios=[1.2, 0.8],
                           hspace=0.30,
                           left=0.06, right=0.96, top=0.95, bottom=0.06)

    line_colors = {
        "Femur_Medial":  '#2ecc71',
        "Femur_Lateral": '#3498db',
        "Tibia_Medial":  '#e67e22',
        "Tibia_Lateral": '#9b59b6',
    }

    # ======= Row 0: ROC 曲线 =======
    ax_roc = fig.add_subplot(gs[0])

    has_roc = False
    for region_name in region_list:
        m = region_results.get(region_name)
        if m is None or m['fpr'] is None:
            continue
        display_name = REGION_CN.get(region_name, region_name)
        color = line_colors.get(region_name, 'gray')
        ax_roc.plot(m['fpr'], m['tpr'], color=color, linewidth=2.2,
                    label=f"{display_name}  (AUC = {m['auc']:.3f},  n = {m['n']})")
        has_roc = True

    ax_roc.plot([0, 1], [0, 1], 'k--', linewidth=0.8, alpha=0.5, label='Random')
    ax_roc.set_xlim([-0.02, 1.02])
    ax_roc.set_ylim([-0.02, 1.02])
    ax_roc.set_xlabel('False Positive Rate  (1 - Specificity)', fontsize=11)
    ax_roc.set_ylabel('True Positive Rate  (Sensitivity)', fontsize=11)
    ax_roc.set_title('ROC Curves — Normal vs Damaged (v8.4+ v2.1)', fontsize=14, fontweight='bold')
    ax_roc.legend(loc='lower right', fontsize=10, frameon=True, fancybox=True)
    ax_roc.grid(True, alpha=0.3)

    if not has_roc:
        ax_roc.text(0.5, 0.5,
                    "Insufficient data for ROC curves\n"
                    "(need both positive and negative samples per region)",
                    ha='center', va='center', fontsize=12, color='#95a5a6',
                    transform=ax_roc.transAxes)

    # ======= Row 1: 二分类指标表格 =======
    ax_tbl = fig.add_subplot(gs[1])
    ax_tbl.axis('off')
    ax_tbl.set_title('Binary Classification Metrics  (Normal vs Damaged)',
                     fontsize=13, fontweight='bold', pad=10)

    # 表头
    col_headers = ["Region", "AUC", "Acc", "Sens", "Spec", "Prec", "F1",
                   "n", "Pos", "Neg", "TP", "FP", "FN", "TN"]
    col_x = np.linspace(0.01, 0.97, len(col_headers))

    y = 0.88
    for i, h in enumerate(col_headers):
        ax_tbl.text(col_x[i], y, h, fontsize=9, fontweight='bold', va='top',
                    ha='center', transform=ax_tbl.transAxes, color='#2c3e50')

    ax_tbl.plot([0.0, 1.0], [y - 0.04, y - 0.04], color='#bdc3c7',
                linewidth=0.8, transform=ax_tbl.transAxes)

    y -= 0.14
    for region_name in region_list:
        m = region_results.get(region_name)
        dn = REGION_CN.get(region_name, region_name)
        color = line_colors.get(region_name, 'gray')

        if m is None:
            vals = [dn] + ["—"] * (len(col_headers) - 1)
        else:
            auc_str = f"{m['auc']:.3f}" if not np.isnan(m['auc']) else "N/A"
            vals = [
                dn, auc_str,
                f"{m['accuracy']:.3f}", f"{m['recall']:.3f}",
                f"{m['specificity']:.3f}", f"{m['precision']:.3f}",
                f"{m['f1']:.3f}",
                str(m['n']), str(m['n_pos']), str(m['n_neg']),
                str(m['tp']), str(m['fp']), str(m['fn']), str(m['tn']),
            ]

        for i, v in enumerate(vals):
            fc = color if i == 0 else '#2c3e50'
            fw = 'bold' if i == 0 else 'normal'
            ax_tbl.text(col_x[i], y, v, fontsize=9, va='top', ha='center',
                        transform=ax_tbl.transAxes, color=fc, fontweight=fw)
        y -= 0.16

    # 底部分割线
    ax_tbl.plot([0.0, 1.0], [y + 0.06, y + 0.06], color='#bdc3c7',
                linewidth=0.5, transform=ax_tbl.transAxes)

    # ---- 保存图片 ----
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, "summary_metrics.png")
    fig.savefig(save_path, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"\n  [Summary] Metrics report image saved: {save_path}")

    # ---- 打印到控制台 ----
    print("\n  ========= Binary Classification (Normal vs Damaged) =========")
    for region_name in region_list:
        m = region_results.get(region_name)
        dn = REGION_CN.get(region_name, region_name)
        if m is None:
            print(f"  {dn:12s}  -- Insufficient data")
        else:
            auc_str = f"{m['auc']:.3f}" if not np.isnan(m['auc']) else "N/A"
            print(f"  {dn:12s}  AUC={auc_str}  Acc={m['accuracy']:.3f}  "
                  f"Sens={m['recall']:.3f}  Spec={m['specificity']:.3f}  "
                  f"F1={m['f1']:.3f}  "
                  f"(n={m['n']}, pos/neg={m['n_pos']}/{m['n_neg']}, "
                  f"TP={m['tp']} FP={m['fp']} FN={m['fn']} TN={m['tn']})")

    print("\n  ========= Grade Classification (0/1/2) =========")
    for region_name in region_list:
        m = region_results.get(region_name)
        dn = REGION_CN.get(region_name, region_name)
        if m is None or 'grade_accuracy' not in m:
            print(f"  {dn:12s}  -- N/A")
        else:
            print(f"  {dn:12s}  GradeAcc={m['grade_accuracy']:.3f}  "
                  f"GT(0/1/2)={m.get('gt_grade0',0)}/{m.get('gt_grade1',0)}/{m.get('gt_grade2',0)}  "
                  f"Pred(0/1/2)={m.get('pred_grade0',0)}/{m.get('pred_grade1',0)}/{m.get('pred_grade2',0)}")
    print("  " + "=" * 52)

    # ---- 保存 CSV ----
    csv_path = os.path.join(output_dir, "summary_metrics.csv")
    rows = []
    for region_name in region_list:
        m = region_results.get(region_name)
        if m is None:
            rows.append({"region": region_name, "n": 0, "note": "insufficient_data"})
            continue
        row_data = {
            "region":      region_name,
            "auc":         m['auc'],
            "accuracy":    m['accuracy'],
            "sensitivity": m['recall'],
            "specificity": m['specificity'],
            "precision":   m['precision'],
            "f1":          m['f1'],
            "n":           m['n'],
            "n_positive":  m['n_pos'],
            "n_negative":  m['n_neg'],
            "tp": m['tp'], "fp": m['fp'],
            "fn": m['fn'], "tn": m['tn'],
            "grade_accuracy": m.get('grade_accuracy', ''),
        }
        for g in [0, 1, 2]:
            row_data[f'gt_grade{g}'] = m.get(f'gt_grade{g}', 0)
            row_data[f'pred_grade{g}'] = m.get(f'pred_grade{g}', 0)
            row_data[f'grade{g}_correct'] = m.get(f'grade{g}_correct', 0)
        rows.append(row_data)
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    print(f"  [Summary] Metrics CSV saved: {csv_path}")

    # ---- 单独生成混淆矩阵图 ----
    render_confusion_matrices(region_results, output_dir)


def render_confusion_matrices(region_results, output_dir):
    """
    单独生成混淆矩阵图 (confusion_matrices.png)。

    布局:
      Row 0: 4 个二分类混淆矩阵 (Normal vs Damaged, Blues) — 显示 Acc
      Row 1: 4 个三分类混淆矩阵 (G0/G1/G2, Oranges) — 显示 Grade Acc
    """
    region_list = list(REGION_NAMES.values())

    fig = plt.figure(figsize=(18, 10), facecolor='white')
    gs = gridspec.GridSpec(2, 4, height_ratios=[1.0, 1.0],
                           hspace=0.40, wspace=0.35,
                           left=0.06, right=0.96, top=0.92, bottom=0.06)

    fig.suptitle('Confusion Matrices — Binary & Grade Classification (v8.4+ v2.1)',
                 fontsize=15, fontweight='bold', y=0.98)

    # ======= Row 0: 二分类混淆矩阵 =======
    binary_labels = ["Normal", "Damaged"]
    for col, region_name in enumerate(region_list):
        ax = fig.add_subplot(gs[0, col])
        m = region_results.get(region_name)
        display_name = REGION_CN.get(region_name, region_name)

        if m is None or 'cm_binary' not in m:
            ax.axis('off')
            ax.text(0.5, 0.5, f"{display_name}\nN/A", ha='center', va='center',
                    fontsize=12, color='#95a5a6', transform=ax.transAxes)
        else:
            acc_val = m.get('accuracy', float('nan'))
            _draw_confusion_matrix(
                ax, m['cm_binary'], binary_labels,
                title=f"{display_name} (Binary)",
                metric_val=acc_val, metric_name="Acc", cmap='Blues'
            )

    # ======= Row 1: 三分类混淆矩阵 =======
    grade_labels = ["G0", "G1", "G2"]
    for col, region_name in enumerate(region_list):
        ax = fig.add_subplot(gs[1, col])
        m = region_results.get(region_name)
        display_name = REGION_CN.get(region_name, region_name)

        if m is None or 'cm_grade' not in m:
            ax.axis('off')
            ax.text(0.5, 0.5, f"{display_name}\nN/A", ha='center', va='center',
                    fontsize=12, color='#95a5a6', transform=ax.transAxes)
        else:
            grade_acc = m.get('grade_accuracy', float('nan'))
            _draw_confusion_matrix(
                ax, m['cm_grade'], grade_labels,
                title=f"{display_name} (Grade)",
                metric_val=grade_acc, metric_name="Grade Acc", cmap='Oranges'
            )

    # ---- 保存 ----
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, "confusion_matrices.png")
    fig.savefig(save_path, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"  [Summary] Confusion matrices image saved: {save_path}")


# ===============================
# 8. 单病例报告生成（含旋转功能）
# ===============================

def generate_report(case_id, image_path, mask_path,
                    pred_dict, gt_dict, output_dir, num_slices=3, rotate_90=True):
    """
    v2.3: 默认旋转90度，精简面板，字体放大
    """
    print(f"  Generating report for: {case_id}" + (" [Rotated 90]" if rotate_90 else ""))

    try:
        image_sitk = sitk.ReadImage(image_path)
        mask_sitk = sitk.ReadImage(mask_path)
        mask_sitk = sitk.Cast(mask_sitk, sitk.sitkUInt8)

        img_array = sitk.GetArrayFromImage(image_sitk)
        mask_array = sitk.GetArrayFromImage(mask_sitk)
    except Exception as e:
        print(f"    [Error] Failed to read images: {e}")
        return False

    pred_case = pred_dict.get(case_id, None)
    gt_case = gt_dict.get(case_id, None)

    slices = select_representative_slices(mask_array, num_slices=num_slices)
    n = len(slices)

    if n == 0:
        print(f"    [Warning] No valid slices found")
        return False

    fig_w = max(5 * n, 12)
    fig = plt.figure(figsize=(fig_w, 12), facecolor='white')

    gs = gridspec.GridSpec(3, n, height_ratios=[1, 1, 0.55],
                           hspace=0.20, wspace=0.06,
                           left=0.03, right=0.92, top=0.95, bottom=0.03)

    heatmap_im = None

    for col, z in enumerate(slices):
        img_slice = img_array[z]
        mask_slice = mask_array[z]

        if rotate_90:
            img_slice, mask_slice = rotate_slice_90_clockwise(img_slice, mask_slice)

        ax_seg = fig.add_subplot(gs[0, col])
        render_segmentation_overlay(ax_seg, img_slice, mask_slice,
                                    title=f"Segmentation (z={z})")

        ax_heat = fig.add_subplot(gs[1, col])
        heatmap_im = render_damage_heatmap(ax_heat, img_slice, mask_slice,
                                           pred_case,
                                           title=f"Damage Heatmap (z={z})",
                                           gt_case=gt_case)

    if heatmap_im is not None:
        cbar_ax = fig.add_axes([0.935, 0.40, 0.015, 0.20])
        cbar = fig.colorbar(heatmap_im, cax=cbar_ax)
        cbar.set_label('Damage Prob', fontsize=10)
        cbar.ax.tick_params(labelsize=9)

    # 右侧 colorbar 上方空白处绘制图例（竖排）
    add_report_legend(fig)

    ax_panel = fig.add_subplot(gs[2, :])
    render_diagnosis_panel(ax_panel, case_id, pred_case, gt_case)

    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, f"report_{case_id_to_pinyin(case_id)}.png")
    fig.savefig(save_path, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"    Saved: {save_path}")
    return True


# ===============================
# 9. 进程池包装函数
# ===============================

def _generate_report_worker(task):
    """
    进程池工作函数：解包参数并调用 generate_report。
    使用单独的包装函数避免序列化整个 pred_dict/gt_dict。
    """
    (cid, image_path, mask_path,
     pred_case, gt_case, output_dir,
     num_slices, rotate_90) = task
    return generate_report(
        cid, image_path, mask_path,
        {cid: pred_case}, {cid: gt_case} if gt_case else {},
        output_dir, num_slices, rotate_90
    )


# ===============================
# 10. 主流程（多进程并行）
# ===============================

def main():
    parser = argparse.ArgumentParser(description='Knee Cartilage Diagnosis Report (v8.4)')
    parser.add_argument('--image_folder', type=str, default='./data/split_v9/test_image',
                        help='3D image folder (nii.gz)')
    parser.add_argument('--mask_folder', type=str, default='./data/split_v9/test_mask',
                        help='3D segmentation mask folder (nii.gz)')
    parser.add_argument('--pred_csv', type=str, default='/mnt/tidal-alsh-share2/dataset/askfollow/usr/yangxin/keyan/repo-dev_v4/repo/data/inference_results_v8.7.csv',
                        help='Prediction CSV from v8 inference pipeline')
    parser.add_argument('--excel', type=str, default=None,
                        help='Excel with ground truth grades (optional)')
    parser.add_argument('--output_dir', type=str, default='./data/report_v8.4',
                        help='Output directory for reports')
    parser.add_argument('--case_id', type=str, default=None,
                        help='Specific case to visualize (default: all)')
    parser.add_argument('--num_slices', type=int, default=3,
                        help='Number of representative slices')
    parser.add_argument('--rotate_90', action='store_true', default=True,
                        help='顺时针旋转图像90度（默认开启）')
    parser.add_argument('--no_rotate', action='store_true',
                        help='禁用图像旋转')
    parser.add_argument('--workers', type=int, default=8,
                        help='Number of parallel workers (default: 4)')
    args = parser.parse_args()
    if args.no_rotate:
        args.rotate_90 = False

    print("=" * 60)
    print("  Knee Cartilage Diagnosis Report Generator (v8.4+ v2.3)")
    if args.rotate_90:
        print("  [Mode: Rotated 90]")
    print(f"  [Workers: {args.workers}]")
    print("=" * 60)

    # 加载数据
    pred_dict = load_predictions(args.pred_csv)
    print(f"Loaded predictions for {len(pred_dict)} cases")

    gt_dict = load_ground_truth(args.excel)
    if gt_dict:
        print(f"Loaded ground truth for {len(gt_dict)} cases")
    else:
        print("No ground truth provided (predictions only)")

    # 确定要处理的病例 — 以 pred_csv 中的病例为准，不处理没有预测结果的病例
    if args.case_id:
        case_ids = [args.case_id]
    else:
        if pred_dict:
            case_ids = sorted(pred_dict.keys())
            print(f"  Using {len(case_ids)} cases from pred_csv (skipping cases without predictions)")
        else:
            if not os.path.isdir(args.image_folder):
                print(f"[Error] Image folder not found: {args.image_folder}")
                sys.exit(1)

            files = [f for f in os.listdir(args.image_folder) if f.endswith('.nii.gz')]
            case_ids = sorted(set(f.replace('.nii.gz', '') for f in files))

            if not case_ids:
                print("[Error] No nii.gz files found")
                sys.exit(1)

    print(f"Processing {len(case_ids)} case(s) with {args.workers} worker(s)...\n")

    # ---- 预收集所有病例的路径和参数（轻量操作，串行执行） ----
    tasks = []
    skipped = 0
    # 图像搜索目录列表 — 在多个目录中查找 nii.gz 文件
    image_search_dirs = [args.image_folder]
    for extra_dir in ['./data/image_3d', './data/split_v9/test_image']:
        if extra_dir not in image_search_dirs and os.path.isdir(extra_dir):
            image_search_dirs.append(extra_dir)
    mask_search_dirs = [args.mask_folder]
    for extra_dir in ['./data/mask_3d', './data/split_v9/test_mask']:
        if extra_dir not in mask_search_dirs and os.path.isdir(extra_dir):
            mask_search_dirs.append(extra_dir)

    def _find_file(cid, search_dirs):
        """在多个目录中搜索病例的 nii.gz 文件"""
        for folder in search_dirs:
            path = os.path.join(folder, f"{cid}.nii.gz")
            if os.path.exists(path):
                return path
            if os.path.isdir(folder):
                candidates = [f for f in os.listdir(folder)
                              if f.startswith(cid) and f.endswith('.nii.gz')]
                if candidates:
                    return os.path.join(folder, candidates[0])
        return None

    for cid in case_ids:
        image_path = _find_file(cid, image_search_dirs)
        mask_path = _find_file(cid, mask_search_dirs)

        if not image_path:
            print(f"  [Skip] Image not found for {cid}")
            skipped += 1
            continue
        if not mask_path:
            print(f"  [Skip] Mask not found for {cid}")
            skipped += 1
            continue

        # 对 pred_dict 的 key 做模糊匹配 (case_id 可能是带后缀的)
        pred_case_id = cid.split('_')[0] if cid not in pred_dict else cid
        pred_case = pred_dict.get(pred_case_id, pred_dict.get(cid, {}))
        gt_case = gt_dict.get(cid, gt_dict.get(cid.split('_')[0],
                          gt_dict.get(cid.split('-')[0], {}))) if gt_dict else {}

        tasks.append((
            cid, image_path, mask_path,
            pred_case, gt_case,
            args.output_dir, args.num_slices, args.rotate_90
        ))

    # ---- 并行生成报告 ----
    success = 0
    n_workers = min(args.workers, len(tasks))

    if n_workers <= 1 or len(tasks) <= 1:
        # 单任务或单 worker，直接串行执行（避免进程池开销）
        for task in tasks:
            if _generate_report_worker(task):
                success += 1
    else:
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            future_to_cid = {
                executor.submit(_generate_report_worker, task): task[0]
                for task in tasks
            }
            for future in as_completed(future_to_cid):
                cid = future_to_cid[future]
                try:
                    if future.result():
                        success += 1
                except Exception as e:
                    print(f"  [Error] Failed for {cid}: {e}")

    print("\n" + "=" * 60)
    print(f"Done! Generated {success}/{len(tasks)} reports"
          + (f" ({skipped} skipped)" if skipped > 0 else ""))
    print(f"Output: {args.output_dir}")
    print("=" * 60)

    # ===== 汇总评估指标（需要同时有 pred 和 gt）=====
    if gt_dict and pred_dict:
        print("\nComputing summary metrics (predictions vs ground truth)...")
        region_results = compute_metrics_per_region(pred_dict, gt_dict)
        has_any = any(v is not None for v in region_results.values())
        if has_any:
            render_summary_report(region_results, args.output_dir)
        else:
            print("  [Warning] No matched pred-GT pairs found; skipping summary.")
    else:
        if not gt_dict:
            print("\nNo ground truth Excel provided — skipping summary metrics.")
        if not pred_dict:
            print("\nNo predictions CSV loaded — skipping summary metrics.")


if __name__ == "__main__":
    main()

