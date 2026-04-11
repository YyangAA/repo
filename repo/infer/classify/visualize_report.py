#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
visualize_report.py
综合可视化诊断报告：生成包含 分割结果、损伤预测、真实标签对比、损伤热力图 的完整展示图。
当提供了 Excel 真实标签时，还会生成汇总评估报告（AUC / ROC 曲线 / 各项分类指标）。

用法:
    python infer/visualize_report.py \
        --image_folder  "./data/image_3d" \
        --mask_folder   "./data/mask_3d" \
        --pred_csv      "./data/inference_results.csv" \
        --output_dir    "./data/report" \
        --excel         "/path/to/ground_truth.xlsx"   (可选: 有真实标签时传入)
        --case_id       "张三"                         (可选: 不传则批量处理全部)
        --rotate_90                                    (可选: 顺时针旋转90度)

输出:
    每个病例生成一张报告图 report_{case_id}.png，包含：
    - Row 0: 代表性切片的分割叠加图（4 区域不同颜色）
    - Row 1: 损伤概率热力图（绿=正常, 红=损伤）
    - Row 2: 诊断结果文本面板（预测 vs 真实标签对比表）
    当有真实标签时额外生成:
    - summary_metrics.png: ROC 曲线 + 指标汇总表
    - summary_metrics.csv: 指标数据表
"""

import os
import sys
import argparse
import numpy as np
import SimpleITK as sitk

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
plt.rcParams['font.sans-serif'] = ['WenQuanYi Micro Hei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# 验证字体可用性
print(f"[Font] Available fonts: {plt.rcParams['font.sans-serif']}")
available_fonts = [f.name for f in fm.fontManager.ttflist]
print(f"[Font] WenQuanYi available: {'WenQuanYi Micro Hei' in available_fonts}")

import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Patch
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
    """读取推理输出的 CSV"""
    if csv_path is None or not os.path.exists(csv_path):
        return {}

    df = pd.read_csv(csv_path)
    pred_dict = {}
    for _, row in df.iterrows():
        cid = str(row["case_id"])
        region = row["region"]
        pred = int(row.get("predicted_label", row.get("predicted_grade", 0)))
        prob = float(row.get("probability_damage", 0.0))

        if cid not in pred_dict:
            pred_dict[cid] = {}
        pred_dict[cid][region] = {"pred": pred, "prob": prob}

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
    # 旋转图像 (逆时针-90度 = 顺时针90度)
    img_rotated = ndimage.rotate(img_slice, angle=-90, reshape=False, order=1)
    # 旋转mask (最近邻插值保持标签值)
    mask_rotated = ndimage.rotate(mask_slice, angle=-90, reshape=False, order=0)
    return img_rotated, mask_rotated


# ===============================
# 4. 绘制分割叠加图
# ===============================

def render_segmentation_overlay(ax, img_slice, mask_slice, title=""):
    """原图 + 彩色分割叠加"""
    ax.imshow(img_slice, cmap='gray', aspect='equal')

    overlay = np.zeros((*mask_slice.shape, 4), dtype=np.float32)
    for label_id, color in REGION_COLORS.items():
        region_mask = (mask_slice == label_id)
        if np.any(region_mask):
            overlay[region_mask] = color

    ax.imshow(overlay, aspect='equal')
    ax.set_title(title, fontsize=9, fontweight='bold', pad=4)
    ax.axis('off')


# ===============================
# 5. 绘制损伤热力图
# ===============================

def render_damage_heatmap(ax, img_slice, mask_slice, pred_case, title=""):
    """根据各区域损伤概率生成热力图叠加"""
    ax.imshow(img_slice, cmap='gray', aspect='equal')

    heat = np.full(mask_slice.shape, np.nan, dtype=np.float32)
    for label_id, region_name in REGION_NAMES.items():
        region_mask = (mask_slice == label_id)
        if not np.any(region_mask):
            continue
        prob = 0.0
        if pred_case and region_name in pred_case:
            prob = pred_case[region_name]["prob"]
        heat[region_mask] = prob

    masked_heat = np.ma.masked_where(np.isnan(heat), heat)
    im = ax.imshow(masked_heat, cmap=DAMAGE_CMAP, vmin=0, vmax=1, alpha=0.7, aspect='equal')
    ax.set_title(title, fontsize=9, fontweight='bold', pad=4)
    ax.axis('off')
    return im


# ===============================
# 6. 诊断结果面板
# ===============================

def render_diagnosis_panel(ax, case_id, pred_case, gt_case):
    """绘制文字对比表：预测结果 vs 真实标签"""
    ax.axis('off')

    # 标题
    ax.text(0.5, 0.97, f"Case: {case_id}", fontsize=13, fontweight='bold',
            ha='center', va='top', transform=ax.transAxes)

    # 列位置
    col_x = [0.03, 0.20, 0.42, 0.58, 0.76, 0.92]
    headers = ["Region", "Predict", "Prob", "GT Grade", "GT Label", "Match"]

    y = 0.82
    for i, h in enumerate(headers):
        ax.text(col_x[i], y, h, fontsize=8.5, fontweight='bold', va='top',
                transform=ax.transAxes,
                color='#2c3e50')

    ax.plot([0.01, 0.99], [y - 0.03, y - 0.03], color='#bdc3c7', linewidth=0.8,
            transform=ax.transAxes)

    y -= 0.10

    for label_id, region_name in REGION_NAMES.items():
        display_name = REGION_CN.get(region_name, region_name)
        color_rgb = REGION_COLORS[label_id][:3]

        # 区域名（带颜色标记）
        ax.text(col_x[0], y, display_name, fontsize=8.5, va='top', color=color_rgb,
                fontweight='bold', transform=ax.transAxes)

        # 预测结果
        if pred_case and region_name in pred_case:
            pred = pred_case[region_name]["pred"]
            prob = pred_case[region_name]["prob"]
            pred_str = "Damaged" if pred == 1 else "Normal"
            pred_color = '#e74c3c' if pred == 1 else '#27ae60'
            prob_str = f"{prob:.1%}"
        else:
            pred_str, pred_color, prob_str = "N/A", '#95a5a6', "-"
            pred = -1

        ax.text(col_x[1], y, pred_str, fontsize=8.5, va='top', color=pred_color,
                fontweight='bold', transform=ax.transAxes)
        ax.text(col_x[2], y, prob_str, fontsize=8.5, va='top',
                transform=ax.transAxes)

        # 真实标签
        if gt_case and region_name in gt_case:
            gt = gt_case[region_name]
            if gt >= 0:
                gt_grade_str = f"Grade {gt}"
                gt_label = "Damaged" if gt > 0 else "Normal"
                gt_label_color = '#e74c3c' if gt > 0 else '#27ae60'
                gt_binary = 1 if gt > 0 else 0
            else:
                gt_grade_str, gt_label, gt_label_color, gt_binary = "N/A", "N/A", '#95a5a6', -1
        else:
            gt_grade_str, gt_label, gt_label_color, gt_binary = "N/A", "N/A", '#95a5a6', -1

        ax.text(col_x[3], y, gt_grade_str, fontsize=8.5, va='top',
                transform=ax.transAxes)
        ax.text(col_x[4], y, gt_label, fontsize=8.5, va='top', color=gt_label_color,
                fontweight='bold', transform=ax.transAxes)

        # 匹配判断
        if gt_binary >= 0 and pred >= 0:
            if pred == gt_binary:
                match_str, match_color = "OK", '#27ae60'
            else:
                match_str, match_color = "MISS", '#e74c3c'
        else:
            match_str, match_color = "-", '#95a5a6'

        ax.text(col_x[5], y, match_str, fontsize=9, va='top', color=match_color,
                fontweight='bold', transform=ax.transAxes)

        y -= 0.13

    # 底部分割线
    y -= 0.02
    ax.plot([0.01, 0.99], [y, y], color='#bdc3c7', linewidth=0.5,
            transform=ax.transAxes)

    # 区域颜色图例
    legend_items = [
        Patch(facecolor=REGION_COLORS[lid][:3],
              label=f"{lid}: {REGION_CN[rn]}")
        for lid, rn in REGION_NAMES.items()
    ]
    ax.legend(handles=legend_items, loc='lower center', fontsize=7,
              ncol=4, frameon=True, fancybox=True,
              bbox_to_anchor=(0.5, -0.02), bbox_transform=ax.transAxes)


# ===============================
# 7. 汇总评估指标
# ===============================

def compute_metrics_per_region(pred_dict, gt_dict):
    """
    汇总所有病例，按区域计算分类评估指标。

    对 pred_dict / gt_dict 的 key 做模糊匹配（去掉 _knee 后缀等）。
    真实标签 grade > 0 → 1 (Damaged), grade == 0 → 0 (Normal)。

    返回: {region_name: metrics_dict or None}
    """
    # 建立 gt_dict 模糊匹配映射
    gt_key_map = {}
    for k in gt_dict:
        gt_key_map[k] = k
        gt_key_map[k.replace("_knee", "").strip()] = k

    region_results = {}

    for region_name in REGION_NAMES.values():
        y_true, y_prob, y_pred = [], [], []

        for case_id, pred_case in pred_dict.items():
            if region_name not in pred_case:
                continue

            # 模糊查找 gt
            lookup_keys = [case_id, case_id.split('_')[0], case_id.replace("_knee", "")]
            gt_case = None
            for lk in lookup_keys:
                if lk in gt_key_map:
                    gt_case = gt_dict[gt_key_map[lk]]
                    break
            if gt_case is None or region_name not in gt_case:
                continue

            gt_grade = gt_case[region_name]
            if gt_grade < 0:        # 缺失标注
                continue

            gt_binary = 1 if gt_grade > 0 else 0
            y_true.append(gt_binary)
            y_prob.append(pred_case[region_name]["prob"])
            y_pred.append(pred_case[region_name]["pred"])

        if len(y_true) < 2:
            region_results[region_name] = None
            continue

        y_true = np.array(y_true)
        y_prob = np.array(y_prob)
        y_pred = np.array(y_pred)

        metrics = {
            'n':     len(y_true),
            'n_pos': int(y_true.sum()),
            'n_neg': int((1 - y_true).sum()),
        }

        # AUC & ROC（需要至少含两个类别）
        if len(np.unique(y_true)) >= 2:
            fpr, tpr, _ = roc_curve(y_true, y_prob)
            metrics['auc'] = auc(fpr, tpr)
            metrics['fpr'] = fpr
            metrics['tpr'] = tpr
        else:
            metrics['auc'] = float('nan')
            metrics['fpr'] = None
            metrics['tpr'] = None

        metrics['accuracy']  = accuracy_score(y_true, y_pred)
        metrics['precision'] = precision_score(y_true, y_pred, zero_division=0)
        metrics['recall']    = recall_score(y_true, y_pred, zero_division=0)     # sensitivity
        metrics['f1']        = f1_score(y_true, y_pred, zero_division=0)

        # Specificity
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()
        metrics['specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        metrics['tp'] = int(tp)
        metrics['tn'] = int(tn)
        metrics['fp'] = int(fp)
        metrics['fn'] = int(fn)

        region_results[region_name] = metrics

    return region_results


def render_summary_report(region_results, output_dir):
    """
    生成汇总评估报告图 (summary_metrics.png) + CSV (summary_metrics.csv)。
    上半部分: 4 条 ROC 曲线；下半部分: 指标汇总表 + 混淆矩阵。
    """

    fig = plt.figure(figsize=(16, 11), facecolor='white')
    gs = gridspec.GridSpec(2, 1, height_ratios=[1.3, 1], hspace=0.28,
                           left=0.07, right=0.96, top=0.93, bottom=0.04)

    # ------- ROC 曲线 -------
    ax_roc = fig.add_subplot(gs[0])

    line_colors = {
        "Femur_Medial":  '#2ecc71',
        "Femur_Lateral": '#3498db',
        "Tibia_Medial":  '#e67e22',
        "Tibia_Lateral": '#9b59b6',
    }

    has_roc = False
    for region_name in REGION_NAMES.values():
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
    ax_roc.set_title('ROC Curves — Per Region', fontsize=14, fontweight='bold')
    ax_roc.legend(loc='lower right', fontsize=10, frameon=True, fancybox=True)
    ax_roc.grid(True, alpha=0.3)

    if not has_roc:
        ax_roc.text(0.5, 0.5,
                    "Insufficient data for ROC curves\n"
                    "(need both positive and negative samples per region)",
                    ha='center', va='center', fontsize=12, color='#95a5a6',
                    transform=ax_roc.transAxes)

    # ------- 指标汇总表 -------
    ax_tbl = fig.add_subplot(gs[1])
    ax_tbl.axis('off')

    ax_tbl.text(0.5, 0.97, "Classification Metrics Summary",
                fontsize=14, fontweight='bold', ha='center', va='top',
                transform=ax_tbl.transAxes)

    col_x = [0.01, 0.13, 0.23, 0.33, 0.43, 0.53, 0.63, 0.72, 0.82, 0.93]
    headers = ["Region", "AUC", "Acc", "Sens", "Spec", "Prec", "F1",
               "N", "Pos/Neg", "TP/FP/FN/TN"]

    y = 0.84
    for i, h in enumerate(headers):
        ax_tbl.text(col_x[i], y, h, fontsize=9.5, fontweight='bold', va='top',
                    transform=ax_tbl.transAxes,
                    color='#2c3e50')
    ax_tbl.plot([0.0, 1.0], [y - 0.04, y - 0.04], color='#bdc3c7',
                linewidth=1, transform=ax_tbl.transAxes)

    y -= 0.14
    for region_name in REGION_NAMES.values():
        m = region_results.get(region_name)
        display_name = REGION_CN.get(region_name, region_name)
        label_id = [k for k, v in REGION_NAMES.items() if v == region_name][0]
        color_rgb = REGION_COLORS[label_id][:3]

        ax_tbl.text(col_x[0], y, display_name, fontsize=10, va='top',
                    color=color_rgb, fontweight='bold',
                    transform=ax_tbl.transAxes)

        if m is None:
            ax_tbl.text(col_x[1], y, "Insufficient data (< 2 samples)",
                        fontsize=9.5, va='top', color='#95a5a6',
                        transform=ax_tbl.transAxes)
        else:
            values = [
                f"{m['auc']:.3f}" if not np.isnan(m['auc']) else "N/A",
                f"{m['accuracy']:.3f}",
                f"{m['recall']:.3f}",
                f"{m['specificity']:.3f}",
                f"{m['precision']:.3f}",
                f"{m['f1']:.3f}",
                f"{m['n']}",
                f"{m['n_pos']}/{m['n_neg']}",
                f"{m['tp']}/{m['fp']}/{m['fn']}/{m['tn']}",
            ]
            for i, v in enumerate(values):
                ax_tbl.text(col_x[i + 1], y, v, fontsize=9.5, va='top',
                            transform=ax_tbl.transAxes)
        y -= 0.15

    # 底部分割线
    y -= 0.02
    ax_tbl.plot([0.0, 1.0], [y, y], color='#bdc3c7', linewidth=0.5,
                transform=ax_tbl.transAxes)

    # ---- 保存图片 ----
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, "summary_metrics.png")
    fig.savefig(save_path, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"\n  [Summary] Metrics report image saved: {save_path}")

    # ---- 同时打印到控制台 ----
    print("\n  ========= Classification Metrics per Region =========")
    for region_name in REGION_NAMES.values():
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
    print("  " + "=" * 52)

    # ---- 保存 CSV ----
    csv_path = os.path.join(output_dir, "summary_metrics.csv")
    rows = []
    for region_name in REGION_NAMES.values():
        m = region_results.get(region_name)
        if m is None:
            rows.append({"region": region_name, "n": 0, "note": "insufficient_data"})
            continue
        rows.append({
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
        })
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    print(f"  [Summary] Metrics CSV saved: {csv_path}")


# ===============================
# 8. 单病例报告生成（含旋转功能）
# ===============================

def generate_report(case_id, image_path, mask_path,
                    pred_dict, gt_dict, output_dir, num_slices=3, rotate_90=False):
    """
    为单个病例生成完整可视化诊断报告。

    参数:
        rotate_90: 是否顺时针旋转90度
    
    布局:
    ┌───────────┬───────────┬───────────┐
    │  分割叠加   │  分割叠加   │  分割叠加   │  ← 代表性切片 + 4色分割
    ├───────────┼───────────┼───────────┤
    │  损伤热力图 │  损伤热力图 │  损伤热力图 │  ← 按损伤概率着色
    ├───────────┴───────────┴───────────┤
    │         诊断结果对比面板              │  ← 预测 vs 真实标签
    └──────────────────────────────────┘
    """
    print(f"  Generating report for: {case_id}" + (" [Rotated 90°]" if rotate_90 else ""))

    try:
        image_sitk = sitk.ReadImage(image_path)
        mask_sitk = sitk.ReadImage(mask_path)
        mask_sitk = sitk.Cast(mask_sitk, sitk.sitkUInt8)

        img_array = sitk.GetArrayFromImage(image_sitk)    # (D, H, W)
        mask_array = sitk.GetArrayFromImage(mask_sitk)     # (D, H, W)
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

    # 创建画布
    fig_w = max(4.5 * n, 10)
    fig = plt.figure(figsize=(fig_w, 13), facecolor='white')

    gs = gridspec.GridSpec(3, n, height_ratios=[1, 1, 0.7],
                           hspace=0.20, wspace=0.06,
                           left=0.03, right=0.92, top=0.95, bottom=0.03)

    heatmap_im = None

    for col, z in enumerate(slices):
        img_slice = img_array[z]
        mask_slice = mask_array[z]
        
        # 如果需要，顺时针旋转90度
        if rotate_90:
            img_slice, mask_slice = rotate_slice_90_clockwise(img_slice, mask_slice)

        # Row 0: 分割叠加
        ax_seg = fig.add_subplot(gs[0, col])
        render_segmentation_overlay(ax_seg, img_slice, mask_slice,
                                    title=f"Segmentation (z={z})")

        # Row 1: 热力图
        ax_heat = fig.add_subplot(gs[1, col])
        heatmap_im = render_damage_heatmap(ax_heat, img_slice, mask_slice,
                                           pred_case,
                                           title=f"Damage Heatmap (z={z})")

    # Colorbar
    if heatmap_im is not None:
        cbar_ax = fig.add_axes([0.935, 0.38, 0.015, 0.22])
        cbar = fig.colorbar(heatmap_im, cax=cbar_ax)
        cbar.set_label('Damage Prob', fontsize=8)
        cbar.ax.tick_params(labelsize=7)

    # Row 2: 诊断面板
    ax_panel = fig.add_subplot(gs[2, :])
    render_diagnosis_panel(ax_panel, case_id, pred_case, gt_case)

    # 保存
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, f"report_{case_id}.png")
    fig.savefig(save_path, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"    Saved: {save_path}")
    return True


# ===============================
# 9. 主流程
# ===============================

def main():
    parser = argparse.ArgumentParser(description='Knee Cartilage Diagnosis Report')
    parser.add_argument('--image_folder', type=str, default='./data/image_3d',
                        help='3D image folder (nii.gz)')
    parser.add_argument('--mask_folder', type=str, default='./data/mask_3d',
                        help='3D segmentation mask folder (nii.gz)')
    parser.add_argument('--pred_csv', type=str, default='./data/inference_results.csv',
                        help='Prediction CSV from inference pipeline')
    parser.add_argument('--excel', type=str, default=None,
                        help='Excel with ground truth grades (optional)')
    parser.add_argument('--output_dir', type=str, default='./data/report',
                        help='Output directory for reports')
    parser.add_argument('--case_id', type=str, default=None,
                        help='Specific case to visualize (default: all)')
    parser.add_argument('--num_slices', type=int, default=3,
                        help='Number of representative slices')
    parser.add_argument('--rotate_90', action='store_true',
                        help='顺时针旋转图像90度（用于冠状位/矢状位调整）')
    args = parser.parse_args()

    print("=" * 60)
    print("  Knee Cartilage Diagnosis Report Generator")
    if args.rotate_90:
        print("  [Mode: 顺时针旋转90度]")
    print("=" * 60)

    # 加载数据
    pred_dict = load_predictions(args.pred_csv)
    print(f"Loaded predictions for {len(pred_dict)} cases")

    gt_dict = load_ground_truth(args.excel)
    if gt_dict:
        print(f"Loaded ground truth for {len(gt_dict)} cases")
    else:
        print("No ground truth provided (predictions only)")

    # 确定要处理的病例
    if args.case_id:
        case_ids = [args.case_id]
    else:
        if not os.path.isdir(args.image_folder):
            print(f"[Error] Image folder not found: {args.image_folder}")
            sys.exit(1)

        files = [f for f in os.listdir(args.image_folder) if f.endswith('.nii.gz')]
        case_ids = sorted(set(f.replace('.nii.gz', '') for f in files))

        if not case_ids:
            print("[Error] No nii.gz files found")
            sys.exit(1)

    print(f"Processing {len(case_ids)} case(s)...\n")

    success = 0
    for cid in case_ids:
        image_path = os.path.join(args.image_folder, f"{cid}.nii.gz")
        mask_path = os.path.join(args.mask_folder, f"{cid}.nii.gz")

        # 模糊匹配
        if not os.path.exists(image_path):
            candidates = [f for f in os.listdir(args.image_folder)
                          if f.startswith(cid) and f.endswith('.nii.gz')]
            if candidates:
                image_path = os.path.join(args.image_folder, candidates[0])
                mask_path = os.path.join(args.mask_folder, candidates[0])

        if not os.path.exists(image_path):
            print(f"  [Skip] Image not found for {cid}")
            continue
        if not os.path.exists(mask_path):
            print(f"  [Skip] Mask not found for {cid}")
            continue

        # 对 pred_dict 的 key 做模糊匹配 (case_id 可能是带后缀的)
        pred_case_id = cid.split('_')[0] if cid not in pred_dict else cid

        ok = generate_report(
            cid, image_path, mask_path,
            {cid: pred_dict.get(pred_case_id, pred_dict.get(cid, {}))},
            gt_dict, args.output_dir, args.num_slices,
            rotate_90=args.rotate_90  # 传递旋转参数
        )
        if ok:
            success += 1

    print("\n" + "=" * 60)
    print(f"Done! Generated {success}/{len(case_ids)} reports")
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