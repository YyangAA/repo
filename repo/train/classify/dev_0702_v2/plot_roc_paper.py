#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
plot_roc_paper.py
论文级 ROC 曲线美化绘制脚本

功能:
  Stage 1 (Normal vs Damaged):
    1. 单区域 ROC 曲线（含 5-fold 阴影 + 置信带 + 最佳阈值标注）
    2. 四区域 ROC 合并对比图
    3. Precision-Recall 曲线（单区域 + 合并）
    4. 综合 Figure（ROC + PR 2×2 面板）
  Stage 2 (Grade 1 vs Grade 2):
    5. 池化 ROC / PR 曲线（全体样本 LOOCV）
    6. 各区域 ROC / PR 子图（标注样本量）
    7. Stage 2 综合面板

模型/数据配置 (脚本顶部常量):
  MODEL_DIR   = ./checkpoint/results_v8.9_0702_v2                 # 输入模型
  FEATURE_DIR = ./train/classify/dev_0702_v2/data_train/feature  # Stage2/回退用特征
  OOF_CSV     = ./data/in_domain_cv_results_v8.9_0702_v2/oof_predictions.csv  # Stage1 数据源
  OUTPUT_DIR  = ./checkpoint/results_v8.9_0702_v2_paper_figures   # 输出目录

Stage 1 数据来源说明:
  Stage 1 直接读取 in_domain_cv_eval.py 产出的 OOF_CSV（oof_predictions.csv）,
  按"聚合所有折 out-of-fold 预测"方式算 ROC/PR, 因此图中 AUC 与
  集内测试表 1 (metrics_summary.csv) 严格一致。此模式下无折间方差,
  故 SHOW_STD=False（不显示 ±std）。若 OOF_CSV 不存在, 则回退到从特征 CSV 重跑 CV。

使用方式 (如何画出本图):
  cd /mnt/tidal-alsh-share2/dataset/askfollow/usr/yangxin/keyan/repo-dev_v4/repo
  # 1) 先确保集内测试 OOF 已生成 (若 oof_predictions.csv 已存在可跳过):
  venv310/bin/python train/classify/dev_0702_v2/in_domain_cv_eval.py
  # 2) 再画图 (Stage1 图 AUC 与表 1 一致):
  venv310/bin/python train/classify/dev_0702_v2/plot_roc_paper.py
  # 输出: checkpoint/results_v8.9_0702_v2_paper_figures/S1_Figure_Panel.png 等
"""

import os
import sys
import numpy as np
import pandas as pd
import joblib
import warnings
warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import matplotlib.patheffects as pe
from sklearn.model_selection import GroupKFold, LeaveOneGroupOut, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import (
    roc_curve, auc, precision_recall_curve, average_precision_score,
    roc_auc_score
)

# ============================================================
# 全局配置
# ============================================================
# 路径
FEATURE_DIR = "./train/classify/dev_0702_v2/data_train/feature"
MODEL_DIR = "./checkpoint/results_v8.9_0702_v2"
OUTPUT_DIR = "./checkpoint/results_v8.9_0702_v2_paper_figures"
SHOW_STD = False  # Stage1 用聚合OOF, std=0, 不显示

# 区域定义
REGIONS = ["Femur_Medial", "Femur_Lateral", "Tibia_Medial", "Tibia_Lateral"]
REGION_LABELS = {
    "Femur_Medial": "MFC (内侧股骨髁)",
    "Femur_Lateral": "LFC (外侧股骨髁)",
    "Tibia_Medial": "MTP (内侧胫骨平台)",
    "Tibia_Lateral": "LTP (外侧胫骨平台)",
}
REGION_LABELS_EN = {
    "Femur_Medial": "MFC (Medial Femoral Condyle)",
    "Femur_Lateral": "LFC (Lateral Femoral Condyle)",
    "Tibia_Medial": "MTP (Medial Tibial Plateau)",
    "Tibia_Lateral": "LTP (Lateral Tibial Plateau)",
}

# 配色方案 (参考 summary_metrics.png 风格)
COLORS = {
    "Femur_Medial":  "#2ecc71",   # 翠绿
    "Femur_Lateral": "#3498db",   # 天蓝
    "Tibia_Medial":  "#e67e22",   # 橙色
    "Tibia_Lateral": "#9b59b6",   # 紫色
}
COLOR_LIGHT = {
    "Femur_Medial":  "#a9dfbf",
    "Femur_Lateral": "#aed6f1",
    "Tibia_Medial":  "#f5cba7",
    "Tibia_Lateral": "#d2b4de",
}

# Matplotlib 全局样式
plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor": "#fafafa",
    "axes.edgecolor": "#cccccc",
    "axes.linewidth": 0.8,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "grid.linestyle": "--",
    "grid.color": "#cccccc",
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 12,
    "legend.fontsize": 10,
    "legend.framealpha": 0.9,
    "legend.edgecolor": "#cccccc",
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.15,
})

# 尝试加载中文字体
try:
    from matplotlib.font_manager import FontProperties
    _cn_fonts = [
        "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
        "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    ]
    CN_FONT = None
    for fp in _cn_fonts:
        if os.path.exists(fp):
            CN_FONT = FontProperties(fname=fp, size=11)
            plt.rcParams["axes.unicode_minus"] = False
            break
except:
    CN_FONT = None

N_INTERP = 200  # 插值点数量（比原始 100 更平滑）


# ============================================================
# 数据加载与 CV 重算
# ============================================================
def load_region_data(region):
    """加载某区域的训练特征数据"""
    cross_csv = os.path.join(FEATURE_DIR, f"{region}_cross_filtered_features.csv")
    normal_csv = os.path.join(FEATURE_DIR, f"{region}_filtered_features.csv")

    csv_path = cross_csv if os.path.exists(cross_csv) else normal_csv
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Feature CSV not found for {region}")

    df_raw = pd.read_csv(csv_path)
    df = df_raw.groupby(["case_id", "region", "grade"]).mean().reset_index()

    drop_cols = [c for c in ["case_id", "region", "grade", "cartilage_missing"] if c in df.columns]
    X = df.drop(columns=drop_cols)
    y = (df["grade"] > 0).astype(int).values
    groups = df["case_id"].values

    if X.isnull().values.any():
        X = X.fillna(0)

    return X, y, groups


def load_model_params(region):
    """加载 v8.7 模型的超参数"""
    params_path = os.path.join(MODEL_DIR, region, "models", "model_params.pkl")
    if os.path.exists(params_path):
        params = joblib.load(params_path)
        return params["C"], params["gamma"], params.get("class_weight", None)
    else:
        print(f"  Warning: model_params.pkl not found for {region}, using defaults")
        return 1, "scale", None


OOF_CSV = "./data/in_domain_cv_results_v8.9_0702_v2/oof_predictions.csv"


def compute_curves_from_oof(region):
    """
    从 in_domain_cv_eval.py 生成的 oof_predictions.csv 读取该区域的
    OOF 预测概率，按“聚合所有折”方式算一条 ROC/PR，
    使图的 AUC 与集内测试表 1 (metrics_summary.csv) 严格一致。
    返回结构与 compute_cv_curves 兼容。
    """
    import pandas as pd
    df = pd.read_csv(OOF_CSV)
    sub = df[df["region"] == region]
    y = sub["true_binary"].values.astype(int)
    prob = sub["oof_prob_damage"].values.astype(float)

    mean_fpr = np.linspace(0, 1, 100)
    mean_recall = np.linspace(0, 1, 100)

    fpr, tpr, _ = roc_curve(y, prob)
    roc_auc = auc(fpr, tpr)
    interp_tpr = np.interp(mean_fpr, fpr, tpr); interp_tpr[0] = 0.0; interp_tpr[-1] = 1.0

    prec, rec, _ = precision_recall_curve(y, prob)
    ap = average_precision_score(y, prob)
    order = np.argsort(rec)
    interp_prec = np.interp(mean_recall, rec[order], prec[order])

    # Youden's J 最佳阈值
    youden = tpr - fpr
    bi = int(np.argmax(youden))
    from sklearn.metrics import roc_curve as _rc
    fpr2, tpr2, thr2 = _rc(y, prob)
    bthr = float(thr2[bi]) if bi < len(thr2) else 0.5
    bfpr, btpr = float(fpr2[bi]), float(tpr2[bi])

    fold_curves = [{
        "fpr": fpr, "tpr": tpr, "roc_auc": roc_auc,
        "prec": prec, "rec": rec, "ap": ap,
        "best_thresh": bthr, "best_fpr": bfpr, "best_tpr": btpr,
    }]

    return {
        "mean_fpr": mean_fpr, "mean_tpr": interp_tpr, "std_tpr": np.zeros_like(mean_fpr),
        "mean_recall": mean_recall, "mean_prec": interp_prec, "std_prec": np.zeros_like(mean_recall),
        "mean_roc_auc": roc_auc, "std_roc_auc": 0.0,
        "mean_pr_auc": ap, "std_pr_auc": 0.0,
        "tprs_interp": [interp_tpr], "precs_interp": [interp_prec],
        "fold_curves": fold_curves,
        "best_thresh": bthr, "best_fpr": bfpr, "best_tpr": btpr,
        "best_threshold": (bthr, bfpr, btpr),
        "n_pos": int(y.sum()), "n_neg": int((1 - y).sum()),
    }


def compute_cv_curves(X, y, groups, C, gamma, class_weight=None, n_splits=5):
    """
    5-Fold GroupKFold CV，返回每折和平均的 ROC / PR 数据
    """
    cv = GroupKFold(n_splits=n_splits)
    mean_fpr = np.linspace(0, 1, N_INTERP)
    mean_recall = np.linspace(0, 1, N_INTERP)

    tprs_interp = []
    precs_interp = []
    auc_roc_list = []
    auc_pr_list = []
    fold_curves = []   # 保存每折原始曲线
    threshold_list = []

    for fold_idx, (train_ix, test_ix) in enumerate(cv.split(X, y, groups=groups)):
        X_train, X_test = X.iloc[train_ix], X.iloc[test_ix]
        y_train, y_test = y[train_ix], y[test_ix]

        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)

        model = SVC(kernel="rbf", C=C, gamma=gamma, probability=True,
                     class_weight=class_weight)
        model.fit(X_train_s, y_train)
        y_prob = model.predict_proba(X_test_s)[:, 1]

        # --- ROC ---
        fpr, tpr, thresholds_roc = roc_curve(y_test, y_prob)
        roc_auc_val = auc(fpr, tpr)
        auc_roc_list.append(roc_auc_val)

        interp_tpr = np.interp(mean_fpr, fpr, tpr)
        interp_tpr[0] = 0.0
        tprs_interp.append(interp_tpr)

        # 最佳阈值 (Youden's J)
        youden_j = tpr - fpr
        best_idx = np.argmax(youden_j)
        best_thresh = thresholds_roc[best_idx]
        best_fpr = fpr[best_idx]
        best_tpr = tpr[best_idx]
        threshold_list.append((best_thresh, best_fpr, best_tpr))

        # --- PR ---
        prec, rec, _ = precision_recall_curve(y_test, y_prob)
        ap_val = average_precision_score(y_test, y_prob)
        auc_pr_list.append(ap_val)

        # PR 需要反转 (recall 降序 → 升序)
        rec_sorted = rec[::-1]
        prec_sorted = prec[::-1]
        interp_prec = np.interp(mean_recall, rec_sorted, prec_sorted)
        precs_interp.append(interp_prec)

        fold_curves.append({
            "fpr": fpr, "tpr": tpr, "roc_auc": roc_auc_val,
            "prec": prec, "rec": rec, "ap": ap_val,
            "best_thresh": best_thresh, "best_fpr": best_fpr, "best_tpr": best_tpr,
        })

    # 汇总
    mean_tpr = np.mean(tprs_interp, axis=0)
    mean_tpr[-1] = 1.0
    std_tpr = np.std(tprs_interp, axis=0)

    mean_prec = np.mean(precs_interp, axis=0)
    std_prec = np.std(precs_interp, axis=0)

    mean_roc_auc = np.mean(auc_roc_list)
    std_roc_auc = np.std(auc_roc_list)
    mean_pr_auc = np.mean(auc_pr_list)
    std_pr_auc = np.std(auc_pr_list)

    # 平均最佳阈值点
    avg_thresh_fpr = np.mean([t[1] for t in threshold_list])
    avg_thresh_tpr = np.mean([t[2] for t in threshold_list])
    avg_thresh_val = np.median([t[0] for t in threshold_list])

    return {
        "mean_fpr": mean_fpr,
        "mean_tpr": mean_tpr,
        "std_tpr": std_tpr,
        "mean_roc_auc": mean_roc_auc,
        "std_roc_auc": std_roc_auc,
        "mean_recall": mean_recall,
        "mean_prec": mean_prec,
        "std_prec": std_prec,
        "mean_pr_auc": mean_pr_auc,
        "std_pr_auc": std_pr_auc,
        "fold_curves": fold_curves,
        "tprs_interp": tprs_interp,
        "precs_interp": precs_interp,
        "auc_roc_list": auc_roc_list,
        "auc_pr_list": auc_pr_list,
        "best_threshold": (avg_thresh_val, avg_thresh_fpr, avg_thresh_tpr),
    }


# ============================================================
# 绘图函数
# ============================================================

def _add_watermark(ax, text="5T MRI · SVM-RBF", alpha=0.06):
    """右下角添加淡水印"""
    ax.text(0.97, 0.03, text, transform=ax.transAxes,
            fontsize=8, color="#888888", alpha=alpha,
            ha="right", va="bottom", style="italic")


def plot_single_roc(region, data, save_path):
    """
    单区域 ROC 曲线
    包含: 5-fold 半透明曲线 + 均值曲线 + ±1σ 置信带 + 最佳阈值点
    """
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    color = COLORS[region]
    color_light = COLOR_LIGHT[region]

    # 对角参考线
    ax.plot([0, 1], [0, 1], ls="--", color="#bdc3c7", lw=1.5, alpha=0.7,
            label="Chance level (AUC = 0.500)")

    # 各折 ROC（半透明）
    for i, fc in enumerate(data["fold_curves"]):
        ax.plot(fc["fpr"], fc["tpr"], color=color, alpha=0.15, lw=1.0)

    # ±1σ 置信带
    upper = np.minimum(data["mean_tpr"] + data["std_tpr"], 1)
    lower = np.maximum(data["mean_tpr"] - data["std_tpr"], 0)
    ax.fill_between(data["mean_fpr"], lower, upper,
                     color=color_light, alpha=0.35,
                     label=f"±1 std. dev.")

    # 均值 ROC 主曲线
    ax.plot(data["mean_fpr"], data["mean_tpr"],
            color=color, lw=2.8, alpha=0.95,
            label=f"Mean ROC (AUC = {data['mean_roc_auc']:.3f})",
            path_effects=[pe.Stroke(linewidth=4.0, foreground="white"), pe.Normal()])

    # 最佳阈值点标注
    bt_val, bt_fpr, bt_tpr = data["best_threshold"]
    ax.scatter([bt_fpr], [bt_tpr], s=90, color=color, edgecolors="white",
               linewidths=1.8, zorder=10)
    ax.annotate(f"Threshold={bt_val:.3f}\n(FPR={bt_fpr:.2f}, TPR={bt_tpr:.2f})",
                xy=(bt_fpr, bt_tpr), xytext=(bt_fpr + 0.12, bt_tpr - 0.12),
                fontsize=8.5, color="#333333",
                arrowprops=dict(arrowstyle="-|>", color="#777777", lw=1.0),
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#cccccc", alpha=0.85))

    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.05])
    ax.set_xlabel("False Positive Rate (1 - Specificity)")
    ax.set_ylabel("True Positive Rate (Sensitivity)")
    ax.set_title(f"ROC Curve — {REGION_LABELS_EN[region]}", fontweight="bold", pad=12)
    ax.legend(loc="lower right", frameon=True)
    _add_watermark(ax)

    fig.tight_layout()
    fig.savefig(save_path, dpi=300, facecolor="white")
    plt.close(fig)
    print(f"  Saved: {save_path}")


def plot_combined_roc(all_data, save_path):
    """
    四区域 ROC 合并图 (论文 Figure 主图)
    """
    fig, ax = plt.subplots(figsize=(7.5, 6.5))

    # 对角参考线
    ax.plot([0, 1], [0, 1], ls="--", color="#bdc3c7", lw=1.5, alpha=0.7,
            label="Chance level")

    for region in REGIONS:
        d = all_data[region]
        color = COLORS[region]
        color_light = COLOR_LIGHT[region]
        label_short = REGION_LABELS_EN[region].split("(")[0].strip()

        # 置信带
        upper = np.minimum(d["mean_tpr"] + d["std_tpr"], 1)
        lower = np.maximum(d["mean_tpr"] - d["std_tpr"], 0)
        ax.fill_between(d["mean_fpr"], lower, upper, color=color_light, alpha=0.2)

        # 均值 ROC
        ax.plot(d["mean_fpr"], d["mean_tpr"],
                color=color, lw=2.5, alpha=0.92,
                label=f"{label_short} (AUC = {d['mean_roc_auc']:.3f})",
                path_effects=[pe.Stroke(linewidth=3.8, foreground="white"), pe.Normal()])

        # 最佳阈值点
        bt_val, bt_fpr, bt_tpr = d["best_threshold"]
        ax.scatter([bt_fpr], [bt_tpr], s=60, color=color, edgecolors="white",
                   linewidths=1.5, zorder=10)

    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.05])
    ax.set_xlabel("False Positive Rate (1 - Specificity)", fontsize=12)
    ax.set_ylabel("True Positive Rate (Sensitivity)", fontsize=12)
    ax.set_title("ROC Curves — Four Knee Cartilage Regions (5-Fold CV)",
                 fontweight="bold", fontsize=13.5, pad=14)
    ax.legend(loc="lower right", frameon=True, fontsize=10)
    _add_watermark(ax)

    fig.tight_layout()
    fig.savefig(save_path, dpi=300, facecolor="white")
    plt.close(fig)
    print(f"  Saved: {save_path}")


def plot_single_pr(region, data, save_path):
    """
    单区域 Precision-Recall 曲线
    """
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    color = COLORS[region]
    color_light = COLOR_LIGHT[region]

    # 各折 PR（半透明）
    for fc in data["fold_curves"]:
        ax.plot(fc["rec"], fc["prec"], color=color, alpha=0.15, lw=1.0)

    # ±1σ 置信带
    upper = np.minimum(data["mean_prec"] + data["std_prec"], 1)
    lower = np.maximum(data["mean_prec"] - data["std_prec"], 0)
    ax.fill_between(data["mean_recall"], lower, upper,
                     color=color_light, alpha=0.35,
                     label="±1 std. dev.")

    # 均值 PR 主曲线
    ax.plot(data["mean_recall"], data["mean_prec"],
            color=color, lw=2.8, alpha=0.95,
            label=f"Mean PR (AP = {data['mean_pr_auc']:.3f})",
            path_effects=[pe.Stroke(linewidth=4.0, foreground="white"), pe.Normal()])

    # 基线 (prevalence)
    # 注：无法直接从 mean curves 还原 prevalence，但可以在标题中标注

    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.05])
    ax.set_xlabel("Recall (Sensitivity)")
    ax.set_ylabel("Precision (Positive Predictive Value)")
    ax.set_title(f"Precision-Recall Curve — {REGION_LABELS_EN[region]}",
                 fontweight="bold", pad=12)
    ax.legend(loc="lower left", frameon=True)
    _add_watermark(ax)

    fig.tight_layout()
    fig.savefig(save_path, dpi=300, facecolor="white")
    plt.close(fig)
    print(f"  Saved: {save_path}")


def plot_combined_pr(all_data, save_path):
    """
    四区域 PR 合并图
    """
    fig, ax = plt.subplots(figsize=(7.5, 6.5))

    for region in REGIONS:
        d = all_data[region]
        color = COLORS[region]
        color_light = COLOR_LIGHT[region]
        label_short = REGION_LABELS_EN[region].split("(")[0].strip()

        # 置信带
        upper = np.minimum(d["mean_prec"] + d["std_prec"], 1)
        lower = np.maximum(d["mean_prec"] - d["std_prec"], 0)
        ax.fill_between(d["mean_recall"], lower, upper, color=color_light, alpha=0.2)

        # 均值 PR
        ax.plot(d["mean_recall"], d["mean_prec"],
                color=color, lw=2.5, alpha=0.92,
                label=f"{label_short} (AP = {d['mean_pr_auc']:.3f})",
                path_effects=[pe.Stroke(linewidth=3.8, foreground="white"), pe.Normal()])

    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.05])
    ax.set_xlabel("Recall (Sensitivity)", fontsize=12)
    ax.set_ylabel("Precision (Positive Predictive Value)", fontsize=12)
    ax.set_title("Precision-Recall Curves — Four Knee Cartilage Regions (5-Fold CV)",
                 fontweight="bold", fontsize=13.5, pad=14)
    ax.legend(loc="lower left", frameon=True, fontsize=10)
    _add_watermark(ax)

    fig.tight_layout()
    fig.savefig(save_path, dpi=300, facecolor="white")
    plt.close(fig)
    print(f"  Saved: {save_path}")


def plot_figure_panel(all_data, save_path):
    """
    论文级 2×2 综合面板：
      (a) 四区域 ROC 合并      (b) 四区域 PR 合并
      (c) 单区域 ROC 小多图     (d) AUC / AP 指标柱状图
    """
    fig = plt.figure(figsize=(15, 13))
    gs = fig.add_gridspec(2, 2, hspace=0.30, wspace=0.28)

    # ---- (a) Combined ROC ----
    ax_a = fig.add_subplot(gs[0, 0])
    ax_a.plot([0, 1], [0, 1], ls="--", color="#bdc3c7", lw=1.2, alpha=0.6)
    for region in REGIONS:
        d = all_data[region]
        c = COLORS[region]
        cl = COLOR_LIGHT[region]
        lbl = REGION_LABELS_EN[region].split("(")[0].strip()
        upper = np.minimum(d["mean_tpr"] + d["std_tpr"], 1)
        lower = np.maximum(d["mean_tpr"] - d["std_tpr"], 0)
        ax_a.fill_between(d["mean_fpr"], lower, upper, color=cl, alpha=0.18)
        ax_a.plot(d["mean_fpr"], d["mean_tpr"], color=c, lw=2.2,
                  label=f"{lbl} (AUC={d['mean_roc_auc']:.3f})",
                  path_effects=[pe.Stroke(linewidth=3.2, foreground="white"), pe.Normal()])
        bt_v, bt_f, bt_t = d["best_threshold"]
        ax_a.scatter([bt_f], [bt_t], s=45, color=c, edgecolors="white", lw=1.2, zorder=10)
    ax_a.set_xlim([-0.02, 1.02]); ax_a.set_ylim([-0.02, 1.05])
    ax_a.set_xlabel("False Positive Rate"); ax_a.set_ylabel("True Positive Rate")
    ax_a.set_title("(a) ROC Curves — Combined", fontweight="bold", fontsize=12)
    ax_a.legend(loc="lower right", fontsize=8.5, frameon=True)

    # ---- (b) Combined PR ----
    ax_b = fig.add_subplot(gs[0, 1])
    for region in REGIONS:
        d = all_data[region]
        c = COLORS[region]
        cl = COLOR_LIGHT[region]
        lbl = REGION_LABELS_EN[region].split("(")[0].strip()
        upper = np.minimum(d["mean_prec"] + d["std_prec"], 1)
        lower = np.maximum(d["mean_prec"] - d["std_prec"], 0)
        ax_b.fill_between(d["mean_recall"], lower, upper, color=cl, alpha=0.18)
        ax_b.plot(d["mean_recall"], d["mean_prec"], color=c, lw=2.2,
                  label=f"{lbl} (AP={d['mean_pr_auc']:.3f})",
                  path_effects=[pe.Stroke(linewidth=3.2, foreground="white"), pe.Normal()])
    ax_b.set_xlim([-0.02, 1.02]); ax_b.set_ylim([-0.02, 1.05])
    ax_b.set_xlabel("Recall"); ax_b.set_ylabel("Precision")
    ax_b.set_title("(b) Precision-Recall Curves — Combined", fontweight="bold", fontsize=12)
    ax_b.legend(loc="lower left", fontsize=8.5, frameon=True)

    # ---- (c) 4 个单区域 ROC 子图 ----
    gs_c = gs[1, 0].subgridspec(2, 2, hspace=0.35, wspace=0.30)
    for i, region in enumerate(REGIONS):
        ax = fig.add_subplot(gs_c[i // 2, i % 2])
        d = all_data[region]
        c = COLORS[region]
        cl = COLOR_LIGHT[region]
        lbl = REGION_LABELS_EN[region].split("(")[0].strip()

        ax.plot([0, 1], [0, 1], ls="--", color="#bdc3c7", lw=1.0, alpha=0.5)
        # 各折
        for fc in d["fold_curves"]:
            ax.plot(fc["fpr"], fc["tpr"], color=c, alpha=0.18, lw=0.8)
        # 置信带
        upper = np.minimum(d["mean_tpr"] + d["std_tpr"], 1)
        lower = np.maximum(d["mean_tpr"] - d["std_tpr"], 0)
        ax.fill_between(d["mean_fpr"], lower, upper, color=cl, alpha=0.25)
        # 主曲线
        ax.plot(d["mean_fpr"], d["mean_tpr"], color=c, lw=2.0,
                path_effects=[pe.Stroke(linewidth=3.0, foreground="white"), pe.Normal()])
        # 最佳阈值
        bt_v, bt_f, bt_t = d["best_threshold"]
        ax.scatter([bt_f], [bt_t], s=35, color=c, edgecolors="white", lw=1.0, zorder=10)

        ax.set_xlim([-0.03, 1.03]); ax.set_ylim([-0.03, 1.06])
        ax.set_title(f"{lbl}\nAUC={d['mean_roc_auc']:.3f}",
                     fontsize=9, fontweight="bold", pad=4)
        ax.tick_params(labelsize=7.5)
        if i // 2 == 1:
            ax.set_xlabel("FPR", fontsize=9)
        if i % 2 == 0:
            ax.set_ylabel("TPR", fontsize=9)

    # ---- (d) AUC / AP 柱状图 ----
    ax_d = fig.add_subplot(gs[1, 1])
    short_names = [REGION_LABELS_EN[r].split("(")[0].strip() for r in REGIONS]
    auc_vals = [all_data[r]["mean_roc_auc"] for r in REGIONS]
    auc_stds = [all_data[r]["std_roc_auc"] for r in REGIONS]
    ap_vals = [all_data[r]["mean_pr_auc"] for r in REGIONS]
    ap_stds = [all_data[r]["std_pr_auc"] for r in REGIONS]

    x = np.arange(len(REGIONS))
    width = 0.32
    bars1 = ax_d.bar(x - width/2, auc_vals, width, yerr=auc_stds,
                      color=[COLORS[r] for r in REGIONS], alpha=0.82,
                      edgecolor="white", linewidth=1.0,
                      error_kw=dict(lw=1.2, capsize=4, capthick=1.0),
                      label="AUC-ROC")
    bars2 = ax_d.bar(x + width/2, ap_vals, width, yerr=ap_stds,
                      color=[COLOR_LIGHT[r] for r in REGIONS], alpha=0.82,
                      edgecolor=[COLORS[r] for r in REGIONS], linewidth=1.2,
                      error_kw=dict(lw=1.2, capsize=4, capthick=1.0),
                      label="AP (PR-AUC)", hatch="///")

    # 数值标注
    for bar, val in zip(bars1, auc_vals):
        ax_d.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.015,
                  f"{val:.3f}", ha="center", va="bottom", fontsize=8, fontweight="bold")
    for bar, val in zip(bars2, ap_vals):
        ax_d.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.015,
                  f"{val:.3f}", ha="center", va="bottom", fontsize=8, fontweight="bold")

    ax_d.set_xticks(x)
    ax_d.set_xticklabels(short_names, fontsize=9)
    ax_d.set_ylim([0, 1.15])
    ax_d.set_ylabel("Score")
    ax_d.set_title("(d) AUC-ROC & AP Comparison", fontweight="bold", fontsize=12)
    ax_d.legend(loc="upper right", fontsize=9, frameon=True)
    # 0.5 参考线
    ax_d.axhline(y=0.5, color="#e74c3c", ls=":", lw=1.0, alpha=0.4)

    fig.suptitle("Stage-1 Binary Classification: Normal vs. Damaged (5-Fold GroupKFold CV)",
                 fontsize=15, fontweight="bold", y=0.995)

    fig.savefig(save_path, dpi=300, facecolor="white")
    plt.close(fig)
    print(f"  Saved: {save_path}")


def print_summary_table(all_data, title="Stage 1"):
    """打印汇总表格"""
    print(f"\n{'=' * 80}")
    print(f"  {title}")
    print(f"{'=' * 80}")
    print(f"{'Region':<22} {'AUC-ROC':>12} {'AP (PR-AUC)':>14} {'Threshold':>12}")
    print("-" * 80)
    for region in all_data:
        d = all_data[region]
        if region in REGION_LABELS_EN:
            label = REGION_LABELS_EN[region].split("(")[0].strip()
        else:
            label = region
        bt_val = d["best_threshold"][0]
        print(f"{label:<22} "
              f"{d['mean_roc_auc']:.3f} ± {d['std_roc_auc']:.3f}   "
              f"{d['mean_pr_auc']:.3f} ± {d['std_pr_auc']:.3f}   "
              f"{bt_val:.4f}")
    print("=" * 80)


# ============================================================
# Stage 2: Grade 1 vs Grade 2 — 池化 + 各区域绘图
# ============================================================

def load_stage2_data():
    """
    加载 Stage 2 数据。
    由于 v8.7 使用 pooled_all_shared 策略，需要合并各区域的 stage2 CSV。
    返回: 池化数据 (X_pool, y_pool, groups_pool, region_labels)
          以及各区域独立数据 dict。
    """
    all_dfs = []
    region_indep = {}

    for region in REGIONS:
        csv_path = os.path.join(FEATURE_DIR, f"{region}_stage2_filtered_features.csv")
        if not os.path.exists(csv_path):
            print(f"  Warning: {csv_path} not found, skipping {region} for Stage2")
            continue
        df_raw = pd.read_csv(csv_path)
        df = df_raw.groupby(["case_id", "region", "grade"]).mean().reset_index()
        all_dfs.append(df)

        # 保存独立数据
        drop_cols = [c for c in ["case_id", "region", "grade", "cartilage_missing"] if c in df.columns]
        X_ind = df.drop(columns=drop_cols)
        y_ind = (df["grade"] == 2).astype(int).values
        groups_ind = df["case_id"].values
        if X_ind.isnull().values.any():
            X_ind = X_ind.fillna(0)
        region_indep[region] = {"X": X_ind, "y": y_ind, "groups": groups_ind,
                                 "n_total": len(y_ind),
                                 "n_g1": int((y_ind == 0).sum()),
                                 "n_g2": int((y_ind == 1).sum())}

    if not all_dfs:
        return None, None, None, None, region_indep

    # 合并所有区域
    df_pooled = pd.concat(all_dfs, ignore_index=True)
    # 使用 v8.7 模型保存的 feature_list（21 个特征，含 region one-hot）
    s2_feat_path = os.path.join(MODEL_DIR, REGIONS[0], "models", "feature_list_stage2.pkl")
    if os.path.exists(s2_feat_path):
        saved_features = joblib.load(s2_feat_path)
        print(f"  Stage2 saved features ({len(saved_features)}): {saved_features[:5]}...")
    else:
        saved_features = None

    # 构建特征矩阵
    drop_cols = [c for c in ["case_id", "region", "grade", "cartilage_missing"] if c in df_pooled.columns]
    base_features = [c for c in df_pooled.columns if c not in drop_cols]

    # 添加 region one-hot
    for r in REGIONS:
        col_name = f"region_{r}"
        df_pooled[col_name] = (df_pooled["region"] == r).astype(float)

    # 用保存的特征列表对齐，缺失的用 0 填充
    if saved_features:
        for feat in saved_features:
            if feat not in df_pooled.columns:
                df_pooled[feat] = 0.0
        X_pool = df_pooled[saved_features].copy()
    else:
        all_cols = base_features + [f"region_{r}" for r in REGIONS]
        X_pool = df_pooled[all_cols].copy()

    y_pool = (df_pooled["grade"] == 2).astype(int).values
    groups_pool = df_pooled["case_id"].values
    region_col = df_pooled["region"].values

    if X_pool.isnull().values.any():
        X_pool = X_pool.fillna(0)

    return X_pool, y_pool, groups_pool, region_col, region_indep


def compute_stage2_cv_curves(X, y, groups, C, gamma, class_weight="balanced", n_splits=5):
    """
    Stage 2 CV：使用 StratifiedKFold（与原训练脚本的 GridSearchCV 内部 CV 一致）。
    
    小样本情况下，StratifiedKFold 比 LOGO 更稳定；
    同时保证每折中两个类别均有代表。

    返回与 compute_cv_curves 兼容的数据结构。
    """
    n_minority = min(int((y == 0).sum()), int((y == 1).sum()))
    actual_splits = min(n_splits, n_minority)
    if actual_splits < 2:
        print(f"  Warning: n_minority={n_minority}, cannot do CV")
        return None

    skf = StratifiedKFold(n_splits=actual_splits, shuffle=True, random_state=42)
    mean_fpr = np.linspace(0, 1, N_INTERP)
    mean_recall = np.linspace(0, 1, N_INTERP)

    tprs_interp = []
    precs_interp = []
    auc_roc_list = []
    auc_pr_list = []
    fold_curves = []
    threshold_list = []

    all_y_true = []
    all_y_prob = []
    all_indices = []

    for fold_idx, (train_ix, test_ix) in enumerate(skf.split(X, y)):
        X_train, X_test = X.iloc[train_ix], X.iloc[test_ix]
        y_train, y_test = y[train_ix], y[test_ix]

        if len(np.unique(y_train)) < 2:
            continue

        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)

        model = SVC(kernel="rbf", C=C, gamma=gamma, probability=True,
                     class_weight=class_weight)
        model.fit(X_train_s, y_train)
        y_prob = model.predict_proba(X_test_s)[:, 1]

        all_y_true.extend(y_test.tolist())
        all_y_prob.extend(y_prob.tolist())
        all_indices.extend(test_ix.tolist())

        # ROC
        fpr, tpr, thresholds_roc = roc_curve(y_test, y_prob)
        roc_auc_val = auc(fpr, tpr)
        auc_roc_list.append(roc_auc_val)

        interp_tpr = np.interp(mean_fpr, fpr, tpr)
        interp_tpr[0] = 0.0
        tprs_interp.append(interp_tpr)

        youden_j = tpr - fpr
        best_idx = np.argmax(youden_j)
        best_thresh = thresholds_roc[best_idx]
        best_fpr_pt = fpr[best_idx]
        best_tpr_pt = tpr[best_idx]
        threshold_list.append((best_thresh, best_fpr_pt, best_tpr_pt))

        # PR
        prec, rec, _ = precision_recall_curve(y_test, y_prob)
        ap_val = average_precision_score(y_test, y_prob)
        auc_pr_list.append(ap_val)

        rec_sorted = rec[::-1]
        prec_sorted = prec[::-1]
        interp_prec = np.interp(mean_recall, rec_sorted, prec_sorted)
        precs_interp.append(interp_prec)

        fold_curves.append({
            "fpr": fpr, "tpr": tpr, "roc_auc": roc_auc_val,
            "prec": prec, "rec": rec, "ap": ap_val,
            "best_thresh": best_thresh, "best_fpr": best_fpr_pt, "best_tpr": best_tpr_pt,
        })

    if not auc_roc_list:
        print("  Warning: Stage2 CV produced no valid folds")
        return None

    # 汇总
    mean_tpr = np.mean(tprs_interp, axis=0)
    mean_tpr[-1] = 1.0
    std_tpr = np.std(tprs_interp, axis=0)

    mean_prec = np.mean(precs_interp, axis=0)
    std_prec = np.std(precs_interp, axis=0)

    mean_roc_auc = np.mean(auc_roc_list)
    std_roc_auc = np.std(auc_roc_list)
    mean_pr_auc = np.mean(auc_pr_list)
    std_pr_auc = np.std(auc_pr_list)

    avg_thresh_fpr = np.mean([t[1] for t in threshold_list])
    avg_thresh_tpr = np.mean([t[2] for t in threshold_list])
    avg_thresh_val = np.median([t[0] for t in threshold_list])

    return {
        "mean_fpr": mean_fpr,
        "mean_tpr": mean_tpr,
        "std_tpr": std_tpr,
        "mean_roc_auc": mean_roc_auc,
        "std_roc_auc": std_roc_auc,
        "mean_recall": mean_recall,
        "mean_prec": mean_prec,
        "std_prec": std_prec,
        "mean_pr_auc": mean_pr_auc,
        "std_pr_auc": std_pr_auc,
        "fold_curves": fold_curves,
        "tprs_interp": tprs_interp,
        "precs_interp": precs_interp,
        "auc_roc_list": auc_roc_list,
        "auc_pr_list": auc_pr_list,
        "best_threshold": (avg_thresh_val, avg_thresh_fpr, avg_thresh_tpr),
        "all_y_true": np.array(all_y_true),
        "all_y_prob": np.array(all_y_prob),
        "all_indices": np.array(all_indices),
    }


def compute_stage2_per_region(all_y_true, all_y_prob, region_col):
    """
    从池化 LOGO-CV 的结果中按区域拆分计算 ROC / PR。
    """
    per_region_data = {}
    mean_fpr = np.linspace(0, 1, N_INTERP)
    mean_recall = np.linspace(0, 1, N_INTERP)

    for region in REGIONS:
        mask = (region_col == region)
        yt = all_y_true[mask]
        yp = all_y_prob[mask]

        if len(yt) < 2 or len(np.unique(yt)) < 2:
            print(f"  {region}: skipped (n={len(yt)}, unique_labels={np.unique(yt)})")
            continue

        fpr, tpr, thresholds_roc = roc_curve(yt, yp)
        roc_auc_val = auc(fpr, tpr)

        interp_tpr = np.interp(mean_fpr, fpr, tpr)
        interp_tpr[0] = 0.0

        youden_j = tpr - fpr
        best_idx = np.argmax(youden_j)

        prec, rec, _ = precision_recall_curve(yt, yp)
        ap_val = average_precision_score(yt, yp)
        rec_sorted = rec[::-1]
        prec_sorted = prec[::-1]
        interp_prec = np.interp(mean_recall, rec_sorted, prec_sorted)

        per_region_data[region] = {
            "mean_fpr": mean_fpr,
            "mean_tpr": interp_tpr,
            "std_tpr": np.zeros_like(mean_fpr),
            "mean_roc_auc": roc_auc_val,
            "std_roc_auc": 0.0,
            "mean_recall": mean_recall,
            "mean_prec": interp_prec,
            "std_prec": np.zeros_like(mean_recall),
            "mean_pr_auc": ap_val,
            "std_pr_auc": 0.0,
            "fold_curves": [{
                "fpr": fpr, "tpr": tpr, "roc_auc": roc_auc_val,
                "prec": prec, "rec": rec, "ap": ap_val,
                "best_thresh": thresholds_roc[best_idx],
                "best_fpr": fpr[best_idx], "best_tpr": tpr[best_idx],
            }],
            "best_threshold": (thresholds_roc[best_idx], fpr[best_idx], tpr[best_idx]),
            "n_samples": len(yt),
            "n_pos": int(yt.sum()),
            "n_neg": int((1 - yt).sum()),
        }

    return per_region_data


def plot_stage2_pooled_roc(pooled_data, save_path):
    """
    Stage 2 单张池化 ROC 图（不拆区域）。
    风格与 Stage 1 单区域 ROC 一致：5-fold 半透明线 + 均值 + 置信带 + 阈值标注。
    """
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    color = "#e74c3c"       # 红色系，与 Stage 1 区分
    color_light = "#f5b7b1"

    d = pooled_data
    n_total = len(d["all_y_true"])
    n_pos = int(d["all_y_true"].sum())

    # 对角参考线
    ax.plot([0, 1], [0, 1], ls="--", color="#bdc3c7", lw=1.5, alpha=0.7,
            label="Chance level (AUC = 0.500)")

    # 各折 ROC（半透明）
    for i, fc in enumerate(d["fold_curves"]):
        ax.plot(fc["fpr"], fc["tpr"], color=color, alpha=0.18, lw=1.0)

    # ±1σ 置信带
    upper = np.minimum(d["mean_tpr"] + d["std_tpr"], 1)
    lower = np.maximum(d["mean_tpr"] - d["std_tpr"], 0)
    ax.fill_between(d["mean_fpr"], lower, upper,
                     color=color_light, alpha=0.35,
                     label="±1 std. dev.")

    # 均值 ROC 主曲线
    ax.plot(d["mean_fpr"], d["mean_tpr"],
            color=color, lw=2.8, alpha=0.95,
            label=f"Mean ROC (AUC = {d['mean_roc_auc']:.3f} ± {d['std_roc_auc']:.3f})",
            path_effects=[pe.Stroke(linewidth=4.0, foreground="white"), pe.Normal()])

    # 最佳阈值点标注
    bt_val, bt_fpr, bt_tpr = d["best_threshold"]
    ax.scatter([bt_fpr], [bt_tpr], s=90, color=color, edgecolors="white",
               linewidths=1.8, zorder=10)
    ax.annotate(f"Threshold = {bt_val:.3f}\n(FPR = {bt_fpr:.2f}, TPR = {bt_tpr:.2f})",
                xy=(bt_fpr, bt_tpr), xytext=(bt_fpr + 0.12, bt_tpr - 0.12),
                fontsize=8.5, color="#333333",
                arrowprops=dict(arrowstyle="-|>", color="#777777", lw=1.0),
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#cccccc", alpha=0.85))

    # 样本量标注
    ax.text(0.55, 0.15, f"n = {n_total}  (Grade 1: {n_total - n_pos}, Grade 2: {n_pos})",
            transform=ax.transAxes, fontsize=9, color="#555555",
            bbox=dict(boxstyle="round,pad=0.3", fc="#f8f8f8", ec="#dddddd", alpha=0.8))

    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.05])
    ax.set_xlabel("False Positive Rate (1 - Specificity)")
    ax.set_ylabel("True Positive Rate (Sensitivity)")
    ax.set_title("Stage-2 ROC: Grade 1 vs. Grade 2\n(Pooled All Regions, 5-Fold StratifiedKFold CV)",
                 fontweight="bold", pad=12, fontsize=12)
    ax.legend(loc="lower right", frameon=True)
    _add_watermark(ax)

    fig.tight_layout()
    fig.savefig(save_path, dpi=300, facecolor="white")
    plt.close(fig)
    print(f"  Saved: {save_path}")


def plot_stage2_combined_roc(pooled_data, per_region_data, region_info, save_path):
    """
    Stage 2 ROC 图: 池化曲线 + 各区域曲线
    """
    fig, ax = plt.subplots(figsize=(8, 6.5))

    # 对角参考线
    ax.plot([0, 1], [0, 1], ls="--", color="#bdc3c7", lw=1.5, alpha=0.7,
            label="Chance level")

    # 池化 ROC (加粗虚线)
    d = pooled_data
    ax.fill_between(d["mean_fpr"],
                     np.maximum(d["mean_tpr"] - d["std_tpr"], 0),
                     np.minimum(d["mean_tpr"] + d["std_tpr"], 1),
                     color="#95a5a6", alpha=0.15)
    n_total = len(d["all_y_true"])
    n_pos = int(d["all_y_true"].sum())
    ax.plot(d["mean_fpr"], d["mean_tpr"],
            color="#2c3e50", lw=3.0, ls="-", alpha=0.9,
            label=f"Pooled (AUC = {d['mean_roc_auc']:.3f} ± {d['std_roc_auc']:.3f}, "
                  f"n={n_total}, G2={n_pos})",
            path_effects=[pe.Stroke(linewidth=4.5, foreground="white"), pe.Normal()])
    bt_v, bt_f, bt_t = d["best_threshold"]
    ax.scatter([bt_f], [bt_t], s=80, color="#2c3e50", edgecolors="white",
               linewidths=1.5, zorder=10, marker="D")

    # 各区域曲线
    for region in REGIONS:
        if region not in per_region_data:
            continue
        rd = per_region_data[region]
        c = COLORS[region]
        lbl = REGION_LABELS_EN[region].split("(")[0].strip()
        info = region_info.get(region, {})
        n = info.get("n_total", rd.get("n_samples", "?"))
        n_g2 = info.get("n_g2", rd.get("n_pos", "?"))

        ax.plot(rd["mean_fpr"], rd["mean_tpr"],
                color=c, lw=2.0, alpha=0.85,
                label=f"{lbl} (AUC = {rd['mean_roc_auc']:.3f}, n={n}, G2={n_g2})",
                path_effects=[pe.Stroke(linewidth=3.2, foreground="white"), pe.Normal()])

        bt_v2, bt_f2, bt_t2 = rd["best_threshold"]
        ax.scatter([bt_f2], [bt_t2], s=45, color=c, edgecolors="white",
                   linewidths=1.2, zorder=10)

    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.05])
    ax.set_xlabel("False Positive Rate (1 - Specificity)", fontsize=12)
    ax.set_ylabel("True Positive Rate (Sensitivity)", fontsize=12)
    ax.set_title("Stage-2 ROC: Grade 1 vs. Grade 2 (LOGO-CV, Pooled)",
                 fontweight="bold", fontsize=13, pad=14)
    ax.legend(loc="lower right", frameon=True, fontsize=9)
    _add_watermark(ax)

    fig.tight_layout()
    fig.savefig(save_path, dpi=300, facecolor="white")
    plt.close(fig)
    print(f"  Saved: {save_path}")


def plot_stage2_combined_pr(pooled_data, per_region_data, region_info, save_path):
    """
    Stage 2 PR 图: 池化曲线 + 各区域曲线
    """
    fig, ax = plt.subplots(figsize=(8, 6.5))

    # 池化 PR
    d = pooled_data
    ax.fill_between(d["mean_recall"],
                     np.maximum(d["mean_prec"] - d["std_prec"], 0),
                     np.minimum(d["mean_prec"] + d["std_prec"], 1),
                     color="#95a5a6", alpha=0.15)
    n_total = len(d["all_y_true"])
    n_pos = int(d["all_y_true"].sum())
    prevalence = n_pos / n_total if n_total > 0 else 0
    ax.axhline(y=prevalence, color="#e74c3c", ls=":", lw=1.0, alpha=0.5,
               label=f"Prevalence = {prevalence:.2f}")
    ax.plot(d["mean_recall"], d["mean_prec"],
            color="#2c3e50", lw=3.0, alpha=0.9,
            label=f"Pooled (AP = {d['mean_pr_auc']:.3f} ± {d['std_pr_auc']:.3f}, "
                  f"n={n_total})",
            path_effects=[pe.Stroke(linewidth=4.5, foreground="white"), pe.Normal()])

    # 各区域
    for region in REGIONS:
        if region not in per_region_data:
            continue
        rd = per_region_data[region]
        c = COLORS[region]
        lbl = REGION_LABELS_EN[region].split("(")[0].strip()
        info = region_info.get(region, {})
        n = info.get("n_total", rd.get("n_samples", "?"))

        ax.plot(rd["mean_recall"], rd["mean_prec"],
                color=c, lw=2.0, alpha=0.85,
                label=f"{lbl} (AP = {rd['mean_pr_auc']:.3f}, n={n})",
                path_effects=[pe.Stroke(linewidth=3.2, foreground="white"), pe.Normal()])

    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.05])
    ax.set_xlabel("Recall (Sensitivity)", fontsize=12)
    ax.set_ylabel("Precision (Positive Predictive Value)", fontsize=12)
    ax.set_title("Stage-2 Precision-Recall: Grade 1 vs. Grade 2 (LOGO-CV, Pooled)",
                 fontweight="bold", fontsize=13, pad=14)
    ax.legend(loc="lower left", frameon=True, fontsize=9)
    _add_watermark(ax)

    fig.tight_layout()
    fig.savefig(save_path, dpi=300, facecolor="white")
    plt.close(fig)
    print(f"  Saved: {save_path}")


def plot_stage2_panel(pooled_data, per_region_data, region_info, save_path):
    """
    Stage 2 综合面板 (2×2):
      (a) 池化 ROC + 各区域     (b) 池化 PR + 各区域
      (c) 各区域 ROC 子图        (d) 样本分布 + 指标柱状图
    """
    fig = plt.figure(figsize=(15, 13))
    gs = fig.add_gridspec(2, 2, hspace=0.30, wspace=0.28)

    # ---- (a) 合并 ROC ----
    ax_a = fig.add_subplot(gs[0, 0])
    ax_a.plot([0, 1], [0, 1], ls="--", color="#bdc3c7", lw=1.2, alpha=0.6)

    d = pooled_data
    ax_a.fill_between(d["mean_fpr"],
                       np.maximum(d["mean_tpr"] - d["std_tpr"], 0),
                       np.minimum(d["mean_tpr"] + d["std_tpr"], 1),
                       color="#95a5a6", alpha=0.12)
    ax_a.plot(d["mean_fpr"], d["mean_tpr"],
              color="#2c3e50", lw=2.8, alpha=0.9,
              label=f"Pooled (AUC={d['mean_roc_auc']:.3f}±{d['std_roc_auc']:.3f})",
              path_effects=[pe.Stroke(linewidth=4.0, foreground="white"), pe.Normal()])
    bt_v, bt_f, bt_t = d["best_threshold"]
    ax_a.scatter([bt_f], [bt_t], s=55, color="#2c3e50", edgecolors="white",
                 lw=1.2, zorder=10, marker="D")

    for region in REGIONS:
        if region not in per_region_data:
            continue
        rd = per_region_data[region]
        c = COLORS[region]
        lbl = REGION_LABELS_EN[region].split("(")[0].strip()
        ax_a.plot(rd["mean_fpr"], rd["mean_tpr"], color=c, lw=1.8, alpha=0.8,
                  label=f"{lbl} (AUC={rd['mean_roc_auc']:.3f})",
                  path_effects=[pe.Stroke(linewidth=2.8, foreground="white"), pe.Normal()])

    ax_a.set_xlim([-0.02, 1.02]); ax_a.set_ylim([-0.02, 1.05])
    ax_a.set_xlabel("False Positive Rate"); ax_a.set_ylabel("True Positive Rate")
    ax_a.set_title("(a) Stage-2 ROC — Pooled + Per-Region", fontweight="bold", fontsize=12)
    ax_a.legend(loc="lower right", fontsize=8, frameon=True)

    # ---- (b) 合并 PR ----
    ax_b = fig.add_subplot(gs[0, 1])
    n_total = len(d["all_y_true"])
    n_pos = int(d["all_y_true"].sum())
    prevalence = n_pos / n_total if n_total > 0 else 0
    ax_b.axhline(y=prevalence, color="#e74c3c", ls=":", lw=1.0, alpha=0.5,
                 label=f"Prevalence={prevalence:.2f}")

    ax_b.fill_between(d["mean_recall"],
                       np.maximum(d["mean_prec"] - d["std_prec"], 0),
                       np.minimum(d["mean_prec"] + d["std_prec"], 1),
                       color="#95a5a6", alpha=0.12)
    ax_b.plot(d["mean_recall"], d["mean_prec"],
              color="#2c3e50", lw=2.8, alpha=0.9,
              label=f"Pooled (AP={d['mean_pr_auc']:.3f}±{d['std_pr_auc']:.3f})",
              path_effects=[pe.Stroke(linewidth=4.0, foreground="white"), pe.Normal()])

    for region in REGIONS:
        if region not in per_region_data:
            continue
        rd = per_region_data[region]
        c = COLORS[region]
        lbl = REGION_LABELS_EN[region].split("(")[0].strip()
        ax_b.plot(rd["mean_recall"], rd["mean_prec"], color=c, lw=1.8, alpha=0.8,
                  label=f"{lbl} (AP={rd['mean_pr_auc']:.3f})",
                  path_effects=[pe.Stroke(linewidth=2.8, foreground="white"), pe.Normal()])

    ax_b.set_xlim([-0.02, 1.02]); ax_b.set_ylim([-0.02, 1.05])
    ax_b.set_xlabel("Recall"); ax_b.set_ylabel("Precision")
    ax_b.set_title("(b) Stage-2 PR — Pooled + Per-Region", fontweight="bold", fontsize=12)
    ax_b.legend(loc="lower left", fontsize=8, frameon=True)

    # ---- (c) 各区域 ROC 子图 ----
    gs_c = gs[1, 0].subgridspec(2, 2, hspace=0.40, wspace=0.30)
    for i, region in enumerate(REGIONS):
        ax = fig.add_subplot(gs_c[i // 2, i % 2])
        ax.plot([0, 1], [0, 1], ls="--", color="#bdc3c7", lw=1.0, alpha=0.5)

        c = COLORS[region]
        cl = COLOR_LIGHT[region]
        lbl = REGION_LABELS_EN[region].split("(")[0].strip()
        info = region_info.get(region, {})
        n = info.get("n_total", "?")
        n_g2 = info.get("n_g2", "?")

        if region in per_region_data:
            rd = per_region_data[region]
            ax.plot(rd["mean_fpr"], rd["mean_tpr"], color=c, lw=2.2,
                    path_effects=[pe.Stroke(linewidth=3.2, foreground="white"), pe.Normal()])
            bt_v2, bt_f2, bt_t2 = rd["best_threshold"]
            ax.scatter([bt_f2], [bt_t2], s=40, color=c, edgecolors="white", lw=1.0, zorder=10)
            auc_str = f"AUC={rd['mean_roc_auc']:.3f}"
        else:
            auc_str = "N/A"

        ax.set_xlim([-0.03, 1.03]); ax.set_ylim([-0.03, 1.06])
        ax.set_title(f"{lbl}\n{auc_str} (n={n}, G2={n_g2})",
                     fontsize=9, fontweight="bold", pad=4)
        ax.tick_params(labelsize=7.5)
        if i // 2 == 1:
            ax.set_xlabel("FPR", fontsize=9)
        if i % 2 == 0:
            ax.set_ylabel("TPR", fontsize=9)

    # ---- (d) 样本分布 + AUC/AP 柱状图 ----
    ax_d = fig.add_subplot(gs[1, 1])
    short_names = []
    auc_vals = []
    ap_vals = []
    n_g1_vals = []
    n_g2_vals = []

    # 加入 Pooled
    short_names.append("Pooled")
    auc_vals.append(pooled_data["mean_roc_auc"])
    ap_vals.append(pooled_data["mean_pr_auc"])
    n_g1_vals.append(n_total - n_pos)
    n_g2_vals.append(n_pos)

    for region in REGIONS:
        lbl = REGION_LABELS_EN[region].split("(")[0].strip()
        short_names.append(lbl)
        info = region_info.get(region, {})
        if region in per_region_data:
            auc_vals.append(per_region_data[region]["mean_roc_auc"])
            ap_vals.append(per_region_data[region]["mean_pr_auc"])
        else:
            auc_vals.append(0)
            ap_vals.append(0)
        n_g1_vals.append(info.get("n_g1", 0))
        n_g2_vals.append(info.get("n_g2", 0))

    x = np.arange(len(short_names))
    width = 0.30
    bar_colors = ["#2c3e50"] + [COLORS[r] for r in REGIONS]
    bar_colors_light = ["#95a5a6"] + [COLOR_LIGHT[r] for r in REGIONS]

    bars1 = ax_d.bar(x - width/2, auc_vals, width,
                      color=bar_colors, alpha=0.82,
                      edgecolor="white", linewidth=1.0,
                      label="AUC-ROC")
    bars2 = ax_d.bar(x + width/2, ap_vals, width,
                      color=bar_colors_light, alpha=0.82,
                      edgecolor=bar_colors, linewidth=1.2,
                      label="AP (PR-AUC)", hatch="///")

    for bar, val in zip(bars1, auc_vals):
        if val > 0:
            ax_d.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.015,
                      f"{val:.3f}", ha="center", va="bottom", fontsize=7.5, fontweight="bold")
    for bar, val in zip(bars2, ap_vals):
        if val > 0:
            ax_d.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.015,
                      f"{val:.3f}", ha="center", va="bottom", fontsize=7.5, fontweight="bold")

    # 在柱子底部标注样本量
    for i_x, (g1, g2) in enumerate(zip(n_g1_vals, n_g2_vals)):
        ax_d.text(i_x, 0.02, f"G1={g1}\nG2={g2}", ha="center", va="bottom",
                  fontsize=7, color="#555555", style="italic")

    ax_d.set_xticks(x)
    ax_d.set_xticklabels(short_names, fontsize=8.5, rotation=15)
    ax_d.set_ylim([0, 1.18])
    ax_d.set_ylabel("Score")
    ax_d.set_title("(d) Stage-2 AUC-ROC & AP by Region", fontweight="bold", fontsize=12)
    ax_d.legend(loc="upper right", fontsize=9, frameon=True)
    ax_d.axhline(y=0.5, color="#e74c3c", ls=":", lw=1.0, alpha=0.4)

    fig.suptitle("Stage-2 Grading Classification: Grade 1 vs. Grade 2 (LOGO-CV, Pooled-All-Shared)",
                 fontsize=14.5, fontweight="bold", y=0.995)

    fig.savefig(save_path, dpi=300, facecolor="white")
    plt.close(fig)
    print(f"  Saved: {save_path}")


# ============================================================
# 主函数
# ============================================================
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ==========================================================
    #  STAGE 1: Normal vs Damaged
    # ==========================================================
    print("=" * 60)
    print("  STAGE 1: ROC / PR Curve Paper Figure Generator")
    print("  Model: v8.7 | CV: 5-Fold GroupKFold")
    print("=" * 60)

    all_data_s1 = {}

    for region in REGIONS:
        print(f"\n>>> [S1] Processing: {region}")

        if os.path.exists(OOF_CSV):
            data = compute_curves_from_oof(region)
            print(f"  [OOF] from {OOF_CSV}  Pos: {data['n_pos']}, Neg: {data['n_neg']}")
        else:
            X, y, groups = load_region_data(region)
            C, gamma, cw = load_model_params(region)
            print(f"  Data: {X.shape}, Pos: {y.sum()}, Neg: {(1-y).sum()}")
            print(f"  Params: C={C}, gamma={gamma}, class_weight={cw}")
            data = compute_cv_curves(X, y, groups, C, gamma, class_weight=cw)
        all_data_s1[region] = data
        print(f"  AUC-ROC: {data['mean_roc_auc']:.3f} ± {data['std_roc_auc']:.3f}")
        print(f"  AP:      {data['mean_pr_auc']:.3f} ± {data['std_pr_auc']:.3f}")

        plot_single_roc(region, data,
                        os.path.join(OUTPUT_DIR, f"S1_{region}_ROC.png"))
        plot_single_pr(region, data,
                       os.path.join(OUTPUT_DIR, f"S1_{region}_PR.png"))

    print("\n>>> [S1] Generating combined plots...")
    plot_combined_roc(all_data_s1, os.path.join(OUTPUT_DIR, "S1_Combined_ROC.png"))
    plot_combined_pr(all_data_s1, os.path.join(OUTPUT_DIR, "S1_Combined_PR.png"))

    print("\n>>> [S1] Generating paper figure panel (2x2)...")
    plot_figure_panel(all_data_s1, os.path.join(OUTPUT_DIR, "S1_Figure_Panel.png"))

    print_summary_table(all_data_s1, title="Stage 1: Normal vs Damaged (5-Fold GroupKFold CV)")

    # ==========================================================
    #  STAGE 2: Grade 1 vs Grade 2
    # ==========================================================
    print("\n\n" + "=" * 60)
    print("  STAGE 2: Grade 1 vs Grade 2")
    print("  Model: v8.7 | Pooled-All-Shared | StratifiedKFold CV")
    print("=" * 60)

    X_pool, y_pool, groups_pool, region_col, region_info = load_stage2_data()

    if X_pool is not None:
        print(f"\n  Pooled Stage2 data: {X_pool.shape}")
        print(f"  G1 = {int((y_pool == 0).sum())}, G2 = {int((y_pool == 1).sum())}")
        print(f"  Unique cases: {len(np.unique(groups_pool))}")

        # Stage 2 模型参数 — 从已保存的模型读取
        s2_model_path = os.path.join(MODEL_DIR, REGIONS[0], "models", "svm_model_stage2.pkl")
        if os.path.exists(s2_model_path):
            s2_model = joblib.load(s2_model_path)
            s2_C = s2_model.C
            s2_gamma = s2_model.gamma
            s2_cw = s2_model.class_weight
            print(f"  Loaded Stage2 model params: C={s2_C}, gamma={s2_gamma}, class_weight={s2_cw}")
        else:
            s2_C, s2_gamma, s2_cw = 0.1, 0.01, "balanced"
            print(f"  Using default Stage2 params: C={s2_C}, gamma={s2_gamma}")

        # 使用 GridSearchCV 寻找最佳参数（与原训练脚本一致）
        from sklearn.model_selection import GridSearchCV
        print(f"\n>>> [S2] Running GridSearchCV to find best params...")
        scaler_search = StandardScaler()
        X_pool_scaled_search = scaler_search.fit_transform(X_pool)
        param_grid_s2 = {
            'C': [0.1, 1, 10, 100],
            'gamma': ['scale', 'auto', 0.01, 0.1, 1]
        }
        n_minority = min(int((y_pool == 0).sum()), int((y_pool == 1).sum()))
        inner_cv = min(5, n_minority)
        grid = GridSearchCV(
            SVC(kernel='rbf', probability=True, class_weight='balanced'),
            param_grid_s2, cv=inner_cv, scoring='accuracy'
        )
        grid.fit(X_pool_scaled_search, y_pool)
        s2_C_best = grid.best_params_['C']
        s2_gamma_best = grid.best_params_['gamma']
        print(f"  GridSearchCV best params: C={s2_C_best}, gamma={s2_gamma_best}, "
              f"best_score(acc)={grid.best_score_:.3f}")

        # 使用与训练一致的参数进行 CV 绘图
        print(f"\n>>> [S2] Running StratifiedKFold CV (5-fold) on pooled data...")
        pooled_data = compute_stage2_cv_curves(
            X_pool, y_pool, groups_pool,
            C=s2_C_best, gamma=s2_gamma_best, class_weight="balanced",
            n_splits=5
        )

        if pooled_data is not None:
            print(f"  Pooled AUC-ROC: {pooled_data['mean_roc_auc']:.3f} ± {pooled_data['std_roc_auc']:.3f}")
            print(f"  Pooled AP:      {pooled_data['mean_pr_auc']:.3f} ± {pooled_data['std_pr_auc']:.3f}")

            # 绘图 — Stage2 只画一张池化 ROC 图（不拆区域）
            print(f"\n>>> [S2] Generating Stage2 pooled ROC figure...")
            plot_stage2_pooled_roc(pooled_data,
                                   os.path.join(OUTPUT_DIR, "S2_Pooled_ROC.png"))

            # 汇总
            print_summary_table({"Pooled": pooled_data},
                                title="Stage 2: Grade 1 vs Grade 2 (StratifiedKFold CV, Pooled)")
        else:
            print("  Stage2 CV failed — not enough classes.")
    else:
        print("  No Stage2 data found, skipping.")

    print(f"\n{'=' * 60}")
    print(f"  All figures saved to: {OUTPUT_DIR}")
    print(f"{'=' * 60}")
    print("Done!")


if __name__ == "__main__":
    main()
