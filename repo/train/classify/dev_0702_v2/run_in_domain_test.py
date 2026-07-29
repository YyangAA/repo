#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_in_domain_test.py
在集内测试集上运行推理并生成评估指标

用法:
  cd /mnt/tidal-alsh-share2/dataset/askfollow/usr/yangxin/keyan/repo-dev_v4/repo
  source venv310/bin/activate
  python train/classify/dev_0702_v2/run_in_domain_test.py
"""

import os
import sys
import argparse
import pandas as pd
import numpy as np
from sklearn.metrics import (
    roc_auc_score, roc_curve, auc,
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix
)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# ===============================
# 路径配置
# ===============================
REPO_ROOT = "/mnt/tidal-alsh-share2/dataset/askfollow/usr/yangxin/keyan/repo-dev_v4/repo"
VENV = os.path.join(REPO_ROOT, "venv310", "bin", "activate")

# 集内测试集路径
TEST_IMAGE_DIR = os.path.join(REPO_ROOT, "data/split_v9/test_image")
TEST_MASK_DIR = os.path.join(REPO_ROOT, "data/split_v9/test_mask")
TEST_GT_EXCEL = os.path.join(REPO_ROOT, "data/split_v9/test_gt.xlsx")
TEST_SPLIT_INFO = os.path.join(REPO_ROOT, "data/split_v9/test_split_info.csv")

# 模型路径
MODEL_DIR = os.path.join(REPO_ROOT, "checkpoint/results_v8.9_0702_v2")

# 输出路径
OUTPUT_DIR = os.path.join(REPO_ROOT, "data/in_domain_test_results_v8.9_0702_v2")
OUTPUT_CSV = os.path.join(OUTPUT_DIR, "in_domain_predictions.csv")
METRICS_CSV = os.path.join(OUTPUT_DIR, "metrics_summary.csv")
CONFUSION_MATRICES_PNG = os.path.join(OUTPUT_DIR, "confusion_matrices.png")
ROC_CURVES_PNG = os.path.join(OUTPUT_DIR, "roc_curves.png")

# 区域定义
REGION_NAMES = {
    1: "Femur_Medial",
    2: "Femur_Lateral",
    3: "Tibia_Medial",
    4: "Tibia_Lateral"
}
REGIONS = list(REGION_NAMES.values())

REGION_CN = {
    "Femur_Medial":  "股骨内侧 (FM)",
    "Femur_Lateral": "股骨外侧 (FL)",
    "Tibia_Medial":  "胫骨内侧 (TM)",
    "Tibia_Lateral": "胫骨外侧 (TL)"
}


def load_ground_truth_from_split_info(split_info_path):
    """
    从 test_split_info.csv 加载真实标签
    
    CSV 格式: filename, 患者姓名, 膝关节, source, FM, TM, FL, TL
    其中 FM/TM/FL/TL 是每个区域的真实分级 (0=Normal, 1=G1, 2=G2)
    """
    df = pd.read_csv(split_info_path)
    
    gt_dict = {}
    for _, row in df.iterrows():
        if row['source'] != 'test':
            continue
        
        case_id = row['filename'].replace('.nii.gz', '')
        gt_dict[case_id] = {
            'Femur_Medial':  int(row['FM']) if pd.notna(row['FM']) else -1,
            'Femur_Lateral': int(row['FL']) if pd.notna(row['FL']) else -1,
            'Tibia_Medial':  int(row['TM']) if pd.notna(row['TM']) else -1,
            'Tibia_Lateral': int(row['TL']) if pd.notna(row['TL']) else -1,
        }
    
    return gt_dict


def load_ground_truth_from_excel(excel_path):
    """从 Excel 加载真实标签 (备用)"""
    if not os.path.exists(excel_path):
        return {}
    
    region_columns = {
        "股骨内侧": "Femur_Medial",
        "股骨外侧": "Femur_Lateral",
        "胫骨内侧": "Tibia_Medial",
        "胫骨外侧": "Tibia_Lateral"
    }
    
    df = pd.read_excel(excel_path)
    
    # 尝试不同的列名格式
    name_col = None
    for col in ['name', '患者姓名', '姓名', 'case_id']:
        if col in df.columns:
            name_col = col
            break
    
    if name_col is None:
        print(f"Warning: Could not find name column in {excel_path}")
        return {}
    
    df["name"] = df[name_col].astype(str).str.replace("_knee", "", regex=False).str.strip()
    
    gt_dict = {}
    for _, row in df.iterrows():
        cid = row["name"]
        gt_dict[cid] = {}
        for col_ch, col_en in region_columns.items():
            if col_ch in df.columns:
                gt_dict[cid][col_en] = int(row[col_ch]) if pd.notna(row[col_ch]) else -1
            else:
                gt_dict[cid][col_en] = -1
    
    return gt_dict


def load_predictions(csv_path):
    """加载推理预测结果"""
    if not os.path.exists(csv_path):
        return {}
    
    df = pd.read_csv(csv_path)
    
    pred_dict = {}
    for _, row in df.iterrows():
        cid = str(row["case_id"])
        region = row["region"]
        
        if cid not in pred_dict:
            pred_dict[cid] = {}
        
        pred_dict[cid][region] = {
            "pred": int(row.get("predicted_label", 0)),
            "prob": float(row.get("probability_damage", 0.0)),
            "grade": int(row.get("predicted_grade", 0)),
            "prob_grade2": float(row.get("probability_grade2", 0.0)),
        }
    
    return pred_dict


def compute_metrics(pred_dict, gt_dict):
    """
    计算每个区域的分类指标
    
    返回: {region_name: metrics_dict}
    """
    region_results = {}
    
    for region_name in REGIONS:
        y_true_binary = []
        y_prob = []
        y_pred_binary = []
        y_true_grade = []
        y_pred_grade = []
        
        for case_id in gt_dict:
            if case_id not in pred_dict:
                continue
            if region_name not in pred_dict[case_id]:
                continue
            if region_name not in gt_dict[case_id]:
                continue
            
            gt_grade = gt_dict[case_id][region_name]
            if gt_grade < 0:
                continue
            
            pred_info = pred_dict[case_id][region_name]
            pred_grade = pred_info.get("grade", pred_info.get("pred", 0))
            
            gt_binary = 1 if gt_grade > 0 else 0
            pred_binary = pred_info.get("pred", 0)
            
            y_true_binary.append(gt_binary)
            y_prob.append(pred_info["prob"])
            y_pred_binary.append(pred_binary)
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
            'n': len(y_true_binary),
            'n_pos': int(y_true_binary.sum()),
            'n_neg': int((1 - y_true_binary).sum()),
        }
        
        # 二分类指标
        if len(np.unique(y_true_binary)) >= 2:
            fpr, tpr, _ = roc_curve(y_true_binary, y_prob)
            metrics['auc'] = auc(fpr, tpr)
            metrics['fpr'] = fpr
            metrics['tpr'] = tpr
        else:
            metrics['auc'] = float('nan')
            metrics['fpr'] = None
            metrics['tpr'] = None
        
        metrics['accuracy'] = accuracy_score(y_true_binary, y_pred_binary)
        metrics['precision'] = precision_score(y_true_binary, y_pred_binary, zero_division=0)
        metrics['recall'] = recall_score(y_true_binary, y_pred_binary, zero_division=0)
        metrics['f1'] = f1_score(y_true_binary, y_pred_binary, zero_division=0)
        
        cm = confusion_matrix(y_true_binary, y_pred_binary, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()
        metrics['specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        metrics['tp'] = int(tp)
        metrics['tn'] = int(tn)
        metrics['fp'] = int(fp)
        metrics['fn'] = int(fn)
        metrics['cm_binary'] = cm
        
        # 三分类指标
        metrics['grade_accuracy'] = float(np.mean(y_true_grade == y_pred_grade))
        metrics['cm_grade'] = confusion_matrix(y_true_grade, y_pred_grade, labels=[0, 1, 2])
        
        for g in [0, 1, 2]:
            metrics[f'gt_grade{g}'] = int((y_true_grade == g).sum())
            metrics[f'pred_grade{g}'] = int((y_pred_grade == g).sum())
            metrics[f'grade{g}_correct'] = int(((y_true_grade == g) & (y_pred_grade == g)).sum())
        
        region_results[region_name] = metrics
    
    return region_results


def plot_roc_curves(region_results, output_path):
    """绘制 ROC 曲线"""
    region_list = list(REGION_NAMES.values())
    
    line_colors = {
        "Femur_Medial":  '#2ecc71',
        "Femur_Lateral": '#3498db',
        "Tibia_Medial":  '#e67e22',
        "Tibia_Lateral": '#9b59b6',
    }
    
    fig, ax = plt.subplots(figsize=(10, 8), facecolor='white')
    
    has_roc = False
    for region_name in region_list:
        m = region_results.get(region_name)
        if m is None or m['fpr'] is None:
            continue
        
        display_name = REGION_CN.get(region_name, region_name)
        color = line_colors.get(region_name, 'gray')
        
        ax.plot(m['fpr'], m['tpr'], color=color, linewidth=2.5,
                label=f"{display_name}  (AUC = {m['auc']:.3f},  n = {m['n']})")
        has_roc = True
    
    ax.plot([0, 1], [0, 1], 'k--', linewidth=1.0, alpha=0.5, label='Random Chance')
    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.02])
    ax.set_xlabel('False Positive Rate  (1 - Specificity)', fontsize=12)
    ax.set_ylabel('True Positive Rate  (Sensitivity)', fontsize=12)
    ax.set_title('ROC Curves — Normal vs Damaged (In-Domain Test Set)', fontsize=14, fontweight='bold')
    ax.legend(loc='lower right', fontsize=10, frameon=True, fancybox=True)
    ax.grid(True, alpha=0.3)
    
    if not has_roc:
        ax.text(0.5, 0.5,
                "Insufficient data for ROC curves\n"
                "(need both positive and negative samples per region)",
                ha='center', va='center', fontsize=12, color='#95a5a6',
                transform=ax.transAxes)
    
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    
    print(f"  [Plot] ROC curves saved: {output_path}")


def plot_confusion_matrices(region_results, output_path):
    """绘制混淆矩阵"""
    region_list = list(REGION_NAMES.values())
    
    fig = plt.figure(figsize=(18, 10), facecolor='white')
    gs = gridspec.GridSpec(2, 4, height_ratios=[1.0, 1.0],
                           hspace=0.40, wspace=0.35,
                           left=0.06, right=0.96, top=0.92, bottom=0.06)
    
    fig.suptitle('Confusion Matrices — Binary & Grade Classification (In-Domain Test Set)',
                 fontsize=15, fontweight='bold', y=0.98)
    
    # 二分类混淆矩阵
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
            cm = m['cm_binary']
            ax.imshow(cm, interpolation='nearest', cmap='Blues', aspect='equal')
            
            thresh = cm.max() / 2.0
            for i in range(2):
                for j in range(2):
                    val = cm[i, j]
                    color = 'white' if val > thresh else 'black'
                    ax.text(j, i, str(val), ha='center', va='center',
                            fontsize=14, fontweight='bold', color=color)
            
            ax.set_xticks(range(2))
            ax.set_yticks(range(2))
            ax.set_xticklabels(binary_labels, fontsize=9)
            ax.set_yticklabels(binary_labels, fontsize=9)
            ax.set_xlabel('Predicted', fontsize=10)
            ax.set_ylabel('True', fontsize=10)
            
            acc_val = m.get('accuracy', float('nan'))
            if not np.isnan(acc_val):
                ax.set_title(f"{display_name} (Binary)\nAcc = {acc_val:.3f}", 
                            fontsize=11, fontweight='bold', pad=8)
            else:
                ax.set_title(f"{display_name} (Binary)", fontsize=11, fontweight='bold', pad=8)
    
    # 三分类混淆矩阵
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
            cm = m['cm_grade']
            ax.imshow(cm, interpolation='nearest', cmap='Oranges', aspect='equal')
            
            thresh = cm.max() / 2.0
            for i in range(3):
                for j in range(3):
                    val = cm[i, j]
                    color = 'white' if val > thresh else 'black'
                    ax.text(j, i, str(val), ha='center', va='center',
                            fontsize=12, fontweight='bold', color=color)
            
            ax.set_xticks(range(3))
            ax.set_yticks(range(3))
            ax.set_xticklabels(grade_labels, fontsize=9)
            ax.set_yticklabels(grade_labels, fontsize=9)
            ax.set_xlabel('Predicted', fontsize=10)
            ax.set_ylabel('True', fontsize=10)
            
            grade_acc = m.get('grade_accuracy', float('nan'))
            if not np.isnan(grade_acc):
                ax.set_title(f"{display_name} (Grade)\nGrade Acc = {grade_acc:.3f}", 
                            fontsize=11, fontweight='bold', pad=8)
            else:
                ax.set_title(f"{display_name} (Grade)", fontsize=11, fontweight='bold', pad=8)
    
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    
    print(f"  [Plot] Confusion matrices saved: {output_path}")


def save_metrics_csv(region_results, output_path):
    """保存指标到 CSV"""
    rows = []
    
    for region_name in REGIONS:
        m = region_results.get(region_name)
        if m is None:
            rows.append({
                "region": region_name,
                "n": 0,
                "note": "insufficient_data"
            })
            continue
        
        row_data = {
            "region": region_name,
            "auc": m['auc'],
            "accuracy": m['accuracy'],
            "sensitivity": m['recall'],
            "specificity": m['specificity'],
            "precision": m['precision'],
            "f1": m['f1'],
            "n": m['n'],
            "n_positive": m['n_pos'],
            "n_negative": m['n_neg'],
            "tp": m['tp'],
            "fp": m['fp'],
            "fn": m['fn'],
            "tn": m['tn'],
            "grade_accuracy": m.get('grade_accuracy', ''),
        }
        
        for g in [0, 1, 2]:
            row_data[f'gt_grade{g}'] = m.get(f'gt_grade{g}', 0)
            row_data[f'pred_grade{g}'] = m.get(f'pred_grade{g}', 0)
            row_data[f'grade{g}_correct'] = m.get(f'grade{g}_correct', 0)
        
        rows.append(row_data)
    
    pd.DataFrame(rows).to_csv(output_path, index=False)
    print(f"  [CSV] Metrics saved: {output_path}")


def print_metrics_summary(region_results):
    """打印指标摘要"""
    print("\n" + "=" * 70)
    print("  In-Domain Test Set — Classification Metrics Summary")
    print("=" * 70)
    
    print("\n  Binary Classification (Normal vs Damaged):")
    print("-" * 70)
    
    for region_name in REGIONS:
        m = region_results.get(region_name)
        dn = REGION_CN.get(region_name, region_name)
        
        if m is None:
            print(f"  {dn:20s}  -- Insufficient data")
        else:
            auc_str = f"{m['auc']:.3f}" if not np.isnan(m['auc']) else "N/A"
            print(f"  {dn:20s}  AUC={auc_str}  Acc={m['accuracy']:.3f}  "
                  f"Sens={m['recall']:.3f}  Spec={m['specificity']:.3f}  "
                  f"Prec={m['precision']:.3f}  F1={m['f1']:.3f}  "
                  f"(n={m['n']}, pos/neg={m['n_pos']}/{m['n_neg']}, "
                  f"TP={m['tp']} FP={m['fp']} FN={m['fn']} TN={m['tn']})")
    
    print("\n  Grade Classification (0/1/2):")
    print("-" * 70)
    
    for region_name in REGIONS:
        m = region_results.get(region_name)
        dn = REGION_CN.get(region_name, region_name)
        
        if m is None or 'grade_accuracy' not in m:
            print(f"  {dn:20s}  -- N/A")
        else:
            print(f"  {dn:20s}  GradeAcc={m['grade_accuracy']:.3f}  "
                  f"GT(0/1/2)={m.get('gt_grade0',0)}/{m.get('gt_grade1',0)}/{m.get('gt_grade2',0)}  "
                  f"Pred(0/1/2)={m.get('pred_grade0',0)}/{m.get('pred_grade1',0)}/{m.get('pred_grade2',0)}  "
                  f"Correct(0/1/2)={m.get('grade0_correct',0)}/{m.get('grade1_correct',0)}/{m.get('grade2_correct',0)}")
    
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(description='Evaluate in-domain test set')
    parser.add_argument('--pred_csv', type=str, default=OUTPUT_CSV,
                        help='Path to predictions CSV (from inference pipeline)')
    parser.add_argument('--split_info', type=str, default=TEST_SPLIT_INFO,
                        help='Path to test_split_info.csv')
    parser.add_argument('--gt_excel', type=str, default=TEST_GT_EXCEL,
                        help='Path to ground truth Excel (optional, fallback)')
    parser.add_argument('--output_dir', type=str, default=OUTPUT_DIR,
                        help='Output directory for results')
    args = parser.parse_args()
    
    print("=" * 70)
    print("  In-Domain Test Set Evaluation")
    print("=" * 70)
    
    # 加载真实标签
    print("\n[1/4] Loading ground truth...")
    gt_dict = load_ground_truth_from_split_info(args.split_info)
    
    if not gt_dict:
        print("  Warning: No test cases found in split_info.csv, trying Excel...")
        gt_dict = load_ground_truth_from_excel(args.gt_excel)
    
    if not gt_dict:
        print("Error: Could not load ground truth from any source")
        sys.exit(1)
    
    print(f"  Loaded ground truth for {len(gt_dict)} cases")
    
    # 加载预测结果
    print("\n[2/4] Loading predictions...")
    pred_dict = load_predictions(args.pred_csv)
    
    if not pred_dict:
        print(f"Error: Could not load predictions from {args.pred_csv}")
        print("  Please run the inference pipeline first:")
        print(f"  bash infer/classify/run_inference.sh")
        print(f"  (with IMAGE_FOLDER={TEST_IMAGE_DIR} and MASK_FOLDER={TEST_MASK_DIR})")
        sys.exit(1)
    
    print(f"  Loaded predictions for {len(pred_dict)} cases")
    
    # 计算指标
    print("\n[3/4] Computing metrics...")
    region_results = compute_metrics(pred_dict, gt_dict)
    
    has_any = any(v is not None for v in region_results.values())
    if not has_any:
        print("Error: No matched pred-GT pairs found")
        sys.exit(1)
    
    # 打印摘要
    print_metrics_summary(region_results)
    
    # 保存结果
    print("\n[4/4] Saving results...")
    os.makedirs(args.output_dir, exist_ok=True)
    
    save_metrics_csv(region_results, os.path.join(args.output_dir, "metrics_summary.csv"))
    plot_roc_curves(region_results, os.path.join(args.output_dir, "roc_curves.png"))
    plot_confusion_matrices(region_results, os.path.join(args.output_dir, "confusion_matrices.png"))
    
    print("\n" + "=" * 70)
    print("  Evaluation Complete!")
    print(f"  Output directory: {args.output_dir}")
    print("=" * 70)


if __name__ == "__main__":
    main()

