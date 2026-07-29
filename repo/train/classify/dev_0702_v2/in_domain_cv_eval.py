#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
in_domain_cv_eval.py
集内测试评估 — GroupKFold 交叉验证的 Out-Of-Fold (OOF) 预测

由于全部 117 例（第二批5T）均用于模型训练，"集内测试"采用
5 折 GroupKFold 交叉验证的 out-of-fold 预测评估：每一折模型在
未见过的 held-out 患者上做预测，聚合所有折的 OOF 预测得到集内
测试指标。这与训练脚本 (3_train_svm_v8.py) 的评估协议完全一致。

Stage 1（二分类 Normal vs Damaged）：
  - 使用训练时的跨区域增强特征 (cross_filtered_features)
  - 复现训练最优参数 (C, gamma)
  - GroupKFold(5) OOF，患者级不泄漏
  - 每折 StandardScaler 独立 fit
  - 阈值：使用训练保存的 threshold.pkl（集内测试用训练阈值）

Stage 2（分级 G1 vs G2）：在 Stage1 判为 Damaged 的样本上评估三分类。
"""

import os
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.model_selection import GroupKFold
from sklearn.metrics import (
    roc_auc_score, roc_curve, auc,
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix
)

REPO = "/mnt/tidal-alsh-share2/dataset/askfollow/usr/yangxin/keyan/repo-dev_v4/repo"
FEATURE_DIR = os.path.join(REPO, "train/classify/dev_0702_v2/data_train/feature")
MODEL_BASE = os.path.join(REPO, "checkpoint/results_v8.9_0702_v2")
OUTPUT_DIR = os.path.join(REPO, "data/in_domain_cv_results_v8.9_0702_v2")

REGION_NAMES = ["Femur_Medial", "Femur_Lateral", "Tibia_Medial", "Tibia_Lateral"]
REGION_ABBR = {
    "Femur_Medial": "Femur Med",
    "Femur_Lateral": "Femur Lat",
    "Tibia_Medial": "Tibia Med",
    "Tibia_Lateral": "Tibia Lat",
}
REGION_COLORS = {
    "Femur_Medial": "#2ecc71",
    "Femur_Lateral": "#3498db",
    "Tibia_Medial": "#e67e22",
    "Tibia_Lateral": "#9b59b6",
}

N_SPLITS = 5
RANDOM_STATE = 42


def load_region_data(region):
    """加载训练时使用的跨区域增强特征（与 3_train_svm_v8.py 一致）"""
    cross_csv = os.path.join(FEATURE_DIR, f"{region}_cross_filtered_features.csv")
    normal_csv = os.path.join(FEATURE_DIR, f"{region}_filtered_features.csv")
    csv_path = cross_csv if os.path.exists(cross_csv) else normal_csv

    df_raw = pd.read_csv(csv_path)
    # 患者级聚合（同 case_id/region/grade 取均值）
    df = df_raw.groupby(['case_id', 'region', 'grade']).mean().reset_index()

    drop_cols = [c for c in ["case_id", "region", "grade", "cartilage_missing"] if c in df.columns]
    X = df.drop(columns=drop_cols)
    y = (df["grade"] > 0).astype(int).values
    groups = df["case_id"].values
    grade = df["grade"].values

    if X.isnull().values.any():
        X = X.fillna(0)

    return X, y, groups, grade, df["case_id"].values


def oof_binary_predict(X, y, groups, C, gamma):
    """GroupKFold OOF 二分类预测（复现训练协议）"""
    oof_prob = np.zeros(len(y))
    oof_mask = np.zeros(len(y), dtype=bool)

    cv = GroupKFold(n_splits=N_SPLITS)
    for train_ix, test_ix in cv.split(X, y, groups=groups):
        X_tr, X_te = X.iloc[train_ix], X.iloc[test_ix]
        y_tr = y[train_ix]

        if len(np.unique(y_tr)) < 2:
            continue

        scaler = StandardScaler()
        X_tr_s = scaler.fit_transform(X_tr)
        X_te_s = scaler.transform(X_te)

        model = SVC(kernel='rbf', C=C, gamma=gamma, probability=True, random_state=RANDOM_STATE)
        model.fit(X_tr_s, y_tr)

        oof_prob[test_ix] = model.predict_proba(X_te_s)[:, 1]
        oof_mask[test_ix] = True

    return oof_prob, oof_mask


def load_stage2_data(region):
    """加载 Stage2 特征（G1 vs G2）"""
    csv_path = os.path.join(FEATURE_DIR, f"{region}_stage2_filtered_features.csv")
    if not os.path.exists(csv_path):
        # pooled fallback
        csv_path = os.path.join(FEATURE_DIR, "pooled_stage2_FL_TL_filtered_features.csv")
        if not os.path.exists(csv_path):
            return None
    df_raw = pd.read_csv(csv_path)
    df = df_raw.groupby(['case_id', 'region', 'grade']).mean().reset_index()
    if region in ["Femur_Medial", "Tibia_Medial"]:
        df = df[df['region'] == region] if 'region' in df.columns else df
    drop_cols = [c for c in ["case_id", "region", "grade", "cartilage_missing"] if c in df.columns]
    X = df.drop(columns=drop_cols)
    y = (df["grade"] == 2).astype(int).values
    groups = df["case_id"].values
    if X.isnull().values.any():
        X = X.fillna(0)
    return X, y, groups, df["case_id"].values, df["grade"].values


def oof_stage2_predict(X, y, groups):
    """GroupKFold OOF Stage2 预测"""
    oof_prob = np.full(len(y), np.nan)
    n_minority = min((y == 0).sum(), (y == 1).sum())
    if n_minority < 2:
        return oof_prob
    n_splits = min(N_SPLITS, n_minority, len(np.unique(groups)))
    if n_splits < 2:
        return oof_prob
    cv = GroupKFold(n_splits=n_splits)
    for train_ix, test_ix in cv.split(X, y, groups=groups):
        y_tr = y[train_ix]
        if len(np.unique(y_tr)) < 2:
            continue
        scaler = StandardScaler()
        X_tr_s = scaler.fit_transform(X.iloc[train_ix])
        X_te_s = scaler.transform(X.iloc[test_ix])
        model = SVC(kernel='rbf', C=1, gamma='scale', probability=True,
                    class_weight='balanced', random_state=RANDOM_STATE)
        model.fit(X_tr_s, y_tr)
        oof_prob[test_ix] = model.predict_proba(X_te_s)[:, 1]
    return oof_prob


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print("=" * 70)
    print("  In-Domain Test (GroupKFold OOF Cross-Validation)")
    print("=" * 70)

    results = {}
    all_pred_rows = []

    for region in REGION_NAMES:
        print(f"\n>>> {region}")
        # 加载训练参数
        mp = joblib.load(os.path.join(MODEL_BASE, region, "models", "model_params.pkl"))
        train_th = joblib.load(os.path.join(MODEL_BASE, region, "models", "threshold.pkl"))
        C, gamma = mp['C'], mp['gamma']
        print(f"    Params: C={C}, gamma={gamma}, train_threshold={train_th:.4f}")

        # Stage 1 OOF
        X, y, groups, grade, case_ids = load_region_data(region)
        oof_prob, mask = oof_binary_predict(X, y, groups, C, gamma)

        y_true = y[mask]
        prob = oof_prob[mask]
        grade_true = grade[mask]
        cids = case_ids[mask]
        y_pred = (prob >= train_th).astype(int)

        # Stage 2 OOF (grade)
        s2 = load_stage2_data(region)
        pred_grade = np.zeros(len(y_true), dtype=int)  # default G0
        # 对判为 Damaged 的样本分级
        if s2 is not None:
            X2, y2, g2groups, s2_cids, s2_grades = s2
            oof_p2 = oof_stage2_predict(X2, y2, g2groups)
            # map case_id -> pg2
            pg2_map = {}
            for i, cid in enumerate(s2_cids):
                if not np.isnan(oof_p2[i]):
                    pg2_map[cid] = oof_p2[i]
        else:
            pg2_map = {}

        for i in range(len(y_true)):
            if y_pred[i] == 1:  # Damaged
                cid = cids[i]
                pg2 = pg2_map.get(cid, 0.0)
                pred_grade[i] = 2 if pg2 >= 0.5 else 1
            else:
                pred_grade[i] = 0

        # 指标
        fpr, tpr, _ = roc_curve(y_true, prob)
        auc_val = auc(fpr, tpr)
        acc = accuracy_score(y_true, y_pred)
        prec = precision_score(y_true, y_pred, zero_division=0)
        rec = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        grade_acc = float(np.mean(grade_true == pred_grade))
        cm_grade = confusion_matrix(grade_true, pred_grade, labels=[0, 1, 2])

        results[region] = {
            'auc': auc_val, 'acc': acc, 'sens': rec, 'spec': spec,
            'prec': prec, 'f1': f1, 'n': len(y_true),
            'pos': int(y_true.sum()), 'neg': int((1 - y_true).sum()),
            'tp': int(tp), 'fp': int(fp), 'fn': int(fn), 'tn': int(tn),
            'fpr': fpr, 'tpr': tpr, 'cm': cm,
            'grade_acc': grade_acc, 'cm_grade': cm_grade,
        }

        print(f"    AUC={auc_val:.3f} Acc={acc:.3f} Sens={rec:.3f} Spec={spec:.3f} "
              f"Prec={prec:.3f} F1={f1:.3f} (n={len(y_true)}, pos/neg={int(y_true.sum())}/{int((1-y_true).sum())})")
        print(f"    Grade Acc={grade_acc:.3f}")

        for i in range(len(y_true)):
            all_pred_rows.append({
                'case_id': cids[i], 'region': region,
                'true_grade': int(grade_true[i]), 'true_binary': int(y_true[i]),
                'oof_prob_damage': float(prob[i]), 'pred_binary': int(y_pred[i]),
                'pred_grade': int(pred_grade[i]), 'threshold': float(train_th),
            })

    # 保存预测明细
    pd.DataFrame(all_pred_rows).to_csv(os.path.join(OUTPUT_DIR, "oof_predictions.csv"), index=False)

    # 保存指标 CSV
    rows = []
    for region in REGION_NAMES:
        m = results[region]
        rows.append({
            'Region': REGION_ABBR[region], 'AUC': round(m['auc'], 3),
            'Acc': round(m['acc'], 3), 'Sens': round(m['sens'], 3),
            'Spec': round(m['spec'], 3), 'Prec': round(m['prec'], 3),
            'F1': round(m['f1'], 3), 'n': m['n'], 'Pos': m['pos'], 'Neg': m['neg'],
            'TP': m['tp'], 'FP': m['fp'], 'FN': m['fn'], 'TN': m['tn'],
            'GradeAcc': round(m['grade_acc'], 3),
        })
    df_metrics = pd.DataFrame(rows)
    df_metrics.to_csv(os.path.join(OUTPUT_DIR, "metrics_summary.csv"), index=False)
    print("\n" + "=" * 70)
    print("  Metrics Summary (In-Domain OOF CV)")
    print("=" * 70)
    print(df_metrics.to_string(index=False))

    # ============ 绘制混淆矩阵（binary + grade），风格对齐外部验证图 ============
    fig = plt.figure(figsize=(18, 9), facecolor='white')
    gs = gridspec.GridSpec(2, 4, hspace=0.42, wspace=0.35,
                           left=0.06, right=0.96, top=0.90, bottom=0.07)
    fig.suptitle('Confusion Matrices — Binary & Grade Classification (In-Domain CV)',
                 fontsize=15, fontweight='bold', y=0.97)
    binary_labels = ["Normal", "Damaged"]
    for col, region in enumerate(REGION_NAMES):
        ax = fig.add_subplot(gs[0, col])
        m = results[region]
        cm = m['cm']
        ax.imshow(cm, interpolation='nearest', cmap='Blues', aspect='equal')
        thresh = cm.max() / 2.0
        for i in range(2):
            for j in range(2):
                ax.text(j, i, str(cm[i, j]), ha='center', va='center',
                        fontsize=15, fontweight='bold',
                        color='white' if cm[i, j] > thresh else 'black')
        ax.set_xticks(range(2)); ax.set_yticks(range(2))
        ax.set_xticklabels(binary_labels, fontsize=9)
        ax.set_yticklabels(binary_labels, fontsize=9)
        ax.set_xlabel('Predicted', fontsize=10); ax.set_ylabel('True', fontsize=10)
        ax.set_title(f"{REGION_ABBR[region]} (Binary)\nAcc = {m['acc']:.3f}",
                     fontsize=11, fontweight='bold', pad=8)
    grade_labels = ["G0", "G1", "G2"]
    for col, region in enumerate(REGION_NAMES):
        ax = fig.add_subplot(gs[1, col])
        m = results[region]
        cm = m['cm_grade']
        ax.imshow(cm, interpolation='nearest', cmap='Oranges', aspect='equal')
        thresh = cm.max() / 2.0
        for i in range(3):
            for j in range(3):
                ax.text(j, i, str(cm[i, j]), ha='center', va='center',
                        fontsize=13, fontweight='bold',
                        color='white' if cm[i, j] > thresh else 'black')
        ax.set_xticks(range(3)); ax.set_yticks(range(3))
        ax.set_xticklabels(grade_labels, fontsize=9)
        ax.set_yticklabels(grade_labels, fontsize=9)
        ax.set_xlabel('Predicted', fontsize=10); ax.set_ylabel('True', fontsize=10)
        ax.set_title(f"{REGION_ABBR[region]} (Grade)\nGrade Acc = {m['grade_acc']:.3f}",
                     fontsize=11, fontweight='bold', pad=8)
    fig.savefig(os.path.join(OUTPUT_DIR, "confusion_matrices.png"),
                dpi=200, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"\n  [Plot] Confusion matrices: {OUTPUT_DIR}/confusion_matrices.png")

    # ============ 绘制指标表格图，风格对齐外部验证图 ============
    fig, ax = plt.subplots(figsize=(14, 3.2), facecolor='white')
    ax.axis('off')
    ax.set_title('Binary Classification Metrics  (Normal vs Damaged) — In-Domain CV',
                 fontsize=14, fontweight='bold', pad=16)
    col_labels = ['Region', 'AUC', 'Acc', 'Sens', 'Spec', 'Prec', 'F1', 'n', 'Pos', 'Neg', 'TP', 'FP', 'FN', 'TN']
    cell_text = []
    row_colors = []
    for region in REGION_NAMES:
        m = results[region]
        cell_text.append([
            REGION_ABBR[region], f"{m['auc']:.3f}", f"{m['acc']:.3f}",
            f"{m['sens']:.3f}", f"{m['spec']:.3f}", f"{m['prec']:.3f}",
            f"{m['f1']:.3f}", str(m['n']), str(m['pos']), str(m['neg']),
            str(m['tp']), str(m['fp']), str(m['fn']), str(m['tn']),
        ])
        row_colors.append(REGION_COLORS[region])
    table = ax.table(cellText=cell_text, colLabels=col_labels,
                     cellLoc='center', loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.8)
    for j in range(len(col_labels)):
        table[(0, j)].set_text_props(fontweight='bold')
    for i, region in enumerate(REGION_NAMES):
        table[(i + 1, 0)].set_text_props(fontweight='bold', color=REGION_COLORS[region])
    fig.savefig(os.path.join(OUTPUT_DIR, "metrics_table.png"),
                dpi=200, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"  [Plot] Metrics table: {OUTPUT_DIR}/metrics_table.png")

    print("\n" + "=" * 70)
    print(f"  Done! Output: {OUTPUT_DIR}")
    print("=" * 70)


if __name__ == "__main__":
    main()

