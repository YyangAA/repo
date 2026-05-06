#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SVM_RBF_save_model_v4.py
v4 改造：支持 lasso_v4 输出的「池化 Stage 2」

变化要点：
  1. Stage 1：与 v3 完全一致 (4 区域独立)，但读取目录改为 feature_v4
  2. Stage 2：
     - 优先读取 PooledStage2_filtered_features.csv（池化版）
     - 池化模式下：训练 1 个统一 SVM，使用 GroupKFold 评估（防止同病人跨折）
     - 同一份池化模型 + scaler + feature_list 会被复制到每个 region 目录下
       （这样下游 4 区分别推理的脚本无需改动，能直接用）
     - 同时在 Pooled_Stage2/ 目录下保存"主模型"作为 source of truth
     - 加 is_pooled_stage2.pkl 标记，推理时可识别
  3. 如果池化 CSV 不存在，自动回退到 v3 的 4 区独立 Stage 2 逻辑
"""

import pandas as pd
import numpy as np
import os
import warnings
import matplotlib.pyplot as plt
import joblib
from sklearn.model_selection import StratifiedKFold, GridSearchCV, GroupKFold, StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import roc_auc_score, accuracy_score, roc_curve, auc

# ===============================
# 0. 设置路径与区域
# ===============================
base_dir = "./train/classify/dev_v3/data_train"
csv_input_dir = "./train/classify/dev_v3/data_train/feature_v4"   # ★ 改为 v4 输出目录
results_output_dir = "./checkpoint/results_260506_v4"

region_names = {
    1: "Femur_Medial",
    2: "Femur_Lateral",
    3: "Tibia_Medial",
    4: "Tibia_Lateral",
}

# ===============================================================
#  STAGE 1: 二分类 SVM (Normal vs Damaged) — 与 v3 一致
# ===============================================================
print("\n" + "#" * 70)
print("  STAGE 1: Training Binary SVM (Normal vs Damaged)")
print("#" * 70)

for r_idx in [1, 2, 3, 4]:
    knee_cartilage = region_names[r_idx]
    print(f"\n>>>> [Stage 1] Training SVM Model for Region: {knee_cartilage} <<<<")

    # 1. 读取数据
    csv_path = os.path.join(csv_input_dir, f"{knee_cartilage}_filtered_features.csv")
    if not os.path.exists(csv_path):
        print(f"Skipping {knee_cartilage}: {csv_path} not found.")
        continue

    df_raw = pd.read_csv(csv_path)

    # 2. 按患者聚合
    df = df_raw.groupby(['case_id', 'region', 'grade']).mean().reset_index()
    print(f"Original rows: {len(df_raw)} -> Patient-level rows: {len(df)}")

    # 3. 特征矩阵
    X = df.drop(columns=["case_id", "region", "grade"])
    y = (df["grade"] > 0).astype(int).values
    groups = df["case_id"].values
    print(f"Data shape: {df.shape}")

    if X.isnull().values.any():
        X = X.fillna(0)

    # 4. 交叉验证评估 (GroupKFold)
    cv_outer = GroupKFold(n_splits=5)
    param_grid = {
        'C': [0.1, 1, 10, 100],
        'gamma': ['scale', 'auto', 0.01, 0.1, 1]
    }

    auc_list = []
    tprs = []
    threshold_list = []
    mean_fpr = np.linspace(0, 1, 100)

    fold = 1
    for train_ix, test_ix in cv_outer.split(X, y, groups=groups):
        X_train, X_test = X.iloc[train_ix], X.iloc[test_ix]
        y_train, y_test = y[train_ix], y[test_ix]

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        grid = GridSearchCV(
            SVC(kernel='rbf', probability=True, class_weight='balanced'),
            param_grid, refit=True, cv=3, scoring='roc_auc'
        )
        grid.fit(X_train_scaled, y_train)

        best_model = grid.best_estimator_
        y_prob = best_model.predict_proba(X_test_scaled)[:, 1]

        current_auc = roc_auc_score(y_test, y_prob)
        auc_list.append(current_auc)

        fpr, tpr, thresholds = roc_curve(y_test, y_prob)
        interp_tpr = np.interp(mean_fpr, fpr, tpr)
        interp_tpr[0] = 0.0
        tprs.append(interp_tpr)

        youden_j = tpr - fpr
        best_idx = np.argmax(youden_j)
        threshold_list.append(thresholds[best_idx])

        print(f"  Fold {fold} AUC: {current_auc:.3f}, Threshold: {thresholds[best_idx]:.3f}")
        fold += 1

    # 5. ROC 曲线
    fig_dir = os.path.join(results_output_dir, knee_cartilage, "figures")
    os.makedirs(fig_dir, exist_ok=True)

    plt.figure(figsize=(6, 5))
    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc = np.std(auc_list)

    plt.plot(mean_fpr, mean_tpr, color='b',
             label=f'Mean ROC (AUC = {mean_auc:.3f} ± {std_auc:.3f})',
             lw=2, alpha=0.8)
    plt.plot([0, 1], [0, 1], linestyle='--', color='r', lw=2, alpha=0.5)
    plt.title(f'ROC Curve - {knee_cartilage} (Stage1)')
    plt.xlabel('False Positive Rate'); plt.ylabel('True Positive Rate')
    plt.legend(loc="lower right"); plt.grid(alpha=0.3)
    plt.savefig(os.path.join(fig_dir, f"{knee_cartilage}_ROC.png"), dpi=300)
    plt.close()

    optimal_threshold = float(np.mean(threshold_list))
    print(f"Completed {knee_cartilage}. Mean AUC: {mean_auc:.3f}")
    print(f"  CV Optimal Threshold: {optimal_threshold:.4f}")

    # 6. 全量训练 + 保存
    final_scaler = StandardScaler()
    X_scaled = final_scaler.fit_transform(X)

    final_grid = GridSearchCV(
        SVC(kernel='rbf', probability=True, class_weight='balanced'),
        param_grid, cv=3, scoring='roc_auc'
    )
    final_grid.fit(X_scaled, y)
    final_model = final_grid.best_estimator_

    model_dir = os.path.join(results_output_dir, knee_cartilage, "models")
    os.makedirs(model_dir, exist_ok=True)

    joblib.dump(final_model, os.path.join(model_dir, "svm_model.pkl"))
    joblib.dump(final_scaler, os.path.join(model_dir, "scaler.pkl"))
    joblib.dump(X.columns.tolist(), os.path.join(model_dir, "feature_list.pkl"))
    joblib.dump(optimal_threshold, os.path.join(model_dir, "threshold.pkl"))

    print(f"✅ [Stage 1] Saved: {model_dir}")


# ===============================================================
#  STAGE 2: 池化 SVM (Grade 1 vs Grade 2)
# ===============================================================
print("\n" + "#" * 70)
print("  STAGE 2: POOLED SVM (Grade 1 vs Grade 2)")
print("#" * 70)

pooled_csv = os.path.join(csv_input_dir, "PooledStage2_filtered_features.csv")
use_pooled = os.path.exists(pooled_csv)

if use_pooled:
    print(f"\n[Stage 2] Pooled CSV detected → using POOLED training mode")
    print(f"  CSV: {pooled_csv}")

    # ===== 1. 读取池化数据 =====
    df_raw = pd.read_csv(pooled_csv)

    # ===== 2. 按患者+区域聚合（保留 region 信息！） =====
    df = df_raw.groupby(['case_id', 'region', 'grade']).mean().reset_index()
    print(f"  Original rows: {len(df_raw)} → Patient-region rows: {len(df)}")
    print(f"  Region distribution:")
    print(df.groupby("region")["grade"].value_counts().unstack(fill_value=0).to_string())

    # ===== 3. 构造特征矩阵 =====
    # 注意：lasso_v4 已经把 region one-hot (regionFlag_*) 选入特征列
    # 这些列就是模型识别"样本来自哪个区域"的关键，必须保留
    X = df.drop(columns=["case_id", "region", "grade"])
    if "cartilage_missing" in X.columns:
        X = X.drop(columns=["cartilage_missing"])
    y = (df["grade"] == 2).astype(int).values
    groups = df["case_id"].values

    n_g1 = int((y == 0).sum())
    n_g2 = int((y == 1).sum())
    print(f"\n  Pooled feature shape: {X.shape}")
    print(f"  Class dist — G1: {n_g1}, G2: {n_g2}")

    if X.isnull().values.any():
        print(f"  Filling NaN with 0 ({X.isnull().sum().sum()} cells)")
        X = X.fillna(0)

    param_grid_s2 = {
        'C': [0.1, 1, 10, 100],
        'gamma': ['scale', 'auto', 0.01, 0.1, 1]
    }

    # ===== 4. 交叉验证 =====
    # 用 StratifiedGroupKFold：既保证类别比例，又防止同一病人跨折
    n_minority = min(n_g1, n_g2)
    n_folds = min(5, n_minority)

    print(f"\n  Running {n_folds}-fold StratifiedGroupKFold CV...")

    auc_list_s2 = []
    acc_list_s2 = []
    tprs_s2 = []
    threshold_list_s2 = []
    mean_fpr = np.linspace(0, 1, 100)

    try:
        cv_outer = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=42)
        cv_iter = cv_outer.split(X, y, groups=groups)
    except Exception:
        # 老版本 sklearn 没有 StratifiedGroupKFold，退回 StratifiedKFold
        print("  StratifiedGroupKFold unavailable, falling back to StratifiedKFold")
        cv_outer = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
        cv_iter = cv_outer.split(X, y)

    fold = 1
    for train_ix, test_ix in cv_iter:
        X_train, X_test = X.iloc[train_ix], X.iloc[test_ix]
        y_train, y_test = y[train_ix], y[test_ix]

        if len(np.unique(y_train)) < 2 or len(np.unique(y_test)) < 2:
            print(f"  Fold {fold}: skipped (single class in train/test)")
            fold += 1
            continue

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            grid = GridSearchCV(
                SVC(kernel='rbf', probability=True, class_weight='balanced'),
                param_grid_s2, refit=True, cv=3, scoring='roc_auc'
            )
            grid.fit(X_train_scaled, y_train)

        best_model = grid.best_estimator_
        y_prob = best_model.predict_proba(X_test_scaled)[:, 1]
        y_pred = (y_prob >= 0.5).astype(int)

        current_auc = roc_auc_score(y_test, y_prob)
        current_acc = accuracy_score(y_test, y_pred)
        auc_list_s2.append(current_auc)
        acc_list_s2.append(current_acc)

        fpr, tpr, thresholds = roc_curve(y_test, y_prob)
        interp_tpr = np.interp(mean_fpr, fpr, tpr)
        interp_tpr[0] = 0.0
        tprs_s2.append(interp_tpr)

        youden_j = tpr - fpr
        best_idx = np.argmax(youden_j)
        threshold_list_s2.append(thresholds[best_idx])

        print(f"  Fold {fold} AUC: {current_auc:.3f}, Acc: {current_acc:.3f}, "
              f"Thr: {thresholds[best_idx]:.3f}")
        fold += 1

    if auc_list_s2:
        mean_tpr_s2 = np.mean(tprs_s2, axis=0)
        mean_tpr_s2[-1] = 1.0
        mean_auc_s2 = auc(mean_fpr, mean_tpr_s2)
        std_auc_s2 = np.std(auc_list_s2)

        print(f"\n  Pooled Stage 2 — Mean AUC: {mean_auc_s2:.3f} ± {std_auc_s2:.3f}")
        print(f"  Pooled Stage 2 — Mean Acc: {np.mean(acc_list_s2):.3f}")

        # 保存池化 ROC 图
        pooled_fig_dir = os.path.join(results_output_dir, "Pooled_Stage2", "figures")
        os.makedirs(pooled_fig_dir, exist_ok=True)

        plt.figure(figsize=(6, 5))
        plt.plot(mean_fpr, mean_tpr_s2, color='g',
                 label=f'Mean ROC (AUC = {mean_auc_s2:.3f} ± {std_auc_s2:.3f})',
                 lw=2, alpha=0.8)
        plt.plot([0, 1], [0, 1], linestyle='--', color='r', lw=2, alpha=0.5)
        plt.title('ROC Curve - Pooled Stage 2 (G1 vs G2)')
        plt.xlabel('False Positive Rate'); plt.ylabel('True Positive Rate')
        plt.legend(loc="lower right"); plt.grid(alpha=0.3)
        plt.savefig(os.path.join(pooled_fig_dir, "Pooled_Stage2_ROC.png"), dpi=300)
        plt.close()

        optimal_threshold_s2 = float(np.mean(threshold_list_s2))
        print(f"  Pooled Stage 2 Threshold (Youden's J): {optimal_threshold_s2:.4f}")
    else:
        optimal_threshold_s2 = 0.5
        print("  ⚠ No valid CV folds; threshold defaults to 0.5")

    # ===== 5. 全量训练 + 保存 =====
    print(f"\n  Training final POOLED Stage 2 model on all {len(X)} samples...")
    final_scaler_s2 = StandardScaler()
    X_scaled_s2 = final_scaler_s2.fit_transform(X)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        final_grid_s2 = GridSearchCV(
            SVC(kernel='rbf', probability=True, class_weight='balanced'),
            param_grid_s2, cv=3, scoring='roc_auc'
        )
        final_grid_s2.fit(X_scaled_s2, y)
    final_model_s2 = final_grid_s2.best_estimator_
    print(f"  Best params: {final_grid_s2.best_params_}")

    # ===== 6. 保存到 Pooled 主目录（source of truth） =====
    pooled_model_dir = os.path.join(results_output_dir, "Pooled_Stage2", "models")
    os.makedirs(pooled_model_dir, exist_ok=True)

    joblib.dump(final_model_s2, os.path.join(pooled_model_dir, "svm_model_stage2.pkl"))
    joblib.dump(final_scaler_s2, os.path.join(pooled_model_dir, "scaler_stage2.pkl"))
    joblib.dump(X.columns.tolist(), os.path.join(pooled_model_dir, "feature_list_stage2.pkl"))
    joblib.dump(optimal_threshold_s2, os.path.join(pooled_model_dir, "threshold_stage2.pkl"))
    joblib.dump(True, os.path.join(pooled_model_dir, "is_pooled_stage2.pkl"))

    print(f"  ✅ Pooled model saved → {pooled_model_dir}")

    # ===== 7. 同时复制到每个 region 目录，保持下游兼容 =====
    print(f"\n  Replicating pooled model to per-region dirs (for downstream compatibility):")
    for r_idx, region_name in region_names.items():
        region_model_dir = os.path.join(results_output_dir, region_name, "models")
        os.makedirs(region_model_dir, exist_ok=True)

        joblib.dump(final_model_s2, os.path.join(region_model_dir, "svm_model_stage2.pkl"))
        joblib.dump(final_scaler_s2, os.path.join(region_model_dir, "scaler_stage2.pkl"))
        joblib.dump(X.columns.tolist(), os.path.join(region_model_dir, "feature_list_stage2.pkl"))
        joblib.dump(optimal_threshold_s2, os.path.join(region_model_dir, "threshold_stage2.pkl"))
        # ★ 标记：这是池化模型的副本，推理时需要给样本附加 region one-hot 特征
        joblib.dump(True, os.path.join(region_model_dir, "is_pooled_stage2.pkl"))
        joblib.dump(region_name, os.path.join(region_model_dir, "stage2_region_tag.pkl"))

        print(f"    → {region_model_dir}")

    print(f"\n✅ [Stage 2 POOLED] All models saved.")
    print(f"   Source of truth: {pooled_model_dir}")
    print(f"   Feature count (incl. region one-hot): {len(X.columns)}")

else:
    # =========================================================
    # 回退路径：4 区独立训练（与 v3 完全一致）
    # =========================================================
    print(f"\n[Stage 2] Pooled CSV not found, fallback to per-region mode.")
    print(f"  Expected: {pooled_csv}")

    for r_idx in [1, 2, 3, 4]:
        knee_cartilage = region_names[r_idx]
        print(f"\n>>>> [Stage 2 / per-region] Region: {knee_cartilage} <<<<")

        csv_path = os.path.join(csv_input_dir, f"{knee_cartilage}_stage2_filtered_features.csv")
        if not os.path.exists(csv_path):
            print(f"Skipping {knee_cartilage}: Stage 2 CSV not found.")
            continue

        df_raw = pd.read_csv(csv_path)
        df = df_raw.groupby(['case_id', 'region', 'grade']).mean().reset_index()
        print(f"Original rows: {len(df_raw)} -> Patient-level rows: {len(df)}")

        X = df.drop(columns=["case_id", "region", "grade"])
        y = (df["grade"] == 2).astype(int).values

        n_g1 = int((y == 0).sum())
        n_g2 = int((y == 1).sum())
        print(f"Data shape: {X.shape}, G1: {n_g1}, G2: {n_g2}")

        if X.isnull().values.any():
            X = X.fillna(0)

        if min(n_g1, n_g2) < 2:
            print(f"  ⚠ Too few minority samples; using default params.")
            final_model_s2 = SVC(kernel='rbf', C=1, gamma='scale',
                                  probability=True, class_weight='balanced')
            scaler_s2 = StandardScaler()
            X_scaled = scaler_s2.fit_transform(X)
            final_model_s2.fit(X_scaled, y)
        else:
            scaler_s2 = StandardScaler()
            X_scaled = scaler_s2.fit_transform(X)
            param_grid_s2 = {'C': [0.1, 1, 10, 100],
                             'gamma': ['scale', 'auto', 0.01, 0.1, 1]}
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                final_grid_s2 = GridSearchCV(
                    SVC(kernel='rbf', probability=True, class_weight='balanced'),
                    param_grid_s2, cv=min(3, min(n_g1, n_g2)), scoring='accuracy'
                )
                final_grid_s2.fit(X_scaled, y)
            final_model_s2 = final_grid_s2.best_estimator_

        model_dir_s2 = os.path.join(results_output_dir, knee_cartilage, "models")
        os.makedirs(model_dir_s2, exist_ok=True)
        joblib.dump(final_model_s2, os.path.join(model_dir_s2, "svm_model_stage2.pkl"))
        joblib.dump(scaler_s2, os.path.join(model_dir_s2, "scaler_stage2.pkl"))
        joblib.dump(X.columns.tolist(), os.path.join(model_dir_s2, "feature_list_stage2.pkl"))
        joblib.dump(False, os.path.join(model_dir_s2, "is_pooled_stage2.pkl"))
        print(f"  ✅ Saved: {model_dir_s2}")


print("\n" + "=" * 70)
print("All regions processed. Stage 1 + Stage 2 models saved.")
print(f"Results directory: {results_output_dir}")
if use_pooled:
    print(f"  Stage 2 mode: POOLED (single model, replicated to per-region dirs)")
else:
    print(f"  Stage 2 mode: PER-REGION (fallback)")
print("=" * 70)