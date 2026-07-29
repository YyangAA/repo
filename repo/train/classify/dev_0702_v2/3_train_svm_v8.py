#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
3_train_svm_v8.py (dev_0702_v2)
SVM-RBF 级联分类器训练 - v2 优化版本

v2 关键改进:
  1. 扩展 SVM 参数搜索空间:
     - C: [0.01, 0.05, 0.1, 0.5, 1, 5, 10, 50, 100] (原: [0.1, 1, 10])
     - gamma: ['scale', 'auto', 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1] (原: ['scale', 0.01, 0.1])
     - 总候选从 9 组扩展到 81 组
  2. Stage2 分池化策略:
     - FM 独立训练 Stage2 模型 (FM G1/G2 区分模式与其他区域不同)
     - TM 独立训练 Stage2 模型 (样本相对充足)
     - FL+TL 继续使用池化模型 (样本少)
  3. 过拟合惩罚函数优化:
     - C_penalty 从 0.015 降到 0.005 (更温和的 C 偏好)
     - 增加 C=0.01 和 C=0.05 的探索机会
  4. 阈值 cap 放宽: [0.30, 0.65] → [0.30, 0.70] (给高 AUC 区域更多自由度)
"""

import pandas as pd
import numpy as np
import os
import warnings
import math
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import joblib
from sklearn.model_selection import StratifiedKFold, GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import roc_auc_score, accuracy_score, roc_curve, auc

# ===============================
# 0. 设置路径与区域
# ===============================
csv_input_dir = "./train/classify/dev_0702_v2/data_train/feature"
results_output_dir = "./checkpoint/results_v8.9_0702_v2"

region_names = {
    1: "Femur_Medial",
    2: "Femur_Lateral",
    3: "Tibia_Medial",
    4: "Tibia_Lateral"
}

REGIONS = list(region_names.values())

# v2: FM 和 TM 使用独立 Stage2, FL+TL 使用池化 Stage2
STAGE2_INDEPENDENT_REGIONS = ["Femur_Medial", "Tibia_Medial"]
STAGE2_POOLED_REGIONS = ["Femur_Lateral", "Tibia_Lateral"]


# ===============================================================
#  v2: 扩展参数搜索空间
# ===============================================================
def _select_svm_params_with_penalty(X, y, groups, random_state=42, class_weight=None):
    """
    v2: 扩展参数搜索空间 + 优化惩罚函数。
    
    参数空间从 9 组扩展到 81 组:
      C: [0.01, 0.05, 0.1, 0.5, 1, 5, 10, 50, 100]
      gamma: ['scale', 'auto', 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1]
    """
    param_candidates = [
        # C × gamma: 9 × 9 = 81 组合
        (C, gamma)
        for C in [0.01, 0.05, 0.1, 0.5, 1, 5, 10, 50, 100]
        for gamma in ['scale', 'auto', 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1]
    ]

    overfit_weight = 2.0
    perfect_penalty = 0.05
    range_weight = 0.01
    C_penalty = 0.015       # v2.1: 恢复到 0.015 (0.005 过低导致总选 C=0.01)
    leak_penalty = 0.05
    gamma_smooth_penalty = 0.02  # v2.1: gamma < 0.01 时额外惩罚 (防止 RBF 过于平滑)
    
    cv_outer = GroupKFold(n_splits=5)
    
    results = []
    
    for C, gamma in param_candidates:
        try:
            cv_aucs = []
            train_aucs = []
            
            for train_ix, test_ix in cv_outer.split(X, y, groups=groups):
                X_train, X_test = X.iloc[train_ix], X.iloc[test_ix]
                y_train, y_test = y[train_ix], y[test_ix]
                
                if len(np.unique(y_train)) < 2:
                    continue
                
                scaler = StandardScaler()
                X_train_scaled = scaler.fit_transform(X_train)
                X_test_scaled = scaler.transform(X_test)
                
                model = SVC(kernel='rbf', C=C, gamma=gamma, probability=True,
                           class_weight=class_weight)
                model.fit(X_train_scaled, y_train)
                
                y_prob_test = model.predict_proba(X_test_scaled)[:, 1]
                y_prob_train = model.predict_proba(X_train_scaled)[:, 1]
                
                cv_aucs.append(roc_auc_score(y_test, y_prob_test))
                train_aucs.append(roc_auc_score(y_train, y_prob_train))
            
            if not cv_aucs:
                continue
                
            mean_cv_auc = np.mean(cv_aucs)
            mean_train_auc = np.mean(train_aucs)
            overfit_gap = mean_train_auc - mean_cv_auc
            
            # 计算全量训练集的概率分布宽度
            scaler_full = StandardScaler()
            X_scaled_full = scaler_full.fit_transform(X)
            model_full = SVC(kernel='rbf', C=C, gamma=gamma, probability=True,
                            class_weight=class_weight)
            model_full.fit(X_scaled_full, y)
            y_prob_full = model_full.predict_proba(X_scaled_full)[:, 1]
            prob_range = float(np.max(y_prob_full) - np.min(y_prob_full))
            
            c_log = math.log10(C) if C > 0 else 0
            
            # v2.1: gamma 惩罚 — gamma 太小导致 RBF 核过于平滑，概率分布平坦
            gamma_penalty = 0.0
            if isinstance(gamma, (int, float)) and gamma < 0.01:
                gamma_penalty = gamma_smooth_penalty * (1.0 - gamma / 0.01)
            
            penalized = (mean_cv_auc
                         - overfit_weight * max(overfit_gap, 0)
                         - perfect_penalty * (1.0 if mean_train_auc >= 0.999 else 0.0)
                         - leak_penalty * (1.0 if mean_cv_auc >= 0.995 else 0.0)
                         + range_weight * prob_range
                         - C_penalty * c_log
                         - gamma_penalty)
            
            results.append({
                'C': C, 'gamma': gamma,
                'cv_auc': mean_cv_auc,
                'train_auc': mean_train_auc,
                'gap': overfit_gap,
                'prob_range': prob_range,
                'penalized': penalized,
            })
            
        except Exception as e:
            continue
    
    if not results:
        print("    Warning: No valid param combination found, using default C=1, gamma=0.01")
        return 1, 0.01, 0, 0, 0, 0
    
    # 过滤无效组合
    valid = [r for r in results if r['train_auc'] >= 0.5 and r['prob_range'] >= 0.20]
    if not valid:
        print("    Warning: No param with prob_range>=0.20, relaxing to >=0.10")
        valid = [r for r in results if r['train_auc'] >= 0.5 and r['prob_range'] >= 0.10]
    if not valid:
        valid = results
    
    valid.sort(key=lambda x: x['penalized'], reverse=True)
    best = valid[0]
    
    print(f"    Best: C={best['C']}, gamma={best['gamma']}, "
          f"cv_auc={best['cv_auc']:.3f}, train_auc={best['train_auc']:.3f}, "
          f"gap={best['gap']:.3f}, prob_range={best['prob_range']:.3f}, "
          f"penalized={best['penalized']:.3f}")
    
    for i, r in enumerate(valid[:3]):
        print(f"    #{i+1}: C={r['C']}, gamma={r['gamma']}, "
              f"penalized={r['penalized']:.3f} (cv={r['cv_auc']:.3f}, "
              f"gap={r['gap']:.3f}, range={r['prob_range']:.3f})")
    
    return best['C'], best['gamma'], best['cv_auc'], best['train_auc'], best['gap'], best['prob_range']


# ===============================================================
#  STAGE 1: 二分类 SVM (Normal vs Damaged) — 跨区域特征增强
# ===============================================================
print("\n" + "#" * 60)
print("  STAGE 1: Training Binary SVM (Normal vs Damaged) [v2]")
print("  Extended param space (81 combos) + Adaptive penalty")
print("#" * 60)

for r_idx in [1, 2, 3, 4]:
    knee_cartilage = region_names[r_idx]
    print(f"\n>>>> [Stage 1] Training SVM Model for Region: {knee_cartilage} <<<<")

    cross_csv = os.path.join(csv_input_dir, f"{knee_cartilage}_cross_filtered_features.csv")
    normal_csv = os.path.join(csv_input_dir, f"{knee_cartilage}_filtered_features.csv")
    
    if os.path.exists(cross_csv):
        csv_path = cross_csv
        print(f"  Using cross-region enhanced features: {cross_csv}")
    elif os.path.exists(normal_csv):
        csv_path = normal_csv
        print(f"  Cross features not found, falling back to normal features: {normal_csv}")
    else:
        print(f"Skipping {knee_cartilage}: No feature CSV found.")
        continue
        
    df_raw = pd.read_csv(csv_path)

    df = df_raw.groupby(['case_id', 'region', 'grade']).mean().reset_index()
    print(f"Original rows: {len(df_raw)} -> Patient-level rows: {len(df)}")

    drop_cols = [c for c in ["case_id", "region", "grade", "cartilage_missing"] if c in df.columns]
    X = df.drop(columns=drop_cols)
    y = (df["grade"] > 0).astype(int).values
    groups = df["case_id"].values

    print(f"Data shape: {df.shape}")

    if X.isnull().values.any():
        X = X.fillna(0)

    cross_cols = [c for c in X.columns if c.startswith('cross_')]
    onehot_cols = [c for c in X.columns if c.startswith('region_')]
    ratio_cols = [c for c in X.columns if c.startswith('ratio_')]
    other_cols = [c for c in X.columns if not c.startswith('cross_') and not c.startswith('region_') and not c.startswith('ratio_')]
    print(f"  Feature breakdown: {len(other_cols)} LASSO + {len(cross_cols)} cross + {len(ratio_cols)} ratio + {len(onehot_cols)} onehot = {len(X.columns)}")

    cw = None

    # ---- 参数搜索 ----
    print(f"\n  Selecting SVM parameters (81 combos) with overfitting penalty...")
    best_C, best_gamma, cv_auc, train_auc, gap, prob_range = _select_svm_params_with_penalty(
        X, y, groups, class_weight=cw
    )

    # ---- 交叉验证评估 ----
    cv_outer = GroupKFold(n_splits=5)
    
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

        model = SVC(kernel='rbf', C=best_C, gamma=best_gamma, probability=True,
                   class_weight=cw)
        model.fit(X_train_scaled, y_train)
        
        y_prob = model.predict_proba(X_test_scaled)[:, 1]

        current_auc = roc_auc_score(y_test, y_prob)
        auc_list.append(current_auc)
        
        fpr, tpr, thresholds = roc_curve(y_test, y_prob)
        interp_tpr = np.interp(mean_fpr, fpr, tpr)
        interp_tpr[0] = 0.0
        tprs.append(interp_tpr)

        youden_j = tpr - fpr
        best_idx = np.argmax(youden_j)
        best_threshold = thresholds[best_idx]
        threshold_list.append(best_threshold)

        print(f"  Fold {fold} AUC: {current_auc:.3f}, Optimal threshold: {best_threshold:.3f}")
        fold += 1

    # 保存 ROC 曲线
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
    plt.title(f'ROC Curve - {knee_cartilage} (v2)')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    
    plt.savefig(os.path.join(fig_dir, f"{knee_cartilage}_ROC.png"), dpi=300)
    plt.close()

    raw_threshold = float(np.median(threshold_list))
    # v2: 阈值 cap 放宽到 [0.30, 0.70]
    THRESH_MIN, THRESH_MAX = 0.30, 0.70
    optimal_threshold = max(THRESH_MIN, min(THRESH_MAX, raw_threshold))
    print(f"Completed {knee_cartilage}. Mean AUC: {mean_auc:.3f}")
    print(f"  CV Optimal Threshold (Youden's J, median): {raw_threshold:.4f}")
    print(f"  Per-fold thresholds: {[f'{t:.3f}' for t in threshold_list]}")
    if optimal_threshold != raw_threshold:
        print(f"  [v2 Threshold Cap] {raw_threshold:.4f} → {optimal_threshold:.4f} (range [{THRESH_MIN}, {THRESH_MAX}])")

    # 保存最终模型
    print(f"Saving final model for {knee_cartilage}...")
    
    final_scaler = StandardScaler()
    X_scaled = final_scaler.fit_transform(X)

    final_model = SVC(kernel='rbf', C=best_C, gamma=best_gamma, probability=True,
                     class_weight=cw)
    final_model.fit(X_scaled, y)

    # v2.2: Platt Scaling 概率校准
    # SVM 的 predict_proba 内部已使用 Platt Scaling，但仅用训练集拟合
    # 使用 CalibratedClassifierCV 做交叉验证校准，得到更可靠的概率
    from sklearn.calibration import CalibratedClassifierCV
    calibrated_model = CalibratedClassifierCV(
        SVC(kernel='rbf', C=best_C, gamma=best_gamma, class_weight=cw),
        method='sigmoid',   # Platt Scaling
        cv=min(5, len(np.unique(groups)))  # GroupKFold 模拟
    )
    calibrated_model.fit(X_scaled, y)

    # 验证校准效果
    y_prob_raw = final_model.predict_proba(X_scaled)[:, 1]
    y_prob_calib = calibrated_model.predict_proba(X_scaled)[:, 1]
    print(f"  [Calibration] Raw prob range: [{y_prob_raw.min():.3f}, {y_prob_raw.max():.3f}]")
    print(f"  [Calibration] Calib prob range: [{y_prob_calib.min():.3f}, {y_prob_calib.max():.3f}]")

    model_dir = os.path.join(results_output_dir, knee_cartilage, "models")
    os.makedirs(model_dir, exist_ok=True)

    # 保存校准模型 (推理时优先使用)
    joblib.dump(calibrated_model, os.path.join(model_dir, "svm_model_calibrated.pkl"))
    joblib.dump(final_model, os.path.join(model_dir, "svm_model.pkl"))
    joblib.dump(final_scaler, os.path.join(model_dir, "scaler.pkl"))
    joblib.dump(X.columns.tolist(), os.path.join(model_dir, "feature_list.pkl"))
    joblib.dump(optimal_threshold, os.path.join(model_dir, "threshold.pkl"))
    
    model_params = {
        "C": best_C,
        "gamma": best_gamma,
        "cv_auc": cv_auc,
        "train_auc": train_auc,
        "overfit_gap": gap,
        "prob_range": prob_range,
        "class_weight": cw,
    }
    joblib.dump(model_params, os.path.join(model_dir, "model_params.pkl"))

    print(f"[Stage 1] Saved successfully: {model_dir}")
    print(f"   Threshold: {optimal_threshold:.4f}")
    print(f"   Params: C={best_C}, gamma={best_gamma}, class_weight={cw}\n")


# ===============================================================
#  STAGE 2: 分级 SVM (Grade 1 vs Grade 2) — v2 分池化策略
# ===============================================================
print("\n" + "#" * 60)
print("  STAGE 2: Training Grade SVM (Grade 1 vs Grade 2) [v2]")
print("  Strategy: FM/TM independent + FL/TL pooled")
print("#" * 60)


def train_independent_stage2(region_name, csv_input_dir, results_output_dir):
    """v2: 为单个区域独立训练 Stage2 模型"""
    csv_path = os.path.join(csv_input_dir, f"{region_name}_stage2_filtered_features.csv")
    if not os.path.exists(csv_path):
        print(f"  Skipping {region_name}: Stage 2 CSV not found")
        return False

    print(f"\n>>>> [Stage 2 Independent] Training for {region_name} <<<<")

    df_raw = pd.read_csv(csv_path)
    df = df_raw.groupby(['case_id', 'region', 'grade']).mean().reset_index()

    drop_cols = [c for c in ["case_id", "region", "grade", "cartilage_missing"] if c in df.columns]
    X = df.drop(columns=drop_cols)
    y = (df["grade"] == 2).astype(int).values

    # 添加 region one-hot
    for r in REGIONS:
        X[f"region_{r}"] = 1.0 if r == region_name else 0.0

    if X.isnull().values.any():
        X = X.fillna(0)

    n_g1 = int((y == 0).sum())
    n_g2 = int((y == 1).sum())
    print(f"  Data: {len(df)} rows, G1={n_g1}, G2={n_g2}, Features: {X.shape[1]}")

    if n_g1 < 2 or n_g2 < 2:
        print(f"  Too few samples for independent training (G1={n_g1}, G2={n_g2})")
        return False

    # 训练
    param_grid_s2 = {
        'C': [0.01, 0.1, 1, 10, 100],
        'gamma': ['scale', 'auto', 0.001, 0.01, 0.1, 1]
    }

    final_scaler = StandardScaler()
    X_scaled = final_scaler.fit_transform(X)

    n_minority = min(n_g1, n_g2)
    if n_minority >= 2:
        inner_cv = min(5, n_minority)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            from sklearn.model_selection import GridSearchCV
            grid = GridSearchCV(
                SVC(kernel='rbf', probability=True, class_weight='balanced'),
                param_grid_s2, cv=inner_cv, scoring='accuracy'
            )
            grid.fit(X_scaled, y)
        final_model = grid.best_estimator_
        print(f"  Best params: {grid.best_params_}")
    else:
        final_model = SVC(kernel='rbf', C=1, gamma='scale',
                        probability=True, class_weight='balanced')
        final_model.fit(X_scaled, y)
        print(f"  Using default params")

    # 保存
    model_dir_s2 = os.path.join(results_output_dir, region_name, "models")
    os.makedirs(model_dir_s2, exist_ok=True)

    joblib.dump(final_model, os.path.join(model_dir_s2, "svm_model_stage2.pkl"))
    joblib.dump(final_scaler, os.path.join(model_dir_s2, "scaler_stage2.pkl"))
    joblib.dump(X.columns.tolist(), os.path.join(model_dir_s2, "feature_list_stage2.pkl"))
    joblib.dump("independent", os.path.join(model_dir_s2, "stage2_source.pkl"))

    print(f"  [Independent] Saved to: {model_dir_s2}")
    print(f"    Features ({len(X.columns)}): {X.columns.tolist()}")
    return True


# v2: FM 和 TM 独立训练
for region_name in STAGE2_INDEPENDENT_REGIONS:
    success = train_independent_stage2(region_name, csv_input_dir, results_output_dir)
    if not success:
        print(f"  {region_name} independent training failed, will try pooled fallback")

# v2: FL+TL 池化训练
pooled_all_csv = os.path.join(csv_input_dir, "pooled_stage2_filtered_features.csv")

if os.path.exists(pooled_all_csv):
    print(f"\n>>>> [Stage 2 Pooled] FL+TL Shared Model <<<<")

    df_pooled = pd.read_csv(pooled_all_csv)
    df_pooled = df_pooled.groupby(['case_id', 'region', 'grade']).mean().reset_index()

    # v2: 只保留 FL+TL 区域的样本做池化
    df_pooled_fl_tl = df_pooled[df_pooled["region"].isin(STAGE2_POOLED_REGIONS)].copy()

    if len(df_pooled_fl_tl) == 0:
        # 如果 pooled_stage2 中没有 FL/TL 数据，回退到全量池化
        print(f"  No FL/TL data in pooled CSV, using all regions for pooled model")
        df_pooled_fl_tl = df_pooled

    drop_cols = [c for c in ["case_id", "region", "grade", "cartilage_missing"] if c in df_pooled_fl_tl.columns]
    X_pool = df_pooled_fl_tl.drop(columns=drop_cols)
    y_pool = (df_pooled_fl_tl["grade"] == 2).astype(int).values

    n_g1 = int((y_pool == 0).sum())
    n_g2 = int((y_pool == 1).sum())
    print(f"Pooled FL+TL data: {len(df_pooled_fl_tl)} rows, G1={n_g1}, G2={n_g2}")
    print(f"Features: {X_pool.shape[1]}")

    # 打印每个区域的样本数
    for r in STAGE2_POOLED_REGIONS:
        sub = df_pooled_fl_tl[df_pooled_fl_tl["region"] == r]
        if len(sub) > 0:
            g1_r = (sub["grade"] == 1).sum()
            g2_r = (sub["grade"] == 2).sum()
            print(f"  {r}: {len(sub)} samples (G1={g1_r}, G2={g2_r})")

    if X_pool.isnull().values.any():
        X_pool = X_pool.fillna(0)

    # 训练池化模型
    param_grid_s2 = {
        'C': [0.01, 0.1, 1, 10, 100],
        'gamma': ['scale', 'auto', 0.001, 0.01, 0.1, 1]
    }

    final_scaler_pool = StandardScaler()
    X_pool_scaled = final_scaler_pool.fit_transform(X_pool)

    n_minority = min(n_g1, n_g2)
    if n_minority >= 2:
        inner_cv = min(5, n_minority)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            from sklearn.model_selection import GridSearchCV
            pool_grid = GridSearchCV(
                SVC(kernel='rbf', probability=True, class_weight='balanced'),
                param_grid_s2, cv=inner_cv, scoring='accuracy'
            )
            pool_grid.fit(X_pool_scaled, y_pool)
        pool_model = pool_grid.best_estimator_
        print(f"  Best params: {pool_grid.best_params_}")
    else:
        pool_model = SVC(kernel='rbf', C=1, gamma='scale',
                        probability=True, class_weight='balanced')
        pool_model.fit(X_pool_scaled, y_pool)
        print(f"  Using default params")

    # 保存共享模型到 FL/TL 区域目录
    for region_name in STAGE2_POOLED_REGIONS:
        model_dir_s2 = os.path.join(results_output_dir, region_name, "models")
        os.makedirs(model_dir_s2, exist_ok=True)

        joblib.dump(pool_model, os.path.join(model_dir_s2, "svm_model_stage2.pkl"))
        joblib.dump(final_scaler_pool, os.path.join(model_dir_s2, "scaler_stage2.pkl"))
        joblib.dump(X_pool.columns.tolist(), os.path.join(model_dir_s2, "feature_list_stage2.pkl"))
        joblib.dump("pooled_fl_tl", os.path.join(model_dir_s2, "stage2_source.pkl"))

        print(f"  [Pooled FL+TL] Saved to: {model_dir_s2}")
        print(f"    Features ({len(X_pool.columns)}): {X_pool.columns.tolist()}")

else:
    print(f"\n>>>> [Stage 2] Pooled CSV not found <<<<")
    print(f"  Falling back to independent training for all regions")

    for region_name in REGIONS:
        csv_path = os.path.join(csv_input_dir, f"{region_name}_stage2_filtered_features.csv")
        if not os.path.exists(csv_path):
            print(f"  Skipping {region_name}: Stage 2 CSV not found")
            continue

        df_raw = pd.read_csv(csv_path)
        df = df_raw.groupby(['case_id', 'region', 'grade']).mean().reset_index()

        drop_cols = [c for c in ["case_id", "region", "grade", "cartilage_missing"] if c in df.columns]
        X = df.drop(columns=drop_cols)
        y = (df["grade"] == 2).astype(int).values

        for r in REGIONS:
            X[f"region_{r}"] = 1.0 if r == region_name else 0.0

        if X.isnull().values.any():
            X = X.fillna(0)

        final_scaler_s2 = StandardScaler()
        X_scaled_s2 = final_scaler_s2.fit_transform(X)

        final_model_s2 = SVC(kernel='rbf', C=1, gamma='scale',
                             probability=True, class_weight='balanced')
        final_model_s2.fit(X_scaled_s2, y)

        model_dir_s2 = os.path.join(results_output_dir, region_name, "models")
        os.makedirs(model_dir_s2, exist_ok=True)

        joblib.dump(final_model_s2, os.path.join(model_dir_s2, "svm_model_stage2.pkl"))
        joblib.dump(final_scaler_s2, os.path.join(model_dir_s2, "scaler_stage2.pkl"))
        joblib.dump(X.columns.tolist(), os.path.join(model_dir_s2, "feature_list_stage2.pkl"))
        joblib.dump("independent_fallback", os.path.join(model_dir_s2, "stage2_source.pkl"))

        print(f"  [Independent Fallback] Saved to: {model_dir_s2}")

# 对于没有独立训练成功的 FM/TM，尝试回退到全量池化
for region_name in STAGE2_INDEPENDENT_REGIONS:
    model_dir_s2 = os.path.join(results_output_dir, region_name, "models")
    s2_model_path = os.path.join(model_dir_s2, "svm_model_stage2.pkl")
    if not os.path.exists(s2_model_path):
        print(f"\n  [Fallback] {region_name} independent Stage2 not found, checking pooled model...")
        # 如果独立训练失败，尝试用全量池化
        if os.path.exists(pooled_all_csv):
            print(f"  Using full pooled model as fallback for {region_name}")
            # 这里需要用全量 pooled 数据训练一个模型
            # 已经在上面的 pooled 部分处理过，只是保存路径不同
            # 暂时跳过，让推理脚本自行处理
            print(f"  WARNING: {region_name} has no Stage2 model! Inference will use Grade 1 default.")


print("\n" + "=" * 60)
print("All regions processed. Stage 1 + Stage 2 models saved. [v2]")
print(f"Results directory: {results_output_dir}")
print("=" * 60)
