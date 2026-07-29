#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
1a_lasso_v3.py (dev_0702_v2)
LASSO 特征筛选 - v2 优化版本

v2 改进:
  1. stability_threshold: 0.4 → 0.3 (小样本下保留更多候选特征)
  2. n_bootstrap: 50 → 100 (更充分的稳定性估计)
  3. max_features: 80 → 100 (给后续 SVM 更多候选)
  4. 对少数类极少的区域(FL)，自动降低 threshold 到 0.25
  5. 增加 gamma=0.001 的细粒度 C 搜索

Stage 1: 二分类 (Normal vs Damaged) 特征选择
Stage 2: 分级 (Grade 1 vs Grade 2) 特征选择
"""

import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegressionCV
from sklearn.pipeline import Pipeline
import argparse

# -------------------------------
# 配置路径
# -------------------------------
INPUT_CSV = "./train/classify/dev_0702_v2/data_train/knee_radiomics_features_3d_integrated.csv"
OUTPUT_DIR = "./train/classify/dev_0702_v2/data_train/feature"

# -------------------------------
# 区域定义
# -------------------------------
REGION_NAMES = {
    1: "Femur_Medial",
    2: "Femur_Lateral",
    3: "Tibia_Medial",
    4: "Tibia_Lateral"
}

# Stage 2 强制保留的 shape 特征
FORCED_SHAPE_FEATURES = [
    "original_shape_VoxelVolume_mean",
    "original_shape_SurfaceArea_mean",
    "original_shape_MeshVolume_mean",
]

# v2: 每区域的稳定性选择参数（根据样本量和类别不平衡自适应）
def get_stability_params(region_name, y):
    """
    根据区域样本量和类别不平衡程度，返回稳定性选择参数。

    - 样本少或极不平衡时：降低 threshold，增加 bootstrap
    - 样本充足时：标准参数
    """
    n_samples = len(y)
    class_counts = pd.Series(y).value_counts()
    minority_ratio = class_counts.min() / n_samples if n_samples > 0 else 1.0

    if minority_ratio < 0.15 or n_samples < 30:
        # 极不平衡区域（如 FL: 3/27）
        return {
            'stability_threshold': 0.25,
            'n_bootstrap': 150,
            'max_features': 100,
        }
    elif minority_ratio < 0.30:
        # 中等不平衡
        return {
            'stability_threshold': 0.30,
            'n_bootstrap': 100,
            'max_features': 100,
        }
    else:
        # 相对均衡
        return {
            'stability_threshold': 0.30,
            'n_bootstrap': 100,
            'max_features': 80,
        }


# -------------------------------
# LASSO 特征选择函数
# -------------------------------
def lasso_feature_selection(X, y, feature_names=None, cv=5, random_state=42,
                              stability=True, n_bootstrap=100, stability_threshold=0.3,
                              max_features=100, region_name=None):
    """
    LASSO feature selection with adaptive stability selection.

    v2 改进:
      - 自适应参数 (threshold/bootstrap/max_features 根据区域调整)
      - 更细粒度的 C 搜索 (logspace(-3, 2, 30))
      - bootstrap 子采样比例从 0.8 提到 0.85 (保留更多样本)
    """
    X = np.array(X)
    y = np.array(y)

    if feature_names is None:
        feature_names = [f"feature_{i}" for i in range(X.shape[1])]

    if stability:
        from sklearn.linear_model import LogisticRegression
        from sklearn.utils import resample

        n_samples, n_features = X.shape
        selection_count = np.zeros(n_features)
        coef_sum = np.zeros(n_features)

        # 标准化全量数据
        scaler_full = StandardScaler()
        X_scaled = scaler_full.fit_transform(X)

        # v2: 更细粒度的 C 搜索 (30 个点, 从 1e-3 到 1e2)
        cv_lasso = LogisticRegressionCV(
            Cs=np.logspace(-3, 2, 30), cv=cv, penalty="l1",
            solver="liblinear", scoring="roc_auc", max_iter=5000,
            random_state=random_state, n_jobs=-1,
        )
        cv_lasso.fit(X_scaled, y)
        best_C = cv_lasso.C_[0]

        # Bootstrap LASSO
        rng = np.random.RandomState(random_state)
        for b in range(n_bootstrap):
            # v2: 子采样 85% 样本 (原 80%)
            n_sub = int(n_samples * 0.85)
            idx = rng.choice(n_samples, size=n_sub, replace=False)
            X_b, y_b = X_scaled[idx], y[idx]

            if len(np.unique(y_b)) < 2:
                continue

            try:
                lasso_b = LogisticRegression(
                    C=best_C, penalty="l1", solver="liblinear",
                    max_iter=5000, random_state=b
                )
                lasso_b.fit(X_b, y_b)
                coef_b = lasso_b.coef_.ravel()
                selected_b = np.where(coef_b != 0)[0]
                selection_count[selected_b] += 1
                coef_sum[selected_b] += np.abs(coef_b[selected_b])
            except Exception:
                continue

        # 按 selection frequency 选特征
        selection_freq = selection_count / n_bootstrap
        stable_idx = np.where(selection_freq >= stability_threshold)[0]

        # 第二轮限制: 最多 max_features 个
        if len(stable_idx) > max_features:
            top_by_freq = np.argsort(-selection_freq[stable_idx])[:max_features]
            stable_idx = stable_idx[top_by_freq]

        # 兜底: 如果 stability selection 选不到任何特征，回退到普通 LASSO
        if len(stable_idx) == 0:
            print(f"    [Stability] No features above {stability_threshold}, falling back to CV LASSO")
            coef = cv_lasso.coef_.ravel()
            stable_idx = np.where(coef != 0)[0]
            # 如果还是 0，用 F-test top-20
            if len(stable_idx) == 0:
                from sklearn.feature_selection import f_classif
                f_scores, _ = f_classif(X_scaled, y)
                f_scores = np.nan_to_num(f_scores, nan=0.0)
                stable_idx = np.argsort(-f_scores)[:20]

        # 输出
        selected_idx = stable_idx
        with np.errstate(invalid='ignore'):
            avg_coef = np.where(selection_count > 0, coef_sum / np.maximum(selection_count, 1), 0)
        selected_coef = avg_coef[selected_idx]
        order = np.argsort(-selection_freq[selected_idx])
        selected_idx = selected_idx[order]
        selected_coef = selected_coef[order]

        X_selected = X[:, selected_idx]
        selected_features = [feature_names[i] for i in selected_idx]

        print(f"    [Stability Selection] {n_bootstrap} bootstrap LASSO, threshold={stability_threshold}, C={best_C:.4f}")
        print(f"    Original features: {X.shape[1]}")
        print(f"    Stable features (freq >= {stability_threshold}): {len(selected_idx)}")
        print(f"    Selection ratio: {len(selected_idx)/X.shape[1]*100:.1f}%")

        class _StableClf:
            pass
        clf = _StableClf()
        clf.best_C = best_C
        clf.selection_freq = selection_freq
        return X_selected, selected_features, selected_coef, clf

    # ========== 原始单次 LASSO (兜底) ==========
    clf = Pipeline([
        ("scaler", StandardScaler()),
        ("lasso", LogisticRegressionCV(
            Cs=np.logspace(-3, 2, 30),
            cv=cv,
            penalty="l1",
            solver="liblinear",
            scoring="roc_auc",
            max_iter=5000,
            random_state=random_state,
            n_jobs=-1,
        ))
    ])
    clf.fit(X, y)
    lasso_model = clf.named_steps["lasso"]
    coef = lasso_model.coef_.ravel()
    selected_idx = np.where(coef != 0)[0]
    X_selected = X[:, selected_idx]
    selected_features = [feature_names[i] for i in selected_idx]
    selected_coef = coef[selected_idx]

    print(f"    Original features: {X.shape[1]}")
    print(f"    Selected features: {len(selected_idx)}")
    print(f"    Selection ratio: {len(selected_idx)/X.shape[1]*100:.1f}%")

    return X_selected, selected_features, selected_coef, clf


def run_stage1(df, output_dir):
    """
    Stage 1: 二分类 LASSO 特征选择 (Normal=0 vs Damaged=1+2)
    """
    print("\n" + "#" * 60)
    print("  STAGE 1: Binary Classification (Normal vs Damaged) [v2]")
    print("#" * 60)

    for key, region_name in REGION_NAMES.items():
        print(f"\n{'='*50}")
        print(f"[Stage 1] Region: {region_name}")
        print(f"{'='*50}")

        df_region = df[df["region"] == region_name].copy().reset_index(drop=True)

        if len(df_region) == 0:
            print(f"  Warning: No samples found for {region_name}")
            continue

        print(f"  Samples in this region: {len(df_region)}")

        # 二分类标签：0 vs (1 + 2)
        df_region["grade"] = (df_region["grade"] > 0).astype(int)

        df_region = df_region.dropna(how='any')
        print(f"  Samples after dropping NaN: {len(df_region)}")

        if len(df_region) < 10:
            print(f"  Warning: Too few samples for {region_name}, skipping...")
            continue

        meta_cols = ["case_id", "region", "grade", "cartilage_missing"]
        feature_names = [c for c in df_region.columns if c not in meta_cols]

        print(f"  Features before selection: {len(feature_names)}")

        X = df_region[feature_names].values
        y = df_region["grade"].values

        class_dist = pd.Series(y).value_counts().sort_index()
        print(f"  Class distribution: {dict(class_dist)}")

        # v2: 自适应稳定性参数
        params = get_stability_params(region_name, y)
        print(f"  [v2] Stability params: threshold={params['stability_threshold']}, "
              f"bootstrap={params['n_bootstrap']}, max_features={params['max_features']}")

        try:
            X_sel, features_sel, coef_sel, model = lasso_feature_selection(
                X, y, feature_names, cv=5,
                stability_threshold=params['stability_threshold'],
                n_bootstrap=params['n_bootstrap'],
                max_features=params['max_features'],
                region_name=region_name,
            )

            result_df = pd.DataFrame({
                "feature": features_sel,
                "coef": coef_sel
            })
            result_df["abs_coef"] = result_df["coef"].abs()
            result_df = result_df.sort_values("abs_coef", ascending=False)

            print(f"\n  Top 10 selected features:")
            for i, row in result_df.head(10).iterrows():
                print(f"    {row['feature']}: {row['coef']:.4f}")

            stable_features = result_df["feature"].tolist()
            cols_to_save = ["case_id", "region", "grade"] + stable_features
            if "cartilage_missing" in df_region.columns:
                cols_to_save.append("cartilage_missing")

            save_path = os.path.join(output_dir, f"{region_name}_filtered_features.csv")
            df_region[cols_to_save].to_csv(save_path, index=False)

            print(f"\n  Saved to: {save_path}")
            print(f"  Shape: {df_region[cols_to_save].shape}")

        except Exception as e:
            print(f"  Error processing {region_name}: {e}")
            import traceback
            traceback.print_exc()
            continue


def run_stage2(df, output_dir):
    """
    Stage 2: 分级 LASSO 特征选择 (Grade 1 vs Grade 2)
    """
    print("\n" + "#" * 60)
    print("  STAGE 2: Grade Classification (Grade 1 vs Grade 2) [v2]")
    print("#" * 60)

    for key, region_name in REGION_NAMES.items():
        print(f"\n{'='*50}")
        print(f"[Stage 2] Region: {region_name}")
        print(f"{'='*50}")

        df_region = df[(df["region"] == region_name) & (df["grade"] > 0)].copy().reset_index(drop=True)

        if len(df_region) == 0:
            print(f"  Warning: No damaged samples found for {region_name}")
            continue

        print(f"  Damaged samples in this region: {len(df_region)}")

        # 标签: Grade 1 → 0, Grade 2 → 1
        df_region["grade_binary"] = (df_region["grade"] == 2).astype(int)

        df_region = df_region.dropna(how='any')
        print(f"  Samples after dropping NaN: {len(df_region)}")

        class_dist = pd.Series(df_region["grade_binary"].values).value_counts().sort_index()
        print(f"  Class distribution (0=G1, 1=G2): {dict(class_dist)}")

        if len(class_dist) < 2:
            print(f"  Warning: Only one class present for {region_name}.")
            print(f"  Skipping LASSO, will use forced shape features only.")
            available_forced = [f for f in FORCED_SHAPE_FEATURES if f in df_region.columns]
            if not available_forced:
                print(f"  Error: No forced shape features available. Skipping region.")
                continue
            cols_to_save = ["case_id", "region", "grade"] + available_forced
            save_path = os.path.join(output_dir, f"{region_name}_stage2_filtered_features.csv")
            df_region[cols_to_save].to_csv(save_path, index=False)
            print(f"  Saved (forced features only): {save_path}")
            continue

        if class_dist.min() < 2:
            print(f"  Warning: Very few samples in one class (min={class_dist.min()}).")
            print(f"  Skipping LASSO, will use forced shape features only.")
            available_forced = [f for f in FORCED_SHAPE_FEATURES if f in df_region.columns]
            if not available_forced:
                continue
            cols_to_save = ["case_id", "region", "grade"] + available_forced
            save_path = os.path.join(output_dir, f"{region_name}_stage2_filtered_features.csv")
            df_region[cols_to_save].to_csv(save_path, index=False)
            continue

        meta_cols = ["case_id", "region", "grade", "grade_binary", "cartilage_missing"]
        feature_names = [c for c in df_region.columns if c not in meta_cols]

        print(f"  Features before selection: {len(feature_names)}")

        X = df_region[feature_names].values
        y = df_region["grade_binary"].values

        # v2: 自适应参数
        params = get_stability_params(region_name, y)

        cv_folds = min(5, class_dist.min())
        if cv_folds < 2:
            cv_folds = 2
        print(f"  Using cv={cv_folds} folds for LASSO")

        lasso_selected = []
        try:
            X_sel, features_sel, coef_sel, model = lasso_feature_selection(
                X, y, feature_names, cv=cv_folds,
                stability_threshold=params['stability_threshold'],
                n_bootstrap=params['n_bootstrap'],
                max_features=params['max_features'],
                region_name=region_name,
            )

            result_df = pd.DataFrame({
                "feature": features_sel,
                "coef": coef_sel
            })
            result_df["abs_coef"] = result_df["coef"].abs()
            result_df = result_df.sort_values("abs_coef", ascending=False)

            print(f"\n  LASSO selected features:")
            for i, row in result_df.iterrows():
                print(f"    {row['feature']}: {row['coef']:.4f}")

            lasso_selected = result_df["feature"].tolist()

        except Exception as e:
            print(f"  LASSO failed: {e}")
            print(f"  Will use forced shape features only.")

        available_forced = [f for f in FORCED_SHAPE_FEATURES if f in df_region.columns]
        final_features = list(dict.fromkeys(lasso_selected + available_forced))

        if not final_features:
            print(f"  Error: No features selected at all. Skipping region.")
            continue

        print(f"\n  Final Stage 2 features ({len(final_features)}):")
        for f in final_features:
            source = "LASSO+forced" if f in lasso_selected and f in available_forced \
                     else "LASSO" if f in lasso_selected \
                     else "forced"
            print(f"    {f}  [{source}]")

        cols_to_save = ["case_id", "region", "grade"] + final_features
        save_path = os.path.join(output_dir, f"{region_name}_stage2_filtered_features.csv")
        df_region[cols_to_save].to_csv(save_path, index=False)

        print(f"\n  Saved to: {save_path}")
        print(f"  Shape: {df_region[cols_to_save].shape}")


if __name__ == "__main__":
    df = pd.read_csv(INPUT_CSV)
    print(f"Loaded {len(df)} rows, {df.shape[1]} columns from {INPUT_CSV}")
    print(f"Regions: {df['region'].unique().tolist()}")
    print(f"Grade distribution: {df['grade'].value_counts().sort_index().to_dict()}")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    run_stage1(df, OUTPUT_DIR)
    run_stage2(df, OUTPUT_DIR)

    print("\n" + "=" * 60)
    print("LASSO feature selection complete! [v2]")
    print(f"Output directory: {OUTPUT_DIR}")
    print("=" * 60)
