#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
lasso_feature_selection.py
LASSO 特征筛选 - 适配 3D 整合特征提取的输出格式

Stage 1: 二分类 (Normal vs Damaged) 特征选择 — 输出 {Region}_filtered_features.csv
Stage 2: 分级 (Grade 1 vs Grade 2) 特征选择 — 输出 {Region}_stage2_filtered_features.csv
         强制保留 VoxelVolume / SurfaceArea / MeshVolume 三个 shape 特征
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
INPUT_CSV = "./train/classify/dev_v3/data_train/knee_radiomics_features_3d_integrated.csv"
OUTPUT_DIR = "./train/classify/dev_v3/data_train/feature"

# -------------------------------
# 区域定义
# -------------------------------
REGION_NAMES = {
    1: "Femur_Medial",
    2: "Femur_Lateral",
    3: "Tibia_Medial",
    4: "Tibia_Lateral"
}

# Stage 2 强制保留的 shape 特征（不会被 LASSO 淘汰）
FORCED_SHAPE_FEATURES = [
    "original_shape_VoxelVolume_mean",
    "original_shape_SurfaceArea_mean",
    "original_shape_MeshVolume_mean",
]

# -------------------------------
# LASSO 特征选择函数
# -------------------------------
def lasso_feature_selection(X, y, feature_names=None, cv=5, random_state=42):
    """
    LASSO feature selection for binary classification
    """
    X = np.array(X)
    y = np.array(y)

    if feature_names is None:
        feature_names = [f"feature_{i}" for i in range(X.shape[1])]

    # Pipeline: 标准化 + LASSO
    clf = Pipeline([
        ("scaler", StandardScaler()),
        ("lasso", LogisticRegressionCV(
            Cs=np.logspace(-3, 1, 20),
            cv=cv,
            penalty="l1",
            solver="liblinear",
            scoring="roc_auc",
            max_iter=5000,
            random_state=random_state,
            n_jobs=-1,
        ))
    ])

    # 训练
    clf.fit(X, y)

    # 取出 LASSO 模型
    lasso_model = clf.named_steps["lasso"]

    # 系数
    coef = lasso_model.coef_.ravel()

    # 非零特征索引
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
    输出: {Region}_filtered_features.csv
    """
    print("\n" + "#" * 60)
    print("  STAGE 1: Binary Classification (Normal vs Damaged)")
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

        try:
            X_sel, features_sel, coef_sel, model = lasso_feature_selection(
                X, y, feature_names, cv=5
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
    仅使用损伤样本 (grade > 0)，强制保留 VoxelVolume / SurfaceArea / MeshVolume。
    输出: {Region}_stage2_filtered_features.csv
    """
    print("\n" + "#" * 60)
    print("  STAGE 2: Grade Classification (Grade 1 vs Grade 2)")
    print("#" * 60)

    for key, region_name in REGION_NAMES.items():
        print(f"\n{'='*50}")
        print(f"[Stage 2] Region: {region_name}")
        print(f"{'='*50}")

        # 只保留损伤样本 (grade = 1 或 2)
        df_region = df[(df["region"] == region_name) & (df["grade"] > 0)].copy().reset_index(drop=True)

        if len(df_region) == 0:
            print(f"  Warning: No damaged samples found for {region_name}")
            continue

        print(f"  Damaged samples in this region: {len(df_region)}")

        # 标签: Grade 1 → 0, Grade 2 → 1 (二分类)
        df_region["grade_binary"] = (df_region["grade"] == 2).astype(int)

        df_region = df_region.dropna(how='any')
        print(f"  Samples after dropping NaN: {len(df_region)}")

        class_dist = pd.Series(df_region["grade_binary"].values).value_counts().sort_index()
        print(f"  Class distribution (0=G1, 1=G2): {dict(class_dist)}")

        # 检查是否有足够样本（两类都至少要有2个）
        if len(class_dist) < 2:
            print(f"  Warning: Only one class present for {region_name}.")
            print(f"  Skipping LASSO, will use forced shape features only.")
            # 仅保留强制特征
            available_forced = [f for f in FORCED_SHAPE_FEATURES if f in df_region.columns]
            if not available_forced:
                print(f"  Error: No forced shape features available. Skipping region.")
                continue
            cols_to_save = ["case_id", "region", "grade"] + available_forced
            save_path = os.path.join(output_dir, f"{region_name}_stage2_filtered_features.csv")
            df_region[cols_to_save].to_csv(save_path, index=False)
            print(f"  Saved (forced features only): {save_path}")
            print(f"  Features: {available_forced}")
            continue

        if class_dist.min() < 2:
            print(f"  Warning: Very few samples in one class (min={class_dist.min()}).")
            print(f"  Skipping LASSO, will use forced shape features only.")
            available_forced = [f for f in FORCED_SHAPE_FEATURES if f in df_region.columns]
            if not available_forced:
                print(f"  Error: No forced shape features available. Skipping region.")
                continue
            cols_to_save = ["case_id", "region", "grade"] + available_forced
            save_path = os.path.join(output_dir, f"{region_name}_stage2_filtered_features.csv")
            df_region[cols_to_save].to_csv(save_path, index=False)
            print(f"  Saved (forced features only): {save_path}")
            print(f"  Features: {available_forced}")
            continue

        meta_cols = ["case_id", "region", "grade", "grade_binary", "cartilage_missing"]
        feature_names = [c for c in df_region.columns if c not in meta_cols]

        print(f"  Features before selection: {len(feature_names)}")

        X = df_region[feature_names].values
        y = df_region["grade_binary"].values

        # LASSO 特征选择
        lasso_selected = []
        try:
            cv_folds = min(5, class_dist.min())
            if cv_folds < 2:
                cv_folds = 2
            print(f"  Using cv={cv_folds} folds for LASSO")

            X_sel, features_sel, coef_sel, model = lasso_feature_selection(
                X, y, feature_names, cv=cv_folds
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

        # 合并: LASSO 选出的特征 + 强制保留的 shape 特征（去重）
        available_forced = [f for f in FORCED_SHAPE_FEATURES if f in df_region.columns]
        final_features = list(dict.fromkeys(lasso_selected + available_forced))  # 去重保序

        if not final_features:
            print(f"  Error: No features selected at all. Skipping region.")
            continue

        print(f"\n  Final Stage 2 features ({len(final_features)}):")
        for f in final_features:
            source = "LASSO+forced" if f in lasso_selected and f in available_forced \
                     else "LASSO" if f in lasso_selected \
                     else "forced"
            print(f"    {f}  [{source}]")

        # 保存
        cols_to_save = ["case_id", "region", "grade"] + final_features
        save_path = os.path.join(output_dir, f"{region_name}_stage2_filtered_features.csv")
        df_region[cols_to_save].to_csv(save_path, index=False)

        print(f"\n  Saved to: {save_path}")
        print(f"  Shape: {df_region[cols_to_save].shape}")


# -------------------------------
# 主流程
# -------------------------------
def main():
    parser = argparse.ArgumentParser(description='LASSO Feature Selection (Stage 1 + Stage 2)')
    parser.add_argument('--input', default=INPUT_CSV, help='Input CSV path')
    parser.add_argument('--output_dir', default=OUTPUT_DIR, help='Output directory')
    args = parser.parse_args()

    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)

    # 读取数据
    print(f"Loading data from: {args.input}")
    df = pd.read_csv(args.input)
    print(f"Total samples: {len(df)}")
    print(f"Total features: {df.shape[1]}")

    # 检查 grade 分布
    print(f"\nOverall grade distribution:")
    print(df["grade"].value_counts().sort_index().to_string())

    # Stage 1: 二分类特征选择
    run_stage1(df.copy(), args.output_dir)

    # Stage 2: 分级特征选择
    run_stage2(df.copy(), args.output_dir)

    print(f"\n{'='*50}")
    print("All stages completed!")
    print(f"{'='*50}")

if __name__ == "__main__":
    main()