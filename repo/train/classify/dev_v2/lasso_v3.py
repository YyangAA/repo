#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
lasso_feature_selection.py
LASSO 特征筛选 - 适配 3D 整合特征提取的输出格式
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
INPUT_CSV = "./train/classify/dev_v2/data_train/knee_radiomics_features_3d_integrated.csv"
OUTPUT_DIR = "./train/classify/dev_v2/data_train/feature"

# -------------------------------
# 区域定义
# -------------------------------
REGION_NAMES = {
    1: "Femur_Medial",
    2: "Femur_Lateral",
    3: "Tibia_Medial",
    4: "Tibia_Lateral"
}

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

# -------------------------------
# 主流程
# -------------------------------
def main():
    parser = argparse.ArgumentParser(description='LASSO Feature Selection')
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
    
    # 按区域处理
    for key, region_name in REGION_NAMES.items():
        print(f"\n{'='*50}")
        print(f"Processing Region: {region_name}")
        print(f"{'='*50}")
        
        # 筛选当前区域
        df_region = df[df["region"] == region_name].copy().reset_index(drop=True)
        
        if len(df_region) == 0:
            print(f"  Warning: No samples found for {region_name}")
            continue
        
        print(f"  Samples in this region: {len(df_region)}")
        
        # ===============================
        # 二分类标签：0 vs (1 + 2)
        # ===============================
        # 将 grade > 0 的设为 1（损伤），grade = 0 的设为 0（正常）
        df_region["grade"] = (df_region["grade"] > 0).astype(int)
        
        # 删除包含 NaN 的行（重要！）
        df_region = df_region.dropna(how='any')
        print(f"  Samples after dropping NaN: {len(df_region)}")
        
        if len(df_region) < 10:
            print(f"  Warning: Too few samples for {region_name}, skipping...")
            continue
        
        # 获取特征列（排除元数据列）
        meta_cols = ["case_id", "region", "grade", "cartilage_missing"]
        feature_names = [c for c in df_region.columns if c not in meta_cols]
        
        print(f"  Features before selection: {len(feature_names)}")
        
        # 准备 X 和 y
        X = df_region[feature_names].values
        y = df_region["grade"].values
        
        # 检查类别分布
        class_dist = pd.Series(y).value_counts().sort_index()
        print(f"  Class distribution: {dict(class_dist)}")
        
        # ===============================
        # LASSO 特征选择
        # ===============================
        try:
            X_sel, features_sel, coef_sel, model = lasso_feature_selection(
                X, y, feature_names, cv=5
            )
            
            # 保存选择结果
            result_df = pd.DataFrame({
                "feature": features_sel,
                "coef": coef_sel
            })
            
            # 按系数绝对值排序
            result_df["abs_coef"] = result_df["coef"].abs()
            result_df = result_df.sort_values("abs_coef", ascending=False)
            
            print(f"\n  Top 10 selected features:")
            for i, row in result_df.head(10).iterrows():
                print(f"    {row['feature']}: {row['coef']:.4f}")
            
            # ===============================
            # 保存筛选后的特征 CSV
            # ===============================
            # 构建要保存的列
            stable_features = result_df["feature"].tolist()
            cols_to_save = ["case_id", "region", "grade"] + stable_features
            
            # 添加 cartilage_missing（如果存在）
            if "cartilage_missing" in df_region.columns:
                cols_to_save.append("cartilage_missing")
            
            save_path = os.path.join(args.output_dir, f"{region_name}_filtered_features.csv")
            df_region[cols_to_save].to_csv(save_path, index=False)
            
            print(f"\n  Saved to: {save_path}")
            print(f"  Shape: {df_region[cols_to_save].shape}")
            
        except Exception as e:
            print(f"  Error processing {region_name}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    print(f"\n{'='*50}")
    print("All regions processed!")
    print(f"{'='*50}")

if __name__ == "__main__":
    main()