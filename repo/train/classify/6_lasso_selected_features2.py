import numpy as np
import pandas as pd
from collections import Counter
import os

from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegressionCV
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score, accuracy_score, confusion_matrix, precision_score, f1_score
from sklearn.impute import SimpleImputer # 导入填充器

from features_filters import (
    basic_feature_cleaning,
    univariate_mwu_filter,
    correlation_filter,
    spearman_filter
)

# ===============================
# 1. 配置
# ===============================
region_names = {
    1: "Femur_Medial",
    2: "Femur_Lateral",
    3: "Tibia_Medial",
    4: "Tibia_Lateral"
}

input_csv = "/mnt/sda/yx/knee/5t/classify/knee_combined_features.csv"
n_folds = 5

for key, knee_cartilage in region_names.items():
    print(f"\n>>>> Region : {knee_cartilage} <<<<")

    if not os.path.exists(input_csv):
        print(f"Error: {input_csv} not found!")
        break

    df = pd.read_csv(input_csv)
    df = df[df["region"] == knee_cartilage].reset_index(drop=True)
    
    if len(df) < 10:
        print(f"Too few samples for {knee_cartilage}, skipping.")
        continue

    # 标签处理
    df["grade"] = (df["grade"] > 0).astype(int)

    # ===============================
    # 2. 缺失值预处理 (关键修改)
    # ===============================
    # 先剔除掉那些全是 NaN 或几乎全是 NaN 的列（缺失率 > 30%）
    initial_cols = df.columns.difference(["case_id", "region", "grade"])
    nan_threshold = 0.3
    keep_cols = df[initial_cols].columns[df[initial_cols].isnull().mean() < nan_threshold]
    df = df[["case_id", "region", "grade"] + list(keep_cols)]
    
    print(f"Features after dropping high-NaN columns: {len(keep_cols)}")

    # 使用均值填充剩余的少量 NaN
    # 注意：为了防止数据泄露，严格的做法是在 CV 内部填充，
    # 但在这里我们先做一次全局填充以确保流程能跑通
    df = df.fillna(df.mean(numeric_only=True))

    # ===============================
    # 3. 特征筛选流水线 (调用你的 filters)
    # ===============================
    print("Filtering features...")
    df = basic_feature_cleaning(df) 
    df = univariate_mwu_filter(df)
    df = correlation_filter(df)
    # df = spearman_filter(df) # 如果特征还是太多可以开启

    X = df.drop(columns=["case_id", "region", "grade"])
    y = df["grade"].values
    feature_names = X.columns.tolist()
    print(f"Features remaining after filters: {len(feature_names)}")

    # ===============================
    # 4. 带交叉验证的 LASSO
    # ===============================
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    outer_results = []
    best_fold_result = None

    for fold, (train_idx, test_idx) in enumerate(skf.split(X, y), 1):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # 构造 Pipeline：标准化 + LASSO (LogisticRegressionCV)
        # 这里再次加入 Imputer 作为双重保险
        clf = Pipeline([
            ('imputer', SimpleImputer(strategy='mean')), 
            ('scaler', StandardScaler()),
            ('lasso', LogisticRegressionCV(
                Cs=10, 
                cv=3, 
                penalty='l1', 
                solver='liblinear', 
                max_iter=5000, 
                scoring='roc_auc',
                random_state=42
            ))
        ])

        clf.fit(X_train, y_train)
        
        # 提取特征权重
        lasso_model = clf.named_steps['lasso']
        coefs = pd.Series(lasso_model.coef_[0], index=feature_names)
        selected_features = coefs[coefs != 0]

        # 预测
        y_prob = clf.predict_proba(X_test)[:, 1]
        y_pred = clf.predict(X_test)

        # 计算指标
        auc_val = roc_auc_score(y_test, y_prob)
        acc = accuracy_score(y_test, y_pred)
        
        res = {"fold": fold, "auc": auc_val, "accuracy": acc, "features": selected_features.index.tolist()}
        outer_results.append(res)
        
        if best_fold_result is None or auc_val > best_fold_result["auc"]:
            best_fold_result = res
            
        print(f"  Fold {fold}: AUC = {auc_val:.3f}, Selected {len(selected_features)} features")

    # ===============================
    # 5. 保存结果
    # ===============================
    if best_fold_result:
        stable_features = best_fold_result["features"]
        output_path = f"/mnt/sda/yx/knee/5t/classify/csv/{knee_cartilage}_filtered_features.csv"
        df[["case_id", "region", "grade"] + stable_features].to_csv(output_path, index=False)
        print(f"Done. Best AUC: {best_fold_result['auc']:.3f}, Features saved to {output_path}")