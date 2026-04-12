import pandas as pd
import numpy as np
import os
import warnings
import matplotlib.pyplot as plt
import joblib  # 将导入放到顶部
from sklearn.model_selection import StratifiedKFold, GridSearchCV, GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import roc_auc_score, accuracy_score, roc_curve, auc

# ===============================
# 0. 设置路径与区域
# ===============================
base_dir = "./train/classify/dev_v3/data_train"
csv_input_dir = "./train/classify/dev_v3/data_train/feature" 
results_output_dir = "./checkpoint/results_260412"

region_names = {
    1: "Femur_Medial",
    2: "Femur_Lateral",
    3: "Tibia_Medial",
    4: "Tibia_Lateral"
}


# ===============================================================
#  STAGE 1: 二分类 SVM (Normal vs Damaged) — 与原流程完全一致
# ===============================================================
print("\n" + "#" * 60)
print("  STAGE 1: Training Binary SVM (Normal vs Damaged)")
print("#" * 60)

for r_idx in [1, 2, 3, 4]:
    knee_cartilage = region_names[r_idx]
    print(f"\n>>>> [Stage 1] Training SVM Model for Region: {knee_cartilage} <<<<")

    # 1. 读取数据
    csv_path = os.path.join(csv_input_dir, f"{knee_cartilage}_filtered_features.csv")
    
    if not os.path.exists(csv_path):
        print(f"Skipping {knee_cartilage}: {csv_path} not found.")
        continue
        
    df_raw = pd.read_csv(csv_path)

    # 2. 按患者聚合数据 (防止过拟合)
    df = df_raw.groupby(['case_id', 'region', 'grade']).mean().reset_index()
    print(f"Original rows: {len(df_raw)} -> Patient-level rows: {len(df)}")

    # 3. 构造特征矩阵 X 和 标签 y
    X = df.drop(columns=["case_id", "region", "grade"])
    y = (df["grade"] > 0).astype(int).values
    groups = df["case_id"].values 
    
    print(f"Data shape: {df.shape}")
    
    if X.isnull().values.any():
        X = X.fillna(0)

    # 4. 交叉验证评估 (用于评估性能，产出 ROC 图)
    # 使用 GroupKFold 确保同一病人不跨折
    cv_outer = GroupKFold(n_splits=5)
    
    param_grid = {
        'C': [0.1, 1, 10, 100],
        'gamma': ['scale', 'auto', 0.01, 0.1, 1]
    }

    auc_list = []
    tprs = []
    threshold_list = []   # 收集每折的最优阈值 (Youden's J)
    mean_fpr = np.linspace(0, 1, 100)
    
    fold = 1
    # 注意：这里改回了 GroupKFold，比 StratifiedKFold 更适合这种聚合后的医学数据
    for train_ix, test_ix in cv_outer.split(X, y, groups=groups):
        X_train, X_test = X.iloc[train_ix], X.iloc[test_ix]
        y_train, y_test = y[train_ix], y[test_ix]

        # 标准化
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # 网格搜索
        grid = GridSearchCV(SVC(kernel='rbf', probability=True), param_grid,
                            refit=True, cv=3, scoring='roc_auc')
        grid.fit(X_train_scaled, y_train)
        
        best_model = grid.best_estimator_
        y_prob = best_model.predict_proba(X_test_scaled)[:, 1]

        # 计算指标
        current_auc = roc_auc_score(y_test, y_prob)
        auc_list.append(current_auc)
        
        # ROC 插值
        fpr, tpr, thresholds = roc_curve(y_test, y_prob)
        interp_tpr = np.interp(mean_fpr, fpr, tpr)
        interp_tpr[0] = 0.0
        tprs.append(interp_tpr)

        # 计算最优阈值 (Youden's J = Sensitivity + Specificity - 1)
        youden_j = tpr - fpr
        best_idx = np.argmax(youden_j)
        best_threshold = thresholds[best_idx]
        threshold_list.append(best_threshold)

        print(f"  Fold {fold} AUC: {current_auc:.3f}, Optimal threshold: {best_threshold:.3f}")
        fold += 1

    # 5. 保存 ROC 曲线
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
    plt.title(f'ROC Curve - {knee_cartilage}')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    
    plt.savefig(os.path.join(fig_dir, f"{knee_cartilage}_ROC.png"), dpi=300)
    plt.close()
    # 计算 CV 平均最优阈值
    optimal_threshold = float(np.mean(threshold_list))
    print(f"Completed {knee_cartilage}. Mean AUC: {mean_auc:.3f}")
    print(f"  CV Optimal Threshold (Youden's J): {optimal_threshold:.4f}")
    print(f"  Per-fold thresholds: {[f'{t:.3f}' for t in threshold_list]}")

    # =======================================================
    # 6. 保存最终模型
    # =======================================================
    print(f"Saving final model for {knee_cartilage}...")
    
    # A. 在当前区域的全量数据上重新拟合标准化器
    final_scaler = StandardScaler()
    X_scaled = final_scaler.fit_transform(X)

    # B. 在当前区域的全量数据上重新训练 SVM
    final_grid = GridSearchCV(SVC(kernel='rbf', probability=True), param_grid, cv=3, scoring='roc_auc')
    final_grid.fit(X_scaled, y)
    final_model = final_grid.best_estimator_

    # C. 保存路径配置
    model_dir = os.path.join(results_output_dir, knee_cartilage, "models")
    os.makedirs(model_dir, exist_ok=True)

    # D. 执行保存
    joblib.dump(final_model, os.path.join(model_dir, "svm_model.pkl"))
    joblib.dump(final_scaler, os.path.join(model_dir, "scaler.pkl"))
    joblib.dump(X.columns.tolist(), os.path.join(model_dir, "feature_list.pkl"))
    # E. 保存最优阈值（CV 平均 Youden's J）
    joblib.dump(optimal_threshold, os.path.join(model_dir, "threshold.pkl"))

    print(f"✅ [Stage 1] Saved successfully: {model_dir}")
    print(f"   Threshold: {optimal_threshold:.4f}\n")


# ===============================================================
#  STAGE 2: 分级 SVM (Grade 1 vs Grade 2) — 级联第二阶段
# ===============================================================
print("\n" + "#" * 60)
print("  STAGE 2: Training Grade SVM (Grade 1 vs Grade 2)")
print("#" * 60)

for r_idx in [1, 2, 3, 4]:
    knee_cartilage = region_names[r_idx]
    print(f"\n>>>> [Stage 2] Training SVM Model for Region: {knee_cartilage} <<<<")

    # 1. 读取 Stage 2 特征数据
    csv_path = os.path.join(csv_input_dir, f"{knee_cartilage}_stage2_filtered_features.csv")

    if not os.path.exists(csv_path):
        print(f"Skipping {knee_cartilage}: Stage 2 CSV not found: {csv_path}")
        continue

    df_raw = pd.read_csv(csv_path)

    # 2. 按患者聚合 (与 Stage 1 一致)
    df = df_raw.groupby(['case_id', 'region', 'grade']).mean().reset_index()
    print(f"Original rows: {len(df_raw)} -> Patient-level rows: {len(df)}")

    # 3. 构造特征矩阵 X 和标签 y
    #    标签: Grade 1 → 0, Grade 2 → 1
    X = df.drop(columns=["case_id", "region", "grade"])
    y = (df["grade"] == 2).astype(int).values
    groups = df["case_id"].values

    n_g1 = int((y == 0).sum())
    n_g2 = int((y == 1).sum())
    print(f"Data shape: {X.shape}, Grade 1: {n_g1}, Grade 2: {n_g2}")

    if X.isnull().values.any():
        X = X.fillna(0)

    # 检查类别数
    if n_g1 < 1 or n_g2 < 1:
        print(f"  ⚠ Warning: Insufficient samples (G1={n_g1}, G2={n_g2}). Skipping CV.")
        print(f"  Will still train a model on all available data.")

    # 4. 交叉验证评估
    #    少数类 <= 3 时，CV 评估不可靠（fold 中可能只有一类），直接跳过
    MIN_MINORITY_FOR_CV = 4
    n_minority = min(n_g1, n_g2)

    param_grid_s2 = {
        'C': [0.1, 1, 10, 100],
        'gamma': ['scale', 'auto', 0.01, 0.1, 1]
    }

    if n_minority >= MIN_MINORITY_FOR_CV:
        n_folds = min(5, n_minority)
        # 使用 StratifiedKFold (样本量小，GroupKFold 可能导致某折只有一类)
        cv_outer = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)

        auc_list_s2 = []
        acc_list_s2 = []

        fold = 1
        for train_ix, test_ix in cv_outer.split(X, y):
            X_train, X_test = X.iloc[train_ix], X.iloc[test_ix]
            y_train, y_test = y[train_ix], y[test_ix]

            # 检查两个类别是否都存在
            if len(np.unique(y_train)) < 2 or len(np.unique(y_test)) < 2:
                print(f"  Fold {fold}: skipped (single class in train or test)")
                fold += 1
                continue

            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)

            # 使用 class_weight='balanced' 处理类别不平衡
            inner_cv = min(3, len(y_train) // 2)
            if inner_cv < 2:
                inner_cv = 2

            # 内部 GridSearch 用 accuracy，避免小样本下 roc_auc 全 NaN
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                grid = GridSearchCV(
                    SVC(kernel='rbf', probability=True, class_weight='balanced'),
                    param_grid_s2, refit=True, cv=inner_cv, scoring='accuracy'
                )
                grid.fit(X_train_scaled, y_train)

            best_model = grid.best_estimator_
            y_prob = best_model.predict_proba(X_test_scaled)[:, 1]
            y_pred = (y_prob >= 0.5).astype(int)

            try:
                current_auc = roc_auc_score(y_test, y_prob)
                auc_list_s2.append(current_auc)
                print(f"  Fold {fold} AUC: {current_auc:.3f}, Acc: {accuracy_score(y_test, y_pred):.3f}")
            except ValueError:
                print(f"  Fold {fold}: AUC not computable (single class), Acc: {accuracy_score(y_test, y_pred):.3f}")
            acc_list_s2.append(accuracy_score(y_test, y_pred))
            fold += 1

        if auc_list_s2:
            print(f"  Mean AUC: {np.mean(auc_list_s2):.3f} ± {np.std(auc_list_s2):.3f}")
        if acc_list_s2:
            print(f"  Mean Acc: {np.mean(acc_list_s2):.3f} ± {np.std(acc_list_s2):.3f}")
    else:
        print(f"  ⚠ Minority class has only {n_minority} samples (< {MIN_MINORITY_FOR_CV}).")
        print(f"  Skipping CV evaluation — directly training on all data.")

    # 5. 在全量数据上训练最终模型
    print(f"\nTraining final Stage 2 model for {knee_cartilage}...")

    final_scaler_s2 = StandardScaler()
    X_scaled_s2 = final_scaler_s2.fit_transform(X)

    # 全量训练：少数类 < 2 无法做 GridSearch，直接用默认参数
    #           少数类 >= 2 但 < MIN_MINORITY_FOR_CV 时用 accuracy 做 scoring
    if n_minority < 2:
        print(f"  Too few samples for GridSearch, using default params (C=1, gamma='scale')")
        final_model_s2 = SVC(kernel='rbf', C=1, gamma='scale',
                             probability=True, class_weight='balanced')
        final_model_s2.fit(X_scaled_s2, y)
    else:
        inner_cv_final = min(3, n_minority)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            final_grid_s2 = GridSearchCV(
                SVC(kernel='rbf', probability=True, class_weight='balanced'),
                param_grid_s2, cv=inner_cv_final, scoring='accuracy'
            )
            final_grid_s2.fit(X_scaled_s2, y)
        final_model_s2 = final_grid_s2.best_estimator_
        print(f"  Best params: {final_grid_s2.best_params_}")

    # 6. 保存 Stage 2 模型
    model_dir_s2 = os.path.join(results_output_dir, knee_cartilage, "models")
    os.makedirs(model_dir_s2, exist_ok=True)

    joblib.dump(final_model_s2, os.path.join(model_dir_s2, "svm_model_stage2.pkl"))
    joblib.dump(final_scaler_s2, os.path.join(model_dir_s2, "scaler_stage2.pkl"))
    joblib.dump(X.columns.tolist(), os.path.join(model_dir_s2, "feature_list_stage2.pkl"))

    print(f"✅ [Stage 2] Saved successfully: {model_dir_s2}")
    print(f"   Model: svm_model_stage2.pkl")
    print(f"   Scaler: scaler_stage2.pkl")
    print(f"   Features ({len(X.columns)}): {X.columns.tolist()}\n")


print("\n" + "=" * 60)
print("All regions processed. Stage 1 + Stage 2 models saved.")
print(f"Results directory: {results_output_dir}")
print("=" * 60)