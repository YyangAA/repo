import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import joblib  # 将导入放到顶部
from sklearn.model_selection import StratifiedKFold, GridSearchCV, GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import roc_auc_score, accuracy_score, roc_curve, auc

# ===============================
# 0. 设置路径与区域
# ===============================
base_dir = "/mnt/sda/yx/knee/5t/classify"
csv_input_dir = "/mnt/sda/yx/knee/5t/classify/csv" 
results_output_dir = "./results"

region_names = {
    1: "Femur_Medial",
    2: "Femur_Lateral",
    3: "Tibia_Medial",
    4: "Tibia_Lateral"
}

# ===============================
# 开始循环处理四个区域
# ===============================
for r_idx in [1, 2, 3, 4]:
    knee_cartilage = region_names[r_idx]
    print(f"\n>>>> Training SVM Model for Region: {knee_cartilage} <<<<")

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
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        interp_tpr = np.interp(mean_fpr, fpr, tpr)
        interp_tpr[0] = 0.0
        tprs.append(interp_tpr)

        print(f"  Fold {fold} AUC: {current_auc:.3f}")
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
    print(f"Completed {knee_cartilage}. Mean AUC: {mean_auc:.3f}")

    # =======================================================
    # 6. 保存最终模型 (关键修改：这部分现在缩进到了循环内部)
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

    print(f"✅ Saved successfully: {model_dir}\n")

print("\nAll regions processed. Results and Models saved in ./results/")