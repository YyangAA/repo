import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import roc_auc_score, accuracy_score, roc_curve, auc, confusion_matrix, ConfusionMatrixDisplay, precision_score, f1_score

# ===============================
# 0. 设置路径与区域
# ===============================
# 统一指向你当前的 Linux 工作目录
base_dir = "/mnt/sda/yx/knee/5t/classify"
csv_input_dir = "/mnt/sda/yx/knee/5t/classify/csv" # 存放之前步骤生成的 filtered_features.csv
results_output_dir = "./results"

region_names = {
    1: "Femur_Medial",
    2: "Femur_Lateral",
    3: "Tibia_Medial",
    4: "Tibia_Lateral"
}

# 循环处理四个区域，或者手动修改索引 [1-4]
for r_idx in [1, 2, 3, 4]:
    knee_cartilage = region_names[r_idx]
    print(f"\n>>>> Training SVM Model for Region: {knee_cartilage} <<<<")

    # ===============================
    # 1. 读取由 LASSO 脚本生成的稳定特征 CSV
    # ===============================
    csv_path = os.path.join(csv_input_dir, f"{knee_cartilage}_filtered_features.csv")
    
    if not os.path.exists(csv_path):
        print(f"Skipping {knee_cartilage}: {csv_path} not found.")
        continue
        
    df_raw = pd.read_csv(csv_path)

    # 将同一个 case_id 的三行数据取平均，合并为一行
    df = df_raw.groupby(['case_id', 'region', 'grade']).mean().reset_index()
    print(f"Original rows: {len(df_raw)} -> Patient-level rows: {len(df)}")

    # 构造 X, y 和 groups
    X = df.drop(columns=["case_id", "region", "grade"])
    y = (df["grade"] > 0).astype(int).values
    groups = df["case_id"].values  # 用于 GroupKFold 确保同一病人不跨折
    print(f"Data shape: {df.shape}")

    # ===============================
    # 2. 构造 X 和 y（二分类：Grade 0 vs Grade 1+2）
    # ===============================
    # 自动剔除元数据列，剩余全是特征
    X = df.drop(columns=["case_id", "region", "grade"])
    y = (df["grade"] > 0).astype(int).values  # 0: 正常, 1: 损伤
    
    # 处理可能的缺失值（虽然之前的步骤应该已经处理过）
    if X.isnull().values.any():
        X = X.fillna(0)

    # ===============================
    # 3. 设置交叉验证与参数网格
    # ===============================
    cv_outer = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    # SVM RBF 核的超参数搜索范围
    param_grid = {
        'C': [0.1, 1, 10, 100],
        'gamma': ['scale', 'auto', 0.01, 0.1, 1]
    }

    auc_list = []
    acc_list = []
    f1_list = []
    
    # 用于绘制平均 ROC 曲线
    tprs = []
    mean_fpr = np.linspace(0, 1, 100)

    # ===============================
    # 4. 外层交叉验证评估
    # ===============================
    fold = 1
    for train_ix, test_ix in cv_outer.split(X, y):
        X_train, X_test = X.iloc[train_ix], X.iloc[test_ix]
        y_train, y_test = y[train_ix], y[test_ix]

        # 标准化
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # 内层网格搜索最佳参数
        grid = GridSearchCV(SVC(kernel='rbf', probability=True), param_grid, 
                            refit=True, cv=3, scoring='roc_auc')
        grid.fit(X_train_scaled, y_train)
        
        best_model = grid.best_estimator_

        # 预测
        y_prob = best_model.predict_proba(X_test_scaled)[:, 1]
        y_pred = best_model.predict(X_test_scaled)

        # 指标计算
        current_auc = roc_auc_score(y_test, y_prob)
        auc_list.append(current_auc)
        acc_list.append(accuracy_score(y_test, y_pred))
        
        # 计算 ROC 曲线插值
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        interp_tpr = np.interp(mean_fpr, fpr, tpr)
        interp_tpr[0] = 0.0
        tprs.append(interp_tpr)

        print(f"  Fold {fold} AUC: {current_auc:.3f}")
        fold += 1

    # ===============================
    # 5. 结果可视化与保存
    # ===============================
    # 创建结果目录
    fig_dir = os.path.join(results_output_dir, knee_cartilage, "figures")
    os.makedirs(fig_dir, exist_ok=True)

    # 绘制并保存 ROC 曲线
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

print("\nAll regions processed. Results saved in ./results/")

