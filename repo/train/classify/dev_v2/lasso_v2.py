import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, spearmanr
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegressionCV
from sklearn.pipeline import Pipeline

def basic_feature_cleaning(df, label_col="grade", missing_thresh=0.5):
    """
    删除 NaN 过多 & 常数特征
    """
    feature_cols = df.columns.difference(
        ["case_id", "region", label_col]
    )

    # 删除 NaN 比例过高的特征
    nan_ratio = df[feature_cols].isna().mean()
    keep_cols = nan_ratio[nan_ratio < missing_thresh].index

    df = df[["case_id", "region", label_col] + list(keep_cols)]

    # 删除常数特征
    nunique = df[keep_cols].nunique()
    keep_cols = nunique[nunique > 1].index

    df = df[["case_id", "region", label_col] + list(keep_cols)]

    return df

def univariate_mwu_filter(df, label_col="grade", p_thresh=0.05):
    """
    Mann-Whitney U 检验
    """
    feature_cols = df.columns.difference(
        ["case_id", "region", label_col]
    )

    selected_features = []

    group0 = df[df[label_col] == 0]
    group1 = df[df[label_col] > 0]

    for feat in feature_cols:
        x0 = group0[feat].dropna()
        x1 = group1[feat].dropna()

        if len(x0) < 5 or len(x1) < 5:
            continue

        _, p = mannwhitneyu(x0, x1, alternative="two-sided")

        if p < p_thresh:
            selected_features.append(feat)

    return df[["case_id", "region", label_col] + selected_features]

def correlation_filter(df, label_col="grade", corr_thresh=0.9, method="pearson"):
    """
    删除高相关特征
    """
    feature_cols = df.columns.difference(
        ["case_id", "region", label_col]
    )

    corr = df[feature_cols].corr(method).abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))

    # drop_cols = [
    #     col for col in upper.columns if any(upper[col] > corr_thresh)
    # ]
    variances = df[feature_cols].var()

    drop_cols = []
    for col in upper.columns:
        correlated = upper.index[upper[col] > corr_thresh].tolist()
        for row in correlated:
            # 删除方差更小的那个
            if variances[row] < variances[col]:
                drop_cols.append(row)
            else:
                drop_cols.append(col)
    drop_cols = list(set(drop_cols))

    keep_cols = [c for c in feature_cols if c not in drop_cols]

    return df[["case_id", "region", label_col] + keep_cols]


def spearman_filter(
    df,
    label_col="grade",
    p_thresh=0.05,
    corr_thresh=0.2
):
    """
    Spearman 相关性筛选
    - p < p_thresh
    - |rho| >= corr_thresh
    """
    feature_cols = df.columns.difference(
        ["case_id", "region", label_col]
    )

    selected_features = []

    y = df[label_col]

    for feat in feature_cols:
        x = df[feat]

        # 去除 NaN
        valid = x.notna() & y.notna()
        if valid.sum() < 10:
            continue

        rho, p = spearmanr(x[valid], y[valid])

        if np.isnan(rho):
            continue

        if (abs(rho) >= corr_thresh) and (p < p_thresh):
            selected_features.append(feat)

    return df[["case_id", "region", label_col] + selected_features]


import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegressionCV


def lasso_feature_selection(X, y, feature_names=None, cv=5, random_state=42):
    """
    LASSO feature selection for binary classification

    """

    X = np.array(X)
    y = np.array(y)

    if feature_names is None:
        feature_names = [f"feature_{i}" for i in range(X.shape[1])]

    # Pipeline
    clf = Pipeline([
        ("scaler", StandardScaler()),
        ("lasso", LogisticRegressionCV(
            Cs = np.logspace(-3, 1, 20),
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

    # 取出LASSO模型
    lasso_model = clf.named_steps["lasso"]

    # 系数
    coef = lasso_model.coef_.ravel()

    # 非零特征
    selected_idx = np.where(coef != 0)[0]

    X_selected = X[:, selected_idx]
    selected_features = [feature_names[i] for i in selected_idx]
    selected_coef = coef[selected_idx]

    print("Original features:", X.shape[1])
    print("Selected features:", len(selected_idx))

    return X_selected, selected_features, selected_coef, clf

if __name__ == "__main__":
    import os

    region_names = {
        1: "Femur_Medial",
        2: "Femur_Lateral",
        3: "Tibia_Medial",
        4: "Tibia_Lateral"
    }

    save_dir = "./train/classify/dev_v2/data_train"
    os.makedirs(save_dir, exist_ok=True)

    df = pd.read_csv("./train/classify/dev_v2/data_train/knee_radiomics_features_3d_integrated.csv")

    for key, knee_cartilage in region_names.items():
        print(f"Region : {knee_cartilage}")

        df_fm = df[df["region"] == knee_cartilage].copy().reset_index(drop=True)

        # ===============================
        # 二分类标签：0 vs (1 + 2)
        # ===============================
        df_fm["grade"] = (df_fm["grade"] > 0).astype(int)
        df_fm = df_fm.dropna(how='any') # !!!!!!

        feature_names = df_fm.columns.difference(
            ["case_id", "region", "grade"]
        )
        X = df_fm[feature_names].values
        y = df_fm["grade"].values

        X_sel, features_sel, coef_sel, model = lasso_feature_selection(
                X,
                y,
                feature_names,
                cv=5,
            )
        
        result = pd.DataFrame({
            "feature": features_sel,
            "coef": coef_sel
        })
        stable_features = result["feature"].tolist()
        cols_to_save = ["case_id", "region", "grade"] + stable_features

        save_path = os.path.join(save_dir, f"{knee_cartilage}_filtered_features.csv")
        df_fm[cols_to_save].to_csv(save_path, index=False)

        


