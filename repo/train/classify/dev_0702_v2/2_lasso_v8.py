#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
2_lasso_v8.py (dev_0702_v2)
LASSO 特征筛选 - v2 优化版本（跨区域特征增强 + Stage2 池化）

v2 改进:
  1. 同步 1a 的 LASSO 参数优化 (threshold 0.3, bootstrap 100, C搜索 30点)
  2. 自适应稳定性参数 (根据区域样本量和不平衡度)
  3. max_features 从 60 提到 80

Stage 1: 对跨区域增强特征做二次 LASSO 选择
Stage 2: 对池化数据做 LASSO 特征选择
"""

import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegressionCV, LogisticRegression
from sklearn.pipeline import Pipeline
import argparse

# -------------------------------
# 配置路径
# -------------------------------
INPUT_CSV = "./train/classify/dev_0702_v2/data_train/knee_radiomics_features_3d_integrated.csv"
OUTPUT_DIR = "./train/classify/dev_0702_v2/data_train/feature"
CROSS_FEATURE_DIR = "./train/classify/dev_0702_v2/data_train/feature_cross"

REGION_NAMES = {
    1: "Femur_Medial",
    2: "Femur_Lateral",
    3: "Tibia_Medial",
    4: "Tibia_Lateral"
}

REGIONS = list(REGION_NAMES.values())

FORCED_SHAPE_FEATURES = [
    "original_shape_VoxelVolume_mean",
    "original_shape_SurfaceArea_mean",
    "original_shape_MeshVolume_mean",
]

IMBALANCE_THRESHOLD = 0.35


def get_stability_params(y):
    """v2: 自适应稳定性参数"""
    n_samples = len(y)
    class_counts = pd.Series(y).value_counts()
    minority_ratio = class_counts.min() / n_samples if n_samples > 0 else 1.0

    if minority_ratio < 0.15 or n_samples < 30:
        return {'stability_threshold': 0.25, 'n_bootstrap': 150, 'max_features': 80}
    elif minority_ratio < 0.30:
        return {'stability_threshold': 0.30, 'n_bootstrap': 100, 'max_features': 80}
    else:
        return {'stability_threshold': 0.30, 'n_bootstrap': 100, 'max_features': 60}


def lasso_feature_selection(X, y, feature_names=None, cv=5, random_state=42,
                              class_weight=None, stability=True, n_bootstrap=100,
                              stability_threshold=0.3, max_features=80):
    """
    v2: LASSO feature selection with adaptive stability selection.
    """
    X = np.array(X)
    y = np.array(y)

    if feature_names is None:
        feature_names = [f"feature_{i}" for i in range(X.shape[1])]

    if stability:
        n_samples, n_features = X.shape
        selection_count = np.zeros(n_features)
        coef_sum = np.zeros(n_features)

        scaler_full = StandardScaler()
        X_scaled = scaler_full.fit_transform(X)

        # v2: 30 个 C 候选, 从 1e-3 到 1e2
        cv_lasso = LogisticRegressionCV(
            Cs=np.logspace(-3, 2, 30), cv=cv, penalty="l1",
            solver="liblinear", scoring="roc_auc", max_iter=5000,
            random_state=random_state, n_jobs=-1, class_weight=class_weight,
        )
        cv_lasso.fit(X_scaled, y)
        best_C = cv_lasso.C_[0]

        rng = np.random.RandomState(random_state)
        for b in range(n_bootstrap):
            n_sub = int(n_samples * 0.85)
            idx = rng.choice(n_samples, size=n_sub, replace=False)
            X_b, y_b = X_scaled[idx], y[idx]

            if len(np.unique(y_b)) < 2:
                continue

            try:
                lasso_b = LogisticRegression(
                    C=best_C, penalty="l1", solver="liblinear",
                    max_iter=5000, random_state=b, class_weight=class_weight
                )
                lasso_b.fit(X_b, y_b)
                coef_b = lasso_b.coef_.ravel()
                selected_b = np.where(coef_b != 0)[0]
                selection_count[selected_b] += 1
                coef_sum[selected_b] += np.abs(coef_b[selected_b])
            except Exception:
                continue

        selection_freq = selection_count / n_bootstrap
        stable_idx = np.where(selection_freq >= stability_threshold)[0]

        if len(stable_idx) > max_features:
            top_by_freq = np.argsort(-selection_freq[stable_idx])[:max_features]
            stable_idx = stable_idx[top_by_freq]

        if len(stable_idx) == 0:
            print(f"    [Stability] No features above {stability_threshold}, falling back to CV LASSO")
            coef = cv_lasso.coef_.ravel()
            stable_idx = np.where(coef != 0)[0]
            if len(stable_idx) == 0:
                from sklearn.feature_selection import f_classif
                f_scores, _ = f_classif(X_scaled, y)
                f_scores = np.nan_to_num(f_scores, nan=0.0)
                stable_idx = np.argsort(-f_scores)[:20]

        with np.errstate(invalid='ignore'):
            avg_coef = np.where(selection_count > 0, coef_sum / np.maximum(selection_count, 1), 0)
        selected_coef = avg_coef[stable_idx]
        order = np.argsort(-selection_freq[stable_idx])
        selected_idx = stable_idx[order]
        selected_coef = selected_coef[order]

        X_selected = X[:, selected_idx]
        selected_features = [feature_names[i] for i in selected_idx]

        print(f"    [Stability] {n_bootstrap} boot, th={stability_threshold}, C={best_C:.4f}")
        print(f"    {X.shape[1]} -> {len(selected_idx)} ({len(selected_idx)/X.shape[1]*100:.1f}%)")

        class _StableClf:
            pass
        clf = _StableClf()
        clf.best_C = best_C
        clf.selection_freq = selection_freq
        return X_selected, selected_features, selected_coef, clf

    # 兜底
    clf = Pipeline([
        ("scaler", StandardScaler()),
        ("lasso", LogisticRegressionCV(
            Cs=np.logspace(-3, 2, 30), cv=cv, penalty="l1",
            solver="liblinear", scoring="roc_auc", max_iter=5000,
            random_state=random_state, n_jobs=-1, class_weight=class_weight,
        ))
    ])
    clf.fit(X, y)
    lasso_model = clf.named_steps["lasso"]
    coef = lasso_model.coef_.ravel()
    selected_idx = np.where(coef != 0)[0]
    X_selected = X[:, selected_idx]
    selected_features = [feature_names[i] for i in selected_idx]
    selected_coef = coef[selected_idx]

    print(f"    Original: {X.shape[1]}, Selected: {len(selected_idx)}")
    return X_selected, selected_features, selected_coef, clf


def should_use_balanced(y):
    class_counts = pd.Series(y).value_counts()
    minority_ratio = class_counts.min() / len(y)
    return minority_ratio < IMBALANCE_THRESHOLD


def run_stage1_cross_lasso(cross_feature_dir, output_dir):
    """Stage 1: 对跨区域增强特征做二次 LASSO 选择 (v2)"""
    print("\n" + "#" * 60)
    print("  STAGE 1 (Cross-Region): LASSO on Enhanced Features [v2]")
    print("#" * 60)

    for key, region_name in REGION_NAMES.items():
        print(f"\n{'='*50}")
        print(f"[Stage 1 Cross] Region: {region_name}")
        print(f"{'='*50}")

        cross_csv = os.path.join(cross_feature_dir, f"{region_name}_cross_features.csv")
        if not os.path.exists(cross_csv):
            print(f"  Warning: {cross_csv} not found, skipping")
            continue

        df = pd.read_csv(cross_csv)
        df = df.groupby(['case_id', 'region', 'grade']).mean().reset_index()

        if len(df) == 0:
            continue

        print(f"  Samples: {len(df)}")

        df["grade"] = (df["grade"] > 0).astype(int)
        df = df.dropna(how='any')
        print(f"  After dropping NaN: {len(df)}")

        meta_cols = ["case_id", "region", "grade", "cartilage_missing"]
        feature_names = [c for c in df.columns if c not in meta_cols]
        print(f"  Features before LASSO: {len(feature_names)}")

        X = df[feature_names].values
        y = df["grade"].values

        class_dist = pd.Series(y).value_counts().sort_index()
        print(f"  Class distribution: {dict(class_dist)}")

        params = get_stability_params(y)
        print(f"  [v2] Stability params: th={params['stability_threshold']}, "
              f"boot={params['n_bootstrap']}, max={params['max_features']}")

        try:
            X_sel, features_sel, coef_sel, model = lasso_feature_selection(
                X, y, feature_names, cv=5, class_weight=None,
                stability_threshold=params['stability_threshold'],
                n_bootstrap=params['n_bootstrap'],
                max_features=params['max_features'],
            )

            result_df = pd.DataFrame({"feature": features_sel, "coef": coef_sel})
            result_df["abs_coef"] = result_df["coef"].abs()
            result_df = result_df.sort_values("abs_coef", ascending=False)

            print(f"\n  Top 15 selected features:")
            for i, row in result_df.head(15).iterrows():
                feat_type = "CROSS" if row['feature'].startswith('cross_') else \
                           "RATIO" if row['feature'].startswith('ratio_') else \
                           "ONEHOT" if row['feature'].startswith('region_') else "LASSO"
                print(f"    {row['feature']}: {row['coef']:.4f}  [{feat_type}]")

            n_cross = len([f for f in features_sel if f.startswith('cross_')])
            n_ratio = len([f for f in features_sel if f.startswith('ratio_')])
            n_onehot = len([f for f in features_sel if f.startswith('region_')])
            n_lasso = len(features_sel) - n_cross - n_ratio - n_onehot
            print(f"\n  Feature breakdown: {n_lasso} LASSO + {n_cross} cross + {n_ratio} ratio + {n_onehot} onehot = {len(features_sel)}")

            stable_features = result_df["feature"].tolist()
            cols_to_save = ["case_id", "region", "grade"] + stable_features
            if "cartilage_missing" in df.columns:
                cols_to_save.append("cartilage_missing")

            save_path = os.path.join(output_dir, f"{region_name}_cross_filtered_features.csv")
            df[cols_to_save].to_csv(save_path, index=False)

            print(f"\n  Saved to: {save_path}")
            print(f"  Shape: {df[cols_to_save].shape}")

        except Exception as e:
            print(f"  Error processing {region_name}: {e}")
            import traceback
            traceback.print_exc()
            continue


def run_stage2_pooled_lasso(cross_feature_dir, output_dir):
    """Stage 2: 对池化数据做 LASSO 特征选择 (v2)"""
    print("\n" + "#" * 60)
    print("  STAGE 2 (Pooled): LASSO on Pooled Features [v2]")
    print("#" * 60)

    # ---- 全4区域池化 LASSO ----
    pooled_csv = os.path.join(cross_feature_dir, "pooled_stage2_features.csv")
    if os.path.exists(pooled_csv):
        print(f"\n{'='*50}")
        print("[Stage 2 Pooled] 4-Region Pooling")
        print(f"{'='*50}")

        df_pooled = pd.read_csv(pooled_csv)
        df_pooled["grade_binary"] = (df_pooled["grade"] == 2).astype(int)

        print(f"  Total damaged samples: {len(df_pooled)}")
        class_dist = pd.Series(df_pooled["grade_binary"]).value_counts().sort_index()
        print(f"  Class distribution (0=G1, 1=G2): {dict(class_dist)}")

        meta_cols = ["case_id", "region", "grade", "grade_binary", "cartilage_missing"]
        feature_names = [c for c in df_pooled.columns if c not in meta_cols]

        # 零方差过滤
        n_before = len(feature_names)
        zero_var_feats = [f for f in feature_names if df_pooled[f].std() < 1e-10]
        if zero_var_feats:
            print(f"  Zero-variance filter: removing {len(zero_var_feats)}/{n_before}")
            feature_names = [f for f in feature_names if f not in zero_var_feats]
        print(f"  Features before LASSO: {len(feature_names)}")

        X = df_pooled[feature_names].values
        y = df_pooled["grade_binary"].values

        cv_folds = min(5, int(class_dist.min()))
        if cv_folds < 2:
            cv_folds = 2

        params = get_stability_params(y)
        print(f"  [v2] Stability params: th={params['stability_threshold']}, "
              f"boot={params['n_bootstrap']}, max={params['max_features']}")

        lasso_selected = []
        try:
            X_sel, features_sel, coef_sel, model = lasso_feature_selection(
                X, y, feature_names, cv=cv_folds, class_weight='balanced',
                stability_threshold=params['stability_threshold'],
                n_bootstrap=params['n_bootstrap'],
                max_features=params['max_features'],
            )

            result_df = pd.DataFrame({"feature": features_sel, "coef": coef_sel})
            result_df["abs_coef"] = result_df["coef"].abs()
            result_df = result_df.sort_values("abs_coef", ascending=False)

            print(f"\n  Selected features:")
            for i, row in result_df.iterrows():
                feat_type = "RATIO" if row['feature'].startswith('ratio_') else \
                           "ONEHOT" if row['feature'].startswith('region_') else "FEAT"
                print(f"    {row['feature']}: {row['coef']:.4f}  [{feat_type}]")

            lasso_selected = result_df["feature"].tolist()

        except Exception as e:
            print(f"  LASSO failed: {e}")

        available_forced = [f for f in FORCED_SHAPE_FEATURES if f in feature_names]
        final_features = list(dict.fromkeys(lasso_selected + available_forced))

        if not final_features:
            print(f"  Error: No features selected!")
        else:
            print(f"\n  Final features ({len(final_features)})")
            cols_to_save = ["case_id", "region", "grade"] + final_features
            save_path = os.path.join(output_dir, "pooled_stage2_filtered_features.csv")
            df_pooled[cols_to_save].to_csv(save_path, index=False)
            print(f"\n  Saved to: {save_path}")
            print(f"  Shape: {df_pooled[cols_to_save].shape}")

    # ---- FL+TL 池化 LASSO ----
    fl_tl_csv = os.path.join(cross_feature_dir, "pooled_stage2_FL_TL_features.csv")
    if os.path.exists(fl_tl_csv):
        print(f"\n{'='*50}")
        print("[Stage 2 Pooled] FL+TL Pooling")
        print(f"{'='*50}")

        df_fl_tl = pd.read_csv(fl_tl_csv)
        df_fl_tl["grade_binary"] = (df_fl_tl["grade"] == 2).astype(int)

        print(f"  Total damaged samples: {len(df_fl_tl)}")
        class_dist = pd.Series(df_fl_tl["grade_binary"]).value_counts().sort_index()
        print(f"  Class distribution (0=G1, 1=G2): {dict(class_dist)}")

        meta_cols = ["case_id", "region", "grade", "grade_binary", "cartilage_missing"]
        feature_names = [c for c in df_fl_tl.columns if c not in meta_cols]

        n_before = len(feature_names)
        zero_var_feats = [f for f in feature_names if df_fl_tl[f].std() < 1e-10]
        if zero_var_feats:
            print(f"  Zero-variance filter: removing {len(zero_var_feats)}/{n_before}")
            feature_names = [f for f in feature_names if f not in zero_var_feats]
        print(f"  Features before LASSO: {len(feature_names)}")

        X = df_fl_tl[feature_names].values
        y = df_fl_tl["grade_binary"].values

        if class_dist.min() >= 2:
            cv_folds = min(5, int(class_dist.min()))
            if cv_folds < 2:
                cv_folds = 2

            params = get_stability_params(y)

            lasso_selected = []
            try:
                X_sel, features_sel, coef_sel, model = lasso_feature_selection(
                    X, y, feature_names, cv=cv_folds, class_weight='balanced',
                    stability_threshold=params['stability_threshold'],
                    n_bootstrap=params['n_bootstrap'],
                    max_features=params['max_features'],
                )

                result_df = pd.DataFrame({"feature": features_sel, "coef": coef_sel})
                result_df["abs_coef"] = result_df["coef"].abs()
                result_df = result_df.sort_values("abs_coef", ascending=False)

                print(f"\n  Selected features:")
                for i, row in result_df.iterrows():
                    print(f"    {row['feature']}: {row['coef']:.4f}")

                lasso_selected = result_df["feature"].tolist()

            except Exception as e:
                print(f"  LASSO failed: {e}")

            available_forced = [f for f in FORCED_SHAPE_FEATURES if f in feature_names]
            final_features = list(dict.fromkeys(lasso_selected + available_forced))

            if final_features:
                cols_to_save = ["case_id", "region", "grade"] + final_features
                save_path = os.path.join(output_dir, "pooled_stage2_FL_TL_filtered_features.csv")
                df_fl_tl[cols_to_save].to_csv(save_path, index=False)
                print(f"\n  Saved to: {save_path}")
        else:
            print(f"  Too few samples for LASSO, using forced features only")
            available_forced = [f for f in FORCED_SHAPE_FEATURES if f in feature_names]
            if available_forced:
                cols_to_save = ["case_id", "region", "grade"] + available_forced
                save_path = os.path.join(output_dir, "pooled_stage2_FL_TL_filtered_features.csv")
                df_fl_tl[cols_to_save].to_csv(save_path, index=False)
                print(f"\n  Saved (forced only): {save_path}")


if __name__ == "__main__":
    run_stage1_cross_lasso(CROSS_FEATURE_DIR, OUTPUT_DIR)
    run_stage2_pooled_lasso(CROSS_FEATURE_DIR, OUTPUT_DIR)

    print("\n" + "=" * 60)
    print("LASSO v2 on enhanced features complete!")
    print(f"Output directory: {OUTPUT_DIR}")
    print("=" * 60)
