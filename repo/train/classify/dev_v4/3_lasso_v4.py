#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
lasso_v4.py
v4 改造重点（针对小样本 + 类不平衡 + Stage2 数据匮乏）：

  1. Stage 1 (二分类: Normal vs Damaged):
     - 4 区域分别 LASSO（保留原结构）
     - 加 class_weight='balanced'（解决 8:1 不平衡）
     - 用 StratifiedKFold 替代默认 KFold
     - 「股骨内侧先验」：股骨内侧数据最充足（75:42），先做它的 LASSO，
       拿到的稳定特征作为「先验集」，强制保留到其他 3 区的最终特征中。
     - 输出 {Region}_filtered_features.csv（与 v3 文件结构兼容）

  2. Stage 2 (分级: Grade1 vs Grade2):
     - 【核心变化】不再 4 区独立！
       原因：Tibia_Medial G2=8、Femur_Lateral G2=2、Tibia_Lateral G2=3
       根本无法独立训练，必须池化。
     - 池化 4 区损伤样本（G1=58, G2=31）
     - 加入 region one-hot（4 维）作为特征
     - 在池化样本上做 LASSO + class_weight + StratifiedKFold
     - 输出统一的 PooledStage2_filtered_features.csv
     - 同时为兼容下游 4 区独立分类脚本，把池化选出的特征写到
       {Region}_stage2_filtered_features.csv（每区文件内容相同特征列）

  3. 兼容性：
     - 输入 CSV 可以是原 get_feature_v3.py 的输出，也可以是
       add_cross_region_features.py 处理后的扩展版本（自动识别 cross_* 列）
    python /mnt/sda/yx/knee/nnUNet/repo/train/classify/dev_v4/lasso_v4.py --input /mnt/sda/yx/knee/nnUNet/repo/train/classify/dev_v4/data_train/knee_radiomics_features_3d_integrated_cross.csv  --output /mnt/sda/yx/knee/nnUNet/repo/train/classify/dev_v4/data_train/feature_v4
"""

import os
import argparse
import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegressionCV
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold


# -------------------------------
# 默认配置
# -------------------------------
INPUT_CSV = "./train/classify/dev_v4/data_train/knee_radiomics_features_3d_integrated_cross.csv"
OUTPUT_DIR = "./train/classify/dev_v4/data_train/feature_v4"

REGION_NAMES = {
    1: "Femur_Medial",
    2: "Femur_Lateral",
    3: "Tibia_Medial",
    4: "Tibia_Lateral",
}

# 「金标准区域」：数据最充足，作为 Stage1 先验来源
GOLD_REGION = "Femur_Medial"

# ★ v4.1 防过拟合参数
MAX_FEATURES_PER_REGION = 30        # Stage1 每个区域 LASSO 输出后的特征数硬上限
MAX_GOLD_PRIOR_SIZE = 20            # 金标准先验传递给其他区域时的 top-K 截断
LASSO_CS = np.logspace(-3, -0.3, 15) # C 范围 [0.001, 0.5]，强制更强正则化避免 p>>n 过拟合

# Stage 2 池化时强制保留的 shape 特征
FORCED_SHAPE_FEATURES = [
    "original_shape_VoxelVolume_mean",
    "original_shape_SurfaceArea_mean",
    "original_shape_MeshVolume_mean",
    # 兼容无 _mean 后缀的版本
    "original_shape_VoxelVolume",
    "original_shape_SurfaceArea",
    "original_shape_MeshVolume",
]

META_COLS = ["case_id", "region", "grade", "cartilage_missing"]


# -------------------------------
# 核心 LASSO 工具函数
# -------------------------------
def lasso_select(X, y, feature_names, cv=5, random_state=42,
                 class_weight="balanced", max_features=None):
    """
    通用 LASSO 二分类特征选择。
    Args:
        max_features: 选完后按 |coef| 截断到 top-K，None=不截断
    返回：(selected_feature_names, coef_array, fitted_pipeline)
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y).astype(int)

    # StratifiedKFold 确保每折类比例一致（重要！）
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=random_state)

    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("lasso", LogisticRegressionCV(
            Cs=LASSO_CS,                  # ★ 收紧 C 范围，避免 p>>n 过拟合
            cv=skf,
            penalty="l1",
            solver="liblinear",
            scoring="roc_auc",
            class_weight=class_weight,    # ★ 解决类不平衡
            max_iter=5000,
            random_state=random_state,
            n_jobs=-1,
        ))
    ])

    pipe.fit(X, y)
    coef = pipe.named_steps["lasso"].coef_.ravel()
    best_C = pipe.named_steps["lasso"].C_[0]
    sel_idx = np.where(np.abs(coef) > 1e-12)[0]

    # ★ 按 |coef| 排序，做 top-K 截断
    if max_features is not None and len(sel_idx) > max_features:
        ranked = sel_idx[np.argsort(-np.abs(coef[sel_idx]))]
        sel_idx = np.sort(ranked[:max_features])
        print(f"    Truncating to top-{max_features} by |coef|")

    sel_feats = [feature_names[i] for i in sel_idx]
    sel_coef = coef[sel_idx]

    print(f"    Input features: {X.shape[1]}, selected: {len(sel_feats)} "
          f"({len(sel_feats)/max(X.shape[1],1)*100:.1f}%), best C: {best_C:.4f}")
    return sel_feats, sel_coef, pipe


def get_feature_columns(df):
    """获取所有数值特征列（排除元数据 & 任何包含 NaN 但其实是 grade 的奇怪列）"""
    return [c for c in df.columns
            if c not in META_COLS and pd.api.types.is_numeric_dtype(df[c])]


def safe_dropna_for_lasso(df_region, feature_cols):
    """
    只在选定的特征列上 dropna，避免一些不相关的 cross_* 列把整行毁掉。
    返回过滤后的 df 和最终特征列列表。
    """
    sub = df_region[META_COLS + [c for c in feature_cols if c in df_region.columns]].copy()
    # 找出 NaN 比例 > 50% 的特征列，直接丢
    nan_ratio = sub[feature_cols].isna().mean()
    bad_cols = nan_ratio[nan_ratio > 0.5].index.tolist()
    if bad_cols:
        print(f"    Dropping {len(bad_cols)} cols with >50% NaN")
        feature_cols = [c for c in feature_cols if c not in bad_cols]
        sub = sub[META_COLS + feature_cols]
    # 行级 dropna
    sub = sub.dropna(how="any").reset_index(drop=True)

    # 去除零方差列
    if len(sub) > 0:
        var = sub[feature_cols].var()
        zero_var = var[var == 0].index.tolist()
        if zero_var:
            print(f"    Dropping {len(zero_var)} zero-variance cols")
            feature_cols = [c for c in feature_cols if c not in zero_var]
    return sub, feature_cols


# -------------------------------
# Stage 1: 4 区域独立 LASSO + 金标准先验
# -------------------------------
def run_stage1(df, output_dir):
    print("\n" + "#" * 70)
    print("  STAGE 1: Binary (Normal vs Damaged) — per-region LASSO + gold prior")
    print("#" * 70)

    stage1_features = {}

    # ===== Step A: 先训「金标准区域」，拿到先验特征集 =====
    print(f"\n[Stage 1] === Gold Region First: {GOLD_REGION} ===")
    df_gold = df[df["region"] == GOLD_REGION].copy().reset_index(drop=True)
    df_gold["grade"] = (df_gold["grade"] > 0).astype(int)

    feat_cols = get_feature_columns(df_gold)
    df_gold, feat_cols = safe_dropna_for_lasso(df_gold, feat_cols)
    print(f"  Samples: {len(df_gold)}, features: {len(feat_cols)}")
    print(f"  Class dist: {dict(pd.Series(df_gold['grade']).value_counts().sort_index())}")

    if len(df_gold) < 10 or len(feat_cols) == 0:
        print(f"  ERROR: gold region {GOLD_REGION} has insufficient data, abort.")
        return stage1_features

    gold_feats, gold_coef, _ = lasso_select(
        df_gold[feat_cols].values, df_gold["grade"].values, feat_cols, cv=5,
        max_features=MAX_FEATURES_PER_REGION
    )
    print(f"  Gold region final feature count: {len(gold_feats)}")
    stage1_features[GOLD_REGION] = gold_feats

    # ★ 金标准先验只传 top-K，避免淹没其他区域的本地信号
    gold_prior_topk = _top_k_features(gold_feats, gold_coef, MAX_GOLD_PRIOR_SIZE)
    print(f"  Gold prior (for other regions, top-{MAX_GOLD_PRIOR_SIZE}): {len(gold_prior_topk)}")

    _save_stage1_csv(df_gold, gold_feats, output_dir, GOLD_REGION)

    # ===== Step B: 其他 3 区，先各自 LASSO，再合并金标准先验 =====
    for _, region_name in REGION_NAMES.items():
        if region_name == GOLD_REGION:
            continue

        print(f"\n[Stage 1] === Region: {region_name} ===")
        df_r = df[df["region"] == region_name].copy().reset_index(drop=True)
        df_r["grade"] = (df_r["grade"] > 0).astype(int)

        feat_cols = get_feature_columns(df_r)
        df_r, feat_cols = safe_dropna_for_lasso(df_r, feat_cols)
        print(f"  Samples: {len(df_r)}, features: {len(feat_cols)}")
        print(f"  Class dist: {dict(pd.Series(df_r['grade']).value_counts().sort_index())}")

        if len(df_r) < 10 or len(feat_cols) == 0:
            print(f"  Skipping {region_name}: not enough data.")
            continue

        try:
            sel_feats, sel_coef, _ = lasso_select(
                df_r[feat_cols].values, df_r["grade"].values, feat_cols, cv=5,
                max_features=MAX_FEATURES_PER_REGION
            )
        except Exception as e:
            print(f"  LASSO failed for {region_name}: {e}")
            sel_feats = []

        # 合并：本地 LASSO 选出的 ∪ 金标准先验 top-K（仅保留该 region 表里实际存在的列）
        prior_in_df = [f for f in gold_prior_topk if f in df_r.columns]
        merged = list(dict.fromkeys(sel_feats + prior_in_df))
        # 合并后再做一次 top-K 截断（防止 merged 又超出预期）
        if len(merged) > MAX_FEATURES_PER_REGION + MAX_GOLD_PRIOR_SIZE:
            merged = merged[:MAX_FEATURES_PER_REGION + MAX_GOLD_PRIOR_SIZE]
        print(f"  Local LASSO: {len(sel_feats)}, gold prior added: "
              f"{len(prior_in_df)} → merged: {len(merged)}")

        stage1_features[region_name] = merged
        _save_stage1_csv(df_r, merged, output_dir, region_name)

    return stage1_features


def _top_k_features(feat_names, coefs, k):
    """按 |coef| 排序取 top-K"""
    if k is None or len(feat_names) <= k:
        return list(feat_names)
    order = np.argsort(-np.abs(coefs))[:k]
    return [feat_names[i] for i in order]


def _save_stage1_csv(df_region, feats, output_dir, region_name):
    feats = [f for f in feats if f in df_region.columns]
    cols = ["case_id", "region", "grade"] + feats
    if "cartilage_missing" in df_region.columns:
        cols.append("cartilage_missing")
    out_path = os.path.join(output_dir, f"{region_name}_filtered_features.csv")
    df_region[cols].to_csv(out_path, index=False)
    print(f"  Saved Stage1 → {out_path}  shape={df_region[cols].shape}")


# -------------------------------
# Stage 2: 池化 LASSO
# -------------------------------
def run_stage2_pooled(df, output_dir, stage1_features):
    print("\n" + "#" * 70)
    print("  STAGE 2: POOLED LASSO (Grade1 vs Grade2)")
    print("  Pooling all 4 regions into one model + region one-hot encoding")
    print("#" * 70)

    # 只取损伤样本
    df_dmg = df[df["grade"] > 0].copy().reset_index(drop=True)
    print(f"\n  Damaged samples (pooled): {len(df_dmg)}")
    print(f"  Per-region damaged count:")
    print(df_dmg.groupby("region")["grade"].value_counts().unstack(fill_value=0).to_string())

    # 候选特征池：取 Stage1 各区域选出的特征的并集（去重）
    all_stage1_feats = set()
    for r, fs in stage1_features.items():
        all_stage1_feats.update(fs)
    candidate_feats = sorted(all_stage1_feats)
    # 加上强制 shape
    forced_in_df = [f for f in FORCED_SHAPE_FEATURES if f in df_dmg.columns]
    candidate_feats = list(dict.fromkeys(candidate_feats + forced_in_df))
    candidate_feats = [c for c in candidate_feats if c in df_dmg.columns]
    print(f"\n  Candidate feature pool (Stage1 union ∪ forced shape): {len(candidate_feats)}")

    # 加 region one-hot
    region_dummies = pd.get_dummies(df_dmg["region"], prefix="regionFlag")
    df_dmg = pd.concat([df_dmg, region_dummies], axis=1)
    region_flag_cols = region_dummies.columns.tolist()
    print(f"  Region one-hot columns: {region_flag_cols}")

    final_feat_cols = candidate_feats + region_flag_cols

    # 安全 dropna
    df_dmg_clean, final_feat_cols = safe_dropna_for_lasso(df_dmg, final_feat_cols)
    print(f"  After NaN cleanup: samples={len(df_dmg_clean)}, features={len(final_feat_cols)}")

    if len(df_dmg_clean) < 15:
        print("  ERROR: too few pooled samples for Stage2.")
        return

    # 标签：Grade1=0, Grade2=1
    y = (df_dmg_clean["grade"] == 2).astype(int).values
    X = df_dmg_clean[final_feat_cols].values
    print(f"  Pooled class dist (G1 vs G2): {dict(pd.Series(y).value_counts().sort_index())}")

    try:
        # Stage2 是池化的，候选特征已经经过 Stage1 筛选，比较干净
        # 仍加上限保护，避免万一过拟合
        sel_feats, sel_coef, _ = lasso_select(
            X, y, final_feat_cols, cv=5, max_features=20
        )
    except Exception as e:
        print(f"  Pooled LASSO failed: {e}")
        return

    # 强制把 shape 特征加回（即使被 LASSO 淘汰）
    final_set = list(dict.fromkeys(sel_feats + forced_in_df))
    final_set = [f for f in final_set if f in df_dmg_clean.columns]

    print(f"\n  Final Stage2 features ({len(final_set)}):")
    for f in final_set:
        coef_val = sel_coef[sel_feats.index(f)] if f in sel_feats else 0.0
        flag = "[LASSO]" if f in sel_feats else "[forced]"
        print(f"    {flag} {f}: coef={coef_val:.4f}")

    # ===== 保存：池化版（推荐使用） =====
    cols_to_save = ["case_id", "region", "grade"] + final_set
    if "cartilage_missing" in df_dmg_clean.columns:
        cols_to_save.append("cartilage_missing")
    pooled_path = os.path.join(output_dir, "PooledStage2_filtered_features.csv")
    df_dmg_clean[cols_to_save].to_csv(pooled_path, index=False)
    print(f"\n  Saved POOLED Stage2 → {pooled_path}")
    print(f"  Shape: {df_dmg_clean[cols_to_save].shape}")

    # ===== 同时为下游兼容性，保存 4 个 region 子集（同一份特征） =====
    print(f"\n  Also saving per-region Stage2 subset CSVs (same feature set, per-region rows):")
    # 注意：region one-hot 列在 per-region 文件里没意义，但保留以保持列对齐
    for _, region_name in REGION_NAMES.items():
        sub = df_dmg_clean[df_dmg_clean["region"] == region_name]
        if len(sub) == 0:
            print(f"    {region_name}: no samples, skip.")
            continue
        out_path = os.path.join(output_dir, f"{region_name}_stage2_filtered_features.csv")
        sub[cols_to_save].to_csv(out_path, index=False)
        print(f"    {region_name}: {len(sub)} samples → {out_path}")


# -------------------------------
# 主流程
# -------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="LASSO Feature Selection v4 (cross-region prior + pooled Stage2)"
    )
    parser.add_argument("--input", default=INPUT_CSV, help="Input CSV path")
    parser.add_argument("--output_dir", default=OUTPUT_DIR, help="Output directory")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Loading data: {args.input}")
    df = pd.read_csv(args.input)
    print(f"  Total rows: {len(df)}")
    print(f"  Total columns: {df.shape[1]}")

    # 输出列分布概览
    cross_cols = [c for c in df.columns if c.startswith("cross_")]
    zwr_cols = [c for c in df.columns if c.endswith("_zwr")]
    print(f"  cross_* features: {len(cross_cols)}")
    print(f"  *_zwr features: {len(zwr_cols)}")

    print("\nOverall grade distribution:")
    print(df.groupby("region")["grade"].value_counts().unstack(fill_value=0).to_string())

    # Stage 1
    stage1_features = run_stage1(df.copy(), args.output_dir)

    # Stage 2 (池化)
    run_stage2_pooled(df.copy(), args.output_dir, stage1_features)

    print("\n" + "=" * 70)
    print("All stages completed!")
    print("=" * 70)
    print(f"Outputs in: {args.output_dir}")
    print(" - Stage 1: {Region}_filtered_features.csv  (per-region)")
    print(" - Stage 2: PooledStage2_filtered_features.csv  (★ recommended)")
    print(" - Stage 2: {Region}_stage2_filtered_features.csv  (compatibility)")


if __name__ == "__main__":
    main()