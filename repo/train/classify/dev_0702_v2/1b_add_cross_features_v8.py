#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
1b_add_cross_features_v8.py (dev_0702_v2)
跨区域特征增强 - v2 优化版本

v2 改进:
  1. 新增解剖比值特征:
     - 同侧间室比值 (FM/TM, FL/TL) — 反映同侧损伤扩散程度
     - 内外侧比值 (FM/FL, TM/TL) — 反映内外侧间室差异
     - 使用 shape 特征 (VoxelVolume, SurfaceArea) 构造比值
  2. 特征聚类去冗余:
     - 对跨区域特征做层次聚类 (相关性 > 0.9 的特征只保留 F-score 最高的)
     - 避免高度相关的特征淹没 LASSO 的有效信号
  3. NaN 填充策略优化:
     - 原始特征 NaN → 中位数填充 (而非填0，避免偏差)
     - 跨区域缺失 → 0 (表示该区域无数据)
  4. TOP-K 从 10 提到 15 (给后续 LASSO 更多候选)

输入:
  - knee_radiomics_features_3d_integrated.csv (原始特征)
  - {Region}_filtered_features.csv (Stage1 LASSO 选择后的特征)

输出:
  - {Region}_cross_features.csv (Stage1 + 跨区域特征 + 比值特征 + region one-hot)
  - pooled_stage2_features.csv (Stage2 池化数据集)
  - pooled_stage2_FL_TL_features.csv (FL+TL 池化数据集)
"""

import os
import pandas as pd
import numpy as np

# -------------------------------
# 配置路径
# -------------------------------
REPO_DIR = "/mnt/tidal-alsh-share2/dataset/askfollow/usr/yangxin/keyan/repo-dev_v4/repo"
INPUT_CSV = "./train/classify/dev_0702_v2/data_train/knee_radiomics_features_3d_integrated.csv"
STAGE1_FEATURE_DIR = "./train/classify/dev_0702_v2/data_train/feature"
STAGE2_FEATURE_DIR = "./train/classify/dev_0702_v2/data_train/feature"
OUTPUT_DIR = "./train/classify/dev_0702_v2/data_train/feature_cross"

REGION_NAMES = {
    1: "Femur_Medial",
    2: "Femur_Lateral",
    3: "Tibia_Medial",
    4: "Tibia_Lateral",
}

REGIONS = list(REGION_NAMES.values())

# v2: TOP-K 从 10 提到 15
CROSS_REGION_TOP_K = 15

# Stage2 池化数据集的特征数量上限
STAGE2_POOL_TOP_N = 120

# v2: 用于构造比值特征的 shape 特征
RATIO_SHAPE_FEATURES = [
    "original_shape_VoxelVolume_mean",
    "original_shape_SurfaceArea_mean",
    "original_shape_MeshVolume_mean",
]

# v2: 解剖配对定义 (用于比值特征)
ANATOMICAL_PAIRS = [
    ("Femur_Medial", "Tibia_Medial"),    # 内侧同室
    ("Femur_Lateral", "Tibia_Lateral"),   # 外侧同室
    ("Femur_Medial", "Femur_Lateral"),    # 股骨内外侧
    ("Tibia_Medial", "Tibia_Lateral"),    # 胫骨内外侧
]


def get_stage1_top_features(region_name, feature_dir, top_k=15):
    """
    v2: 从原始特征用 MI + F-test 联合排序选 top-K
    增加 top_k 到 15，给后续 LASSO 更多候选
    """
    import numpy as np
    from sklearn.feature_selection import f_classif, mutual_info_classif

    raw_csv = INPUT_CSV
    if not os.path.exists(raw_csv):
        print(f"  Warning: raw CSV not found, falling back to LASSO output")
        csv_path = os.path.join(feature_dir, f"{region_name}_filtered_features.csv")
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
            meta_cols = ["case_id", "region", "grade", "cartilage_missing"]
            feat_cols = [c for c in df.columns if c not in meta_cols]
            return feat_cols[:top_k]
        return []

    df_raw = pd.read_csv(raw_csv)
    df_region = df_raw[df_raw["region"] == region_name].copy()
    meta_cols = ["case_id", "region", "grade", "cartilage_missing"]
    feat_cols = [c for c in df_region.columns if c not in meta_cols]

    # v2: NaN 用中位数填充 (而非填0)
    for col in feat_cols:
        if df_region[col].isnull().any():
            med = df_region[col].median()
            df_region[col] = df_region[col].fillna(med if pd.notna(med) else 0.0)

    X = df_region[feat_cols].values
    y = (df_region["grade"] > 0).astype(int).values

    # 零方差过滤
    var = X.std(axis=0)
    valid_mask = var > 1e-10
    X_valid = X[:, valid_mask]
    feat_valid = [feat_cols[i] for i, v in enumerate(valid_mask) if v]

    # 两种单变量分数
    f_scores, _ = f_classif(X_valid, y)
    f_scores = np.nan_to_num(f_scores, nan=0.0)
    mi_scores = mutual_info_classif(X_valid, y, random_state=42, n_neighbors=5)

    # rank average
    from scipy.stats import rankdata
    f_rank = rankdata(f_scores)
    mi_rank = rankdata(mi_scores)
    combined = (f_rank + mi_rank) / 2.0

    top_idx = np.argsort(-combined)[:top_k]
    top_feats = [feat_valid[i] for i in top_idx]

    print(f"  {region_name} top-{len(top_feats)} features (F+MI rank): {top_feats[:3]}...")
    return top_feats


def build_anatomical_ratio_features(df_original):
    """
    v2 新增: 构造解剖比值特征。

    对每个病例，计算同侧间室和内外侧间室的 shape 特征比值。
    比值特征能捕捉区域间的相对大小变化，比绝对值更具区分力。

    生成特征名: ratio_{pair}_{feature}
    例如: ratio_FM_TM_VoxelVolume = FM_VoxelVolume / TM_VoxelVolume
    """
    print("\n  Building anatomical ratio features...")

    ratio_rows = []

    for case_id in df_original["case_id"].unique():
        case_df = df_original[df_original["case_id"] == case_id]

        # 构建区域 → shape 特征值 映射
        region_shapes = {}
        for _, row in case_df.iterrows():
            r = row["region"]
            region_shapes[r] = {}
            for feat in RATIO_SHAPE_FEATURES:
                if feat in row.index and pd.notna(row[feat]):
                    region_shapes[r][feat] = float(row[feat])
                else:
                    region_shapes[r][feat] = None

        # 为每条样本附加比值特征
        for _, row in case_df.iterrows():
            current_region = row["region"]
            ratio_feats = {}

            for r_a, r_b in ANATOMICAL_PAIRS:
                shapes_a = region_shapes.get(r_a, {})
                shapes_b = region_shapes.get(r_b, {})

                for feat in RATIO_SHAPE_FEATURES:
                    val_a = shapes_a.get(feat)
                    val_b = shapes_b.get(feat)
                    feat_short = feat.replace("original_shape_", "").replace("_mean", "")

                    # 比值: a/b，防止除0
                    if val_a is not None and val_b is not None and val_b > 1e-6:
                        ratio = val_a / val_b
                    else:
                        ratio = 1.0  # 缺失时填1.0 (中性值)

                    ratio_name = f"ratio_{r_a.split('_')[0]}_{r_b.split('_')[0]}_{feat_short}"
                    ratio_feats[ratio_name] = ratio

            ratio_rows.append({
                "case_id": case_id,
                "region": current_region,
                **ratio_feats,
            })

    df_ratios = pd.DataFrame(ratio_rows)
    n_ratio = len([c for c in df_ratios.columns if c.startswith("ratio_")])
    print(f"  Created {n_ratio} ratio features for {len(df_ratios)} rows")
    return df_ratios


def cluster_redundant_features(df, threshold=0.90):
    """
    v2 新增: 对跨区域特征做相关性聚类去冗余。

    对相关性 > threshold 的特征对，只保留 F-score (对二分类标签) 更高的那个。
    这样避免高度相关的特征淹没 LASSO 的信号。

    只对 cross_ 开头的特征做聚类，不影响原始 LASSO 特征。
    """
    from scipy.cluster.hierarchy import linkage, fcluster
    from scipy.spatial.distance import squareform
    from sklearn.feature_selection import f_classif

    cross_cols = [c for c in df.columns if c.startswith("cross_")]
    if len(cross_cols) < 2:
        return df, cross_cols

    X_cross = df[cross_cols].fillna(0).values

    # 计算相关矩阵
    corr = np.corrcoef(X_cross.T)
    corr = np.nan_to_num(corr, nan=0.0)
    # 距离 = 1 - |corr|
    dist = 1 - np.abs(corr)
    np.fill_diagonal(dist, 0)
    dist = np.clip(dist, 0, 2)

    # 层次聚类
    condensed = squareform(dist, checks=False)
    Z = linkage(condensed, method='average')
    labels = fcluster(Z, t=1 - threshold, criterion='distance')

    # 每个聚类中选 F-score 最高的
    y = (df["grade"] > 0).astype(int).values if "grade" in df.columns else None
    if y is None or len(np.unique(y)) < 2:
        # 无法计算 F-score，每个聚类取第一个
        selected = []
        for cluster_id in set(labels):
            members = [cross_cols[i] for i, l in enumerate(labels) if l == cluster_id]
            selected.append(members[0])
        return df, selected

    f_scores, _ = f_classif(X_cross, y)
    f_scores = np.nan_to_num(f_scores, nan=0.0)

    selected = []
    n_before = len(cross_cols)
    for cluster_id in set(labels):
        members_idx = [i for i, l in enumerate(labels) if l == cluster_id]
        # 选 F-score 最高的
        best_idx = max(members_idx, key=lambda i: f_scores[i])
        selected.append(cross_cols[best_idx])

    n_removed = n_before - len(selected)
    print(f"  [Cluster Dedup] {n_before} cross features → {len(selected)} (removed {n_removed} redundant, corr>{threshold})")

    return df, selected


def build_cross_region_features(df_original, region_top_features_full, df_ratios):
    """
    构建跨区域特征增强数据集 (v2)。
    """
    REGIONS = list(REGION_NAMES.values())

    # 预处理：为每个区域创建 case_id -> features 的映射
    region_feature_maps = {}

    for region_name in REGIONS:
        region_df = df_original[df_original["region"] == region_name].copy()
        top_feats = region_top_features_full.get(region_name, [])

        case_map = {}
        for _, row in region_df.iterrows():
            case_id = row["case_id"]
            case_map[case_id] = {}
            for feat in top_feats:
                cross_feat_name = f"cross_{region_name}_{feat}"
                if feat in row.index:
                    val = row[feat]
                    if pd.notna(val):
                        case_map[case_id][cross_feat_name] = val
                    else:
                        case_map[case_id][cross_feat_name] = 0.0
                else:
                    case_map[case_id][cross_feat_name] = 0.0
        region_feature_maps[region_name] = case_map

    # 构建增强数据集
    all_enhanced = []

    for region_name in REGIONS:
        region_df = df_original[df_original["region"] == region_name].copy()
        all_source_regions = REGIONS

        for idx, row in region_df.iterrows():
            enhanced_row = row.copy()
            case_id = row["case_id"]

            # 附加所有4个区域的 top 特征
            for source_region in all_source_regions:
                source_map = region_feature_maps.get(source_region, {})
                cross_feats = source_map.get(case_id, {})
                for cross_feat_name, cross_feat_val in cross_feats.items():
                    enhanced_row[cross_feat_name] = cross_feat_val

            # Stage1 region one-hot
            for r in REGIONS:
                enhanced_row[f"region_{r}"] = 1.0 if r == region_name else 0.0

            # v2: 附加比值特征
            ratio_row = df_ratios[(df_ratios["case_id"] == case_id) & (df_ratios["region"] == region_name)]
            if len(ratio_row) > 0:
                for col in df_ratios.columns:
                    if col.startswith("ratio_"):
                        enhanced_row[col] = ratio_row.iloc[0][col]

            all_enhanced.append(enhanced_row)

    df_enhanced = pd.DataFrame(all_enhanced)
    return df_enhanced


def run_stage1_cross_features(df_original, feature_dir, output_dir):
    """Stage1: 为每个区域创建跨区域特征增强数据集 (v2)"""
    print("\n" + "#" * 60)
    print("  STAGE 1: Cross-Region Feature Enhancement [v2]")
    print(f"  Top-{CROSS_REGION_TOP_K} from ALL 4 regions + Anatomical Ratios + Cluster Dedup")
    print("#" * 60)

    REGIONS = list(REGION_NAMES.values())

    # v2: 构建比值特征
    df_ratios = build_anatomical_ratio_features(df_original)

    # 获取每个区域的 top-K 特征
    region_top_features_full = {}
    for region_name in REGIONS:
        print(f"\nExtracting top features for {region_name}...")
        top_feats = get_stage1_top_features(region_name, feature_dir, top_k=CROSS_REGION_TOP_K)
        region_top_features_full[region_name] = top_feats

    # 构建跨区域特征
    print(f"\nBuilding cross-region features (top-{CROSS_REGION_TOP_K} from all 4 regions)...")
    df_enhanced = build_cross_region_features(df_original, region_top_features_full, df_ratios)

    # v2: 对跨区域特征做聚类去冗余
    df_enhanced, deduped_cross_cols = cluster_redundant_features(df_enhanced, threshold=0.90)

    # 保存每个区域的增强数据
    for region_name in REGIONS:
        s1_csv = os.path.join(feature_dir, f"{region_name}_filtered_features.csv")
        if not os.path.exists(s1_csv):
            print(f"  Skipping {region_name}: Stage1 CSV not found")
            continue

        df_s1 = pd.read_csv(s1_csv)
        meta_cols_s1 = ["case_id", "region", "grade", "cartilage_missing"]
        lasso_feats = [c for c in df_s1.columns if c not in meta_cols_s1]

        region_enhanced = df_enhanced[df_enhanced["region"] == region_name].copy().reset_index(drop=True)

        # v2: 使用去冗余后的跨区域特征
        all_cross_cols = deduped_cross_cols

        # 比值特征
        ratio_cols = [c for c in region_enhanced.columns if c.startswith("ratio_")]

        available_lasso = [f for f in lasso_feats if f in region_enhanced.columns]
        missing_lasso = [f for f in lasso_feats if f not in region_enhanced.columns]
        if missing_lasso:
            print(f"  Warning: {len(missing_lasso)} LASSO features missing in enhanced data for {region_name}")

        onehot_cols = [f"region_{r}" for r in REGIONS if f"region_{r}" in region_enhanced.columns]
        cols_to_save = ["case_id", "region", "grade"] + available_lasso + all_cross_cols + ratio_cols + onehot_cols
        if "cartilage_missing" in region_enhanced.columns:
            cols_to_save.append("cartilage_missing")

        cols_to_save = [c for c in cols_to_save if c in region_enhanced.columns]

        df_output = region_enhanced[cols_to_save].copy()

        # v2: NaN 填充策略 — 数值特征用中位数，跨区域缺失用0
        for col in df_output.columns:
            if col in ["case_id", "region", "grade", "cartilage_missing"]:
                continue
            if df_output[col].isnull().any():
                if col.startswith("cross_") or col.startswith("ratio_"):
                    df_output[col] = df_output[col].fillna(0.0)
                else:
                    med = df_output[col].median()
                    df_output[col] = df_output[col].fillna(med if pd.notna(med) else 0.0)

        save_path = os.path.join(output_dir, f"{region_name}_cross_features.csv")
        df_output.to_csv(save_path, index=False)

        n_lasso = len(available_lasso)
        n_cross = len([c for c in all_cross_cols if c in df_output.columns])
        n_ratio = len([c for c in ratio_cols if c in df_output.columns])
        n_onehot = len([c for c in onehot_cols if c in df_output.columns])
        n_total_feat = n_lasso + n_cross + n_ratio + n_onehot

        print(f"  {region_name}: {len(df_output)} rows, "
              f"{n_lasso} LASSO + {n_cross} cross + {n_ratio} ratio + {n_onehot} onehot = {n_total_feat} features → {save_path}")


def run_stage2_pooled_features(df_original, feature_dir, output_dir):
    """Stage2: 创建池化数据集 (v2)"""
    from sklearn.feature_selection import f_classif

    print("\n" + "#" * 60)
    print(f"  STAGE 2: Pooled Feature Dataset [v2]")
    print(f"  ANOVA F-test top-{STAGE2_POOL_TOP_N} + Forced Shape + Ratio Features")
    print("#" * 60)

    REGIONS = list(REGION_NAMES.values())
    FORCED_SHAPE = [
        "original_shape_VoxelVolume_mean",
        "original_shape_SurfaceArea_mean",
        "original_shape_MeshVolume_mean",
    ]

    # ========== Step 1: 收集所有 damaged 样本 ==========
    df_damaged = df_original[df_original["grade"] > 0].copy().reset_index(drop=True)
    print(f"\n  Total damaged samples: {len(df_damaged)}")
    print(f"  Grade distribution: {df_damaged['grade'].value_counts().sort_index().to_dict()}")
    print(f"  Per-region breakdown:")
    for region_name in REGIONS:
        sub = df_damaged[df_damaged["region"] == region_name]
        if len(sub) > 0:
            g1 = (sub["grade"] == 1).sum()
            g2 = (sub["grade"] == 2).sum()
            print(f"    {region_name}: {len(sub)} samples (G1={g1}, G2={g2})")

    # ========== Step 2: 提取所有原始特征列 ==========
    meta_cols = ["case_id", "region", "grade", "cartilage_missing"]
    raw_feat_cols = [c for c in df_damaged.columns if c not in meta_cols]
    print(f"\n  Raw feature columns: {len(raw_feat_cols)}")

    # v2: NaN 用中位数填充
    for col in raw_feat_cols:
        if df_damaged[col].isnull().any():
            med = df_damaged[col].median()
            df_damaged[col] = df_damaged[col].fillna(med if pd.notna(med) else 0.0)

    # ========== Step 3: 零方差过滤 ==========
    valid_feat_cols = [c for c in raw_feat_cols if df_damaged[c].std() > 1e-10]
    print(f"  After zero-variance filter: {len(valid_feat_cols)} (removed {len(raw_feat_cols) - len(valid_feat_cols)})")

    # ========== Step 4: ANOVA F-test 选 top-N ==========
    X = df_damaged[valid_feat_cols].values
    y = (df_damaged["grade"] == 2).astype(int).values

    f_scores, _ = f_classif(X, y)
    f_scores = np.nan_to_num(f_scores, nan=0.0)

    top_n = min(STAGE2_POOL_TOP_N, len(valid_feat_cols))
    top_idx = np.argsort(f_scores)[::-1][:top_n]
    top_features = [valid_feat_cols[i] for i in top_idx]

    # 强制保留 shape 特征
    for sf in FORCED_SHAPE:
        if sf in valid_feat_cols and sf not in top_features:
            top_features.append(sf)
            print(f"  [Forced] Added shape feature: {sf}")

    print(f"  Selected {len(top_features)} features (top-{top_n} by F-score + forced shape)")
    print(f"  Top 10 features by F-score:")
    for i, idx in enumerate(top_idx[:10]):
        print(f"    {i+1}. {valid_feat_cols[idx]}: F={f_scores[idx]:.2f}")

    # ========== Step 5: 构建池化数据集 ==========
    # v2: 附加比值特征
    df_ratios = build_anatomical_ratio_features(df_original)
    df_damaged_ratios = df_ratios.merge(
        df_damaged[["case_id", "region"]].drop_duplicates(),
        on=["case_id", "region"],
        how="right"
    )

    pooled_rows = []
    for _, row in df_damaged.iterrows():
        new_row = {"case_id": row["case_id"], "region": row["region"], "grade": row["grade"]}
        for feat in top_features:
            new_row[feat] = row[feat]
        for r in REGIONS:
            new_row[f"region_{r}"] = 1.0 if r == row["region"] else 0.0
        # v2: 附加比值特征
        ratio_match = df_ratios[(df_ratios["case_id"] == row["case_id"]) & (df_ratios["region"] == row["region"])]
        if len(ratio_match) > 0:
            for col in df_ratios.columns:
                if col.startswith("ratio_"):
                    new_row[col] = ratio_match.iloc[0][col]
        pooled_rows.append(new_row)

    df_pooled = pd.DataFrame(pooled_rows)

    # NaN 兜底
    feat_cols_all = [c for c in df_pooled.columns if c not in ["case_id", "region", "grade"]]
    for col in feat_cols_all:
        if df_pooled[col].isnull().any():
            df_pooled[col] = df_pooled[col].fillna(0.0)

    save_path = os.path.join(output_dir, "pooled_stage2_features.csv")
    df_pooled.to_csv(save_path, index=False)

    n_g1 = (df_pooled["grade"] == 1).sum()
    n_g2 = (df_pooled["grade"] == 2).sum()
    n_onehot = len([c for c in feat_cols_all if c.startswith("region_")])
    n_ratio = len([c for c in feat_cols_all if c.startswith("ratio_")])
    n_feat = len(feat_cols_all) - n_onehot - n_ratio

    print(f"\n  Pooled dataset: {len(df_pooled)} rows (G1={n_g1}, G2={n_g2}), "
          f"{n_feat} features + {n_onehot} onehot + {n_ratio} ratio = {len(feat_cols_all)} total")
    print(f"  Saved to: {save_path}")

    # ========== Step 6: FL+TL 池化数据集 ==========
    df_fl_tl = df_pooled[df_pooled["region"].isin(["Femur_Lateral", "Tibia_Lateral"])].copy()
    if len(df_fl_tl) > 0:
        save_path_fl_tl = os.path.join(output_dir, "pooled_stage2_FL_TL_features.csv")
        df_fl_tl.to_csv(save_path_fl_tl, index=False)
        n_g1_fl = (df_fl_tl["grade"] == 1).sum()
        n_g2_fl = (df_fl_tl["grade"] == 2).sum()
        print(f"\n  FL+TL Pooled dataset: {len(df_fl_tl)} rows (G1={n_g1_fl}, G2={n_g2_fl})")
        print(f"  Saved to: {save_path_fl_tl}")


if __name__ == "__main__":
    df = pd.read_csv(INPUT_CSV)
    print(f"Loaded {len(df)} rows, {df.shape[1]} columns from {INPUT_CSV}")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    run_stage1_cross_features(df, STAGE1_FEATURE_DIR, OUTPUT_DIR)
    run_stage2_pooled_features(df, STAGE2_FEATURE_DIR, OUTPUT_DIR)

    print("\n" + "=" * 60)
    print("Cross-region feature enhancement complete! [v2]")
    print(f"Output directory: {OUTPUT_DIR}")
    print("=" * 60)
