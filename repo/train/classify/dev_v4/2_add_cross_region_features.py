#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
add_cross_region_features.py
读取 get_feature_v3.py 输出的特征 CSV，按病例为单位构造跨区域对比特征：
  1. 内外侧对比（Medial vs Lateral）：
       - 股骨：Femur_Medial 与 Femur_Lateral 的差/比
       - 胫骨：Tibia_Medial 与 Tibia_Lateral 的差/比
  2. 股胫对比（Femur vs Tibia）：
       - 内侧：Femur_Medial 与 Tibia_Medial 的差/比
       - 外侧：Femur_Lateral 与 Tibia_Lateral 的差/比
  3. 同病例 4 区均值偏离：
       - 各区域特征 / 4 区域均值

输出：
  - 在每行（每个区域行）追加上述跨区域特征列
  - 列名：cross_<comparison>_<原特征名>
  - 同时输出一个区域内 z-score 归一化的版本，用于解决"区域基线差异"
 python /mnt/sda/yx/knee/nnUNet/repo/train/classify/dev_v4/add_cross_region_features.py  --input /mnt/sda/yx/knee/nnUNet/repo/train/classify/dev_v4/data_train/knee_radiomics_features_3d_integrated.csv  --output /mnt/sda/yx/knee/nnUNet/repo/train/classify/dev_v4/data_train/knee_radiomics_features_3d_integrated_cross.csv  
"""

import os
import argparse
import pandas as pd
import numpy as np

# 区域分组
REGIONS = ["Femur_Medial", "Femur_Lateral", "Tibia_Medial", "Tibia_Lateral"]
META_COLS = ["case_id", "region", "grade", "cartilage_missing"]

# 临床上有意义的对比对（A vs B）
CONTRAST_PAIRS = [
    # 内外侧对比（同骨）
    ("Femur_Medial", "Femur_Lateral", "femur_MedVsLat"),
    ("Tibia_Medial", "Tibia_Lateral", "tibia_MedVsLat"),
    # 股胫对比（同间室）
    ("Femur_Medial", "Tibia_Medial", "medial_FemVsTib"),
    ("Femur_Lateral", "Tibia_Lateral", "lateral_FemVsTib"),
]


def get_numeric_feature_cols(df):
    """提取真正的数值特征列（排除元数据）"""
    cols = [c for c in df.columns if c not in META_COLS]
    # 只保留数值列
    numeric_cols = []
    for c in cols:
        if pd.api.types.is_numeric_dtype(df[c]):
            numeric_cols.append(c)
    return numeric_cols


def pivot_case_features(df, feature_cols):
    """
    把长表（每行 case+region）转成宽表（每行 case，每列 region_feat）。
    便于做跨区域对比计算。
    """
    df_w = df.pivot_table(
        index="case_id",
        columns="region",
        values=feature_cols,
        aggfunc="first"   # 每个 case+region 只有一行
    )
    # 多级列名: (feature, region) → 合并为 region__feature
    df_w.columns = [f"{region}__{feat}" for feat, region in df_w.columns]
    df_w = df_w.reset_index()
    return df_w


def build_contrast_features(df_wide, feature_cols):
    """
    基于宽表构造跨区域对比特征：
      - cross_<pair_name>_diff_<feat>
      - cross_<pair_name>_ratio_<feat>
    返回 case_id 索引的 DataFrame，列为新特征。
    """
    new_cols = {"case_id": df_wide["case_id"].values}

    for region_a, region_b, pair_name in CONTRAST_PAIRS:
        for feat in feature_cols:
            col_a = f"{region_a}__{feat}"
            col_b = f"{region_b}__{feat}"
            if col_a not in df_wide.columns or col_b not in df_wide.columns:
                continue
            a = df_wide[col_a].values.astype(float)
            b = df_wide[col_b].values.astype(float)

            # 差值
            new_cols[f"cross_{pair_name}_diff_{feat}"] = a - b
            # 比值（带平滑因子防 0）
            denom = np.where(np.abs(b) < 1e-12, np.sign(b) * 1e-12 + 1e-12, b)
            new_cols[f"cross_{pair_name}_ratio_{feat}"] = a / denom

    return pd.DataFrame(new_cols)


def build_deviation_features(df_wide, feature_cols):
    """
    构造"同病例 4 区均值偏离"特征。
    对每个特征，先算病例的 4 区域均值，再算各区域相对偏离：
      cross_devFromMean_<region>_<feat> = region_value / case_mean
    """
    new_cols = {"case_id": df_wide["case_id"].values}

    for feat in feature_cols:
        # 收集该特征在 4 个区域的列
        region_cols = [f"{r}__{feat}" for r in REGIONS if f"{r}__{feat}" in df_wide.columns]
        if len(region_cols) < 2:
            continue
        # 计算 4 区均值（忽略 NaN）
        case_mean = df_wide[region_cols].mean(axis=1, skipna=True).values
        # 每个区域相对均值的偏离比
        for r in REGIONS:
            col = f"{r}__{feat}"
            if col not in df_wide.columns:
                continue
            v = df_wide[col].values.astype(float)
            denom = np.where(np.abs(case_mean) < 1e-12,
                             np.sign(case_mean) * 1e-12 + 1e-12,
                             case_mean)
            new_cols[f"cross_devFromMean_{r}_{feat}"] = v / denom

    return pd.DataFrame(new_cols)


def attach_cross_features_to_long(df_long, df_contrast, df_deviation):
    """
    把 case 级别的跨区域特征（contrast 和 deviation）拼回长表。
      - contrast 特征：每个病例只有一份，拼到该病例的所有区域行上
      - deviation 特征：包含 region 信息，需要按 (case, region) 取出对应列
    """
    # ---- contrast 特征：直接 merge on case_id（每行重复 4 次）
    df_out = df_long.merge(df_contrast, on="case_id", how="left")

    # ---- deviation 特征：要为每行选出对应 region 的列
    # 列名形如 cross_devFromMean_<region>_<feat>
    dev_cols = [c for c in df_deviation.columns if c.startswith("cross_devFromMean_")]
    if not dev_cols:
        return df_out

    # 解析每个 dev 列的 region
    # cross_devFromMean_Femur_Medial_<feat>
    # 因为 region 名本身含下划线，需要严格按 REGIONS 匹配
    dev_long_records = []
    df_dev_indexed = df_deviation.set_index("case_id")

    # 对每个区域准备一个映射：原始 dev 列名 → 该 region 在长表中应该用的列名
    rename_per_region = {}
    for r in REGIONS:
        prefix = f"cross_devFromMean_{r}_"
        rename = {}
        for col in dev_cols:
            if col.startswith(prefix):
                feat_name = col[len(prefix):]
                rename[col] = f"cross_devFromMean_{feat_name}"
        rename_per_region[r] = rename

    # 按区域填充
    for r in REGIONS:
        rename = rename_per_region[r]
        if not rename:
            continue
        # 只取属于该 region 的 dev 列，重命名为统一名（去掉 region 前缀）
        df_dev_r = df_dev_indexed[list(rename.keys())].rename(columns=rename).reset_index()
        df_dev_r["region"] = r
        dev_long_records.append(df_dev_r)

    df_dev_long = pd.concat(dev_long_records, ignore_index=True)
    df_out = df_out.merge(df_dev_long, on=["case_id", "region"], how="left")

    return df_out


def add_within_region_zscore(df, feature_cols):
    """
    按区域做 z-score 归一化，消除"区域基线差异"。
    输出的归一化列名加 _zwr 后缀（z-score within region）。
    注意：跨区域特征不做归一化（它们已经是相对量）。
    """
    new_df = df.copy()
    for r in REGIONS:
        mask = new_df["region"] == r
        if mask.sum() == 0:
            continue
        sub = new_df.loc[mask, feature_cols]
        mu = sub.mean()
        sd = sub.std().replace(0, 1.0)  # 防止除零
        z = (sub - mu) / sd
        # 加后缀
        z = z.rename(columns={c: f"{c}_zwr" for c in feature_cols})
        # 插入到原表（按相同 mask 的行）
        for col in z.columns:
            if col not in new_df.columns:
                new_df[col] = np.nan
            new_df.loc[mask, col] = z[col].values
    return new_df


def main():
    parser = argparse.ArgumentParser(description="Add cross-region features (long-table CSV).")
    parser.add_argument("--input", required=True, help="Input long-table feature CSV (from get_feature_v3.py)")
    parser.add_argument("--output", required=True, help="Output CSV with cross-region features")
    parser.add_argument("--add_zscore", action="store_true",
                        help="Additionally add per-region z-score normalized features (_zwr columns)")
    args = parser.parse_args()

    print(f"Loading: {args.input}")
    df = pd.read_csv(args.input)
    print(f"  Long-table shape: {df.shape}")

    # 找出真正的特征列
    feature_cols = get_numeric_feature_cols(df)
    print(f"  Numeric feature columns: {len(feature_cols)}")

    # 检查所有 region 都齐了
    region_set = set(df["region"].unique())
    missing = set(REGIONS) - region_set
    if missing:
        print(f"  WARNING: missing regions in data: {missing}")

    # 1) 构造宽表
    print("Pivoting to wide table...")
    df_wide = pivot_case_features(df, feature_cols)
    print(f"  Wide-table shape: {df_wide.shape}")

    # 2) 构造跨区域对比特征
    print("Building contrast features (medial vs lateral, femur vs tibia)...")
    df_contrast = build_contrast_features(df_wide, feature_cols)
    print(f"  Contrast features added: {df_contrast.shape[1] - 1}")

    # 3) 构造同病例 4 区均值偏离
    print("Building case-level mean deviation features...")
    df_deviation = build_deviation_features(df_wide, feature_cols)
    print(f"  Deviation features added: {df_deviation.shape[1] - 1}")

    # 4) 拼回长表
    print("Merging back to long table...")
    df_out = attach_cross_features_to_long(df, df_contrast, df_deviation)

    # 5) 可选：加区域内 z-score
    if args.add_zscore:
        print("Adding per-region z-score normalized features (_zwr suffix)...")
        df_out = add_within_region_zscore(df_out, feature_cols)

    # 6) 保存
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    df_out.to_csv(args.output, index=False)
    print(f"\nDone. Output shape: {df_out.shape}")
    print(f"Saved to: {args.output}")

    # 统计信息
    cross_cols = [c for c in df_out.columns if c.startswith("cross_")]
    zwr_cols = [c for c in df_out.columns if c.endswith("_zwr")]
    print(f"\nNew column statistics:")
    print(f"  cross_* columns: {len(cross_cols)}")
    if zwr_cols:
        print(f"  *_zwr columns: {len(zwr_cols)}")


if __name__ == "__main__":
    main()