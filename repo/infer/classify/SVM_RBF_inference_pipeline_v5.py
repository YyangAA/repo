#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SVM_RBF_inference_pipeline_v5.py
适配 dev_v3 + v4 训练方式的推理脚本（级联二分类 + 跨区域特征 + 池化 Stage2）：

【与 v4 推理脚本相比的关键改动】
  1. 特征提取严格对齐 get_feature_v3.py：
     - 去掉 _std=0 列，只输出 _mean
     - Wavelet 仅保留沿 Z 方向为 L 的 4 个方向（LLL/LHL/HLL/HHL）
  2. 新增"跨区域特征构造"步骤（与 add_cross_region_features.py 一致）：
     - 内外侧对比 / 股胫对比的 diff & ratio 特征
     - 同病例 4 区均值偏离特征（cross_devFromMean_*）
     这些 cross_* 特征是 lasso_v4 / SVM_save_v4 训练时的候选池
  3. Stage 2 推理支持池化模型：
     - 检查 model_dir 下是否存在 is_pooled_stage2.pkl 标记
     - 若标记为 True，给样本附加 regionFlag_* one-hot 后再推理
     - 否则回退到 4 区独立模型推理（兼容老逻辑）
  4. 默认模型路径改为 ./checkpoint/results_v4
"""

import os
import sys
import argparse
import logging

import pandas as pd
import numpy as np
import SimpleITK as sitk
from radiomics import featureextractor
import joblib

logging.getLogger("radiomics").setLevel(logging.ERROR)

# ===============================
# 配置路径（默认值，可通过命令行参数覆盖）
# ===============================
IMAGE_FOLDER = "./data/image_3d"
MASK_FOLDER = "./data/mask_3d"
MODEL_BASE_DIR = "./checkpoint/results_v4"
OUTPUT_CSV = "./data/inference_results_v5.csv"

# ===============================
# 区域定义
# ===============================
REGION_NAMES = {
    1: "Femur_Medial",
    2: "Femur_Lateral",
    3: "Tibia_Medial",
    4: "Tibia_Lateral",
}
REGIONS = list(REGION_NAMES.values())

META_COLS = ["case_id", "region", "cartilage_missing"]

# ===============================
# Wavelet 方向过滤（与 get_feature_v3.py 严格一致）
# ===============================
WAVELET_KEEP_DIRECTIONS = {"LLL", "LHL", "HLL", "HHL"}

# ===============================
# 跨区域特征对（与 add_cross_region_features.py 严格一致）
# ===============================
CONTRAST_PAIRS = [
    ("Femur_Medial", "Femur_Lateral", "femur_MedVsLat"),
    ("Tibia_Medial", "Tibia_Lateral", "tibia_MedVsLat"),
    ("Femur_Medial", "Tibia_Medial", "medial_FemVsTib"),
    ("Femur_Lateral", "Tibia_Lateral", "lateral_FemVsTib"),
]


# ===============================
# 1. 特征提取器配置（与 get_feature_v3.py 完全对齐）
# ===============================

def get_3d_extractor(enable_wavelet=True):
    """
    创建 3D 特征提取器，与训练 get_feature_v3.py 完全对齐。
    """
    params = {
        "binWidth": 25,
        "normalize": True,
        "normalizeScale": 100,
        "interpolator": "sitkBSpline",
        "resampledPixelSpacing": [1, 1, 1],

        "imageType": {
            "Original": {},
        },

        "featureClass": {
            "shape": [],      # 启用 shape（VoxelVolume / SurfaceArea / MeshVolume...）
            "firstorder": [],
            "glcm": [],
            "glrlm": [],
            "glszm": [],
            "ngtdm": [],
        },
    }

    if enable_wavelet:
        params["imageType"]["Wavelet"] = {"wavelet": "haar"}

    extractor = featureextractor.RadiomicsFeatureExtractor(**params)
    extractor.settings["force2D"] = False  # 3D 模式

    if enable_wavelet:
        extractor.enableImageTypeByName("Wavelet")

    return extractor


# ===============================
# 2. 单区域 3D 特征提取（与 get_feature_v3.py 完全对齐）
# ===============================

def extract_region_features_3d(image, mask, region_label, region_name,
                                extractor_orig, extractor_wav=None):
    """
    3D 整体提取特征。
    - 只输出 _mean 后缀（与 v3 训练一致，去掉 _std=0 冗余列）
    - Wavelet 仅保留 Z 方向为 L 的 4 个方向（LLL / LHL / HLL / HHL）
    """
    mask_np = sitk.GetArrayFromImage(mask)
    roi_mask = (mask_np == region_label)

    if np.sum(roi_mask) < 20:
        return None

    features = {}

    # 2.1 Original 特征（含 shape）
    try:
        result_orig = extractor_orig.execute(image, mask, label=region_label)
        for k, v in result_orig.items():
            if k.startswith("original_"):
                features[k + "_mean"] = float(v)
    except Exception as e:
        print(f"  Error extracting original features for {region_name}: {e}")
        return None

    # 2.2 Wavelet 特征（仅保留 Z 低通方向）
    if extractor_wav is not None:
        try:
            result_wav = extractor_wav.execute(image, mask, label=region_label)
            for k, v in result_wav.items():
                if not k.startswith("wavelet-"):
                    continue
                # wavelet-XYZ_<class>_<feat>
                direction = k.split("_", 1)[0].replace("wavelet-", "")
                if (
                    len(direction) == 3
                    and direction[2] == "L"
                    and direction in WAVELET_KEEP_DIRECTIONS
                ):
                    features[k + "_mean"] = float(v)
        except Exception as e:
            print(f"  Error extracting wavelet features for {region_name}: {e}")

    return pd.Series(features)


def extract_all_features_for_case(image_path, mask_path, case_id,
                                   extractor_orig, extractor_wav):
    """为单个病例提取所有区域的全部特征（3D 模式）"""
    print(f"Processing: {case_id} ...")

    try:
        image = sitk.ReadImage(image_path)
        mask = sitk.ReadImage(mask_path)
        mask = sitk.Cast(mask, sitk.sitkUInt8)
        mask.CopyInformation(image)
    except Exception as e:
        print(f"Error reading {case_id}: {e}")
        return []

    all_features = []

    for label_id, region_name in REGION_NAMES.items():
        feats = extract_region_features_3d(
            image, mask, label_id, region_name,
            extractor_orig, extractor_wav,
        )

        missing_flag = 0
        if feats is None:
            missing_flag = 1
            feats = pd.Series(dtype=float)
            print(f"  Warning: {region_name} insufficient ROI for feature extraction")

        feats["case_id"] = case_id
        feats["region"] = region_name
        feats["cartilage_missing"] = missing_flag

        all_features.append(feats)

    return all_features


# ===============================
# 3. 跨区域特征构造（与 add_cross_region_features.py 完全对齐）
# ===============================

def get_numeric_feature_cols(df):
    cols = [c for c in df.columns if c not in META_COLS]
    return [c for c in cols if pd.api.types.is_numeric_dtype(df[c])]


def pivot_case_features(df, feature_cols):
    """长表 → 宽表（每行 case，每列 region__feat）"""
    df_w = df.pivot_table(
        index="case_id",
        columns="region",
        values=feature_cols,
        aggfunc="first",
    )
    df_w.columns = [f"{region}__{feat}" for feat, region in df_w.columns]
    df_w = df_w.reset_index()
    return df_w


def build_contrast_features(df_wide, feature_cols):
    """构造 cross_<pair>_diff_/ratio_<feat>"""
    new_cols = {"case_id": df_wide["case_id"].values}

    for region_a, region_b, pair_name in CONTRAST_PAIRS:
        for feat in feature_cols:
            col_a = f"{region_a}__{feat}"
            col_b = f"{region_b}__{feat}"
            if col_a not in df_wide.columns or col_b not in df_wide.columns:
                continue
            a = df_wide[col_a].values.astype(float)
            b = df_wide[col_b].values.astype(float)

            new_cols[f"cross_{pair_name}_diff_{feat}"] = a - b
            denom = np.where(np.abs(b) < 1e-12, np.sign(b) * 1e-12 + 1e-12, b)
            new_cols[f"cross_{pair_name}_ratio_{feat}"] = a / denom

    return pd.DataFrame(new_cols)


def build_deviation_features(df_wide, feature_cols):
    """构造 cross_devFromMean_<region>_<feat>"""
    new_cols = {"case_id": df_wide["case_id"].values}

    for feat in feature_cols:
        region_cols = [
            f"{r}__{feat}" for r in REGIONS if f"{r}__{feat}" in df_wide.columns
        ]
        if len(region_cols) < 2:
            continue
        case_mean = df_wide[region_cols].mean(axis=1, skipna=True).values
        for r in REGIONS:
            col = f"{r}__{feat}"
            if col not in df_wide.columns:
                continue
            v = df_wide[col].values.astype(float)
            denom = np.where(
                np.abs(case_mean) < 1e-12,
                np.sign(case_mean) * 1e-12 + 1e-12,
                case_mean,
            )
            new_cols[f"cross_devFromMean_{r}_{feat}"] = v / denom

    return pd.DataFrame(new_cols)


def attach_cross_features_to_long(df_long, df_contrast, df_deviation):
    """把 case 级跨区域特征拼回长表"""
    df_out = df_long.merge(df_contrast, on="case_id", how="left")

    dev_cols = [c for c in df_deviation.columns if c.startswith("cross_devFromMean_")]
    if not dev_cols:
        return df_out

    df_dev_indexed = df_deviation.set_index("case_id")
    rename_per_region = {}
    for r in REGIONS:
        prefix = f"cross_devFromMean_{r}_"
        rename = {}
        for col in dev_cols:
            if col.startswith(prefix):
                feat_name = col[len(prefix):]
                rename[col] = f"cross_devFromMean_{feat_name}"
        rename_per_region[r] = rename

    dev_long_records = []
    for r in REGIONS:
        rename = rename_per_region[r]
        if not rename:
            continue
        df_dev_r = (
            df_dev_indexed[list(rename.keys())]
            .rename(columns=rename)
            .reset_index()
        )
        df_dev_r["region"] = r
        dev_long_records.append(df_dev_r)

    df_dev_long = pd.concat(dev_long_records, ignore_index=True)
    df_out = df_out.merge(df_dev_long, on=["case_id", "region"], how="left")
    return df_out


def add_cross_region_features(df_long):
    """对外封装：在长表 df 上追加跨区域特征列"""
    feature_cols = get_numeric_feature_cols(df_long)
    if not feature_cols:
        print("  Warning: no numeric feature columns found, skipping cross features")
        return df_long

    df_wide = pivot_case_features(df_long, feature_cols)
    df_contrast = build_contrast_features(df_wide, feature_cols)
    df_deviation = build_deviation_features(df_wide, feature_cols)
    df_out = attach_cross_features_to_long(df_long, df_contrast, df_deviation)

    n_cross = sum(1 for c in df_out.columns if c.startswith("cross_"))
    print(f"  Cross-region features added: {n_cross}")
    return df_out


# ===============================
# 4. 通用：对齐特征 + 标准化 + 预测
# ===============================

def _align_features(features_df, feature_list):
    """按 feature_list 对齐特征矩阵；缺失列填 0；返回 X 和缺失列名 list"""
    X = pd.DataFrame(index=features_df.index)
    missing = []
    for feat in feature_list:
        if feat in features_df.columns:
            X[feat] = features_df[feat]
        else:
            X[feat] = 0
            missing.append(feat)
    X = X.fillna(0)
    return X, missing


# ===============================
# 5. Stage 1 推理（4 区独立，不变）
# ===============================

def load_model_and_predict_stage1(region_name, features_df, model_base_dir):
    """加载 Stage 1 模型并进行二分类预测（使用训练时 Youden's J 阈值）"""
    model_dir = os.path.join(model_base_dir, region_name, "models")

    model_path = os.path.join(model_dir, "svm_model.pkl")
    scaler_path = os.path.join(model_dir, "scaler.pkl")
    feat_list_path = os.path.join(model_dir, "feature_list.pkl")
    threshold_path = os.path.join(model_dir, "threshold.pkl")

    if not os.path.exists(model_path):
        print(f"  Error: Stage 1 model not found for {region_name} at {model_path}")
        return None, None, 0.5

    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    feature_list = joblib.load(feat_list_path)

    if os.path.exists(threshold_path):
        threshold = float(joblib.load(threshold_path))
        print(f"  [Stage 1] Loaded model for {region_name}, "
              f"{len(feature_list)} features, threshold={threshold:.4f}")
    else:
        threshold = 0.5
        print(f"  [Stage 1] Loaded model for {region_name}, "
              f"{len(feature_list)} features, threshold=0.5 (default)")

    X, missing = _align_features(features_df, feature_list)
    if missing:
        print(f"  Warning: {len(missing)} Stage 1 features missing, filled with 0")
        if len(missing) <= 5:
            print(f"    Missing: {missing}")
        else:
            print(f"    Missing (first 5): {missing[:5]}...")

    X_scaled = scaler.transform(X)
    y_prob = model.predict_proba(X_scaled)[:, 1]
    y_pred = (y_prob >= threshold).astype(int)
    return y_pred, y_prob, threshold


# ===============================
# 6. Stage 2 推理（支持池化 + 兼容 4 区独立）
# ===============================

def _attach_region_onehot(features_df, feature_list, region_name):
    """
    给 features_df 附加 regionFlag_* one-hot 列，
    target_region == region_name 的列置 1，其余置 0。
    只新增 feature_list 中以 'regionFlag_' 开头的列。
    """
    df = features_df.copy()
    for col in feature_list:
        if not col.startswith("regionFlag_"):
            continue
        target_region = col[len("regionFlag_"):]
        df[col] = 1 if target_region == region_name else 0
    return df


def load_model_and_predict_stage2(region_name, features_df, model_base_dir):
    """
    加载 Stage 2 模型并对 Damaged 样本进行分级预测。

    自动检测：
      - is_pooled_stage2.pkl 存在且为 True → 池化模型，附加 regionFlag_* one-hot
      - 否则 → 4 区独立模型，原逻辑

    返回:
        y_pred: 0 = Grade 1, 1 = Grade 2
        y_prob: Grade 2 概率
        is_pooled: bool 标记
    """
    model_dir = os.path.join(model_base_dir, region_name, "models")

    model_path = os.path.join(model_dir, "svm_model_stage2.pkl")
    scaler_path = os.path.join(model_dir, "scaler_stage2.pkl")
    feat_list_path = os.path.join(model_dir, "feature_list_stage2.pkl")
    is_pooled_path = os.path.join(model_dir, "is_pooled_stage2.pkl")
    region_tag_path = os.path.join(model_dir, "stage2_region_tag.pkl")

    if not os.path.exists(model_path):
        print(f"  [Stage 2] Model not found for {region_name}, defaulting to Grade 1")
        return None, None, False

    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    feature_list = joblib.load(feat_list_path)

    is_pooled = False
    if os.path.exists(is_pooled_path):
        try:
            is_pooled = bool(joblib.load(is_pooled_path))
        except Exception as e:
            print(f"  [Stage 2] Failed to load is_pooled flag: {e}, treating as per-region")
            is_pooled = False

    region_tag = None
    if is_pooled and os.path.exists(region_tag_path):
        try:
            region_tag = joblib.load(region_tag_path)
        except Exception:
            region_tag = None

    mode_str = "POOLED" if is_pooled else "PER-REGION"
    print(f"  [Stage 2 | {mode_str}] Loaded model for {region_name}, "
          f"expecting {len(feature_list)} features"
          + (f" (pooled tag={region_tag})" if region_tag else ""))

    # 池化模型：附加 regionFlag_* one-hot（用本次推理的 region_name 作为 target）
    if is_pooled:
        n_flag_cols = sum(1 for c in feature_list if c.startswith("regionFlag_"))
        if n_flag_cols > 0:
            features_df = _attach_region_onehot(features_df, feature_list, region_name)
            print(f"    Attached {n_flag_cols} regionFlag_* one-hot column(s) "
                  f"(active = regionFlag_{region_name})")
        else:
            print(f"    Note: pooled flag is set but feature_list contains no "
                  f"regionFlag_* columns; skipping one-hot attachment")

    X, missing = _align_features(features_df, feature_list)
    if missing:
        print(f"  Warning: {len(missing)} Stage 2 features missing, filled with 0")
        if len(missing) <= 10:
            for mf in missing:
                print(f"    Missing: {mf}")
        else:
            print(f"    Missing (first 5): {missing[:5]}...")

    X_scaled = scaler.transform(X)
    y_prob = model.predict_proba(X_scaled)[:, 1]
    y_pred = (y_prob >= 0.5).astype(int)
    return y_pred, y_prob, is_pooled


# ===============================
# 7. 主流程
# ===============================

def main():
    parser = argparse.ArgumentParser(
        description="Knee Cartilage Inference (v5 - Cascaded + CrossRegion + Pooled Stage2)"
    )
    parser.add_argument("--image_folder", default=IMAGE_FOLDER,
                        help="Path to 3D image folder (.nii.gz)")
    parser.add_argument("--mask_folder", default=MASK_FOLDER,
                        help="Path to 3D mask folder (.nii.gz)")
    parser.add_argument("--model_dir", default=MODEL_BASE_DIR,
                        help="Path to trained models (default: ./checkpoint/results_v4)")
    parser.add_argument("--output", default=OUTPUT_CSV,
                        help="Output CSV path")
    args = parser.parse_args()

    # 初始化提取器
    print("Initializing 3D feature extractors (v3-aligned: shape on, wavelet Z-low only)...")
    extractor_orig = get_3d_extractor(enable_wavelet=False)
    extractor_wav = get_3d_extractor(enable_wavelet=True)

    # ---------- Step 1: 特征提取 ----------
    print("\n" + "=" * 60)
    print("Step 1: Extracting 3D features (aligned with get_feature_v3.py)...")
    print("=" * 60)

    all_case_features = []
    for filename in sorted(os.listdir(args.image_folder)):
        if not filename.endswith(".nii.gz"):
            continue

        file_prefix = filename.replace(".nii.gz", "")
        case_id = file_prefix.split("_")[0]

        image_path = os.path.join(args.image_folder, filename)
        mask_path = os.path.join(args.mask_folder, file_prefix + ".nii.gz")

        if not os.path.exists(mask_path):
            print(f"Warning: Mask not found for {file_prefix}")
            continue

        case_feats = extract_all_features_for_case(
            image_path, mask_path, case_id,
            extractor_orig, extractor_wav,
        )
        all_case_features.extend(case_feats)

    if not all_case_features:
        print("No features extracted. Please check input paths.")
        sys.exit(1)

    df_features = pd.DataFrame(all_case_features)

    # 调整列顺序：meta cols 在前
    meta_first = [c for c in META_COLS if c in df_features.columns]
    other_cols = [c for c in df_features.columns if c not in meta_first]
    df_features = df_features[meta_first + other_cols]

    print(f"\nExtracted features for {len(df_features)} region samples")
    print(f"Feature matrix shape: {df_features.shape}")

    shape_cols = [c for c in df_features.columns if "shape" in c.lower()]
    print(f"Shape features extracted: {len(shape_cols)}")

    # ---------- Step 2: 跨区域特征构造 ----------
    print("\n" + "=" * 60)
    print("Step 2: Building cross-region features "
          "(contrast diff/ratio + devFromMean)...")
    print("=" * 60)
    df_features = add_cross_region_features(df_features)
    print(f"Feature matrix shape after cross features: {df_features.shape}")

    # 保存原始 + 跨区域特征（调试用）
    raw_output = args.output.replace(".csv", "_raw_features_with_cross.csv")
    os.makedirs(os.path.dirname(os.path.abspath(args.output)) or ".", exist_ok=True)
    df_features.to_csv(raw_output, index=False)
    print(f"Raw + cross features saved to: {raw_output}")

    # ---------- Step 3: Stage 1 推理 ----------
    print("\n" + "=" * 60)
    print("Step 3: Stage 1 — SVM inference (Normal vs Damaged)...")
    print("=" * 60)

    results = []
    region_thresholds = {}

    for region_name in REGIONS:
        print(f"\nProcessing region: {region_name}")
        region_df = df_features[df_features["region"] == region_name].copy()

        if len(region_df) == 0:
            print(f"  No samples found for {region_name}")
            continue

        y_pred, y_prob, threshold = load_model_and_predict_stage1(
            region_name, region_df, args.model_dir
        )

        if y_pred is None:
            continue

        region_df["predicted_label"] = y_pred
        region_df["probability_damage"] = y_prob
        region_df["threshold_used"] = threshold
        region_thresholds[region_name] = threshold
        results.append(region_df)

    if not results:
        print("No predictions made. Please check model paths.")
        sys.exit(1)

    df_results = pd.concat(results, ignore_index=True)

    # ---------- Step 4: Stage 2 推理（支持池化） ----------
    print("\n" + "=" * 60)
    print("Step 4: Stage 2 — Grade classification (Grade1 vs Grade2)...")
    print("=" * 60)

    predicted_grades = []
    grade2_probs = []
    grade_reasons = []
    region_stage2_modes = {}  # region -> 'pooled' / 'per-region' / 'no-model'

    for region_name in REGIONS:
        print(f"\nProcessing region: {region_name}")

        region_mask = df_results["region"] == region_name
        damaged_mask = region_mask & (df_results["predicted_label"] == 1)

        damaged_indices = df_results[damaged_mask].index.tolist()
        n_damaged = len(damaged_indices)

        if n_damaged == 0:
            print(f"  No damaged samples for Stage 2")
            region_stage2_modes[region_name] = "no-damaged"
            continue

        print(f"  {n_damaged} damaged samples to grade")

        damaged_df = df_results.loc[damaged_indices]
        s2_pred, s2_prob, is_pooled = load_model_and_predict_stage2(
            region_name, damaged_df, args.model_dir
        )

        if s2_pred is not None:
            mode_str = "pooled" if is_pooled else "per-region"
            region_stage2_modes[region_name] = mode_str
            for i, idx in enumerate(damaged_indices):
                grade = 2 if s2_pred[i] == 1 else 1
                prob_g2 = float(s2_prob[i])
                predicted_grades.append((idx, grade))
                grade2_probs.append((idx, prob_g2))
                reason = f"stage2_{mode_str}_prob_g2={prob_g2:.3f}"
                grade_reasons.append((idx, reason))
                print(f"    {df_results.loc[idx, 'case_id']}: "
                      f"Grade {grade} (prob_G2={prob_g2:.3f})")
        else:
            region_stage2_modes[region_name] = "no-model"
            for idx in damaged_indices:
                predicted_grades.append((idx, 1))
                grade2_probs.append((idx, 0.0))
                grade_reasons.append((idx, "no_stage2_model"))

    # 组装最终 grade
    df_results["predicted_grade"] = 0  # 默认 Normal → Grade 0
    df_results["probability_grade2"] = 0.0
    df_results["grade_reason"] = "stage1_normal"

    for idx, grade in predicted_grades:
        df_results.loc[idx, "predicted_grade"] = grade
    for idx, prob in grade2_probs:
        df_results.loc[idx, "probability_grade2"] = prob
    for idx, reason in grade_reasons:
        df_results.loc[idx, "grade_reason"] = reason

    # ---------- Step 5: 保存结果 ----------
    print("\n" + "=" * 60)
    print("Step 5: Saving results...")
    print("=" * 60)

    output_cols = [
        "case_id",
        "region",
        "cartilage_missing",
        "predicted_label",       # 0=Normal, 1=Damaged
        "probability_damage",    # Stage 1 SVM 损伤概率
        "threshold_used",        # Youden's J 最优阈值
        "predicted_grade",       # 0=Normal, 1=Grade1, 2=Grade2
        "probability_grade2",    # Stage 2 Grade 2 概率
        "grade_reason",          # 分级依据
    ]
    output_cols = [c for c in output_cols if c in df_results.columns]

    df_output = df_results[output_cols].copy()
    df_output.to_csv(args.output, index=False)

    detailed_output = args.output.replace(".csv", "_detailed.csv")
    df_results.to_csv(detailed_output, index=False)

    print(f"\nResults saved:")
    print(f"  Summary:  {args.output}")
    print(f"  Detailed: {detailed_output}")
    print(f"  Raw+Cross: {raw_output}")

    # ---------- 打印统计 ----------
    print("\n" + "=" * 60)
    print("Inference Summary:")
    print("=" * 60)
    print(f"Total region samples: {len(df_output)}")
    print(f"  Grade 0 (Normal):        {(df_output['predicted_grade']==0).sum()}")
    print(f"  Grade 1 (Mild damage):   {(df_output['predicted_grade']==1).sum()}")
    print(f"  Grade 2 (Severe damage): {(df_output['predicted_grade']==2).sum()}")
    print(f"  Missing cartilage:       {(df_output['cartilage_missing']==1).sum()}")

    print("\nPer-region threshold / Stage2 mode / grade distribution:")
    for region_name in REGIONS:
        rdf = df_output[df_output["region"] == region_name]
        g0 = (rdf["predicted_grade"] == 0).sum()
        g1 = (rdf["predicted_grade"] == 1).sum()
        g2 = (rdf["predicted_grade"] == 2).sum()
        thr = region_thresholds.get(region_name, 0.5)
        mode = region_stage2_modes.get(region_name, "n/a")
        print(f"  {region_name:16s}  threshold={thr:.4f}  "
              f"stage2_mode={mode:10s}  G0={g0}  G1={g1}  G2={g2}")


if __name__ == "__main__":
    main()