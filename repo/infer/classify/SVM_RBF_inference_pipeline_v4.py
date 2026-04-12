#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SVM_RBF_inference_pipeline_v2.py
适配 dev_v2 训练方式的推理脚本（级联二分类版本）：
  - Stage 1: SVM 二分类 (Normal vs Damaged) — 使用 radiomics 特征
    · 使用训练时 CV 计算的最优阈值 (Youden's J)，而非固定 0.5
  - Stage 2: SVM 分级 (Grade 1 vs Grade 2) — 使用 LASSO 筛选特征 + shape 特征
  - 3D 整体特征提取（与 get_feature_v2.py 完全一致）
  - 启用 shape 特征提取（VoxelVolume / SurfaceArea / MeshVolume 等）
"""

import os
import sys
import pandas as pd
import numpy as np
import SimpleITK as sitk
from radiomics import featureextractor
import joblib
import logging
import argparse

logging.getLogger("radiomics").setLevel(logging.ERROR)

# ===============================
# 配置路径（默认值，可通过命令行参数覆盖）
# ===============================
IMAGE_FOLDER = "./data/image_3d"
MASK_FOLDER = "./data/mask_3d"
MODEL_BASE_DIR = "./checkpoint/results_260410"
OUTPUT_CSV = "./data/inference_results.csv"

# ===============================
# 区域定义
# ===============================
REGION_NAMES = {
    1: "Femur_Medial",
    2: "Femur_Lateral",
    3: "Tibia_Medial",
    4: "Tibia_Lateral"
}

# ===============================
# 1. 特征提取器配置（与 dev_v2/get_feature_v2.py 完全一致）
# ===============================

def get_3d_extractor(enable_wavelet=True):
    """
    创建 3D 特征提取器，与训练时 get_feature_v2.py 完全对齐：
    - 3D 模式 (force2D=False)
    - 体素重采样 resampledPixelSpacing=[1,1,1]
    - 启用 shape 特征（用于 Stage 2 分级）
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
            "shape": [],      # 启用 shape 特征（VoxelVolume、SurfaceArea 等）
            "firstorder": [],
            "glcm": [],
            "glrlm": [],
            "glszm": [],
            "ngtdm": []
        }
    }

    if enable_wavelet:
        params["imageType"]["Wavelet"] = {"wavelet": "haar"}

    extractor = featureextractor.RadiomicsFeatureExtractor(**params)
    extractor.settings['force2D'] = False  # 3D 模式

    if enable_wavelet:
        extractor.enableImageTypeByName("Wavelet")

    return extractor


# ===============================
# 2. 特征提取（3D 整体，与训练完全一致）
# ===============================

def extract_region_features_3d(image, mask, region_label, region_name,
                                extractor_orig, extractor_wav=None):
    """
    3D 整体提取特征，输出 _mean/_std 格式（与 get_feature_v2.py 一致）
    - _mean = 3D 提取的原始值
    - _std = 0（因为是 3D 整体提取，无跨层聚合）
    """
    mask_np = sitk.GetArrayFromImage(mask)
    roi_mask = (mask_np == region_label)

    if np.sum(roi_mask) < 20:
        return None

    features = {}

    # 提取 Original 特征（包括 shape）
    try:
        result_orig = extractor_orig.execute(image, mask, label=region_label)
        for k, v in result_orig.items():
            if k.startswith("original_"):
                features[k + "_mean"] = float(v)
                features[k + "_std"] = 0.0
    except Exception as e:
        print(f"  Error extracting original features for {region_name}: {e}")
        return None

    # 提取 Wavelet 特征
    if extractor_wav is not None:
        try:
            result_wav = extractor_wav.execute(image, mask, label=region_label)
            for k, v in result_wav.items():
                if k.startswith("wavelet-"):
                    features[k + "_mean"] = float(v)
                    features[k + "_std"] = 0.0
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
            extractor_orig, extractor_wav
        )

        missing_flag = 0
        if feats is None:
            missing_flag = 1
            feats = pd.Series(dtype=float)
            print(f"  Warning: {region_name} has insufficient ROI for feature extraction")

        feats["case_id"] = case_id
        feats["region"] = region_name
        feats["cartilage_missing"] = missing_flag

        all_features.append(feats)

    return all_features


# ===============================
# 3. Stage 1: 二分类模型推理 (Normal vs Damaged)
# ===============================

def load_model_and_predict_stage1(region_name, features_df, model_base_dir):
    """
    加载 Stage 1 模型并进行二分类预测。
    使用训练时 CV 计算的最优阈值 (Youden's J)，而非固定 0.5。
    """
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

    # 加载最优阈值，若不存在则回退到 0.5
    if os.path.exists(threshold_path):
        threshold = float(joblib.load(threshold_path))
        print(f"  [Stage 1] Loaded model for {region_name}, "
              f"{len(feature_list)} features, threshold={threshold:.4f}")
    else:
        threshold = 0.5
        print(f"  [Stage 1] Loaded model for {region_name}, "
              f"{len(feature_list)} features, threshold=0.5 (default, no threshold.pkl found)")

    # 对齐特征
    X = pd.DataFrame(index=features_df.index)
    missing_features = []

    for feat in feature_list:
        if feat in features_df.columns:
            X[feat] = features_df[feat]
        else:
            X[feat] = 0
            missing_features.append(feat)

    if missing_features:
        print(f"  Warning: {len(missing_features)} features missing, filled with 0")
        if len(missing_features) <= 10:
            print(f"    Missing: {missing_features}")
        else:
            print(f"    Missing (first 5): {missing_features[:5]}...")

    X = X.fillna(0)
    X_scaled = scaler.transform(X)

    y_prob = model.predict_proba(X_scaled)[:, 1]
    y_pred = (y_prob >= threshold).astype(int)

    return y_pred, y_prob, threshold


# ===============================
# 4. Stage 2: 分级模型推理 (Grade 1 vs Grade 2)
# ===============================

def load_model_and_predict_stage2(region_name, features_df, model_base_dir):
    """
    加载 Stage 2 模型并对 Damaged 样本进行分级预测。
    
    返回:
        y_pred: 0 = Grade 1, 1 = Grade 2
        y_prob: Grade 2 的概率
        如果模型不存在返回 None, None
    """
    model_dir = os.path.join(model_base_dir, region_name, "models")

    model_path = os.path.join(model_dir, "svm_model_stage2.pkl")
    scaler_path = os.path.join(model_dir, "scaler_stage2.pkl")
    feat_list_path = os.path.join(model_dir, "feature_list_stage2.pkl")

    if not os.path.exists(model_path):
        print(f"  [Stage 2] Model not found for {region_name}, defaulting to Grade 1")
        return None, None

    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    feature_list = joblib.load(feat_list_path)

    print(f"  [Stage 2] Loaded model for {region_name}, expecting {len(feature_list)} features")
    print(f"    Features: {feature_list}")

    # 对齐特征
    X = pd.DataFrame(index=features_df.index)
    missing_features = []

    for feat in feature_list:
        if feat in features_df.columns:
            X[feat] = features_df[feat]
        else:
            X[feat] = 0
            missing_features.append(feat)

    if missing_features:
        print(f"  Warning: {len(missing_features)} Stage 2 features missing, filled with 0")
        for mf in missing_features:
            print(f"    Missing: {mf}")

    X = X.fillna(0)
    X_scaled = scaler.transform(X)

    y_prob = model.predict_proba(X_scaled)[:, 1]  # Grade 2 的概率
    y_pred = (y_prob >= 0.5).astype(int)  # 0=Grade1, 1=Grade2

    return y_pred, y_prob


# ===============================
# 5. 主流程
# ===============================

def main():
    parser = argparse.ArgumentParser(description='Knee Cartilage Inference (v2 - Cascaded Classification)')
    parser.add_argument('--image_folder', default=IMAGE_FOLDER, help='Path to 3D image folder')
    parser.add_argument('--mask_folder', default=MASK_FOLDER, help='Path to 3D mask folder')
    parser.add_argument('--model_dir', default=MODEL_BASE_DIR, help='Path to trained models')
    parser.add_argument('--output', default=OUTPUT_CSV, help='Output CSV path')
    args = parser.parse_args()

    # 初始化 3D 提取器（启用 shape 特征）
    print("Initializing 3D feature extractors (with shape features)...")
    extractor_orig = get_3d_extractor(enable_wavelet=False)
    extractor_wav = get_3d_extractor(enable_wavelet=True)

    # Step 1: 提取所有特征
    print("\n" + "=" * 50)
    print("Step 1: Extracting 3D features from new data...")
    print("=" * 50)

    all_case_features = []

    for filename in sorted(os.listdir(args.image_folder)):
        if not filename.endswith(".nii.gz"):
            continue

        file_prefix = filename.replace(".nii.gz", "")
        case_id = file_prefix.split('_')[0]

        image_path = os.path.join(args.image_folder, filename)
        mask_path = os.path.join(args.mask_folder, file_prefix + ".nii.gz")

        if not os.path.exists(mask_path):
            print(f"Warning: Mask not found for {file_prefix}")
            continue

        case_feats = extract_all_features_for_case(
            image_path, mask_path, case_id,
            extractor_orig, extractor_wav
        )
        all_case_features.extend(case_feats)

    if not all_case_features:
        print("No features extracted. Please check input paths.")
        sys.exit(1)

    df_features = pd.DataFrame(all_case_features)

    meta_cols = ["case_id", "region", "cartilage_missing"]
    other_cols = [c for c in df_features.columns if c not in meta_cols]
    df_features = df_features[meta_cols + other_cols]

    print(f"\nExtracted features for {len(df_features)} region samples")
    print(f"Feature matrix shape: {df_features.shape}")

    # 检查 shape 特征是否提取成功
    shape_cols = [c for c in df_features.columns if 'shape' in c.lower()]
    print(f"Shape features extracted: {len(shape_cols)}")

    # 保存原始特征（用于调试）
    raw_output = args.output.replace(".csv", "_raw_features.csv")
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    df_features.to_csv(raw_output, index=False)
    print(f"Raw features saved to: {raw_output}")

    # Step 2: Stage 1 — SVM 二分类 (Normal vs Damaged)
    print("\n" + "=" * 50)
    print("Step 2: Stage 1 — SVM inference (Normal vs Damaged)...")
    print("=" * 50)

    results = []
    region_thresholds = {}  # 记录每个区域使用的阈值

    for region_name in REGION_NAMES.values():
        print(f"\nProcessing region: {region_name}")

        region_df = df_features[df_features["region"] == region_name].copy()

        if len(region_df) == 0:
            print(f"  No samples found for {region_name}")
            continue

        y_pred, y_prob, threshold = load_model_and_predict_stage1(region_name, region_df, args.model_dir)

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

    # Step 3: Stage 2 — SVM 分级 (Grade 1 vs Grade 2)
    print("\n" + "=" * 50)
    print("Step 3: Stage 2 — Grade classification (Grade 1 vs 2)...")
    print("=" * 50)

    predicted_grades = []
    grade2_probs = []
    grade_reasons = []

    for region_name in REGION_NAMES.values():
        print(f"\nProcessing region: {region_name}")

        # 获取该区域被判定为 Damaged 的样本索引
        region_mask = df_results["region"] == region_name
        damaged_mask = region_mask & (df_results["predicted_label"] == 1)

        damaged_indices = df_results[damaged_mask].index.tolist()
        n_damaged = len(damaged_indices)

        if n_damaged == 0:
            print(f"  No damaged samples for Stage 2")
            continue

        print(f"  {n_damaged} damaged samples to grade")

        # 加载 Stage 2 模型并预测
        damaged_df = df_results.loc[damaged_indices]
        s2_pred, s2_prob = load_model_and_predict_stage2(region_name, damaged_df, args.model_dir)

        if s2_pred is not None:
            for i, idx in enumerate(damaged_indices):
                grade = 2 if s2_pred[i] == 1 else 1
                prob_g2 = s2_prob[i]
                predicted_grades.append((idx, grade))
                grade2_probs.append((idx, prob_g2))
                reason = f"stage2_prob_g2={prob_g2:.3f}"
                grade_reasons.append((idx, reason))
                print(f"    {df_results.loc[idx, 'case_id']}: "
                      f"Grade {grade} (prob_G2={prob_g2:.3f})")
        else:
            # 没有 Stage 2 模型，所有 Damaged 默认 Grade 1
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

    # Step 4: 保存结果
    print("\n" + "=" * 50)
    print("Step 4: Saving results...")
    print("=" * 50)

    output_cols = [
        "case_id",
        "region",
        "cartilage_missing",
        "predicted_label",       # 0=Normal, 1=Damaged (Stage 1 SVM 二分类)
        "probability_damage",    # Stage 1 SVM 损伤概率
        "threshold_used",        # 使用的最优阈值 (Youden's J)
        "predicted_grade",       # 0=Normal, 1=Grade1(轻度), 2=Grade2(严重)
        "probability_grade2",    # Stage 2 SVM Grade 2 概率
        "grade_reason",          # 分级依据
    ]

    df_output = df_results[output_cols].copy()
    df_output.to_csv(args.output, index=False)

    detailed_output = args.output.replace(".csv", "_detailed.csv")
    df_results.to_csv(detailed_output, index=False)

    print(f"\nResults saved:")
    print(f"  Summary: {args.output}")
    print(f"  Detailed: {detailed_output}")

    # 打印统计信息
    print("\n" + "=" * 50)
    print("Inference Summary:")
    print("=" * 50)
    print(f"Total region samples: {len(df_output)}")
    print(f"  Grade 0 (Normal):        {(df_output['predicted_grade']==0).sum()}")
    print(f"  Grade 1 (Mild damage):   {(df_output['predicted_grade']==1).sum()}")
    print(f"  Grade 2 (Severe damage): {(df_output['predicted_grade']==2).sum()}")
    print(f"  Missing cartilage:       {(df_output['cartilage_missing']==1).sum()}")

    print("\nThreshold & Grade Distribution per Region:")
    for region_name in REGION_NAMES.values():
        rdf = df_output[df_output["region"] == region_name]
        g0 = (rdf["predicted_grade"] == 0).sum()
        g1 = (rdf["predicted_grade"] == 1).sum()
        g2 = (rdf["predicted_grade"] == 2).sum()
        thr = region_thresholds.get(region_name, 0.5)
        print(f"  {region_name:16s}  threshold={thr:.4f}  Grade0={g0}  Grade1={g1}  Grade2={g2}")


if __name__ == "__main__":
    main()