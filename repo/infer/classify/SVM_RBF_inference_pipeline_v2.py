#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SVM_RBF_inference_pipeline_v2.py
适配 dev_v2 训练方式的推理脚本：
  - 使用 3D 整体特征提取（与 get_feature_v2.py 完全一致）
  - 输出格式：feature_mean = 3D提取值, feature_std = 0
  - 不提取 shape 特征
  - 小波使用 3D 分解（8个方向：HHH, HHL, HLH, HLL, LHH, LHL, LLH, LLL）
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
MODEL_BASE_DIR = "./checkpoint/results"
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
    - 不使用 shape 特征
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

    # 提取 Original 特征
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
# 3. 模型推理
# ===============================

def load_model_and_predict(region_name, features_df, model_base_dir):
    """加载对应区域的模型并进行预测"""
    model_dir = os.path.join(model_base_dir, region_name, "models")

    model_path = os.path.join(model_dir, "svm_model.pkl")
    scaler_path = os.path.join(model_dir, "scaler.pkl")
    feat_list_path = os.path.join(model_dir, "feature_list.pkl")

    if not os.path.exists(model_path):
        print(f"  Error: Model not found for {region_name} at {model_path}")
        return None, None

    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    feature_list = joblib.load(feat_list_path)

    print(f"  Loaded model for {region_name}, expecting {len(feature_list)} features")

    # 对齐特征（确保列顺序和名称与训练时一致）
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
    # y_pred = model.predict(X_scaled)

    # 注意：不使用 model.predict()，因为 SVM 的 predict() 基于决策超平面，
    # 其阈值和 predict_proba() 的 0.5 不一定对齐，会导致概率>0.5却预测为Normal的矛盾。
    # 统一使用概率阈值 0.5 来判断标签，保证预测标签和概率的一致性。
    y_pred = (y_prob >= 0.5).astype(int)

    return y_pred, y_prob


# ===============================
# 4. 主流程
# ===============================

def main():
    parser = argparse.ArgumentParser(description='Knee Cartilage Inference (v2 - 3D features)')
    parser.add_argument('--image_folder', default=IMAGE_FOLDER, help='Path to 3D image folder')
    parser.add_argument('--mask_folder', default=MASK_FOLDER, help='Path to 3D mask folder')
    parser.add_argument('--model_dir', default=MODEL_BASE_DIR, help='Path to trained models')
    parser.add_argument('--output', default=OUTPUT_CSV, help='Output CSV path')
    args = parser.parse_args()

    # 初始化 3D 提取器（只创建一次，复用）
    print("Initializing 3D feature extractors...")
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

    # 保存原始特征（用于调试）
    raw_output = args.output.replace(".csv", "_raw_features.csv")
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    df_features.to_csv(raw_output, index=False)
    print(f"Raw features saved to: {raw_output}")

    # Step 2: 对每个区域进行推理
    print("\n" + "=" * 50)
    print("Step 2: Running inference...")
    print("=" * 50)

    results = []

    for region_name in REGION_NAMES.values():
        print(f"\nProcessing region: {region_name}")

        region_df = df_features[df_features["region"] == region_name].copy()

        if len(region_df) == 0:
            print(f"  No samples found for {region_name}")
            continue

        y_pred, y_prob = load_model_and_predict(region_name, region_df, args.model_dir)

        if y_pred is None:
            continue

        region_df["predicted_label"] = y_pred
        region_df["probability_damage"] = y_prob
        region_df["predicted_grade"] = y_pred

        results.append(region_df)

    # Step 3: 合并并保存结果
    print("\n" + "=" * 50)
    print("Step 3: Saving results...")
    print("=" * 50)

    if not results:
        print("No predictions made. Please check model paths.")
        sys.exit(1)

    df_results = pd.concat(results, ignore_index=True)

    output_cols = [
        "case_id",
        "region",
        "cartilage_missing",
        "predicted_label",
        "probability_damage",
        "predicted_grade"
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
    print(f"Total samples: {len(df_output)}")
    print(f"Predicted damaged (grade>0): {(df_output['predicted_label']==1).sum()}")
    print(f"Predicted normal (grade=0): {(df_output['predicted_label']==0).sum()}")
    print(f"Missing cartilage regions: {(df_output['cartilage_missing']==1).sum()}")

    print("\nBy Region:")
    summary = df_output.groupby("region").agg({
        "predicted_label": ["count", "sum", "mean"],
        "probability_damage": "mean"
    }).round(3)
    print(summary)


if __name__ == "__main__":
    main()