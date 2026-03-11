#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
inference.py
使用训练好的 SVM 模型对新的一批数据进行推理（无标签）
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
# 配置路径
# ===============================
# 输入路径（新数据）
IMAGE_FOLDER = "/mnt/sda/yx/knee/5t/data_unzipped/第二批/image_3d"
MASK_FOLDER = "/mnt/sda/yx/knee/5t/data_unzipped/第二批/mask_3d"

# 模型路径（训练时保存的）
MODEL_BASE_DIR = "/mnt/sda/yx/knee/5t/classify/results"  # 或者你训练时使用的完整路径

# 输出路径
OUTPUT_CSV = "/mnt/sda/yx/knee/5t/classify/inference_results.csv"

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
# 1. 特征提取配置（必须与训练时完全一致）
# ===============================

def get_original_extractor():
    """Original 特征提取器"""
    params = {
        "force2D": True,
        "force2Ddimension": 2,
        "binWidth": 25,
        "normalize": True,
        "normalizeScale": 100,
        "interpolator": "sitkBSpline",
        "featureClass": {
            "firstorder": [],
            "glcm": [],
            "glrlm": [],
            "glszm": [],
            "ngtdm": []
        }
    }
    return featureextractor.RadiomicsFeatureExtractor(**params)

def get_wavelet_extractor():
    """Wavelet 特征提取器"""
    params = {
        "force2D": True,
        "force2Ddimension": 2,
        "binWidth": 25,
        "normalize": True,
        "normalizeScale": 100,
        "interpolator": "sitkBSpline",
        "imageType": {
            "Original": {},
            "Wavelet": {"wavelet": "haar"}
        },
        "featureClass": {
            "firstorder": [],
            "glcm": [],
            "glrlm": [],
            "glszm": [],
            "ngtdm": []
        }
    }
    extractor = featureextractor.RadiomicsFeatureExtractor(**params)
    extractor.enableImageTypeByName("Wavelet")
    return extractor

def get_shape_extractor():
    """Shape 特征提取器"""
    params = {
        "interpolator": "sitkBSpline",
        "featureClass": {
            "shape": []
        }
    }
    return featureextractor.RadiomicsFeatureExtractor(**params)

# ===============================
# 2. 特征提取函数
# ===============================

def extract_original_features(image, mask, region_label):
    """提取 Original 特征（逐层提取后聚合）"""
    extractor = get_original_extractor()
    num_slices = image.GetSize()[2]
    slice_features = []

    for z in range(num_slices):
        img_slice = image[:, :, z]
        mask_slice = mask[:, :, z]
        mask_np = sitk.GetArrayFromImage(mask_slice)

        if np.sum(mask_np == region_label) < 20:
            continue

        try:
            result = extractor.execute(img_slice, mask_slice, label=region_label)
            feat = {}
            for k, v in result.items():
                if k.startswith("original_"):
                    feat[k] = float(v)
            slice_features.append(feat)
        except Exception as e:
            continue

    if len(slice_features) == 0:
        return None

    df_feats = pd.DataFrame(slice_features)
    feats_mean = df_feats.mean().add_suffix("_mean")
    feats_std = df_feats.std().add_suffix("_std")
    return pd.concat([feats_mean, feats_std])

def extract_wavelet_features(image, mask, region_label):
    """提取 Wavelet 特征（逐层提取后聚合）"""
    extractor = get_wavelet_extractor()
    num_slices = image.GetSize()[2]
    slice_features = []

    for z in range(num_slices):
        img_slice = image[:, :, z]
        mask_slice = mask[:, :, z]
        
        # 增加切片维度
        img_slice = sitk.JoinSeries(img_slice)
        mask_slice = sitk.JoinSeries(mask_slice)

        if np.sum(sitk.GetArrayFromImage(mask_slice) == region_label) < 20:
            continue

        try:
            result = extractor.execute(img_slice, mask_slice, label=region_label)
            feat = {}
            for k, v in result.items():
                if k.startswith("wavelet-"):
                    feat[k] = float(v)
            slice_features.append(feat)
        except Exception as e:
            continue

    if len(slice_features) == 0:
        return None

    df_feats = pd.DataFrame(slice_features)
    feats_mean = df_feats.mean().add_suffix("_mean")
    feats_std = df_feats.std().add_suffix("_std")
    return pd.concat([feats_mean, feats_std])

def extract_shape_features(image, mask, region_label):
    """提取 3D Shape 特征"""
    extractor = get_shape_extractor()
    mask_np = sitk.GetArrayFromImage(mask)

    if np.sum(mask_np == region_label) < 50:
        return None

    try:
        result = extractor.execute(image, mask, label=region_label)
        feat_dict = {}
        for k, v in result.items():
            if k.startswith("original_shape"):
                feat_dict[k] = float(v)
        return pd.Series(feat_dict)
    except Exception as e:
        print(f"Error extracting shape: {e}")
        return None

# ===============================
# 3. 完整的特征提取流程
# ===============================

def extract_all_features_for_case(image_path, mask_path, case_id):
    """为单个病例提取所有区域的全部特征"""
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
        # 提取三种特征
        orig_feats = extract_original_features(image, mask, label_id)
        wav_feats = extract_wavelet_features(image, mask, label_id)
        shape_feats = extract_shape_features(image, mask, label_id)

        # 检查是否有特征缺失
        missing_flag = 0
        if orig_feats is None or wav_feats is None:
            missing_flag = 1
            print(f"  Warning: {region_name} has insufficient slices for texture features")

        # 合并特征（使用空 Series 填充缺失）
        if orig_feats is None:
            orig_feats = pd.Series(dtype=float)
        if wav_feats is None:
            wav_feats = pd.Series(dtype=float)
        if shape_feats is None:
            shape_feats = pd.Series(dtype=float)

        # 合并所有特征
        combined = pd.concat([orig_feats, wav_feats, shape_feats])
        combined["case_id"] = case_id
        combined["region"] = region_name
        
        # 标记是否缺失
        combined["cartilage_missing"] = missing_flag
        
        all_features.append(combined)

    return all_features

# ===============================
# 4. 模型推理
# ===============================

def load_model_and_predict(region_name, features_df):
    """加载对应区域的模型并进行预测"""
    model_dir = os.path.join(MODEL_BASE_DIR, region_name, "models")
    
    # 检查模型文件是否存在
    model_path = os.path.join(model_dir, "svm_model.pkl")
    scaler_path = os.path.join(model_dir, "scaler.pkl")
    feat_list_path = os.path.join(model_dir, "feature_list.pkl")
    
    if not os.path.exists(model_path):
        print(f"  Error: Model not found for {region_name} at {model_path}")
        return None, None
    
    # 加载模型
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    feature_list = joblib.load(feat_list_path)
    
    print(f"  Loaded model for {region_name}, expecting {len(feature_list)} features")
    
    # 准备特征矩阵（只使用模型需要的特征）
    # 移除元数据列
    meta_cols = ["case_id", "region", "grade", "cartilage_missing"]
    available_cols = [c for c in features_df.columns if c not in meta_cols]
    
    # 对齐特征（确保列顺序和名称与训练时一致）
    X = pd.DataFrame(index=features_df.index)
    missing_features = []
    
    for feat in feature_list:
        if feat in features_df.columns:
            X[feat] = features_df[feat]
        else:
            X[feat] = 0  # 缺失的特征填充0
            missing_features.append(feat)
    
    if missing_features:
        print(f"  Warning: {len(missing_features)} features missing, filled with 0")
        print(f"    Missing: {missing_features[:5]}...")  # 只显示前5个
    
    # 处理 NaN
    X = X.fillna(0)
    
    # 标准化
    X_scaled = scaler.transform(X)
    
    # 预测概率和类别
    y_prob = model.predict_proba(X_scaled)[:, 1]
    y_pred = model.predict(X_scaled)
    
    return y_pred, y_prob

# ===============================
# 5. 主流程
# ===============================

def main():
    parser = argparse.ArgumentParser(description='Knee Cartilage Inference')
    parser.add_argument('--image_folder', default=IMAGE_FOLDER, help='Path to image folder')
    parser.add_argument('--mask_folder', default=MASK_FOLDER, help='Path to mask folder')
    parser.add_argument('--model_dir', default=MODEL_BASE_DIR, help='Path to trained models')
    parser.add_argument('--output', default=OUTPUT_CSV, help='Output CSV path')
    args = parser.parse_args()
    
    # global MODEL_BASE_DIR
    # MODEL_BASE_DIR = args.model_dir
    model_base_dir = args.model_dir
    
    # Step 1: 提取所有特征
    print("=" * 50)
    print("Step 1: Extracting features from new data...")
    print("=" * 50)
    
    all_case_features = []
    
    # 遍历所有图像文件
    for filename in os.listdir(args.image_folder):
        if not filename.endswith(".nii.gz"):
            continue
        
        file_prefix = filename.replace(".nii.gz", "")
        case_id = file_prefix.split('_')[0]  # 或者直接用 file_prefix 作为 case_id
        
        image_path = os.path.join(args.image_folder, filename)
        mask_path = os.path.join(args.mask_folder, file_prefix + ".nii.gz")
        
        if not os.path.exists(mask_path):
            print(f"Warning: Mask not found for {file_prefix}")
            continue
        
        case_feats = extract_all_features_for_case(image_path, mask_path, case_id)
        all_case_features.extend(case_feats)
    
    if not all_case_features:
        print("No features extracted. Please check input paths.")
        sys.exit(1)
    
    # 合并所有特征
    df_features = pd.DataFrame(all_case_features)
    
    # 确保列顺序合理
    meta_cols = ["case_id", "region", "cartilage_missing"]
    other_cols = [c for c in df_features.columns if c not in meta_cols]
    df_features = df_features[meta_cols + other_cols]
    
    print(f"\nExtracted features for {len(df_features)} region samples")
    print(f"Feature matrix shape: {df_features.shape}")
    
    # 保存原始特征（可选，用于调试）
    raw_output = args.output.replace(".csv", "_raw_features.csv")
    df_features.to_csv(raw_output, index=False)
    print(f"Raw features saved to: {raw_output}")
    
    # Step 2: 对每个区域进行推理
    print("\n" + "=" * 50)
    print("Step 2: Running inference...")
    print("=" * 50)
    
    results = []
    
    for region_name in REGION_NAMES.values():
        print(f"\nProcessing region: {region_name}")
        
        # 筛选该区域的特征
        region_df = df_features[df_features["region"] == region_name].copy()
        
        if len(region_df) == 0:
            print(f"  No samples found for {region_name}")
            continue
        
        # 加载模型并预测
        y_pred, y_prob = load_model_and_predict(region_name, region_df)
        
        if y_pred is None:
            continue
        
        # 添加预测结果
        region_df["predicted_label"] = y_pred
        region_df["probability_damage"] = y_prob
        region_df["predicted_grade"] = y_pred  # 二分类：0=正常, 1=损伤
        
        results.append(region_df)
    
    # Step 3: 合并并保存结果
    print("\n" + "=" * 50)
    print("Step 3: Saving results...")
    print("=" * 50)
    
    if not results:
        print("No predictions made. Please check model paths.")
        sys.exit(1)
    
    df_results = pd.concat(results, ignore_index=True)
    
    # 选择输出列
    output_cols = [
        "case_id", 
        "region", 
        "cartilage_missing",
        "predicted_label",
        "probability_damage",
        "predicted_grade"
    ]
    
    # 添加其他你感兴趣的列...
    df_output = df_results[output_cols].copy()
    
    # 保存结果
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    df_output.to_csv(args.output, index=False)
    
    # 同时保存详细结果（包含所有特征和预测）
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
    
    # 按区域统计
    print("\nBy Region:")
    summary = df_output.groupby("region").agg({
        "predicted_label": ["count", "sum", "mean"],
        "probability_damage": "mean"
    }).round(3)
    print(summary)

if __name__ == "__main__":
    main()