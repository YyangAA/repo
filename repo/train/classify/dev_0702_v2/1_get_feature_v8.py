#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
1_get_feature_v8.py (dev_0702)
dev_0702 版本特征提取 - 基于 dev_v8 逻辑，使用增强后的训练数据

训练数据: classify_train_data_nii_image + classify_train_data_nii_mask
训练标签: 第二批5T.xlsx
输出: knee_radiomics_features_3d_integrated.csv (已由 0_augment.py 预先生成)

本脚本在 dev_0702 中仅做路径适配，实际特征提取与 dev_v8 一致。
如果已有增强后的特征CSV，可直接跳过此步骤。
"""

import os
import pandas as pd
import numpy as np
import SimpleITK as sitk
from radiomics import featureextractor
import logging

logging.getLogger("radiomics").setLevel(logging.ERROR)

# -------------------------------
# 配置路径
# -------------------------------
REPO_DIR = "/mnt/tidal-alsh-share2/dataset/askfollow/usr/yangxin/keyan/repo-dev_v4/repo"
IMAGE_FOLDER = os.path.join(REPO_DIR, "data/classify_train_data_nii_image")
MASK_FOLDER = os.path.join(REPO_DIR, "data/classify_train_data_nii_mask")
EXCEL_FILE = os.path.join(REPO_DIR, "data/第二批5T.xlsx")
OUTPUT_CSV = "./train/classify/dev_0702/data_train/knee_radiomics_features_3d_integrated.csv"

# -------------------------------
# 区域定义
# -------------------------------
REGION_NAMES = {
    1: "Femur_Medial",
    2: "Femur_Lateral",
    3: "Tibia_Medial",
    4: "Tibia_Lateral"
}

REGION_COLUMNS = {
    "股骨内侧": "Femur_Medial",
    "股骨外侧": "Femur_Lateral",
    "胫骨内侧": "Tibia_Medial",
    "胫骨外侧": "Tibia_Lateral"
}

# -------------------------------
# 1. 读取Excel损伤分级
# -------------------------------
def read_grade_info(excel_file):
    """读取Excel中的损伤分级信息"""
    df_grade = pd.read_excel(excel_file)
    
    df_grade["name"] = df_grade["患者姓名"].astype(str).str.replace("_knee", "", regex=False).str.strip()
    
    grade_dict = {}
    for _, row in df_grade.iterrows():
        case_id = row["name"]
        grade_dict[case_id] = {}
        for col_ch, col_en in REGION_COLUMNS.items():
            grade_dict[case_id][col_en] = row[col_ch]
    
    return grade_dict

# -------------------------------
# 2. 配置PyRadiomics提取器（3D配置）
# -------------------------------
def get_3d_extractor(enable_wavelet=True):
    """创建3D特征提取器"""
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
            "shape": [],
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
    extractor.settings['force2D'] = False
    
    if enable_wavelet:
        extractor.enableImageTypeByName("Wavelet")
    
    return extractor

# -------------------------------
# 3. 特征提取核心函数
# -------------------------------
def extract_region_features_3d(image, mask, region_label, region_name,
                                extractor_orig, extractor_wav=None):
    """3D整体提取特征，输出_mean和_std格式"""
    mask_np = sitk.GetArrayFromImage(mask)
    roi_mask = (mask_np == region_label)
    
    if np.sum(roi_mask) < 20:
        return None
    
    features = {}
    
    try:
        result_orig = extractor_orig.execute(image, mask, label=region_label)
        for k, v in result_orig.items():
            if k.startswith("original_"):
                features[k + "_mean"] = float(v)
                features[k + "_std"] = 0.0
    except Exception as e:
        print(f"  Error extracting original features for {region_name}: {e}")
        return None
    
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

# -------------------------------
# 4. 批量处理
# -------------------------------
def extract_all_features(image_folder, mask_folder, grade_dict, output_csv):
    """批量提取所有病例的所有区域特征"""
    
    extractor_orig = get_3d_extractor(enable_wavelet=False)
    extractor_wav = get_3d_extractor(enable_wavelet=True)
    
    feature_template = []
    template_generated = False
    
    print("Generating feature template from first valid case...")
    
    for filename in os.listdir(image_folder):
        if not filename.endswith(".nii.gz") or template_generated:
            continue
        
        file_prefix = filename.replace(".nii.gz", "")
        case_id_for_excel = file_prefix.split('_')[0]
        
        image_path = os.path.join(image_folder, filename)
        mask_path = os.path.join(mask_folder, file_prefix + ".nii.gz")
        
        if not os.path.exists(mask_path):
            continue
        if case_id_for_excel not in grade_dict:
            continue
        
        try:
            image = sitk.ReadImage(image_path)
            mask = sitk.ReadImage(mask_path)
            mask = sitk.Cast(mask, sitk.sitkUInt8)
            mask.CopyInformation(image)
            
            for label_id, region_name in REGION_NAMES.items():
                mask_np = sitk.GetArrayFromImage(mask)
                if np.sum(mask_np == label_id) < 20:
                    continue
                
                result_template = extractor_wav.execute(image, mask, label=label_id)
                
                for k in result_template.keys():
                    if k.startswith("original_") or k.startswith("wavelet-"):
                        feature_template.append(k + "_mean")
                        feature_template.append(k + "_std")
                
                if feature_template:
                    template_generated = True
                    print(f"  Template generated from {case_id_for_excel} {region_name}")
                    print(f"  Total features in template: {len(feature_template)}")
                    break
                    
        except Exception as e:
            print(f"  Failed to generate template from {case_id_for_excel}: {e}")
            continue
    
    if not template_generated:
        print("ERROR: Could not generate feature template from any case!")
        return

    # 开始批量处理
    all_features = []
    
    for filename in os.listdir(image_folder):
        if not filename.endswith(".nii.gz"):
            continue
        
        file_prefix = filename.replace(".nii.gz", "")
        case_id_for_excel = file_prefix.split('_')[0]
        
        image_path = os.path.join(image_folder, filename)
        mask_path = os.path.join(mask_folder, file_prefix + ".nii.gz")
        
        if not os.path.exists(mask_path):
            print(f"Warning: Mask not found for {file_prefix}")
            continue
        
        if case_id_for_excel not in grade_dict:
            print(f"Warning: Grade info not found for {case_id_for_excel}")
            continue
        
        print(f"Processing: {file_prefix} ...")
        
        try:
            image = sitk.ReadImage(image_path)
            mask = sitk.ReadImage(mask_path)
            mask = sitk.Cast(mask, sitk.sitkUInt8)
            mask.CopyInformation(image)
        except Exception as e:
            print(f"Error reading {file_prefix}: {e}")
            continue
        
        for label_id, region_name in REGION_NAMES.items():
            feats = extract_region_features_3d(
                image, mask, label_id, region_name,
                extractor_orig, extractor_wav
            )
            
            if feats is None:
                feats = pd.Series(
                    data=[np.nan] * len(feature_template),
                    index=feature_template
                )
                feats["cartilage_missing"] = 1
            else:
                feats["cartilage_missing"] = 0
            
            feats["case_id"] = case_id_for_excel
            feats["region"] = region_name
            feats["grade"] = grade_dict[case_id_for_excel][region_name]
            
            meta_cols = ["case_id", "region", "grade", "cartilage_missing"]
            other_cols = [c for c in feats.index if c not in meta_cols]
            feats = feats[meta_cols + other_cols]
            
            all_features.append(feats)
    
    # 保存结果
    if all_features:
        df_final = pd.DataFrame(all_features)
        os.makedirs(os.path.dirname(output_csv), exist_ok=True)
        df_final.to_csv(output_csv, index=False)
        print(f"\nFinished! Total {len(df_final)} rows saved to {output_csv}")
        print(f"Feature matrix shape: {df_final.shape}")
    else:
        print("No features extracted!")


if __name__ == "__main__":
    # dev_0702: 检查是否已有增强特征CSV
    if os.path.exists(OUTPUT_CSV):
        df_existing = pd.read_csv(OUTPUT_CSV)
        # 检查是否有增强样本 (case_id 包含 _aug)
        has_aug = df_existing['case_id'].str.contains('_aug', na=False).any()
        if has_aug:
            print(f"[dev_0702] Augmented features already exist: {OUTPUT_CSV}")
            print(f"  Total rows: {len(df_existing)}, cases: {df_existing['case_id'].nunique()}")
            print("  Skipping feature extraction (using pre-augmented data).")
            import sys
            sys.exit(0)
    
    # 否则执行原始特征提取
    grade_dict = read_grade_info(EXCEL_FILE)
    extract_all_features(IMAGE_FOLDER, MASK_FOLDER, grade_dict, OUTPUT_CSV)
