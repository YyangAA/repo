#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
integrated_radiomics_extractor.py
整合版本：使用新代码的3D提取逻辑，输出旧代码的格式（mean/std聚合 + 无shape特征）
修复：dummy mask 创建问题（PyRadiomics 3.0+ 全1 mask bug）
"""

import os
import pandas as pd
import numpy as np
import SimpleITK as sitk
from radiomics import featureextractor
import logging

logging.getLogger("radiomics").setLevel(logging.ERROR)

# -------------------------------
# 配置路径（根据你的实际情况修改）
# -------------------------------
IMAGE_FOLDER = "/mnt/sda/yx/knee/5t/data_unzipped/第二批/image_3d"
MASK_FOLDER = "/mnt/sda/yx/knee/5t/data_unzipped/第二批/mask_3d"
EXCEL_FILE = "/mnt/sda/yx/knee/5t/classify/影像组学代码/第二批5T.xlsx"
OUTPUT_CSV = "/mnt/sda/yx/knee/5t/classify/knee_radiomics_features_3d_integrated.csv"

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
# 1. 读取Excel损伤分级（兼容旧代码逻辑）
# -------------------------------
def read_grade_info(excel_file):
    """读取Excel中的损伤分级信息"""
    df_grade = pd.read_excel(excel_file)
    
    # 统一处理姓名，去除_knee后缀和空格（兼容旧代码）
    df_grade["name"] = df_grade["患者姓名"].astype(str).str.replace("_knee", "", regex=False).str.strip()
    
    grade_dict = {}
    for _, row in df_grade.iterrows():
        case_id = row["name"]
        grade_dict[case_id] = {}
        for col_ch, col_en in REGION_COLUMNS.items():
            grade_dict[case_id][col_en] = row[col_ch]
    
    return grade_dict

# -------------------------------
# 2. 配置PyRadiomics提取器（新代码的3D配置）
# -------------------------------
def get_3d_extractor(enable_wavelet=True):
    """
    创建3D特征提取器
    使用新代码的配置：3D模式 + 体素重采样
    """
    params = {
        "binWidth": 25,
        "normalize": True,
        "normalizeScale": 100,
        "interpolator": "sitkBSpline",
        "resampledPixelSpacing": [1, 1, 1],  # 新代码的3D重采样
        
        "imageType": {
            "Original": {},
        },
        
        "featureClass": {
            # "shape": [],  # ← 注释掉：与旧代码训练保持一致，不使用shape
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
    extractor.settings['force2D'] = False  # 3D模式
    
    if enable_wavelet:
        extractor.enableImageTypeByName("Wavelet")
    
    return extractor

# -------------------------------
# 3. 特征提取核心函数（新代码的3D逻辑 + 旧代码的输出格式）
# -------------------------------
def extract_region_features_3d(image, mask, region_label, region_name, 
                                extractor_orig, extractor_wav=None):
    """
    3D整体提取特征，但输出mean/std格式（兼容旧代码）
    
    关键适配：
    - 使用3D整体提取（新代码逻辑，更稳定）
    - 但输出_mean和_std两列（旧代码格式）
    - _mean = 提取的原始值
    - _std = 0（因为是3D整体，无跨层变异）
    """
    mask_np = sitk.GetArrayFromImage(mask)
    roi_mask = (mask_np == region_label)
    
    # 检查ROI是否存在
    if np.sum(roi_mask) < 20:
        return None
    
    features = {}
    
    # 3.1 提取Original特征
    try:
        result_orig = extractor_orig.execute(image, mask, label=region_label)
        
        for k, v in result_orig.items():
            if k.startswith("original_"):
                # 适配旧格式：生成 _mean 和 _std
                features[k + "_mean"] = float(v)
                features[k + "_std"] = 0.0  # 3D整体提取，std设为0
    except Exception as e:
        print(f"  Error extracting original features for {region_name}: {e}")
        return None
    
    # 3.2 提取Wavelet特征（如果启用）
    if extractor_wav is not None:
        try:
            result_wav = extractor_wav.execute(image, mask, label=region_label)
            
            for k, v in result_wav.items():
                if k.startswith("wavelet-"):
                    # 适配旧格式
                    features[k + "_mean"] = float(v)
                    features[k + "_std"] = 0.0
        except Exception as e:
            print(f"  Error extracting wavelet features for {region_name}: {e}")
    
    return pd.Series(features)

# -------------------------------
# 4. 批量处理（兼容旧代码的文件名逻辑）
# -------------------------------
def extract_all_features(image_folder, mask_folder, grade_dict, output_csv):
    """批量提取所有病例的所有区域特征"""
    
    # 初始化提取器
    extractor_orig = get_3d_extractor(enable_wavelet=False)
    extractor_wav = get_3d_extractor(enable_wavelet=True)
    
    # ===============================
    # 修复：获取特征模板的方式
    # 原方法：使用 dummy image/mask 会导致 PyRadiomics 3.0+ 的 bug
    # 新方法：使用第一个实际病例的第一个有效区域来生成模板
    # ===============================
    
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
            
            # 尝试每个区域，直到找到一个有效的
            for label_id, region_name in REGION_NAMES.items():
                mask_np = sitk.GetArrayFromImage(mask)
                if np.sum(mask_np == label_id) < 20:
                    continue
                
                # 提取特征作为模板
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
        print("Trying fallback method...")
        feature_template = build_fallback_template()

    # 开始批量处理
    all_features = []
    
    for filename in os.listdir(image_folder):
        if not filename.endswith(".nii.gz"):
            continue
        
        # 解析文件名（兼容旧代码：常会儒_0_0000.nii.gz → case_id=常会儒）
        file_prefix = filename.replace(".nii.gz", "")
        case_id_for_excel = file_prefix.split('_')[0]
        
        # 构建路径
        image_path = os.path.join(image_folder, filename)
        mask_path = os.path.join(mask_folder, file_prefix + ".nii.gz")
        
        # 检查文件和分级
        if not os.path.exists(mask_path):
            print(f"Warning: Mask not found for {file_prefix}")
            continue
        
        if case_id_for_excel not in grade_dict:
            print(f"Warning: Grade info not found for {case_id_for_excel}")
            continue
        
        print(f"Processing: {file_prefix} ...")
        
        # 读取图像
        try:
            image = sitk.ReadImage(image_path)
            mask = sitk.ReadImage(mask_path)
            mask = sitk.Cast(mask, sitk.sitkUInt8)
            mask.CopyInformation(image)
        except Exception as e:
            print(f"Error reading {file_prefix}: {e}")
            continue
        
        # 处理每个区域
        for label_id, region_name in REGION_NAMES.items():
            feats = extract_region_features_3d(
                image, mask, label_id, region_name,
                extractor_orig, extractor_wav
            )
            
            if feats is None:
                # 缺失处理：用NaN填充
                feats = pd.Series(
                    data=[np.nan] * len(feature_template),
                    index=feature_template
                )
                feats["cartilage_missing"] = 1
            else:
                feats["cartilage_missing"] = 0
            
            # 添加元数据
            feats["case_id"] = case_id_for_excel
            feats["region"] = region_name
            feats["grade"] = grade_dict[case_id_for_excel][region_name]
            
            # 列排序（与旧代码一致）
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
        
        # 统计信息
        orig_feats = [c for c in df_final.columns if c.startswith("original_")]
        wav_feats = [c for c in df_final.columns if c.startswith("wavelet-")]
        print(f"  Original features: {len(orig_feats)}")
        print(f"  Wavelet features: {len(wav_feats)}")
    else:
        print("No features extracted!")

def build_fallback_template():
    """
    备用方案：如果无法从实际数据生成模板，手动构建特征名列表
    基于 PyRadiomics 默认启用的特征类
    """
    feature_classes = {
        'firstorder': ['Mean', 'Median', 'Std', 'Skewness', 'Kurtosis', 'Minimum', 'Maximum', 
                      'Entropy', 'Energy', 'TotalEnergy', 'Uniformity'],
        'glcm': ['Autocorrelation', 'JointAverage', 'ClusterProminence', 'ClusterShade', 
                'ClusterTendency', 'Contrast', 'Correlation', 'DifferenceAverage',
                'DifferenceEntropy', 'DifferenceVariance', 'Dissimilarity', 'JointEnergy',
                'JointEntropy', 'IMC1', 'IMC2', 'Id', 'Idm', 'Idmn', 'Idn', 'InverseVariance',
                'MaximumProbability', 'SumAverage', 'SumEntropy', 'SumSquares'],
        'glrlm': ['ShortRunEmphasis', 'LongRunEmphasis', 'GreyLevelNonUniformity',
                 'RunLengthNonUniformity', 'RunPercentage', 'GreyLevelVariance',
                 'RunVariance', 'RunEntropy', 'LowGreyLevelRunEmphasis',
                 'HighGreyLevelRunEmphasis', 'ShortRunLowGreyLevelEmphasis',
                 'ShortRunHighGreyLevelEmphasis', 'LongRunLowGreyLevelEmphasis',
                 'LongRunHighGreyLevelEmphasis'],
        'glszm': ['SmallAreaEmphasis', 'LargeAreaEmphasis', 'GreyLevelNonUniformity',
                 'SizeZoneNonUniformity', 'ZonePercentage', 'GreyLevelVariance',
                 'ZoneVariance', 'ZoneEntropy', 'LowGreyLevelZoneEmphasis',
                 'HighGreyLevelZoneEmphasis', 'SmallAreaLowGreyLevelEmphasis',
                 'SmallAreaHighGreyLevelEmphasis', 'LargeAreaLowGreyLevelEmphasis',
                 'LargeAreaHighGreyLevelEmphasis'],
        'ngtdm': ['Coarseness', 'Contrast', 'Busyness', 'Complexity', 'Strength']
    }
    
    template = []
    
    # Original 特征
    for cls, feats in feature_classes.items():
        for feat in feats:
            template.append(f'original_{cls}_{feat}_mean')
            template.append(f'original_{cls}_{feat}_std')
    
    # Wavelet 特征（8个分解 + 近似）
    wavelet_types = ['HHH', 'HHL', 'HLH', 'HLL', 'LHH', 'LHL', 'LLH', 'LLL']
    for wtype in wavelet_types:
        for cls, feats in feature_classes.items():
            for feat in feats:
                template.append(f'wavelet-{wtype}_{cls}_{feat}_mean')
                template.append(f'wavelet-{wtype}_{cls}_{feat}_std')
    
    print(f"Fallback template built with {len(template)} features")
    return template

# -------------------------------
# 5. 主函数
# -------------------------------
if __name__ == "__main__":
    # 读取分级信息
    grade_dict = read_grade_info(EXCEL_FILE)
    print(f"Loaded grade info for {len(grade_dict)} cases")
    
    # 提取特征
    extract_all_features(IMAGE_FOLDER, MASK_FOLDER, grade_dict, OUTPUT_CSV)
    
    print("\nAll done!")