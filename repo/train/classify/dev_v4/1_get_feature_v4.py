#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
integrated_radiomics_extractor.py
整合版本：使用3D整体提取逻辑，直接输出特征原名（无 _mean/_std 后缀）
修复：dummy mask 创建问题（PyRadiomics 3.0+ 全1 mask bug）
v3:
  - 去除冗余的 _std=0 列，特征维度减半
  - Wavelet 保留沿 Z(D) 方向是 L 的 4 个方向（LLL, LHL, HLL, HHL）
    去掉沿 Z 高频的 4 个不稳定方向（LLH, LHH, HLH, HHH）
    原因：数据 D=3，沿 Z 高频统计不稳定；但沿 X/Y(832) 高频是稳定的，应保留
  - 多进程并行提取
"""

# Wavelet 方向过滤规则：
# PyRadiomics 3D Wavelet 命名 = wavelet-XYZ_<feature_class>_<feature>
# X/Y/Z 各位为 L(低通) 或 H(高通)，第 3 位对应 Z(深度) 方向
# 数据 D=3，Z 方向高频不稳定，所以只保留第 3 位为 L 的 4 个方向
WAVELET_KEEP_DIRECTIONS = {"LLL", "LHL", "HLL", "HHL"}

import os
import pandas as pd
import numpy as np
import SimpleITK as sitk
from radiomics import featureextractor
import logging
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import cpu_count

try:
    from tqdm import tqdm
except ImportError:
    # 如果未安装 tqdm，用简单替代
    def tqdm(iterable, **kwargs):
        return iterable

logging.getLogger("radiomics").setLevel(logging.ERROR)

# -------------------------------
# 配置路径（根据你的实际情况修改）
# -------------------------------
IMAGE_FOLDER = "/mnt/sda/yx/knee/5t/data_unzipped/第二批/image_3d"
MASK_FOLDER = "/mnt/sda/yx/knee/5t/data_unzipped/第二批/mask_3d"
EXCEL_FILE = "/mnt/sda/yx/knee/5t/classify/影像组学代码/第二批5T.xlsx"
OUTPUT_CSV = "./train/classify/dev_v4/data_train/knee_radiomics_features_3d_integrated.csv"

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
    恢复 PyRadiomics 自带 normalize（全局归一化），经验证更稳定
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
            "shape": [],  # 启用 shape 特征（VoxelVolume, SurfaceArea 等）
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
# 3. 特征提取核心函数（3D整体提取，只保留有效特征）
# -------------------------------
def extract_region_features_3d(image, mask, region_label, region_name,
                                extractor_orig, extractor_wav=None):
    """
    3D整体提取特征。
    
    优化：
    - 去掉冗余的 _std 列（3D整体提取只有一个值，std恒为0）
    - Wavelet 保留沿 Z 方向是 L 的 4 个方向（LLL, LHL, HLL, HHL），去掉沿 Z 高频的 4 个不稳定方向
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
                features[k + "_mean"] = float(v)
    except Exception as e:
        print(f"  Error extracting original features for {region_name}: {e}")
        return None
    
    # 3.2 提取Wavelet特征（保留 Z 方向为 L 的 4 个方向：LLL, LHL, HLL, HHL）
    if extractor_wav is not None:
        try:
            result_wav = extractor_wav.execute(image, mask, label=region_label)
            
            for k, v in result_wav.items():
                if not k.startswith("wavelet-"):
                    continue
                # 解析方向标签：wavelet-XYZ_<class>_<feat>
                direction = k.split("_", 1)[0].replace("wavelet-", "")
                # 只保留 3 位且第 3 位为 L 的方向（沿 Z 低通，避免 D=3 时高频不稳定）
                if len(direction) == 3 and direction[2] == "L" and direction in WAVELET_KEEP_DIRECTIONS:
                    features[k + "_mean"] = float(v)
        except Exception as e:
            print(f"  Error extracting wavelet features for {region_name}: {e}")
    
    return pd.Series(features)

# -------------------------------
# 4. 单病例处理（用于多进程 worker）
# -------------------------------
def process_single_case(args):
    """
    单个病例的特征提取（在子进程中运行）。
    每个 worker 独立创建 extractor，避免跨进程共享问题。
    """
    image_path, mask_path, case_id_for_excel, feature_template, grade_dict = args
    
    # 每个 worker 独立创建提取器
    extractor_orig = get_3d_extractor(enable_wavelet=False)
    extractor_wav = get_3d_extractor(enable_wavelet=True)
    
    results = []
    
    try:
        image = sitk.ReadImage(image_path)
        mask = sitk.ReadImage(mask_path)
        mask = sitk.Cast(mask, sitk.sitkUInt8)
        mask.CopyInformation(image)
    except Exception as e:
        print(f"Error reading {case_id_for_excel}: {e}")
        return results
    
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
        
        results.append(feats)
    
    return results


# -------------------------------
# 5. 批量处理（多进程并行）
# -------------------------------
def extract_all_features(image_folder, mask_folder, grade_dict, output_csv, n_workers=None):
    """
    批量提取所有病例的所有区域特征（多进程并行）。
    
    Args:
        n_workers: 并行 worker 数量，默认 CPU 核心数 - 1
    """
    if n_workers is None:
        n_workers = max(1, cpu_count() - 1)
    
    # 初始化提取器（仅用于模板生成）
    extractor_orig = get_3d_extractor(enable_wavelet=False)
    extractor_wav = get_3d_extractor(enable_wavelet=True)
    
    # ===============================
    # Step 1: 收集有效病例列表
    # ===============================
    valid_cases = []
    for filename in sorted(os.listdir(image_folder)):
        if not filename.endswith(".nii.gz"):
            continue
        
        file_prefix = filename.replace(".nii.gz", "")
        case_id_for_excel = file_prefix.split('_')[0]
        
        image_path = os.path.join(image_folder, filename)
        mask_path = os.path.join(mask_folder, file_prefix + ".nii.gz")
        
        if not os.path.exists(mask_path):
            continue
        if case_id_for_excel not in grade_dict:
            continue
        
        valid_cases.append((image_path, mask_path, case_id_for_excel))
    
    print(f"Total valid cases: {len(valid_cases)}")
    if len(valid_cases) == 0:
        print("No valid cases found!")
        return
    
    # ===============================
    # Step 2: 用第一个有效病例生成特征模板
    # ===============================
    feature_template = []
    template_generated = False
    
    print("Generating feature template from first valid case...")
    
    for image_path, mask_path, case_id_for_excel in valid_cases:
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
                    # Original 保留；Wavelet 保留 Z 方向为 L 的 4 个方向
                    if k.startswith("original_"):
                        feature_template.append(k + "_mean")
                    elif k.startswith("wavelet-"):
                        direction = k.split("_", 1)[0].replace("wavelet-", "")
                        if len(direction) == 3 and direction[2] == "L" and direction in WAVELET_KEEP_DIRECTIONS:
                            feature_template.append(k + "_mean")
                
                if feature_template:
                    template_generated = True
                    print(f"  Template generated from {case_id_for_excel} {region_name}")
                    print(f"  Total features in template: {len(feature_template)}")
                    break
                    
        except Exception as e:
            print(f"  Failed to generate template from {case_id_for_excel}: {e}")
            continue
        
        if template_generated:
            break
    
    if not template_generated:
        print("ERROR: Could not generate feature template from any case!")
        print("Trying fallback method...")
        feature_template = build_fallback_template()
    
    # ===============================
    # Step 3: 多进程并行提取特征
    # ===============================
    print(f"\nStarting parallel extraction with {n_workers} workers...")
    
    all_features = []
    
    # 构造 worker 参数（每个 case 传入 grade_dict 和 feature_template）
    worker_args = [
        (img_path, msk_path, cid, feature_template, grade_dict)
        for img_path, msk_path, cid in valid_cases
    ]
    
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = {executor.submit(process_single_case, args): args[-2] for args in worker_args}
        
        for future in tqdm(as_completed(futures), total=len(futures), desc="Extracting features"):
            case_id = futures[future]
            try:
                case_results = future.result()
                all_features.extend(case_results)
            except Exception as e:
                print(f"  Error processing case {case_id}: {e}")
    
    # ===============================
    # Step 4: 保存结果
    # ===============================
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
    备用方案：如果无法从实际数据生成模板，手动构建特征名列表。
    只保留 _mean（去掉冗余 _std），Wavelet 保留 Z 方向为 L 的 4 个方向。
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
    
    # Original 特征（只保留 _mean）
    for cls, feats in feature_classes.items():
        for feat in feats:
            template.append(f'original_{cls}_{feat}_mean')
    
    # Wavelet 特征：保留 Z 方向为 L 的 4 个方向（LLL, LHL, HLL, HHL）
    # 沿 Z 高频方向（第3位为H）在 D=3 时不稳定，已排除
    for direction in sorted(WAVELET_KEEP_DIRECTIONS):
        for cls, feats in feature_classes.items():
            for feat in feats:
                template.append(f'wavelet-{direction}_{cls}_{feat}_mean')
    
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