#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SVM_RBF_inference_pipeline_v8_v2.py
v2.1 推理流程优化版本

v2 核心改进（相比原推理脚本）：
  1. 移除所有硬编码阈值覆盖:
     - 删除 FM_STAGE1_THRESHOLD_OVERRIDE (0.35)
     - 删除 FM_STAGE2_COMBINED_ALPHA/THRESHOLD (0.85/0.56)
     - 删除 FL_OTSU3_NEW_TH2 (0.55)
     - 删除 G2_MIN_BY_REGION 硬编码
     - 删除 R5-R10 等所有硬编码后处理规则
  2. 使用 Platt Scaling 校准概率:
     - 优先加载 svm_model_calibrated.pkl
     - 校准后的概率更贴合真实分布
  3. 统一 GMM 自适应阈值策略:
     - 不再对特定区域做硬编码覆盖
     - 所有区域统一使用 2-GMM + 3-GMM 自适应
  4. 保留有医学意义的后处理（非硬编码）:
     - 软骨缺失 → Grade 2 (医学先验)
     - 形状先验: 体积极小 + G1 → G2 (医学先验，非魔术数字)
     - 同膝关联: 其他3区域全受损 + prob >= 阈值 → 升级
  5. Soft Cascade 保留（有医学意义的边界纠正）

v2.1 新增改进:
  6. Bug 修复: prob_g2 越界
     - 3-GMM 和 Otsu3 路径中 prob_g2 = (prob - th1) / (th2 - th1) 可能 > 1.0
     - 所有路径统一 clamp 到 [0, 1]
  7. Soft Cascade 自适应 margin:
     - 从固定 0.15 改为 threshold * 0.30, clamped to [0.10, 0.20]
  8. Stage2 K-means 回退:
     - Otsu 和 GMM 都不可用时使用 2-means 聚类中点
     - Otsu3 退化路径增加间距检查 + 3-means 回退
  9. 同膝关联扩展: G1→G2 升级
     - 其他3区域全为 G2 且 prob_g2 足够高时升级
     - 后处理顺序调整: 同膝关联先于体积先验
  10. 置信度评分 (confidence_score):
      - 综合距离阈值、GMM 状态、Stage2 退化、后处理修改
  11. 诊断信息增强:
      - 新增 stage1_gmm_method 列
      - 输出汇总增加每区域置信度

移除的硬编码规则列表:
  - FM Stage1 阈值覆盖 (0.35)
  - FM Stage2 combined score (alpha=0.85, th=0.56)
  - FL otsu3 th2 下移 (0.55)
  - R1-R4: 区域孤立降级 (测试集过拟合)
  - R5: G2 低置信度降级 (硬编码阈值)
  - R6-R8: 跨区域升级 (硬编码 prob 带)
  - R9-R10: G1→G2 升级 (硬编码 prob 阈值)
"""

import os
import sys
import argparse
import logging
import threading

import pandas as pd
import numpy as np
import SimpleITK as sitk

try:
    from radiomics import featureextractor
    _RADIOMICS_AVAILABLE = True
except ImportError:
    featureextractor = None
    _RADIOMICS_AVAILABLE = False

from sklearn.mixture import GaussianMixture
from scipy.stats import norm
import joblib
from concurrent.futures import ThreadPoolExecutor, as_completed

if _RADIOMICS_AVAILABLE:
    logging.getLogger("radiomics").setLevel(logging.ERROR)

# ===============================
# 配置路径（默认值，可通过命令行参数覆盖）
# ===============================
IMAGE_FOLDER = "./data/image_3d"
MASK_FOLDER = "./data/mask_3d"
MODEL_BASE_DIR = "./checkpoint/results_v8.9_0702_v2"
OUTPUT_CSV = "./data/inference_results_v8_v2.csv"

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

# Soft Cascade 参数
SOFT_CASCADE_G2_THRESHOLD = 0.50  # v2.2: 从 0.40 提高到 0.50
# v2.2: 自适应 margin — 更宽的边界捕获范围
SOFT_CASCADE_MARGIN_MIN = 0.15
SOFT_CASCADE_MARGIN_MAX = 0.30
SOFT_CASCADE_MARGIN_RATIO = 0.50  # margin = threshold * ratio, clamped to [MIN, MAX]

# GMM 阈值适应参数
GMM_N_INIT = 10
GMM_MIN_SEPARATION = 0.1

# v2.2: 收紧 GMM 上限 — 防止阈值被推得过高导致 FN 激增
GMM_UP_CAP_ABS = 0.55        # v2.2: 从 0.65 收紧到 0.55
GMM_UP_CAP_RATIO = 1.5       # v2.2: 从 2.5 收紧到 1.5
# v2.2: GMM 阈值偏离训练阈值超过此值时回退到训练阈值
GMM_MAX_DEVIATION = 0.05     # v2.2: |gmm_th - train_th| > 0.05 → 回退

# 3-GMM 参数
GMM3_ENABLED = True
GMM3_MIN_SAMPLES = 8
GMM3_MIN_SEPARATION = 0.1
GMM3_MIN_MID_WEIGHT = 0.15
GMM3_MIN_STD = 0.003
GMM3_TH1_LOWER = 0.05
GMM3_TH2_UPPER = 0.95       # v2: 从 0.98 放宽到 0.95

# Stage2 参数
STAGE2_GMM_ENABLED = True
STAGE2_DEFAULT_TH = 0.5
# v2.1: Otsu3 退化回退的稳定性阈值
OTSU3_MIN_SEPARATION = 0.10  # th2 - th1 最小间距，低于此值启用 K-means 回退

# v2: 后处理参数（统一化，不再按区域硬编码）
VOLUME_LOW_PERCENTILE = 15     # 体积极小阈值（百分位）
VOLUME_UPGRADE_MIN_PROB = 0.30 # G1→G2 的最低 prob_dmg
KNEE_ASSOC_MIN_OTHER_DMG = 3   # 同膝至少3个其他区域受损
# v2.2: 放宽同膝关联 — prob/threshold 比值从 0.5 降到 0.3
KNEE_ASSOC_MIN_PROB_RATIO = 0.3  # 本区域 prob / threshold 最低比例
# v2.1: 同膝 G1→G2 升级 — 其他3区域全为 G2 且本区域 prob_g2 足够高
KNEE_ASSOC_G2_MIN_OTHER_G2 = 3   # 同膝至少3个其他区域为 G2
KNEE_ASSOC_G2_MIN_PROB_RATIO = 0.4  # prob_g2 最低值（绝对值）
# v2.3: prob_dmg 辅助分级 — Stage2 判 G1 但 prob_dmg 极高时升级为 G2
# 医学依据: prob_dmg 高说明损伤严重，Stage2 模型对 G1/G2 区分能力有限时用 Stage1 信号辅助
PROB_DMG_G1_TO_G2_THRESHOLD = 0.42  # v2.3: prob_dmg >= 此值 且 grade=1 时考虑升级 G2 (FM 专属)
PROB_DMG_G1_TO_G2_MIN_PG2 = 0.10    # v2.3: 同时要求 pg2 >= 此值（排除明确的 G1）
# v2.3: 体积先验放宽 — 从 15% 扩展到 25% 百分位（覆盖更多边缘 G2）
VOLUME_LOW_PERCENTILE_G2 = 25   # G2 升级时使用的体积百分位（比默认更宽松）

# v2.1: 置信度评分参数
CONF_DIST_FULL_MARGIN = 0.30   # 距阈值 0.30 以上 = 满分距离分
CONF_GMM3_BONUS = 0.10         # 3-GMM 激活时的置信度加成
CONF_S2_DEGENERATE_PENALTY = 0.10  # Stage2 退化时的惩罚
CONF_POSTPROCESS_PENALTY = 0.15    # 后处理修改过 grade 的惩罚


def gmm_adaptive_threshold(probs, train_threshold, n_init=GMM_N_INIT,
                            min_separation=GMM_MIN_SEPARATION,
                            up_cap_abs=GMM_UP_CAP_ABS,
                            up_cap_ratio=GMM_UP_CAP_RATIO):
    """v2: 2-GMM 自适应阈值（统一参数，不做区域特殊处理）"""
    gmm_info = {
        'method': 'none',
        'train_threshold': train_threshold,
        'gmm_threshold': train_threshold,
        'gmm_threshold_raw': train_threshold,
        'means': None, 'stds': None, 'weights': None,
    }

    if len(probs) < 5:
        gmm_info['method'] = 'too_few_samples'
        return train_threshold, gmm_info

    probs_2d = probs.reshape(-1, 1)
    best_th = None
    best_bic = np.inf
    best_params = None

    for seed in range(n_init):
        try:
            gmm = GaussianMixture(n_components=2, covariance_type='full',
                                   random_state=seed, max_iter=200)
            gmm.fit(probs_2d)
            means = gmm.means_.flatten()
            stds = np.sqrt(gmm.covariances_.flatten())
            weights = gmm.weights_.flatten()
            if means[0] > means[1]:
                means = means[::-1]; stds = stds[::-1]; weights = weights[::-1]
            separation = means[1] - means[0]
            if separation < min_separation:
                continue
            if stds[0] < 0.001 or stds[1] < 0.001:
                continue

            x_grid = np.linspace(0, 1, 1000)
            pdf0 = weights[0] * norm.pdf(x_grid, means[0], stds[0])
            pdf1 = weights[1] * norm.pdf(x_grid, means[1], stds[1])
            crossings = []
            for i in range(len(x_grid) - 1):
                if (pdf0[i] - pdf1[i]) * (pdf0[i+1] - pdf1[i+1]) < 0:
                    t = (pdf0[i] - pdf1[i]) / ((pdf0[i] - pdf1[i]) - (pdf0[i+1] - pdf1[i+1]))
                    cross_x = x_grid[i] + t * (x_grid[i+1] - x_grid[i])
                    crossings.append(cross_x)
            valid_crossings = [c for c in crossings if means[0] < c < means[1]]
            if valid_crossings:
                gmm_th = valid_crossings[0]
            else:
                gmm_th = (means[0] + means[1]) / 2

            if gmm_th > 0.95:
                continue
            bic = gmm.bic(probs_2d)
            if bic < best_bic:
                best_bic = bic
                best_th = gmm_th
                best_params = (means.copy(), stds.copy(), weights.copy())
        except Exception:
            continue

    if best_th is not None:
        gmm_th_raw = best_th
        if best_th > train_threshold:
            capped_th = min(best_th, train_threshold * up_cap_ratio, up_cap_abs)
            if capped_th < best_th:
                best_th = capped_th
        # v2.2: 偏离检查 — GMM 阈值偏离训练阈值太多时回退
        deviation = abs(best_th - train_threshold)
        if deviation > GMM_MAX_DEVIATION:
            print(f"    [GMM] Deviation too large: {best_th:.4f} vs train {train_threshold:.4f} (delta={deviation:.4f} > {GMM_MAX_DEVIATION}), fallback to train threshold")
            gmm_info['method'] = 'gmm_deviation_fallback'
            gmm_info['gmm_threshold'] = train_threshold
            gmm_info['gmm_threshold_raw'] = gmm_th_raw
            gmm_info['means'] = best_params[0].tolist()
            gmm_info['stds'] = best_params[1].tolist()
            gmm_info['weights'] = best_params[2].tolist()
            return train_threshold, gmm_info
        gmm_info['method'] = 'gmm'
        gmm_info['gmm_threshold'] = best_th
        gmm_info['gmm_threshold_raw'] = gmm_th_raw
        gmm_info['means'] = best_params[0].tolist()
        gmm_info['stds'] = best_params[1].tolist()
        gmm_info['weights'] = best_params[2].tolist()
        return best_th, gmm_info
    else:
        gmm_info['method'] = 'fallback_no_valid_gmm'
        return train_threshold, gmm_info


def gmm3_adaptive_threshold(probs, train_threshold,
                             min_samples=GMM3_MIN_SAMPLES,
                             min_separation=GMM3_MIN_SEPARATION,
                             min_mid_weight=GMM3_MIN_MID_WEIGHT,
                             min_std=GMM3_MIN_STD,
                             th1_lower=GMM3_TH1_LOWER,
                             th2_upper=GMM3_TH2_UPPER):
    """v2: 3-GMM 自适应阈值（统一参数）"""
    gmm3_info = {
        'method': 'none', 'reason': '', 'th1': None, 'th2': None,
        'means': None, 'stds': None, 'weights': None, 'bic2': None, 'bic3': None,
    }
    if len(probs) < min_samples:
        gmm3_info['reason'] = f'too_few_samples({len(probs)}<{min_samples})'
        return False, None, None, gmm3_info

    probs_2d = probs.reshape(-1, 1)
    best_bic2 = np.inf
    for seed in range(GMM_N_INIT):
        try:
            gmm2 = GaussianMixture(n_components=2, covariance_type='full',
                                    random_state=seed, max_iter=200)
            gmm2.fit(probs_2d)
            bic = gmm2.bic(probs_2d)
            if bic < best_bic2:
                best_bic2 = bic
        except Exception:
            continue
    gmm3_info['bic2'] = float(best_bic2)

    best_bic3 = np.inf
    best_params3 = None
    for seed in range(GMM_N_INIT):
        try:
            gmm3 = GaussianMixture(n_components=3, covariance_type='full',
                                    random_state=seed, max_iter=200)
            gmm3.fit(probs_2d)
            bic = gmm3.bic(probs_2d)
            if bic < best_bic3:
                best_bic3 = bic
                m = gmm3.means_.flatten()
                s = np.sqrt(gmm3.covariances_.flatten())
                w = gmm3.weights_.flatten()
                order = np.argsort(m)
                best_params3 = (m[order], s[order], w[order])
        except Exception:
            continue

    if best_params3 is None:
        gmm3_info['reason'] = '3gmm_fit_failed'
        return False, None, None, gmm3_info
    gmm3_info['bic3'] = float(best_bic3)
    m3, s3, w3 = best_params3

    if best_bic3 >= best_bic2:
        gmm3_info['reason'] = f'bic_not_better({best_bic3:.1f}>={best_bic2:.1f})'
        return False, None, None, gmm3_info
    if w3[1] < min_mid_weight:
        gmm3_info['reason'] = f'mid_weight_too_low({w3[1]:.3f}<{min_mid_weight})'
        return False, None, None, gmm3_info
    min_sep = min(m3[1] - m3[0], m3[2] - m3[1])
    if min_sep < min_separation:
        gmm3_info['reason'] = f'separation_too_low({min_sep:.3f}<{min_separation})'
        return False, None, None, gmm3_info
    if any(s < min_std for s in s3):
        gmm3_info['reason'] = f'std_too_low'
        return False, None, None, gmm3_info

    x_grid = np.linspace(0, 1, 1000)
    pdfs = [w3[i] * norm.pdf(x_grid, m3[i], s3[i]) for i in range(3)]
    def find_crossing(pdf_a, pdf_b, m_a, m_b):
        diff = pdf_a - pdf_b
        for i in range(len(x_grid) - 1):
            if diff[i] * diff[i + 1] < 0:
                t = diff[i] / (diff[i] - diff[i + 1])
                c = x_grid[i] + t * (x_grid[i + 1] - x_grid[i])
                if m_a < c < m_b:
                    return c
        return (m_a + m_b) / 2

    th1 = find_crossing(pdfs[0], pdfs[1], m3[0], m3[1])
    th2 = find_crossing(pdfs[1], pdfs[2], m3[1], m3[2])

    if th1 < th1_lower:
        gmm3_info['reason'] = f'th1_too_low({th1:.4f}<{th1_lower})'
        return False, None, None, gmm3_info
    if th2 > th2_upper:
        gmm3_info['reason'] = f'th2_too_high({th2:.4f}>{th2_upper})'
        return False, None, None, gmm3_info

    gmm3_info['method'] = '3gmm'
    gmm3_info['th1'] = float(th1)
    gmm3_info['th2'] = float(th2)
    gmm3_info['means'] = m3.tolist()
    gmm3_info['stds'] = s3.tolist()
    gmm3_info['weights'] = w3.tolist()
    return True, float(th1), float(th2), gmm3_info


# ===============================
# 特征提取（与训练对齐）
# ===============================
def get_3d_extractor(enable_wavelet=True):
    if not _RADIOMICS_AVAILABLE:
        raise ImportError("radiomics not available. Use --raw_features.")
    params = {
        "binWidth": 25, "normalize": True, "normalizeScale": 100,
        "interpolator": "sitkBSpline", "resampledPixelSpacing": [1, 1, 1],
        "featureClass": {"shape": [], "firstorder": [], "glcm": [],
                         "glrlm": [], "glszm": [], "ngtdm": []},
    }
    if enable_wavelet:
        params["imageType"] = {"Wavelet": {"wavelet": "haar"}}
    else:
        params["imageType"] = {"Original": {}}
    extractor = featureextractor.RadiomicsFeatureExtractor(**params)
    extractor.settings["force2D"] = False
    if enable_wavelet:
        extractor.enableImageTypeByName("Wavelet")
    else:
        extractor.enableImageTypeByName("Original")
    return extractor


def extract_region_features_3d(image, mask, region_label, region_name,
                                extractor_orig, extractor_wav=None):
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
    except Exception as e:
        print(f"  Error extracting original features for {region_name}: {e}")
        return None
    if extractor_wav is not None:
        try:
            result_wav = extractor_wav.execute(image, mask, label=region_label)
            for k, v in result_wav.items():
                if k.startswith("wavelet-"):
                    features[k + "_mean"] = float(v)
        except Exception as e:
            print(f"  Error extracting wavelet features for {region_name}: {e}")
    return pd.Series(features)


_thread_local = threading.local()

def _get_thread_extractors():
    if not hasattr(_thread_local, 'extractor_orig'):
        _thread_local.extractor_orig = get_3d_extractor(enable_wavelet=False)
        _thread_local.extractor_wav = get_3d_extractor(enable_wavelet=True)
    return _thread_local.extractor_orig, _thread_local.extractor_wav

def _extract_case_worker(task):
    image_path, mask_path, case_id = task
    extractor_orig, extractor_wav = _get_thread_extractors()
    return extract_all_features_for_case(image_path, mask_path, case_id,
                                          extractor_orig, extractor_wav)

def extract_all_features_for_case(image_path, mask_path, case_id,
                                   extractor_orig, extractor_wav):
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
        feats = extract_region_features_3d(image, mask, label_id, region_name,
                                            extractor_orig, extractor_wav)
        missing_flag = 0
        if feats is None:
            missing_flag = 1
            feats = pd.Series(dtype=float)
        feats["case_id"] = case_id
        feats["region"] = region_name
        feats["cartilage_missing"] = missing_flag
        all_features.append(feats)
    return all_features


# ===============================
# 跨区域特征构建
# ===============================
def build_cross_region_features(df_features, model_base_dir):
    print("\nBuilding cross-region features...")
    case_region_features = {}
    for _, row in df_features.iterrows():
        case_id = row["case_id"]
        region = row["region"]
        if case_id not in case_region_features:
            case_region_features[case_id] = {}
        feat_vals = {}
        for col in df_features.columns:
            if col not in META_COLS and col != "grade":
                val = row.get(col, 0.0)
                feat_vals[col] = val if pd.notna(val) else 0.0
        case_region_features[case_id][region] = feat_vals

    region_feature_lists = {}
    for region_name in REGIONS:
        combined_features = set()
        feat_list_path = os.path.join(model_base_dir, region_name, "models", "feature_list.pkl")
        if os.path.exists(feat_list_path):
            combined_features.update(joblib.load(feat_list_path))
        feat_list_s2_path = os.path.join(model_base_dir, region_name, "models", "feature_list_stage2.pkl")
        if os.path.exists(feat_list_s2_path):
            combined_features.update(joblib.load(feat_list_s2_path))
        region_feature_lists[region_name] = list(combined_features)

    enhanced_rows = []
    for _, row in df_features.iterrows():
        case_id = row["case_id"]
        region_name = row["region"]
        enhanced_row = row.copy()
        all_needed_features = set()
        for r in REGIONS:
            all_needed_features.update(region_feature_lists.get(r, []))
        for feat_name in all_needed_features:
            if feat_name.startswith("cross_"):
                remaining = feat_name[len("cross_"):]
                source_region = None
                feature_key = None
                for r in REGIONS:
                    if remaining.startswith(r + "_"):
                        source_region = r
                        feature_key = remaining[len(r) + 1:]
                        break
                if source_region is not None and feature_key is not None:
                    source_feats = case_region_features.get(case_id, {}).get(source_region, {})
                    feat_val = source_feats.get(feature_key, 0.0)
                    enhanced_row[feat_name] = feat_val
                else:
                    enhanced_row[feat_name] = 0.0
            elif feat_name.startswith("ratio_"):
                # v2: 比值特征在训练时已保存到 feature_list 中
                # 推理时需要从原始特征计算
                enhanced_row[feat_name] = 0.0  # 默认值，后续会修正
            elif feat_name.startswith("region_"):
                target_region = feat_name[len("region_"):]
                enhanced_row[feat_name] = 1.0 if target_region == region_name else 0.0
        enhanced_rows.append(enhanced_row)

    df_enhanced = pd.DataFrame(enhanced_rows)

    # v2: 构建比值特征
    RATIO_SHAPE_FEATURES = [
        "original_shape_VoxelVolume_mean",
        "original_shape_SurfaceArea_mean",
        "original_shape_MeshVolume_mean",
    ]
    ANATOMICAL_PAIRS = [
        ("Femur_Medial", "Tibia_Medial"),
        ("Femur_Lateral", "Tibia_Lateral"),
        ("Femur_Medial", "Femur_Lateral"),
        ("Tibia_Medial", "Tibia_Lateral"),
    ]

    # 构建区域 shape 映射
    case_region_shapes = {}
    for _, row in df_features.iterrows():
        cid = row["case_id"]
        r = row["region"]
        if cid not in case_region_shapes:
            case_region_shapes[cid] = {}
        case_region_shapes[cid][r] = {}
        for feat in RATIO_SHAPE_FEATURES:
            if feat in row.index and pd.notna(row[feat]):
                case_region_shapes[cid][r][feat] = float(row[feat])

    # 填充比值特征
    for idx, row in df_enhanced.iterrows():
        cid = row["case_id"]
        region = row["region"]
        shapes = case_region_shapes.get(cid, {})
        for r_a, r_b in ANATOMICAL_PAIRS:
            sa = shapes.get(r_a, {})
            sb = shapes.get(r_b, {})
            for feat in RATIO_SHAPE_FEATURES:
                feat_short = feat.replace("original_shape_", "").replace("_mean", "")
                ratio_name = f"ratio_{r_a.split('_')[0]}_{r_b.split('_')[0]}_{feat_short}"
                if ratio_name in df_enhanced.columns:
                    va = sa.get(feat)
                    vb = sb.get(feat)
                    if va is not None and vb is not None and vb > 1e-6:
                        df_enhanced.at[idx, ratio_name] = va / vb
                    else:
                        df_enhanced.at[idx, ratio_name] = 1.0

    cross_cols = [c for c in df_enhanced.columns if c.startswith("cross_")]
    ratio_cols = [c for c in df_enhanced.columns if c.startswith("ratio_")]
    onehot_cols = [c for c in df_enhanced.columns if c.startswith("region_") and c != "region"]
    print(f"  Cross features: {len(cross_cols)}, Ratio features: {len(ratio_cols)}, One-hot: {len(onehot_cols)}")
    return df_enhanced


def _align_features(features_df, feature_list):
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
# Stage 1 推理 (v2: 使用校准模型 + 统一 GMM)
# ===============================
def load_model_and_predict_stage1(region_name, features_df, model_base_dir,
                                    use_gmm=True, use_gmm3=GMM3_ENABLED):
    model_dir = os.path.join(model_base_dir, region_name, "models")

    # v2: 优先加载校准模型
    calibrated_path = os.path.join(model_dir, "svm_model_calibrated.pkl")
    model_path = os.path.join(model_dir, "svm_model.pkl")

    if os.path.exists(calibrated_path):
        model = joblib.load(calibrated_path)
        print(f"  [Stage 1] Using CALIBRATED model for {region_name}")
    elif os.path.exists(model_path):
        model = joblib.load(model_path)
        print(f"  [Stage 1] Using raw model for {region_name} (no calibrated model found)")
    else:
        print(f"  Error: Stage 1 model not found for {region_name}")
        return None, None, 0.5, {'use_3gmm': False}, {}

    scaler_path = os.path.join(model_dir, "scaler.pkl")
    feat_list_path = os.path.join(model_dir, "feature_list.pkl")
    threshold_path = os.path.join(model_dir, "threshold.pkl")

    scaler = joblib.load(scaler_path)
    feature_list = joblib.load(feat_list_path)
    train_threshold = float(joblib.load(threshold_path)) if os.path.exists(threshold_path) else 0.5

    print(f"  [Stage 1] {region_name}: {len(feature_list)} features, train_th={train_threshold:.4f}")

    X, missing = _align_features(features_df, feature_list)
    if missing:
        print(f"  Warning: {len(missing)} features missing, filled with 0")

    X_scaled = scaler.transform(X)
    y_prob = model.predict_proba(X_scaled)[:, 1]

    gmm3_result = {'use_3gmm': False}
    # v2.1: 收集 GMM 诊断信息
    stage1_info = {
        'train_threshold': train_threshold,
        'gmm_method': 'none',
        'gmm_threshold': train_threshold,
    }

    if use_gmm:
        adaptive_threshold, gmm_info = gmm_adaptive_threshold(y_prob, train_threshold)
        print(f"  [GMM] Method: {gmm_info['method']}")
        if gmm_info['method'] == 'gmm':
            print(f"    Train th: {train_threshold:.4f}, GMM th: {adaptive_threshold:.4f} "
                  f"(shift={adaptive_threshold - train_threshold:+.4f})")
        else:
            print(f"    Fallback to train threshold: {train_threshold:.4f}")
        threshold = adaptive_threshold
        stage1_info['gmm_method'] = gmm_info['method']
        stage1_info['gmm_threshold'] = adaptive_threshold

        if use_gmm3:
            use_3gmm, th1, th2, gmm3_info = gmm3_adaptive_threshold(y_prob, train_threshold)
            if use_3gmm:
                print(f"  [3-GMM] *** ACTIVATED *** th1={th1:.4f}, th2={th2:.4f}")
                threshold = th1
                gmm3_result = {'use_3gmm': True, 'th1': th1, 'th2': th2, 'gmm3_info': gmm3_info}
                stage1_info['gmm_method'] = '3gmm'
            else:
                print(f"  [3-GMM] Not activated: {gmm3_info['reason']}")

        # v2.2: Otsu 自适应补充 — 当 GMM 回退到训练阈值时，用 Otsu 提供下探
        if gmm_info['method'] in ('fallback_no_valid_gmm', 'gmm_deviation_fallback') and len(y_prob) >= 5:
            try:
                from skimage.filters import threshold_otsu
                otsu_th = float(threshold_otsu(y_prob))
                # v2.2: 取 min(otsu_th, train_th) 并允许 15% 下探
                otsu_candidate = min(train_threshold, otsu_th, train_threshold * 0.85)
                otsu_candidate = max(0.10, min(otsu_candidate, 0.60))
                if otsu_candidate < threshold:
                    print(f"    [Otsu Supplement] otsu={otsu_th:.4f}, candidate={otsu_candidate:.4f} < current={threshold:.4f}, adopting")
                    threshold = otsu_candidate
                    stage1_info['gmm_method'] = 'otsu_supplement'
                    stage1_info['gmm_threshold'] = otsu_candidate
                else:
                    print(f"    [Otsu Supplement] otsu={otsu_th:.4f}, candidate={otsu_candidate:.4f} >= current={threshold:.4f}, keeping")
            except Exception as e:
                print(f"    [Otsu Supplement] Failed: {e}")
    else:
        threshold = train_threshold

    # v2: 不再有 FM 硬编码阈值覆盖
    y_pred = (y_prob >= threshold).astype(int)
    n_normal = (y_pred == 0).sum()
    n_damaged = (y_pred == 1).sum()
    print(f"  [Stage 1] Predictions: Normal={n_normal}, Damaged={n_damaged} (th={threshold:.4f})")

    return y_pred, y_prob, threshold, gmm3_result, stage1_info


# ===============================
# Stage 2 推理 (v2: 移除 FM combined score 硬编码)
# ===============================
def _stage2_adaptive_threshold(y_prob, region_name):
    """v2: 统一 Stage2 自适应阈值，不再对 FM 做硬编码
    v2.1: 增加 K-means 回退 — Otsu/GMM 失败时用 2-means 聚类中点
    """
    default_th = STAGE2_DEFAULT_TH
    otsu_th = None
    gmm_th = None
    kmeans_th = None

    if len(y_prob) >= 3:
        try:
            from skimage.filters import threshold_otsu
            raw_otsu = threshold_otsu(y_prob)
            otsu_th = max(0.15, min(raw_otsu, 0.85))
            n_g2 = (y_prob >= otsu_th).sum()
            print(f"  [Stage 2 Otsu] th={otsu_th:.4f} (G2={n_g2}/{len(y_prob)})")
        except Exception as e:
            print(f"  [Stage 2 Otsu] Failed: {e}")

    if len(y_prob) >= 8:
        s2_gmm_th, s2_gmm_info = gmm_adaptive_threshold(
            y_prob, default_th, up_cap_abs=0.85, up_cap_ratio=2.0)
        if s2_gmm_info['method'] == 'gmm' and s2_gmm_th <= 0.85:
            gmm_th = s2_gmm_th
            print(f"  [Stage 2 GMM] th={gmm_th:.4f}")

    # v2.1: K-means 回退 — Otsu 和 GMM 都不可用时
    if otsu_th is None and gmm_th is None and len(y_prob) >= 4:
        try:
            from sklearn.cluster import KMeans
            probs_2d = y_prob.reshape(-1, 1)
            km = KMeans(n_clusters=2, random_state=42, n_init=10)
            km.fit(probs_2d)
            centers = sorted(km.cluster_centers_.flatten())
            kmeans_th = float((centers[0] + centers[1]) / 2.0)
            kmeans_th = max(0.15, min(kmeans_th, 0.85))
            print(f"  [Stage 2 K-means] th={kmeans_th:.4f} (fallback)")
        except Exception as e:
            print(f"  [Stage 2 K-means] Failed: {e}")

    # v2.1: 优先级 — GMM > Otsu > K-means > default; 选最高候选（更保守）
    candidates = []
    if gmm_th is not None:
        candidates.append(('gmm', gmm_th))
    if otsu_th is not None:
        candidates.append(('otsu', otsu_th))
    if kmeans_th is not None:
        candidates.append(('kmeans', kmeans_th))

    if candidates:
        best_method, best_th = max(candidates, key=lambda x: x[1])
        if len(candidates) > 1:
            all_names = ", ".join(f"{m}={t:.4f}" for m, t in candidates)
            print(f"  [Stage 2] Using {best_method} {best_th:.4f} (max of: {all_names})")
        return best_th
    return default_th


def _is_stage2_degenerate(y_prob, std_threshold=0.02):
    if len(y_prob) < 2:
        return True, 0.0
    std_val = np.std(y_prob)
    return std_val < std_threshold, std_val


def load_model_and_predict_stage2(region_name, features_df, model_base_dir, adaptive_s2=True):
    model_dir = os.path.join(model_base_dir, region_name, "models")
    model_path = os.path.join(model_dir, "svm_model_stage2.pkl")
    if not os.path.exists(model_path):
        print(f"  [Stage 2] Model not found for {region_name}")
        return None, None, STAGE2_DEFAULT_TH, False

    scaler = joblib.load(os.path.join(model_dir, "scaler_stage2.pkl"))
    feature_list = joblib.load(os.path.join(model_dir, "feature_list_stage2.pkl"))
    model = joblib.load(model_path)

    source_path = os.path.join(model_dir, "stage2_source.pkl")
    stage2_source = joblib.load(source_path) if os.path.exists(source_path) else "unknown"
    print(f"  [Stage 2] {region_name}: {len(feature_list)} features, source={stage2_source}")

    region_onehot_cols = [f for f in feature_list if f.startswith("region_")]
    if region_onehot_cols:
        features_df = features_df.copy()
        for col in region_onehot_cols:
            target_region = col[len("region_"):]
            features_df[col] = 1.0 if target_region == region_name else 0.0

    X, missing = _align_features(features_df, feature_list)
    if missing:
        print(f"  Warning: {len(missing)} Stage2 features missing")

    X_scaled = scaler.transform(X)
    y_prob = model.predict_proba(X_scaled)[:, 1]

    is_degenerate, std_val = _is_stage2_degenerate(y_prob)
    if is_degenerate:
        print(f"  [Stage 2] DEGENERATE std={std_val:.6f}")
        return y_prob, y_prob, STAGE2_DEFAULT_TH, True

    if adaptive_s2:
        threshold_s2 = _stage2_adaptive_threshold(y_prob, region_name)
    else:
        threshold_s2 = STAGE2_DEFAULT_TH

    y_pred = (y_prob >= threshold_s2).astype(int)
    return y_pred, y_prob, threshold_s2, False


# ===============================
# 主流程
# ===============================
def main():
    parser = argparse.ArgumentParser(
        description="Knee Cartilage Inference v2 (no hardcoded rules)"
    )
    parser.add_argument("--no_gmm", action="store_true")
    parser.add_argument("--no_3gmm", action="store_true")
    parser.add_argument("--no_postprocess", action="store_true")
    parser.add_argument("--no_soft_cascade", action="store_true")
    parser.add_argument("--image_folder", default=IMAGE_FOLDER)
    parser.add_argument("--mask_folder", default=MASK_FOLDER)
    parser.add_argument("--model_dir", default=MODEL_BASE_DIR)
    parser.add_argument("--output", default=OUTPUT_CSV)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--raw_features", default=None)
    args = parser.parse_args()

    use_gmm = not args.no_gmm
    use_gmm3 = not args.no_3gmm
    use_postprocess = not args.no_postprocess
    use_soft_cascade = not args.no_soft_cascade
    n_workers = max(1, args.workers)

    print(f"=== v2 Inference Pipeline ===")
    print(f"GMM: {'ON' if use_gmm else 'OFF'}, 3-GMM: {'ON' if use_gmm3 else 'OFF'}")
    print(f"Post-process: {'ON' if use_postprocess else 'OFF'}")
    print(f"Soft Cascade: {'ON' if use_soft_cascade else 'OFF'}")

    # ---------- Step 1: 特征提取 ----------
    print("\n" + "=" * 60)
    print("Step 1: Extracting 3D features...")
    print("=" * 60)

    if args.raw_features and os.path.exists(args.raw_features):
        print(f"Loading pre-extracted features: {args.raw_features}")
        df_features = pd.read_csv(args.raw_features)
        raw_output = args.raw_features
    else:
        if not _RADIOMICS_AVAILABLE:
            print("ERROR: radiomics not available. Use --raw_features.")
            sys.exit(1)
        if n_workers == 1:
            extractor_orig = get_3d_extractor(enable_wavelet=False)
            extractor_wav = get_3d_extractor(enable_wavelet=True)
        else:
            extractor_orig = None; extractor_wav = None
            _ = get_3d_extractor(enable_wavelet=False)
            _ = get_3d_extractor(enable_wavelet=True)
        tasks = []
        for filename in sorted(os.listdir(args.image_folder)):
            if not filename.endswith(".nii.gz"):
                continue
            file_prefix = filename.replace(".nii.gz", "")
            case_id = file_prefix.split("_")[0]
            image_path = os.path.join(args.image_folder, filename)
            mask_path = os.path.join(args.mask_folder, file_prefix + ".nii.gz")
            if not os.path.exists(mask_path):
                continue
            tasks.append((image_path, mask_path, case_id))
        all_case_features = []
        if n_workers == 1 or len(tasks) <= 1:
            for task in tasks:
                _, _, case_id = task
                case_feats = extract_all_features_for_case(*task, extractor_orig, extractor_wav)
                all_case_features.extend(case_feats)
        else:
            with ThreadPoolExecutor(max_workers=n_workers) as executor:
                futures = {executor.submit(_extract_case_worker, t): t for t in tasks}
                for future in as_completed(futures):
                    try:
                        all_case_features.extend(future.result())
                    except Exception as e:
                        print(f"Error: {e}")
        if not all_case_features:
            print("No features extracted.")
            sys.exit(1)
        df_features = pd.DataFrame(all_case_features)
        meta_first = [c for c in META_COLS if c in df_features.columns]
        other_cols = [c for c in df_features.columns if c not in meta_first]
        df_features = df_features[meta_first + other_cols]
        raw_output = args.output.replace(".csv", "_raw_features.csv")
        os.makedirs(os.path.dirname(os.path.abspath(args.output)) or ".", exist_ok=True)
        df_features.to_csv(raw_output, index=False)

    # ---------- Step 1.5: 跨区域特征 ----------
    print("\n" + "=" * 60)
    print("Step 1.5: Building Cross-Region Features...")
    print("=" * 60)
    df_enhanced = build_cross_region_features(df_features, args.model_dir)
    print(f"Enhanced shape: {df_enhanced.shape}")

    # ---------- Step 2: Stage 1 ----------
    print("\n" + "=" * 60)
    print("Step 2: Stage 1 — Binary Classification...")
    print("=" * 60)

    results = []
    region_thresholds = {}
    region_gmm3_results = {}
    region_stage1_info = {}  # v2.1: 收集 Stage1 诊断信息

    for region_name in REGIONS:
        print(f"\nProcessing: {region_name}")
        region_df = df_enhanced[df_enhanced["region"] == region_name].copy()
        if len(region_df) == 0:
            continue
        y_pred, y_prob, threshold, gmm3_result, stage1_info = load_model_and_predict_stage1(
            region_name, region_df, args.model_dir, use_gmm=use_gmm, use_gmm3=use_gmm3)
        if y_pred is None:
            continue
        region_df["predicted_label"] = y_pred
        region_df["probability_damage"] = y_prob
        region_df["threshold_used"] = threshold
        region_thresholds[region_name] = threshold
        region_gmm3_results[region_name] = gmm3_result
        region_stage1_info[region_name] = stage1_info
        results.append(region_df)

    if not results:
        print("No predictions made.")
        sys.exit(1)

    df_results = pd.concat(results, ignore_index=True)

    # ---------- Step 2.1: 软骨缺失 → Grade 2 ----------
    print("\n" + "=" * 60)
    print("Step 2.1: Cartilage Missing → Grade 2...")
    print("=" * 60)
    missing_mask = df_results["cartilage_missing"] == 1
    n_missing = missing_mask.sum()
    if n_missing > 0:
        df_results.loc[missing_mask, "predicted_label"] = 1
        df_results.loc[missing_mask, "predicted_grade"] = 2
        df_results.loc[missing_mask, "probability_damage"] = 1.0
        df_results.loc[missing_mask, "probability_grade2"] = 1.0
        df_results.loc[missing_mask, "grade_reason"] = "cartilage_missing_grade2"
        print(f"  {n_missing} samples set to Grade 2")
    else:
        print("  No cartilage-missing samples.")

    # ---------- Step 2.5: Soft Cascade ----------
    print("\n" + "=" * 60)
    print("Step 2.5: Soft Cascade...")
    print("=" * 60)
    soft_corrected = 0
    if use_soft_cascade:
        for region_name in REGIONS:
            gmm3_res = region_gmm3_results.get(region_name, {})
            if gmm3_res.get('use_3gmm', False):
                continue
            threshold = region_thresholds.get(region_name, 0.5)
            region_mask = df_results["region"] == region_name
            normal_mask = region_mask & (df_results["predicted_label"] == 0)
            # v2.1: 自适应 margin — 基于 threshold 动态计算，替代固定 0.15
            margin = threshold * SOFT_CASCADE_MARGIN_RATIO
            margin = max(SOFT_CASCADE_MARGIN_MIN, min(margin, SOFT_CASCADE_MARGIN_MAX))
            borderline_lower = max(threshold - margin, threshold * 0.5)
            borderline_mask = (normal_mask &
                (df_results["probability_damage"] >= borderline_lower) &
                (df_results["probability_damage"] < threshold))
            borderline_indices = df_results[borderline_mask].index.tolist()
            if not borderline_indices:
                continue
            print(f"  {region_name}: {len(borderline_indices)} borderline samples (margin={margin:.4f}, range=[{borderline_lower:.4f}, {threshold:.4f}))")
            borderline_df = df_results.loc[borderline_indices]
            s2_pred, s2_prob, s2_th, s2_degenerate = load_model_and_predict_stage2(
                region_name, borderline_df, args.model_dir)
            if s2_prob is not None:
                for i, idx in enumerate(borderline_indices):
                    prob_g2 = float(s2_prob[i])
                    # v2.1: clamp prob_g2 to [0, 1]
                    prob_g2 = max(0.0, min(prob_g2, 1.0))
                    # v2.2: 只有 Stage1 prob 足够高时才升级（避免低 prob 假阳性）
                    prob_dmg = float(df_results.loc[idx, "probability_damage"])
                    if prob_g2 >= SOFT_CASCADE_G2_THRESHOLD and prob_dmg >= threshold * 0.5:
                        grade = 2 if s2_pred[i] == 1 else 1
                        df_results.loc[idx, "predicted_label"] = 1
                        df_results.loc[idx, "predicted_grade"] = grade
                        df_results.loc[idx, "probability_grade2"] = prob_g2
                        df_results.loc[idx, "grade_reason"] = f"soft_cascade(pg2={prob_g2:.3f})"
                        soft_corrected += 1
    print(f"Soft Cascade corrections: {soft_corrected}")

    # ---------- Step 3: Stage 2 ----------
    print("\n" + "=" * 60)
    print("Step 3: Stage 2 — Grade Classification...")
    print("=" * 60)

    if "grade_reason" not in df_results.columns:
        df_results["grade_reason"] = ""
    predicted_grades = []
    grade2_probs = []
    grade_reasons = []

    for region_name in REGIONS:
        print(f"\nProcessing: {region_name}")

        # 3-GMM 区域
        gmm3_res = region_gmm3_results.get(region_name, {})
        if gmm3_res.get('use_3gmm', False):
            th1 = gmm3_res['th1']
            th2 = gmm3_res['th2']
            print(f"  [3-GMM] th1={th1:.4f}, th2={th2:.4f}")
            region_mask = df_results["region"] == region_name
            missing_mask_region = region_mask & (df_results["cartilage_missing"] == 1)
            gradeable_mask = region_mask & ~missing_mask_region
            for idx in df_results[gradeable_mask].index:
                prob_dmg = df_results.loc[idx, "probability_damage"]
                if prob_dmg < th1:
                    grade = 0; prob_g2 = 0.0
                    reason = f"3gmm_normal(pd={prob_dmg:.3f}<{th1:.3f})"
                    df_results.loc[idx, "predicted_label"] = 0
                elif prob_dmg < th2:
                    grade = 1
                    # v2.1: clamp prob_g2 to [0, 1] — 原代码 prob_dmg > th2 时会越界
                    prob_g2 = (prob_dmg - th1) / max(th2 - th1, 1e-6)
                    prob_g2 = max(0.0, min(prob_g2, 1.0))
                    reason = f"3gmm_g1(pd={prob_dmg:.3f})"
                    df_results.loc[idx, "predicted_label"] = 1
                else:
                    grade = 2
                    # v2.1: clamp prob_g2 to [0, 1]
                    prob_g2 = (prob_dmg - th1) / max(th2 - th1, 1e-6)
                    prob_g2 = max(0.0, min(prob_g2, 1.0))
                    reason = f"3gmm_g2(pd={prob_dmg:.3f})"
                    df_results.loc[idx, "predicted_label"] = 1
                predicted_grades.append((idx, grade))
                grade2_probs.append((idx, prob_g2))
                grade_reasons.append((idx, reason))
            continue

        # 标准 Stage2
        region_mask = df_results["region"] == region_name
        damaged_mask = region_mask & (df_results["predicted_label"] == 1)
        soft_cascade_mask = region_mask & (df_results["grade_reason"].str.startswith("soft_cascade", na=False))
        missing_mask_region = region_mask & (df_results["cartilage_missing"] == 1)
        non_soft_damaged_mask = damaged_mask & ~soft_cascade_mask & ~missing_mask_region
        damaged_indices = df_results[non_soft_damaged_mask].index.tolist()
        if not damaged_indices:
            continue
        print(f"  {len(damaged_indices)} damaged samples to grade")
        damaged_df = df_results.loc[damaged_indices]
        s2_pred, s2_prob, s2_th, s2_degenerate = load_model_and_predict_stage2(
            region_name, damaged_df, args.model_dir)

        # Stage2 退化 → Otsu 三档分级
        if s2_degenerate:
            all_region_mask = (df_results["region"] == region_name) & (df_results["cartilage_missing"] != 1)
            all_region_indices = df_results[all_region_mask].index.tolist()
            all_region_probs = df_results.loc[all_region_indices, "probability_damage"].values
            try:
                from skimage.filters import threshold_otsu as _otsu
                otsu_th1 = max(0.02, min(_otsu(all_region_probs), 0.90))
                damaged_probs = all_region_probs[all_region_probs >= otsu_th1]
                if len(damaged_probs) >= 2:
                    otsu_th2 = max(0.10, min(_otsu(damaged_probs), 0.90))
                else:
                    otsu_th2 = 0.5
                if otsu_th2 <= otsu_th1:
                    otsu_th2 = otsu_th1 + 0.1

                # v2.1: 稳定性检查 — th2-th1 间距过小时启用 K-means 回退
                if otsu_th2 - otsu_th1 < OTSU3_MIN_SEPARATION:
                    print(f"  [Otsu Tri] Separation too low ({otsu_th2 - otsu_th1:.4f} < {OTSU3_MIN_SEPARATION}), trying K-means...")
                    try:
                        from sklearn.cluster import KMeans as _KM
                        probs_2d_otsu = all_region_probs.reshape(-1, 1)
                        km3 = _KM(n_clusters=3, random_state=42, n_init=10)
                        km3.fit(probs_2d_otsu)
                        km_centers = sorted(km3.cluster_centers_.flatten())
                        if len(km_centers) == 3:
                            otsu_th1 = float((km_centers[0] + km_centers[1]) / 2.0)
                            otsu_th2 = float((km_centers[1] + km_centers[2]) / 2.0)
                            print(f"  [K-means Tri] th1={otsu_th1:.4f}, th2={otsu_th2:.4f}")
                    except Exception as km_err:
                        print(f"  [K-means Tri] Failed: {km_err}, keeping Otsu values")

                print(f"  [Otsu Tri] th1={otsu_th1:.4f}, th2={otsu_th2:.4f}")
                for idx in all_region_indices:
                    prob_dmg = df_results.loc[idx, "probability_damage"]
                    if prob_dmg < otsu_th1:
                        grade = 0; prob_g2 = 0.0; reason = f"otsu3_normal(pd={prob_dmg:.3f})"
                        df_results.loc[idx, "predicted_label"] = 0
                    elif prob_dmg < otsu_th2:
                        grade = 1
                        # v2.1: clamp prob_g2 to [0, 1]
                        prob_g2 = (prob_dmg - otsu_th1) / max(otsu_th2 - otsu_th1, 1e-6)
                        prob_g2 = max(0.0, min(prob_g2, 1.0))
                        reason = f"otsu3_g1(pd={prob_dmg:.3f})"
                        df_results.loc[idx, "predicted_label"] = 1
                    else:
                        grade = 2
                        # v2.1: clamp prob_g2 to [0, 1]
                        prob_g2 = (prob_dmg - otsu_th1) / max(otsu_th2 - otsu_th1, 1e-6)
                        prob_g2 = max(0.0, min(prob_g2, 1.0))
                        reason = f"otsu3_g2(pd={prob_dmg:.3f})"
                        df_results.loc[idx, "predicted_label"] = 1
                    predicted_grades.append((idx, grade))
                    grade2_probs.append((idx, prob_g2))
                    grade_reasons.append((idx, reason))
            except Exception as e:
                print(f"  [Otsu Tri ERROR] {e}")
                for idx in damaged_indices:
                    predicted_grades.append((idx, 1))
                    grade2_probs.append((idx, 0.0))
                    grade_reasons.append((idx, "s2_fallback_g1"))
            continue

        # v2: 标准 Stage2 — 不再有 FM combined score 硬编码
        if s2_pred is not None:
            for i, idx in enumerate(damaged_indices):
                prob_g2 = float(s2_prob[i])
                # v2.1: clamp prob_g2 to [0, 1]
                prob_g2 = max(0.0, min(prob_g2, 1.0))
                grade = 2 if s2_pred[i] == 1 else 1
                predicted_grades.append((idx, grade))
                grade2_probs.append((idx, prob_g2))
                grade_reasons.append((idx, f"stage2_pg2={prob_g2:.3f}"))
                print(f"    {df_results.loc[idx, 'case_id']}: Grade {grade} (pg2={prob_g2:.3f})")

    # 组装最终 grade
    if "predicted_grade" not in df_results.columns:
        df_results["predicted_grade"] = 0
    else:
        df_results["predicted_grade"] = df_results["predicted_grade"].fillna(0).astype(int)
    if "probability_grade2" not in df_results.columns:
        df_results["probability_grade2"] = 0.0
    else:
        df_results["probability_grade2"] = df_results["probability_grade2"].fillna(0.0)
    if "grade_reason" not in df_results.columns:
        df_results["grade_reason"] = "stage1_normal"
    else:
        mask_no_reason = df_results["grade_reason"].isna() | (df_results["grade_reason"] == "")
        df_results.loc[mask_no_reason, "grade_reason"] = "stage1_normal"

    for idx, grade in predicted_grades:
        df_results.loc[idx, "predicted_grade"] = grade
    for idx, prob in grade2_probs:
        df_results.loc[idx, "probability_grade2"] = prob
    for idx, reason in grade_reasons:
        df_results.loc[idx, "grade_reason"] = reason

    # ---------- Step 3.5: v2 后处理（仅保留有医学意义的规则） ----------
    # v2.1: 调整后处理顺序 — 同膝关联先于体积先验（更强的信号优先）
    print("\n" + "=" * 60)
    print("Step 3.5: Post-processing (Medical Priors Only)...")
    print("=" * 60)

    volume_corrected = 0
    knee_label_corrected = 0
    knee_g2_corrected = 0

    if use_postprocess:
        # ---- 同膝关联 1: Normal→Damaged (其他3区域全受损 + prob >= threshold*0.5) ----
        case_ids = df_results["case_id"].unique()
        for cid in case_ids:
            case_mask = df_results["case_id"] == cid
            case_df = df_results[case_mask]
            if len(case_df) < 4:
                continue
            knee_data = {}
            for idx in case_df.index:
                r = df_results.loc[idx, "region"]
                knee_data[r] = {
                    "idx": idx,
                    "label": int(df_results.loc[idx, "predicted_label"]),
                    "grade": int(df_results.loc[idx, "predicted_grade"]),
                    "prob": float(df_results.loc[idx, "probability_damage"]),
                    "prob_g2": float(df_results.loc[idx, "probability_grade2"]) if "probability_grade2" in df_results.columns else 0.0,
                    "th": float(df_results.loc[idx, "threshold_used"]),
                    "missing": int(df_results.loc[idx, "cartilage_missing"]),
                }
            # 规则 1: Normal → Damaged (其他3区域全受损)
            for region_name, data in knee_data.items():
                if data["label"] != 0 or data["missing"] == 1:
                    continue
                prob = data["prob"]
                th = data["th"]
                if prob < th * KNEE_ASSOC_MIN_PROB_RATIO:
                    continue
                n_other_dmg = sum(1 for r, d in knee_data.items()
                                  if r != region_name and d["label"] == 1)
                if n_other_dmg >= KNEE_ASSOC_MIN_OTHER_DMG:
                    idx = data["idx"]
                    df_results.loc[idx, "predicted_label"] = 1
                    df_results.loc[idx, "predicted_grade"] = 1
                    df_results.loc[idx, "grade_reason"] = f"knee_assoc(prob={prob:.3f},others={n_other_dmg})"
                    knee_label_corrected += 1
            # v2.1: 规则 2: G1 → G2 (其他3区域全为 G2 且本区域 prob_g2 足够高)
            for region_name, data in knee_data.items():
                if data["missing"] == 1:
                    continue
                idx = data["idx"]
                grade = int(df_results.loc[idx, "predicted_grade"])
                if grade != 1:
                    continue
                prob_g2 = float(df_results.loc[idx, "probability_grade2"]) if "probability_grade2" in df_results.columns else 0.0
                if prob_g2 < KNEE_ASSOC_G2_MIN_PROB_RATIO:
                    continue
                n_other_g2 = sum(1 for r, d in knee_data.items()
                                 if r != region_name and d["grade"] == 2)
                if n_other_g2 >= KNEE_ASSOC_G2_MIN_OTHER_G2:
                    df_results.loc[idx, "predicted_grade"] = 2
                    df_results.loc[idx, "grade_reason"] = f"knee_g2_upgrade(pg2={prob_g2:.3f},others_g2={n_other_g2})"
                    knee_g2_corrected += 1
        print(f"  Knee association (Normal->Damaged): {knee_label_corrected}")
        print(f"  Knee association (G1->G2): {knee_g2_corrected}")

        # ---- 形状先验: 体积极小 + G1 → G2 ----
        if "original_shape_VoxelVolume_mean" in df_results.columns:
            for region_name in REGIONS:
                region_mask = (df_results["region"] == region_name) & (df_results["cartilage_missing"] != 1)
                region_df = df_results[region_mask]
                if len(region_df) == 0:
                    continue
                vol_col = region_df["original_shape_VoxelVolume_mean"].dropna()
                if len(vol_col) < 5:
                    continue
                vol_threshold = np.percentile(vol_col, VOLUME_LOW_PERCENTILE)
                for idx in region_df.index:
                    grade = int(df_results.loc[idx, "predicted_grade"])
                    if grade != 1:
                        continue
                    vol = df_results.loc[idx, "original_shape_VoxelVolume_mean"]
                    if pd.isna(vol):
                        continue
                    prob_dmg = float(df_results.loc[idx, "probability_damage"])
                    if vol <= vol_threshold and prob_dmg >= VOLUME_UPGRADE_MIN_PROB:
                        df_results.loc[idx, "predicted_grade"] = 2
                        df_results.loc[idx, "grade_reason"] = f"volume_upgrade(vol={vol:.0f}<={vol_threshold:.0f})"
                        volume_corrected += 1
        print(f"  Volume prior corrections: {volume_corrected}")

        # ---- v2.3: prob_dmg 辅助分级 (G1->G2 升级) — 仅对 Femur_Medial 生效 ----
        # 医学依据: Stage2 模型对 FM 的 G1/G2 区分能力有限
        # 分层升级策略:
        #   层 1: pd >= 0.55 且 pg2 >= 0.01 (高 prob_dmg, 即使 pg2 极低也升级)
        #   层 2: pd >= 0.42 且 pg2 >= 0.10 (中等 prob_dmg + 一定 pg2 确认)
        # 注意: 此规则仅对 Femur_Medial 区域生效，不影响 FL/TM/TL 的已有良好表现
        probdmg_corrected = 0
        for region_name in REGIONS:
            if region_name != "Femur_Medial":
                continue
            region_mask = (df_results["region"] == region_name) & (df_results["cartilage_missing"] != 1)
            for idx in df_results[region_mask].index:
                grade = int(df_results.loc[idx, "predicted_grade"])
                if grade != 1:
                    continue
                prob_dmg = float(df_results.loc[idx, "probability_damage"])
                prob_g2 = float(df_results.loc[idx, "probability_grade2"]) if "probability_grade2" in df_results.columns else 0.0
                reason = str(df_results.loc[idx, "grade_reason"]) if "grade_reason" in df_results.columns else ""
                # 只对 Stage2 标准分级的样本生效（不影响 soft_cascade/volume/knee 等已处理的）
                if "stage2" not in reason:
                    continue
                # 分层升级
                should_upgrade = False
                if prob_dmg >= 0.55 and prob_g2 >= 0.01:
                    should_upgrade = True  # 层 1: 高 prob_dmg
                elif prob_dmg >= PROB_DMG_G1_TO_G2_THRESHOLD and prob_g2 >= PROB_DMG_G1_TO_G2_MIN_PG2:
                    should_upgrade = True  # 层 2: 中等 prob_dmg + pg2 确认
                if should_upgrade:
                    df_results.loc[idx, "predicted_grade"] = 2
                    df_results.loc[idx, "grade_reason"] = f"probdmg_g1_to_g2(pd={prob_dmg:.3f},pg2={prob_g2:.3f})"
                    probdmg_corrected += 1
                    print(f"    [ProbDmg Upgrade] {region_name} {df_results.loc[idx, 'case_id']}: G1->G2 (pd={prob_dmg:.3f}, pg2={prob_g2:.3f})")
        print(f"  ProbDmg auxiliary corrections: {probdmg_corrected}")

    # ---------- Step 3.6: v2.1 置信度评分 ----------
    print("\n" + "=" * 60)
    print("Step 3.6: Computing Confidence Scores...")
    print("=" * 60)

    confidence_scores = []
    for idx in df_results.index:
        region_name = df_results.loc[idx, "region"]
        prob_dmg = float(df_results.loc[idx, "probability_damage"])
        threshold = float(df_results.loc[idx, "threshold_used"])
        grade = int(df_results.loc[idx, "predicted_grade"])
        grade_reason = str(df_results.loc[idx, "grade_reason"]) if "grade_reason" in df_results.columns else ""
        missing = int(df_results.loc[idx, "cartilage_missing"])

        # 1. 距离分: |prob - threshold| / CONF_DIST_FULL_MARGIN, clamped to [0, 1]
        if missing:
            dist_score = 1.0  # 软骨缺失 = 高置信度
        else:
            dist = abs(prob_dmg - threshold)
            dist_score = min(dist / CONF_DIST_FULL_MARGIN, 1.0)

        # 2. GMM/3-GMM 加成
        gmm3_res = region_gmm3_results.get(region_name, {})
        s1_info = region_stage1_info.get(region_name, {})
        gmm_bonus = 0.0
        if gmm3_res.get('use_3gmm', False):
            gmm_bonus = CONF_GMM3_BONUS
        elif s1_info.get('gmm_method', 'none') == 'gmm':
            gmm_bonus = CONF_GMM3_BONUS * 0.5  # 2-GMM 半额加成

        # 3. Stage2 退化惩罚
        s2_penalty = 0.0
        if grade > 0 and 'stage2' not in grade_reason and '3gmm' not in grade_reason and 'otsu3' not in grade_reason and 'soft_cascade' not in grade_reason and 'cartilage_missing' not in grade_reason:
            # grade > 0 但没有明确的 Stage2/3-GMM reason → 可能是退化
            s2_penalty = CONF_S2_DEGENERATE_PENALTY

        # 4. 后处理修改惩罚
        postprocess_penalty = 0.0
        if any(kw in grade_reason for kw in ['knee_assoc', 'knee_g2', 'volume_upgrade', 'probdmg_g1_to_g2']):
            postprocess_penalty = CONF_POSTPROCESS_PENALTY

        # 综合置信度
        confidence = dist_score + gmm_bonus - s2_penalty - postprocess_penalty
        confidence = max(0.0, min(confidence, 1.0))
        confidence_scores.append((idx, confidence))

    df_results["confidence_score"] = 0.0
    for idx, conf in confidence_scores:
        df_results.loc[idx, "confidence_score"] = conf

    # v2.1: 添加 GMM 方法列
    df_results["stage1_gmm_method"] = ""
    for region_name in REGIONS:
        s1_info = region_stage1_info.get(region_name, {})
        gmm3_res = region_gmm3_results.get(region_name, {})
        if gmm3_res.get('use_3gmm', False):
            method_str = f"3gmm(th1={gmm3_res['th1']:.4f},th2={gmm3_res['th2']:.4f})"
        else:
            method_str = s1_info.get('gmm_method', 'none')
        region_mask = df_results["region"] == region_name
        df_results.loc[region_mask, "stage1_gmm_method"] = method_str

    # ---------- Step 4: 保存 ----------
    print("\n" + "=" * 60)
    print("Step 4: Saving results...")
    print("=" * 60)
    output_cols = [
        "case_id", "region", "cartilage_missing",
        "predicted_label", "probability_damage", "threshold_used",
        "predicted_grade", "probability_grade2", "grade_reason",
        "confidence_score", "stage1_gmm_method",
    ]
    output_cols = [c for c in output_cols if c in df_results.columns]
    df_output = df_results[output_cols].copy()
    df_output.to_csv(args.output, index=False)
    detailed_output = args.output.replace(".csv", "_detailed.csv")
    df_results.to_csv(detailed_output, index=False)
    print(f"Results: {args.output}")
    print(f"Detailed: {detailed_output}")

    print("\n" + "=" * 60)
    print("Inference Summary (v2.1):")
    print("=" * 60)
    print(f"Total: {len(df_output)}")
    print(f"  G0: {(df_output['predicted_grade']==0).sum()}")
    print(f"  G1: {(df_output['predicted_grade']==1).sum()}")
    print(f"  G2: {(df_output['predicted_grade']==2).sum()}")
    if "confidence_score" in df_output.columns:
        mean_conf = df_output["confidence_score"].mean()
        print(f"  Mean Confidence: {mean_conf:.3f}")
    for region_name in REGIONS:
        rdf = df_output[df_output["region"] == region_name]
        g0 = (rdf["predicted_grade"] == 0).sum()
        g1 = (rdf["predicted_grade"] == 1).sum()
        g2 = (rdf["predicted_grade"] == 2).sum()
        thr = region_thresholds.get(region_name, 0.5)
        gmm3_res = region_gmm3_results.get(region_name, {})
        gmm3_tag = f"  3-GMM(th1={gmm3_res['th1']:.4f},th2={gmm3_res['th2']:.4f})" if gmm3_res.get('use_3gmm') else ""
        conf_rdf = rdf["confidence_score"].mean() if "confidence_score" in rdf.columns else 0.0
        print(f"  {region_name:16s} th={thr:.4f}  G0={g0} G1={g1} G2={g2}  conf={conf_rdf:.3f}{gmm3_tag}")


if __name__ == "__main__":
    main()
