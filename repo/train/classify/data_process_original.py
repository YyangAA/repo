import os
import pandas as pd
import numpy as np
import SimpleITK as sitk
from radiomics import featureextractor

import logging
logging.getLogger("radiomics").setLevel(logging.ERROR)

# -------------------------------
# 1. 配置路径 (已根据你的要求修改)
# -------------------------------
image_folder = "/mnt/sda/yx/knee/5t/data_unzipped/第二批/image_3d"
mask_folder = "/mnt/sda/yx/knee/5t/data_unzipped/第二批/mask_3d"
excel_file = "/mnt/sda/yx/knee/5t/classify/影像组学代码/第二批5T.xlsx"
output_csv = "/mnt/sda/yx/knee/5t/classify/knee_radiomics_original_features.csv"

# 确保输出目录存在
os.makedirs(os.path.dirname(output_csv), exist_ok=True)

# -------------------------------
# 2. PyRadiomics 参数
# -------------------------------
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

extractor = featureextractor.RadiomicsFeatureExtractor(**params)

# -------------------------------
# 3. 读取 Excel 损伤分级
# -------------------------------
df_grade = pd.read_excel(excel_file)

# 这里的 name 处理逻辑：确保和文件名中的前缀一致
df_grade["name"] = df_grade["患者姓名"].astype(str).str.replace("_knee", "", regex=False).str.strip()

region_columns = {
    "股骨内侧": "Femur_Medial",
    "股骨外侧": "Femur_Lateral",
    "胫骨内侧": "Tibia_Medial",
    "胫骨外侧": "Tibia_Lateral"
}

grade_dict = {}
for _, row in df_grade.iterrows():
    case_id = row["name"]
    grade_dict[case_id] = {}
    for col_ch, col_en in region_columns.items():
        grade_dict[case_id][col_en] = row[col_ch]

# -------------------------------
# 4. 构建特征模板 (同原逻辑)
# -------------------------------
def get_feature_template():
    dummy_image = sitk.GetImageFromArray(np.random.rand(32,32))
    dummy_mask = sitk.GetImageFromArray(np.ones((32,32), dtype=np.uint8))
    dummy_image.SetSpacing((1.0, 1.0))
    dummy_mask.SetSpacing((1.0, 1.0))
    result = extractor.execute(dummy_image, dummy_mask, label=1)
    keys = [k for k in result.keys() if k.startswith("original_")]
    cols = []
    for k in keys:
        cols.append(k + "_mean")
        cols.append(k + "_std")
    return cols

# FEATURE_TEMPLATE = get_feature_template()
FEATURE_TEMPLATE = [
    "original_firstorder_Mean_mean", "original_firstorder_Mean_std",
    "original_firstorder_Maximum_mean", "original_firstorder_Maximum_std",
    "original_glcm_Autocorrelation_mean", "original_glcm_Autocorrelation_std",
    "original_glrlm_ShortRunEmphasis_mean", "original_glrlm_ShortRunEmphasis_std"
]
# -------------------------------
# 5. 定义单区域特征提取函数 (同原逻辑)
# -------------------------------
def extract_region_radiomics(image, mask, region_label, region_name):
    num_slices = image.GetSize()[2]
    slice_features = []

    for z in range(num_slices):
        img_slice = image[:, :, z]
        mask_slice = mask[:, :, z]
        mask_np = sitk.GetArrayFromImage(mask_slice)

        # 如果该层中该区域像素太少，跳过
        if np.sum(mask_np == region_label) < 20:
            continue

        try:
            result = extractor.execute(img_slice, mask_slice, label=region_label)
            # --- 关键修改：将 numpy array 转换为 float 标量 ---
            feat = {}
            for k, v in result.items():
                if k.startswith("original_"):
                    # 使用 float() 将 array(-53.04) 转为 -53.04
                    feat[k] = float(v) 
            # ----------------------------------------------
            slice_features.append(feat)
        except Exception as e:
            print(f"Error on slice {z}: {e}")
            continue

    if len(slice_features) == 0:
        return None

    # 现在 df_feats 里的每一列都是纯数值，mean() 不会再报错
    df_feats = pd.DataFrame(slice_features)
    feats_mean = df_feats.mean().add_suffix("_mean")
    feats_std = df_feats.std().add_suffix("_std")
    return pd.concat([feats_mean, feats_std])
# -------------------------------
# 6. 批量处理 (修改了文件名匹配逻辑)
# -------------------------------
all_features = []

region_names = {
    1: "Femur_Medial",
    2: "Femur_Lateral",
    3: "Tibia_Medial",
    4: "Tibia_Lateral"
}

# 遍历图像文件夹
for filename in os.listdir(image_folder):
    # 匹配图像文件：常会儒_0_0000.nii.gz
    if not filename.endswith(".nii.gz"):
        continue
    
    # 提取 case_id（如：常会儒_0）
    # 注意：根据你的描述，Excel里的name可能是"常会儒"，也可能是"常会儒_0"
    # 这里我们提取 "常会儒_0" 作为查找标识
    file_prefix = filename.replace(".nii.gz", "") 
    
    # 这里的 case_id_for_excel 取 "_" 之前的部分，即 "常会儒"
    case_id_for_excel = file_prefix.split('_')[0]

    # 构建 Mask 路径：常会儒_0.nii.gz
    mask_filename = file_prefix + ".nii.gz"
    mask_path = os.path.join(mask_folder, mask_filename)
    image_path = os.path.join(image_folder, filename)

    # ---------------------------
    # 检查文件和分级是否存在
    # ---------------------------
    if not os.path.exists(mask_path):
        print(f"Warning: Mask not found for {file_prefix}, path: {mask_path}")
        continue
    
    if case_id_for_excel not in grade_dict:
        print(f"Warning: Grade info not found for Excel Name: {case_id_for_excel}, skipped.")
        continue

    # ---------------------------
    # 读取并处理
    # ---------------------------
    print(f"Processing: {file_prefix} ...")
    image = sitk.ReadImage(image_path)
    mask = sitk.ReadImage(mask_path)
    mask = sitk.Cast(mask, sitk.sitkUInt8)
    mask.CopyInformation(image)

    for label_id, region_name in region_names.items():
        feats = extract_region_radiomics(image, mask, label_id, region_name)

        if feats is None:
            feats = pd.Series(
                data=[np.nan] * len(FEATURE_TEMPLATE),
                index=FEATURE_TEMPLATE
            )
            feats["cartilage_missing"] = 1
        else:
            feats["cartilage_missing"] = 0

        # 写入元数据
        feats["case_id"] = case_id_for_excel
        feats["region"] = region_name
        feats["grade"] = grade_dict[case_id_for_excel][region_name]

        # 排序
        cols = ["case_id", "region", "grade", "cartilage_missing"] + \
               [c for c in feats.index if c not in ["case_id", "region", "grade", "cartilage_missing"]]
        feats = feats[cols]

        all_features.append(feats)

# -------------------------------
# 7. 保存 CSV
# -------------------------------
if all_features:
    df_final = pd.DataFrame(all_features)
    df_final.to_csv(output_csv, index=False)
    print(f"Finished! Total {len(df_final)} rows saved to {output_csv}")
else:
    print("No features extracted. Please check your file paths and names.")