import os
import pandas as pd
import numpy as np
import SimpleITK as sitk
from radiomics import featureextractor

import logging
logging.getLogger("radiomics").setLevel(logging.ERROR)

# -------------------------------
# 1. 配置路径
# -------------------------------
image_folder = "/mnt/sda/yx/knee/5t/data_unzipped/第二批/image_3d"
mask_folder = "/mnt/sda/yx/knee/5t/data_unzipped/第二批/mask_3d"
excel_file = "/mnt/sda/yx/knee/5t/classify/影像组学代码/第二批5T.xlsx"
output_csv = "/mnt/sda/yx/knee/5t/classify/knee_radiomics_wavelet_features.csv"

# 确保输出目录存在
os.makedirs(os.path.dirname(output_csv), exist_ok=True)

# -------------------------------
# 2. PyRadiomics 参数 (启用小波变换)
# -------------------------------
params = {
    "force2D": True,
    "force2Ddimension": 2,
    "binWidth": 25,
    "normalize": True,
    "normalizeScale": 100,
    "interpolator": "sitkBSpline",
    "imageType": {
        "Original": {},
        "Wavelet": {"wavelet": "haar"} # 启用 Haar 小波
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

# -------------------------------
# 3. 读取 Excel 损伤分级
# -------------------------------
df_grade = pd.read_excel(excel_file)
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
# 4. 构建特征模板
# -------------------------------
def get_feature_template():
    # 模拟一个小波特征列表用于初始化
    # 创建 3D 数组 (depth=1, height=32, width=32) 避开 Radiomics 的 2D 校验报错
    dummy_image = sitk.GetImageFromArray(np.random.rand(1, 32, 32))
    # 确保 Mask 中有 label=1 的像素
    mask_array = np.zeros((1, 32, 32), dtype=np.uint8)
    mask_array[0, 5:25, 5:25] = 1 
    dummy_mask = sitk.GetImageFromArray(mask_array)
    
    dummy_image.SetSpacing((1.0, 1.0, 1.0))
    dummy_mask.SetSpacing((1.0, 1.0, 1.0))
    
    try:
        result = extractor.execute(dummy_image, dummy_mask, label=1)
        # 小波特征以 wavelet- 开头
        keys = [k for k in result.keys() if k.startswith("wavelet-")]
        cols = []
        for k in keys:
            cols.append(k + "_mean")
            cols.append(k + "_std")
        return cols
    except Exception as e:
        print(f"Template generation failed: {e}")
        # 如果还是报错，返回一个空列表或预设的基础列表，防止程序崩溃
        return []

print("Generating feature template...")
FEATURE_TEMPLATE = get_feature_template()
if not FEATURE_TEMPLATE:
    print("Warning: Feature template is empty. Check Radiomics settings.")

# -------------------------------
# 5. 定义单区域特征提取函数
# -------------------------------
def extract_region_radiomics(image, mask, region_label, region_name):
    num_slices = image.GetSize()[2]
    slice_features = []

    for z in range(num_slices):
        img_slice = image[:, :, z]
        mask_slice = mask[:, :, z]
        
        # 增加切片维度：从 (H, W) 变为 (1, H, W)，符合 Radiomics 期望的输入格式
        img_slice = sitk.JoinSeries(img_slice)
        mask_slice = sitk.JoinSeries(mask_slice)

        if np.sum(sitk.GetArrayFromImage(mask_slice) == region_label) < 20:
            continue

        try:
            # 执行提取
            result = extractor.execute(img_slice, mask_slice, label=region_label)
            
            feat = {}
            for k, v in result.items():
                if k.startswith("wavelet-"):
                    feat[k] = float(v) 
            slice_features.append(feat)
        except Exception as e:
            # 捕获 list.remove 错误，跳过该层
            # print(f"Skipping slice {z} due to error: {e}")
            continue

    if len(slice_features) == 0:
        return None

    df_feats = pd.DataFrame(slice_features)
    # 确保只对数值列求均值
    feats_mean = df_feats.mean().add_suffix("_mean")
    feats_std = df_feats.std().add_suffix("_std")
    return pd.concat([feats_mean, feats_std])

# -------------------------------
# 6. 批量处理
# -------------------------------
all_features = []
region_names = {1: "Femur_Medial", 2: "Femur_Lateral", 3: "Tibia_Medial", 4: "Tibia_Lateral"}

for filename in os.listdir(image_folder):
    if not filename.endswith(".nii.gz"):
        continue
    
    file_prefix = filename.replace(".nii.gz", "") 
    case_id_for_excel = file_prefix.split('_')[0]

    mask_filename = file_prefix + ".nii.gz"
    mask_path = os.path.join(mask_folder, mask_filename)
    image_path = os.path.join(image_folder, filename)

    if not os.path.exists(mask_path):
        print(f"Warning: Mask not found for {file_prefix}")
        continue
    
    if case_id_for_excel not in grade_dict:
        print(f"Warning: Grade info missing for {case_id_for_excel}")
        continue

    print(f"Processing Wavelet: {file_prefix} ...")
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

        feats["case_id"] = case_id_for_excel
        feats["region"] = region_name
        feats["grade"] = grade_dict[case_id_for_excel][region_name]

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
    print("No features extracted.")