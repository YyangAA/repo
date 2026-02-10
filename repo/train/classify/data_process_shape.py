import os
import pandas as pd
import numpy as np
import SimpleITK as sitk
from radiomics import featureextractor
import logging

# -------------------------------
# 0. 配置日志
# -------------------------------
logging.getLogger("radiomics").setLevel(logging.ERROR)

# -------------------------------
# 1. 配置路径 (根据你的 NII.GZ 需求)
# -------------------------------
image_folder = "/mnt/sda/yx/knee/5t/data_unzipped/第二批/image_3d"
mask_folder = "/mnt/sda/yx/knee/5t/data_unzipped/第二批/mask_3d"
excel_file = "/mnt/sda/yx/knee/5t/classify/影像组学代码/第二批5T.xlsx"
output_csv = "/mnt/sda/yx/knee/5t/classify/knee_radiomics_shape_features.csv"

# 确保输出目录存在
os.makedirs(os.path.dirname(output_csv), exist_ok=True)

# -------------------------------
# 2. PyRadiomics 参数 (形态学特征专供)
# -------------------------------
params = {
    # 3D 形态学不需要 force2D
    "interpolator": "sitkBSpline",
    "featureClass": {
        "shape": []  # 启用 3D 形态学特征
    }
}

extractor = featureextractor.RadiomicsFeatureExtractor(**params)

# -------------------------------
# 3. 读取 Excel 损伤分级 (匹配你的 Excel 格式)
# -------------------------------
df_grade = pd.read_excel(excel_file)

# 统一处理姓名，去除空格
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
# 4. 获取特征模板
# -------------------------------
def get_shape_template():
    # 预定义的形态学特征名，用于缺失值填充
    return [
        "original_shape_VoxelVolume", "original_shape_SurfaceArea", 
        "original_shape_SurfaceVolumeRatio", "original_shape_Sphericity",
        "original_shape_Maximum3DDiameter", "original_shape_MeshVolume",
        "original_shape_MajorAxisLength", "original_shape_MinorAxisLength",
        "original_shape_LeastAxisLength", "original_shape_Elongation",
        "original_shape_Flatness"
    ]

FEATURE_TEMPLATE = get_shape_template()

# -------------------------------
# 5. 批量处理
# -------------------------------
all_features = []

region_names = {
    1: "Femur_Medial",
    2: "Femur_Lateral",
    3: "Tibia_Medial",
    4: "Tibia_Lateral"
}

for filename in os.listdir(image_folder):
    # 匹配图像：常会儒_0_0000.nii.gz
    if not filename.endswith(".nii.gz"):
        continue
    
    file_prefix = filename.replace(".nii.gz", "") 
    case_id_for_excel = file_prefix.split('_')[0]

    # 构建 Mask：常会儒_0.nii.gz
    mask_path = os.path.join(mask_folder, file_prefix + ".nii.gz")
    image_path = os.path.join(image_folder, filename)

    if not os.path.exists(mask_path):
        print(f"Warning: Mask not found for {file_prefix}")
        continue
    
    if case_id_for_excel not in grade_dict:
        print(f"Warning: Grade info missing for {case_id_for_excel}")
        continue

    print(f"Processing Shape: {file_prefix} ...")
    image = sitk.ReadImage(image_path)
    mask = sitk.ReadImage(mask_path)
    mask = sitk.Cast(mask, sitk.sitkUInt8)
    mask.CopyInformation(image)

    mask_np = sitk.GetArrayFromImage(mask)

    for label_id, region_name in region_names.items():
        # 检查整个 3D 体积中的像素点数
        if np.sum(mask_np == label_id) < 50:
            # 缺失处理
            feats = pd.Series(data=[np.nan] * len(FEATURE_TEMPLATE), index=FEATURE_TEMPLATE)
            feats["cartilage_missing"] = 1
        else:
            try:
                # 提取 3D 形态学特征
                result = extractor.execute(image, mask, label=label_id)
                
                # 转换格式
                feat_dict = {}
                for k, v in result.items():
                    if k.startswith("original_shape"):
                        feat_dict[k] = float(v)
                
                feats = pd.Series(feat_dict)
                feats["cartilage_missing"] = 0
            except Exception as e:
                print(f"Error extracting {region_name} for {file_prefix}: {e}")
                feats = pd.Series(data=[np.nan] * len(FEATURE_TEMPLATE), index=FEATURE_TEMPLATE)
                feats["cartilage_missing"] = 1

        # 注入元数据
        feats["case_id"] = case_id_for_excel
        feats["region"] = region_name
        feats["grade"] = grade_dict[case_id_for_excel][region_name]

        # 排序列顺序
        cols = ["case_id", "region", "grade", "cartilage_missing"] + \
               [c for c in feats.index if c not in ["case_id", "region", "grade", "cartilage_missing"]]
        feats = feats[cols]

        all_features.append(feats)

# -------------------------------
# 6. 保存结果
# -------------------------------
if all_features:
    df_final = pd.DataFrame(all_features)
    df_final.to_csv(output_csv, index=False)
    print(f"Shape features saved to {output_csv}")
else:
    print("No shape features extracted.")