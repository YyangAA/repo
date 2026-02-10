import os
import pandas as pd
import numpy as np
import SimpleITK as sitk
import joblib
from radiomics import featureextractor
import logging

# 静默 PyRadiomics 日志
logging.getLogger("radiomics").setLevel(logging.ERROR)
# 这里目前还有一些问题，不算跑通了，因为特征提取后需要的特征名字要和训练的时候模型特征提取的特征名字完全一致，才能正确对齐输入特征和模型权重，
# 否则就会报错或者预测结果不对，这个代码还需要调整，我预想的它调整完我们的框架就搭完了
# 配置区
# ===============================
BASE_DIR = "/mnt/sda/yx/knee/5t/classify"
RESULTS_DIR = os.path.join(BASE_DIR, "results")

# 定义四个区域
REGION_NAMES = {
    1: "Femur_Medial",
    2: "Femur_Lateral",
    3: "Tibia_Medial",
    4: "Tibia_Lateral"
}

# ===============================
# 第一步：定义特征提取器 (复用你之前的参数)
# ===============================
def get_extractors():
    # 1. 原始 + 小波特征提取器
    params_tex = {
        "binWidth": 25,
        "normalize": True,
        "normalizeScale": 100,
        "interpolator": "sitkBSpline",
        "imageType": {
            "Original": {},
            "Wavelet": {"wavelet": "haar"}
        },
        "featureClass": {
            "firstorder": [], "glcm": [], "glrlm": [], "glszm": [], "ngtdm": []
        }
    }
    extractor_tex = featureextractor.RadiomicsFeatureExtractor(**params_tex)
    extractor_tex.enableImageTypeByName("Wavelet")

    # 2. 形态学特征提取器
    params_shape = {
        "interpolator": "sitkBSpline",
        "featureClass": {"shape": []}
    }
    extractor_shape = featureextractor.RadiomicsFeatureExtractor(**params_shape)
    
    return extractor_tex, extractor_shape

# ===============================
# 第二步：提取单例特征
# ===============================
def extract_single_patient_features(image_path, mask_path):
    print(f"正在提取特征...\nImage: {os.path.basename(image_path)}")
    
    image = sitk.ReadImage(image_path)
    mask = sitk.ReadImage(mask_path)
    mask = sitk.Cast(mask, sitk.sitkUInt8)
    mask.CopyInformation(image)

    extractor_tex, extractor_shape = get_extractors()
    
    patient_rows = []

    # 遍历四个区域
    for label_id, region_name in REGION_NAMES.items():
        # --- A. 提取形态学 (3D 整体) ---
        try:
            # 检查像素点是否足够
            mask_np = sitk.GetArrayFromImage(mask)
            if np.sum(mask_np == label_id) < 50:
                print(f"  - {region_name}: 软骨体积过小，跳过")
                continue

            res_shape = extractor_shape.execute(image, mask, label=label_id)
            feats_shape = {k: float(v) for k, v in res_shape.items() if k.startswith("original_shape")}
        except Exception as e:
            print(f"  - {region_name} Shape Error: {e}")
            feats_shape = {}

        # --- B. 提取纹理/小波 (逐层提取后取均值) ---
        # 注意：这里模拟了你之前 data_process_wavelet.py 的逻辑
        num_slices = image.GetSize()[2]
        slice_feats_list = []
        
        for z in range(num_slices):
            img_slice = image[:, :, z]
            mask_slice = mask[:, :, z]
            
            # 转为伪3D以适应 Radiomics
            img_slice = sitk.JoinSeries(img_slice)
            mask_slice = sitk.JoinSeries(mask_slice)
            
            if np.sum(sitk.GetArrayFromImage(mask_slice) == label_id) < 20:
                continue
            
            try:
                res_tex = extractor_tex.execute(img_slice, mask_slice, label=label_id)
                # 提取 Original 和 Wavelet
                f_dict = {k: float(v) for k, v in res_tex.items() 
                          if k.startswith("original_") or k.startswith("wavelet-")}
                slice_feats_list.append(f_dict)
            except:
                continue

        # 如果这一层提取成功，计算均值
        if slice_feats_list:
            df_slices = pd.DataFrame(slice_feats_list)
            # 计算均值和标准差 (对应你训练时的 mean/std)
            feats_mean = df_slices.mean().to_dict()
            feats_std = df_slices.std().to_dict()
            
            # 合并所有特征
            full_feats = {**feats_shape, **feats_mean, **feats_std}
            full_feats["region"] = region_name
            patient_rows.append(full_feats)
        else:
             print(f"  - {region_name}: 纹理特征提取失败 (可能是ROI太小)")

    if not patient_rows:
        return None
    
    return pd.DataFrame(patient_rows)

# ===============================
# 第三步：加载模型并预测
# ===============================
def predict_pipeline(image_path, mask_path):
    # 1. 现场提取特征
    df_features = extract_single_patient_features(image_path, mask_path)
    
    if df_features is None or df_features.empty:
        print("❌ 无法提取有效特征，预测终止。")
        return

    print("\n特征提取完成，开始推理...\n" + "-"*30)
    
    results = []
    
    # 2. 遍历每个区域进行预测
    for idx, region in REGION_NAMES.items():
        # 获取该区域在 DataFrame 中的数据
        region_data = df_features[df_features["region"] == region]
        
        if region_data.empty:
            results.append({"区域": region, "诊断": "无法评估 (ROI缺失)", "概率": "-"})
            continue
            
        # 3. 加载该区域的专属模型文件
        model_dir = os.path.join(RESULTS_DIR, region, "models")
        try:
            model = joblib.load(os.path.join(model_dir, "svm_model.pkl"))
            scaler = joblib.load(os.path.join(model_dir, "scaler.pkl"))
            feature_list = joblib.load(os.path.join(model_dir, "feature_list.pkl"))
        except FileNotFoundError:
            results.append({"区域": region, "诊断": "模型未训练", "概率": "-"})
            continue

        # 4. 特征对齐 (Feature Alignment) - 核心步骤！
        # 即使提取了1000个特征，这里只取出模型训练时用到的那几十个 LASSO 特征
        try:
            # 自动填充缺失特征为 0 (防止因形状太怪导致某个小波特征没算出来)
            X_input = region_data.reindex(columns=feature_list, fill_value=0)
            
            # 标准化
            X_scaled = scaler.transform(X_input)
            
            # 预测
            prob = model.predict_proba(X_scaled)[0, 1]
            pred = model.predict(X_scaled)[0]
            
            diagnosis = "⚠️ 损伤" if pred == 1 else "✅ 正常"
            results.append({"区域": region, "诊断": diagnosis, "概率": f"{prob:.2%}"})
            
        except Exception as e:
            results.append({"区域": region, "诊断": f"错误: {str(e)}", "概率": "-"})

    # 5. 打印报告
    res_df = pd.DataFrame(results)
    print(res_df)

# ===============================
# 主执行区
# ===============================
if __name__ == "__main__":
    # 输入一个新的、模型从未见过的病人文件路径
    task_id = "张金生"
    test_image = f"/mnt/sda/yx/knee/5t/data_unzipped/第二批/image_3d/{task_id}.nii.gz"
    test_mask  = f"/mnt/sda/yx/knee/5t/data_unzipped/第二批/mask_3d/{task_id}.nii.gz"
    
    if os.path.exists(test_image) and os.path.exists(test_mask):
        predict_pipeline(test_image, test_mask)
    else:
        print("请修改代码底部的 test_image 和 test_mask 为真实路径！")