import pandas as pd
import joblib
import os
import numpy as np

# ===============================
# 1. 基础配置
# ===============================
BASE_DIR = "/mnt/sda/yx/knee/5t/classify"
CSV_DIR = os.path.join(BASE_DIR, "csv")        # 特征文件所在目录
RESULTS_DIR = os.path.join(BASE_DIR, "results") # 模型保存目录

# 指定你要预测的病人ID
TARGET_PATIENT_ID = "樊明利" 

# 定义四个区域
region_names = {
    1: "Femur_Medial",
    2: "Femur_Lateral",
    3: "Tibia_Medial",
    4: "Tibia_Lateral"
}

# ===============================
# 2. 定义预测函数
# ===============================
def predict_cartilage_damage(patient_features_df, region_name):
    """
    加载指定区域的模型，对输入特征进行推理
    """
    # 构造模型文件的绝对路径
    model_path = os.path.join(RESULTS_DIR, region_name, "models", "svm_model.pkl")
    scaler_path = os.path.join(RESULTS_DIR, region_name, "models", "scaler.pkl")
    feature_list_path = os.path.join(RESULTS_DIR, region_name, "models", "feature_list.pkl")

    # 检查模型文件是否存在
    if not os.path.exists(model_path):
        return None, None, "模型文件未找到"

    try:
        # 加载模型组件
        model = joblib.load(model_path)
        scaler = joblib.load(scaler_path)
        selected_features = joblib.load(feature_list_path)

        # 提取特征 (Pandas会自动按列名对齐，不用担心顺序，只要列名存在即可)
        # 注意：这里可能会报 KeyError，如果 CSV 里缺少某些特征
        X_input = patient_features_df[selected_features]

        # 标准化 (必须使用训练时的 scaler)
        X_scaled = scaler.transform(X_input)

        # 推理
        prob = model.predict_proba(X_scaled)[0, 1] # 获取属于 "损伤(1)" 类的概率
        pred = model.predict(X_scaled)[0]           # 获取预测类别 (0 或 1)

        return pred, prob, "Success"

    except Exception as e:
        return None, None, f"推理出错: {str(e)}"

# ===============================
# 3. 主程序：循环四个区域进行预测
# ===============================
print(f"=========================================================")
print(f"正在对患者 [{TARGET_PATIENT_ID}] 进行全膝关节软骨损伤预测")
print(f"=========================================================\n")

summary_report = []

for idx, region in region_names.items():
    print(f"正在处理区域: {region} ...")
    
    # 1. 读取该区域的特征文件
    csv_file = os.path.join(CSV_DIR, f"{region}_filtered_features.csv")
    
    if not os.path.exists(csv_file):
        print(f"  [跳过] 找不到特征文件: {csv_file}")
        continue
        
    df_region = pd.read_csv(csv_file)
    
    # 2. 筛选出该病人的数据
    patient_data = df_region[df_region['case_id'] == TARGET_PATIENT_ID].copy()
    
    if len(patient_data) == 0:
        print(f"  [提示] 该区域未找到患者 [{TARGET_PATIENT_ID}] 的数据 (可能该区域软骨缺失或提取失败)")
        summary_report.append({"区域": region, "结果": "无数据", "概率": "N/A"})
        continue

    # 3. 关键步骤：数据聚合 (Patient-Level Averaging)
    # 因为训练时我们是把同一个病人的多行数据取平均训练的，推理时必须保持一致！
    # numeric_only=True 防止对字符串列求均值报错
    patient_data_avg = patient_data.groupby(['case_id', 'region']).mean(numeric_only=True).reset_index()
    
    # 4. 调用预测函数
    pred, prob, msg = predict_cartilage_damage(patient_data_avg, region)
    
    if msg != "Success":
        print(f"  [错误] {msg}")
        summary_report.append({"区域": region, "结果": "错误", "概率": "N/A"})
    else:
        result_str = "⚠️ 有损伤" if pred == 1 else "✅ 正常"
        prob_str = f"{prob:.2%}"
        print(f"  -> 预测结果: {result_str} (损伤概率: {prob_str})")
        summary_report.append({"区域": region, "结果": result_str, "概率": prob_str})
    
    print("-" * 30)

# ===============================
# 4. 输出最终汇总报告
# ===============================
print("\n" + "="*30)
print(f"患者 [{TARGET_PATIENT_ID}] 诊断汇总报告")
print("="*30)
df_report = pd.DataFrame(summary_report)
print(df_report)