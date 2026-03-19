import os
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
from tqdm import tqdm  # 如果没有安装tqdm，可以去掉这个库以及下方的tqdm包装
# 这个脚本的功能是读取 nnUNet 的原始图像和推理结果，生成对比图并保存。适用于检查 nnUNet 推理质量。
# ================= 配置区域 =================

# 1. 原始图像文件夹 (输入给 nnUNet 的 imagesTs 或 imagesTr)
# 注意：这里必须包含 _0000.nii.gz 的原始文件
raw_image_dir = "/mnt/sda/yx/knee/nnUNet/nnUNet_raw/Dataset9999_Knee2D/imagesTr"

# 2. 推理结果文件夹 (nnUNetv2_predict 的 -o 输出目录)
predict_dir = "./data/inference_output"
# predict_dir = "/mnt/sda/yx/knee/nnUNet/inference_output"
# predict_dir = "/mnt/sda/yx/knee/5t/data_unzipped/第二批/inference_output"

# 3. 可视化结果保存路径 (会自动创建文件夹)
vis_output_dir = "./data/vis_results_old"

# 4. 设置 Case ID
#   - 如果想看单张：写具体名字，如 case_id = "常会儒_0"
#   - 如果想看全部：写 None 或 "" (空字符串)
case_id = None 

# ===========================================

def show_result(case_id, save_dir):
    """
    读取原始图像和预测结果，生成对比图并保存
    """
    # 构造文件路径
    # 原始图像通常带有 _0000 后缀
    raw_path = os.path.join(raw_image_dir, f"{case_id}_0000.nii.gz")
    # 推理结果通常就是 case_id.nii.gz
    pred_path = os.path.join(predict_dir, f"{case_id}.nii.gz")
    
    # 检查文件是否存在
    if not os.path.exists(raw_path):
        print(f"[跳过] 找不到原始图像: {raw_path}")
        return
    if not os.path.exists(pred_path):
        print(f"[跳过] 找不到推理结果: {pred_path}")
        return

    try:
        # 读取数据
        img_obj = nib.load(raw_path)
        pred_obj = nib.load(pred_path)
        
        # 获取数据矩阵
        img_data = img_obj.get_fdata()
        pred_data = pred_obj.get_fdata()
        
        # 维度处理：如果是 (H, W, 1) -> 转为 (H, W)
        if img_data.ndim == 3:
            img_data = img_data[:, :, 0]
        if pred_data.ndim == 3:
            pred_data = pred_data[:, :, 0]

        # --- 开始画图 ---
        plt.figure(figsize=(18, 6))
        
        # 左边：原始图像
        plt.subplot(1, 3, 1)
        plt.imshow(img_data, cmap='gray')
        plt.title(f"Original", fontsize=12) # 避免中文标题乱码，使用英文通用标题
        plt.axis('off')
        
        # 右边：预测结果覆盖
        plt.subplot(1, 3, 2)
        plt.imshow(img_data, cmap='gray') # 先画底图
        
        # 画Mask
        # masked_where(pred_data == 0) 会把背景透明化
        # alpha=0.5 设置半透明
        # cmap='jet' 使用彩色显示不同类别
        plt.imshow(np.ma.masked_where(pred_data == 0, pred_data), cmap='jet', alpha=0.5)
        plt.title(f"Prediction Overlay", fontsize=12)
        plt.axis('off')

        plt.subplot(1, 3, 3)
        # 技巧：先画一个全黑背景，这样视觉效果更好（否则背景是白色的）
        plt.imshow(np.zeros_like(img_data), cmap='gray')
        # 再画 Mask，这次 alpha=1.0 (不透明)，颜色与中间图保持一致
        plt.imshow(np.ma.masked_where(pred_data == 0, pred_data), cmap='jet', alpha=1.0)
        plt.title(f"Mask Only", fontsize=12)
        plt.axis('off')
        
        # 调整子图间距，防止挤在一起
        plt.tight_layout()
        
        # 保存结果
        save_name = f"vis_{case_id}.png"
        save_path = os.path.join(save_dir, save_name)
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
        plt.close() # 这一步很重要，批量处理时防止内存溢出
        
        # print(f"已保存: {save_name}") # 如果用tqdm进度条，这行可以注释掉

    except Exception as e:
        print(f"[错误] 处理 {case_id} 时出错: {e}")

def main():
    # 创建保存目录
    os.makedirs(vis_output_dir, exist_ok=True)
    
    if case_id:
        # === 模式 1: 单个病例 ===
        print(f"正在处理单个病例: {case_id} ...")
        show_result(case_id, vis_output_dir)
        print(f"完成。结果保存在: {vis_output_dir}")
        
    else:
        # === 模式 2: 批量处理 ===
        print(f"未指定 Case ID，开始扫描 {predict_dir} 下的所有结果...")
        
        # 获取推理目录下所有的 .nii.gz 文件
        files = [f for f in os.listdir(predict_dir) if f.endswith(".nii.gz")]
        
        if len(files) == 0:
            print("未找到任何 .nii.gz 文件。")
            return
            
        print(f"共发现 {len(files)} 个推理结果，开始生成可视化图...")
        
        # 遍历处理 (使用 tqdm 显示进度条，如果没有安装 tqdm，直接用 for f in files:)
        try:
            iterator = tqdm(files, desc="Processing")
        except NameError:
            iterator = files # 如果没装 tqdm，这就只是普通的 list
            
        for f in iterator:
            # 文件名通常是 CaseID.nii.gz，我们需要去掉后缀拿到 ID
            # 方法: 用 replace 或者切片
            current_id = f.replace(".nii.gz", "")
            
            show_result(current_id, vis_output_dir)
            
        print("-" * 30)
        print(f"全部完成！所有图片已保存在: {vis_output_dir}")

if __name__ == "__main__":
    main()