import os
import numpy as np
import SimpleITK as sitk

def convert_mhd_to_2d_slices(source_dir, output_dir):
    """
    本代码将mhd和raw格式的3D医学图像数据转换为2D切片，并保存为npy格式。适用于nnUNet训练数据准备。
    遍历文件夹，读取 3D .mhd 数据，将其按切片维度拆分，并保存为 2D .npy 格式。
    """
    
    # 1. 定义输出路径
    images_out_dir = os.path.join(output_dir, "imagesTr")
    labels_out_dir = os.path.join(output_dir, "labelsTr")
    
    os.makedirs(images_out_dir, exist_ok=True)
    os.makedirs(labels_out_dir, exist_ok=True)
    
    print(f"开始处理数据 (3D -> 2D 切片模式)...")
    print(f"源目录: {source_dir}")
    print(f"输出目录: {output_dir}")
    
    total_slices = 0
    processed_cases = 0
    
    # 2. 遍历源目录
    for root, dirs, files in os.walk(source_dir):
        for file in files:
            if file.endswith("_image.mhd"):
                # 获取 Case ID (如 0001)
                case_id = file.split('_')[0]
                
                image_path = os.path.join(root, file)
                # 构造对应的标签路径
                label_filename = file.replace("_image.mhd", "_mask-label.mhd")
                label_path = os.path.join(root, label_filename)
                
                if not os.path.exists(label_path):
                    print(f"[警告] 找不到标签: {label_path}，跳过。")
                    continue
                
                try:
                    # 3. 读取数据 (SimpleITK 读取顺序通常为 z, y, x)
                    itk_img = sitk.ReadImage(image_path)
                    img_array = sitk.GetArrayFromImage(itk_img)  # Shape: (Depth, H, W)
                    
                    itk_label = sitk.ReadImage(label_path)
                    label_array = sitk.GetArrayFromImage(itk_label) # Shape: (Depth, H, W)
                    
                    # 检查图像和标签的切片数量是否一致
                    if img_array.shape[0] != label_array.shape[0]:
                        print(f"[错误] {case_id} 图像和标签切片数不一致: Img {img_array.shape} vs Label {label_array.shape}")
                        continue
                    
                    # 4. 按切片（Depth 维度）进行遍历并保存
                    # img_array.shape[0] 就是切片数量 (例如 3)
                    depth = img_array.shape[0]
                    
                    for i in range(depth):
                        # 提取单张切片 (512, 512)
                        slice_img = img_array[i, :, :]
                        slice_label = label_array[i, :, :]
                        
                        # 构造切片文件名: ID_索引.npy (例如 0001_0.npy)
                        save_name = f"{case_id}_{i}.npy"
                        
                        np.save(os.path.join(images_out_dir, save_name), slice_img)
                        np.save(os.path.join(labels_out_dir, save_name), slice_label)
                        
                        total_slices += 1
                    
                    print(f"[成功] Case {case_id}: 拆分为 {depth} 个切片 (Shape: {img_array.shape} -> {depth} x {slice_img.shape})")
                    processed_cases += 1
                    
                except Exception as e:
                    print(f"[错误] 处理 {file} 时发生错误: {e}")

    print("-" * 30)
    print(f"处理完成！")
    print(f"共处理 {processed_cases} 个病例，生成了 {total_slices} 个 2D npy 文件。")
    print(f"数据保存在: {output_dir}")


if __name__ == "__main__":
    # ================= 配置区域 =================
    
    # 请在这里填写你的原始数据所在文件夹路径
    # 比如：r"D:\MedicalData\RawData" 或 "./my_data"
    SOURCE_PATH = r"/mnt/sda/yx/knee/yx" 
    
    # 请在这里填写你想保存的路径
    OUTPUT_PATH = r"./npy_all"
    
    # ===========================================
    convert_mhd_to_2d_slices(SOURCE_PATH, OUTPUT_PATH)