import os
import numpy as np
import nibabel as nib
from pathlib import Path
# 这个脚本的功能是将之前准备好的 2D npy 数据转换成 nnU-Net 需要的 nii.gz 格式，并按照 nnU-Net 的目录结构进行组织。注意修改task_name
# ================= 配置路径 =================

# 1. 你的原始推理数据（只有图像）所在的路径
source_images_dir = "/mnt/sda/yx/knee/5t/npy_from_dicom_2/imagesTr"

# 2. nnU-Net 目标路径
# 假设任务ID是 600，任务名是 Knee2D
# 既然是推理数据，通常建议放在 imagesTs (Test set) 文件夹下
nnUNet_raw = os.environ.get('nnUNet_raw') 
if nnUNet_raw is None:
    nnUNet_raw = "/mnt/sda/yx/knee/nnUNet/nnUnet_raw/" 

task_name = "Dataset1000208_Knee2D"
target_base = os.path.join(nnUNet_raw, task_name)

# 输出目录改为 imagesTs (用于存放待推理的测试集)
target_imagesTs = os.path.join(target_base, "imagesTr")

# ===========================================

def make_nifti(data):
    """
    将 numpy 数组转换为 Nifti 对象
    """
    # 2D npy 通常是 (H, W)，我们需要把它转成 (H, W, 1) 让它看起来像个单层切片
    if len(data.shape) == 2:
        data = data[:, :, np.newaxis]
    
    # 创建仿射矩阵 (Identity matrix)
    affine = np.eye(4)
    img = nib.Nifti1Image(data, affine)
    return img

def main():
    # 只创建 imagesTs 目录
    os.makedirs(target_imagesTs, exist_ok=True)

    print(f"正在转换推理数据到: {target_imagesTs}")

    # 获取所有图片文件
    if not os.path.exists(source_images_dir):
        print(f"[错误] 源目录不存在: {source_images_dir}")
        return

    img_files = [f for f in os.listdir(source_images_dir) if f.endswith(".npy")]
    
    if len(img_files) == 0:
        print("[警告] 源目录下没有找到 .npy 文件")
        return

    count = 0
    for img_file in img_files:
        try:
            # 1. 解析文件名
            case_id = img_file.replace(".npy", "")
            
            img_path = os.path.join(source_images_dir, img_file)

            # 2. 加载 NPY 图像
            img_np = np.load(img_path).astype(np.float32)

            # 3. 转换为 NIfTI
            nii_img = make_nifti(img_np)

            # 4. 保存到 nnU-Net 的 imagesTs 目录
            # 必须严格遵守命名规则: CaseID_0000.nii.gz
            save_path = os.path.join(target_imagesTs, f"{case_id}_0000.nii.gz")
            nib.save(nii_img, save_path)
            
            count += 1
            if count % 100 == 0:
                print(f"已处理 {count} 张图像...")

        except Exception as e:
            print(f"[错误] 处理 {img_file} 时出错: {e}")

    print("-" * 30)
    print("转换完成！")
    print(f"共转换 {count} 个文件。")
    print(f"推理数据已保存在: {target_imagesTs}")
    print("提示：你可以使用 nnUNetv2_predict 命令对该文件夹进行推理。")

if __name__ == "__main__":
    main()