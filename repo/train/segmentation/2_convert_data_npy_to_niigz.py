import os
import numpy as np
import nibabel as nib
import shutil
from pathlib import Path
# 这个脚本的功能是将之前准备好的 2D npy 数据转换成 nnU-Net 需要的 nii.gz 格式，并按照 nnU-Net 的目录结构进行组织。
# ================= 配置路径 =================
# 原始 npy 数据路径
source_images_dir = "/mnt/sda/yx/knee/npy_all/imagesTr"
source_masks_dir = "/mnt/sda/yx/knee/npy_all/labelsTr"

source_images_dir = "/mnt/sda/yx/knee/5t/npy_from_dicom/imagesTr"
source_masks_dir = "/mnt/sda/yx/knee/5t/npy_from_dicom/labelsTr"
# nnU-Net 目标路径 (请修改为你自己的 nnUNet_raw 路径)
# 假设任务ID是 600，任务名是 Knee2D
nnUNet_raw = os.environ.get('nnUNet_raw') # 自动读取环境变量
if nnUNet_raw is None:
    # 如果环境变量没读到，手动写死路径：
    nnUNet_raw = "/mnt/sda/yx/knee/nnUNet/nnUnet_raw/" 

task_name = "Dataset100_Knee2D"
target_base = os.path.join(nnUNet_raw, task_name)

target_imagesTr = os.path.join(target_base, "imagesTr")
target_labelsTr = os.path.join(target_base, "labelsTr")

# ===========================================

def make_nifti(data, spacing=(1.0, 1.0, 1.0)):
    # 确保数据是浮点型(image)或整型(label)
    # 2D npy 通常是 (H, W)，我们需要把它转成 (H, W, 1) 让它看起来像个单层切片
    if len(data.shape) == 2:
        data = data[:, :, np.newaxis]
    
    # 创建仿射矩阵 (Identity matrix)，因为是从npy转来的，没有真实空间坐标
    affine = np.eye(4)
    img = nib.Nifti1Image(data, affine)
    return img

def main():
    os.makedirs(target_imagesTr, exist_ok=True)
    os.makedirs(target_labelsTr, exist_ok=True)

    print(f"正在转换数据到: {target_base}")

    # 获取所有图片文件
    img_files = [f for f in os.listdir(source_images_dir) if f.endswith(".npy")]

    for img_file in img_files:
        # 1. 解析文件名
        # 输入: 063_slice2.npy
        case_id = img_file.replace(".npy", "")  # -> 063_slice2
        
        # 对应的 Mask 文件名
        # 输入规则: 063_slice2_mask.npy
        mask_file = f"{case_id}.npy"
        
        img_path = os.path.join(source_images_dir, img_file)
        mask_path = os.path.join(source_masks_dir, mask_file)

        if not os.path.exists(mask_path):
            print(f"跳过: 找不到对应的标签文件 {mask_file}")
            continue

        # 2. 加载 NPY
        img_np = np.load(img_path).astype(np.float32)
        mask_np = np.load(mask_path)  # 先不转类型，保持原样读取

        # === 新增/修改：强制二值化 ===
        # 不管原图是 255 还是 1，2，3，只要大于 0 都变成 1
        # mask_np = np.where(mask_np > 0, 1, 0).astype(np.uint8) 
        # ==========================

        # 3. 转换为 NIfTI
        nii_img = make_nifti(img_np)
        nii_mask = make_nifti(mask_np)

        # 4. 保存到 nnU-Net 目录
        # 图片必须以 _0000.nii.gz 结尾
        nib.save(nii_img, os.path.join(target_imagesTr, f"{case_id}_0000.nii.gz"))
        # 标签文件名必须与 case_id 完全一致
        nib.save(nii_mask, os.path.join(target_labelsTr, f"{case_id}.nii.gz"))

    print("转换完成！")
    
    # 自动生成 dataset.json (简单的版本)
    generate_json(target_base, len(img_files))

def generate_json(output_folder, num_training):
    import json
    json_dict = {
        "name": "Knee2D",
        "description": "2D Knee Cartilage Segmentation from npy",
        "tensorImageSize": "3D", # 虽然是2D，但nnU-Net内部逻辑有时需要这个标记
        "reference": "None",
        "licence": "None",
        "release": "0.0",
        "modality": {
            "0": "MRI"
        },
        "labels": {
             "background": 0, 
             "Femoral_Medial": 1, 
             "Femoral_Lateral": 2, 
             "Tibial_Medial": 3, 
             "Tibial_Lateral": 4 
        },
        "numTraining": num_training,
        "file_ending": ".nii.gz",
        "channel_names": {
            "0": "MRI"
        }
    }
    with open(os.path.join(output_folder, "dataset.json"), 'w') as f:
        json.dump(json_dict, f, indent=4)
    print("dataset.json 已生成")

if __name__ == "__main__":
    main()