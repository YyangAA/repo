import os
import re
import SimpleITK as sitk
import numpy as np
def stack_slices_to_nii(input_folder, output_folder):
    """
    这个脚本的功能是将之前准备好的 2D nii.gz 切片数据重新堆叠成 3D nii.gz 格式。适用于将 nnUNet 的 2D 推理结果或训练数据切片重新组合成 3D 图像。
    为下一步的分类做准备，注意修改输入输出路径。
    支持同时处理以下两种格式并重构为 3D:
    1. 名字_序号_0000.nii.gz (nnU-Net 图像)
    2. 名字_序号.nii.gz      (掩码或普通切片)
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # 1. 获取所有 .nii.gz 文件
    all_files = [f for f in os.listdir(input_folder) if f.endswith('.nii.gz')]
    
    # 2. 核心正则表达式
    # ^(.*?)   -> 最小匹配前缀（名字）
    # _(\d+)   -> 匹配序号
    # (?:_0000)? -> 可选匹配 _0000 (?: 表示不捕获该组)
    # \.nii\.gz$ -> 结尾
    pattern = re.compile(r"^(.*?)_(\d+)(?:_0000)?\.nii\.gz$")
    
    file_groups = {}
    for f in all_files:
        match = pattern.match(f)
        if match:
            prefix = match.group(1)
            index = int(match.group(2))
            if prefix not in file_groups:
                file_groups[prefix] = []
            file_groups[prefix].append((index, f))

    print(f"--- 扫描完成：检测到 {len(file_groups)} 个病例组 ---")

    # 3. 逐组堆叠
    for prefix, files in file_groups.items():
        # 严格按数字序号排序
        files.sort(key=lambda x: x[0])
        
        print(f"正在重构 [{prefix}]: 共 {len(files)} 层 (序号从 {files[0][0]} 到 {files[-1][0]})")
        
        slice_list = []
        for _, filename in files:
            file_path = os.path.join(input_folder, filename)
            img_slice = sitk.ReadImage(file_path)
            
            # 将 SimpleITK 图像转为 Numpy 数组
            ary = sitk.GetArrayFromImage(img_slice)
            
            # 兼容性处理：如果是 (1, H, W) 则转为 (H, W)
            if ary.ndim == 3 and ary.shape[0] == 1:
                ary = ary[0, :, :]
            slice_list.append(ary)
        
        # 堆叠成 3D 数组 (Z, Y, X)
        stacked_array = np.stack(slice_list, axis=0)
        combined_img = sitk.GetImageFromArray(stacked_array)
        
        # 4. 继承元数据
        first_slice = sitk.ReadImage(os.path.join(input_folder, files[0][1]))
        combined_img.SetDirection(first_slice.GetDirection())
        combined_img.SetOrigin(first_slice.GetOrigin())
        
        # 处理 Spacing
        orig_spacing = list(first_slice.GetSpacing())
        if len(orig_spacing) == 2:
            # 如果是 2D 切片，假设层厚为 1.0 (你可以根据实际情况修改)
            orig_spacing.append(1.0) 
        combined_img.SetSpacing(orig_spacing)

        # 5. 保存结果
        output_path = os.path.join(output_folder, f"{prefix}.nii.gz")
        sitk.WriteImage(combined_img, output_path)
        
    print(f"\n所有处理已完成，输出目录: {output_folder}")

# ---------------------------------------------------------
# 使用示例：你可以通过修改路径运行两次，或者写个循环
# ---------------------------------------------------------

# 处理图像 (带 _0000 的)
image_in = "/mnt/sda/yx/knee/nnUNet/nnUNet_raw/Dataset9999_Knee2D/imagesTr"
image_out = "./data/image_3d"
stack_slices_to_nii(image_in, image_out)

# 处理掩码 (不带 _0000 的)
mask_in = "./data/inference_output"
mask_out = "./data/mask_3d"
stack_slices_to_nii(mask_in, mask_out)