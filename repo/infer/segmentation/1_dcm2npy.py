# -*- coding: utf-8 -*-
import os
import numpy as np
import SimpleITK as sitk

def convert_dcm_to_2d_slices(source_dir, output_dir):
    """
    本代码将DICOM序列格式的3D医学图像数据转换为2D切片，并保存为npy格式。适用于nnUNet训练数据准备。
    遍历文件夹，读取 DICOM Series 数据，将其按切片维度拆分，并保存为 2D .npy 格式。
     - source_dir: 包含多个DICOM Series的根目录，每个Series在一个独立的子文件夹中。
     - output_dir: 输出目录，内部会自动创建 imagesTr 和 labelsTr 子目录
    """
    # Output directories
    images_out_dir = os.path.join(output_dir, "imagesTr")
    labels_out_dir = os.path.join(output_dir, "labelsTr")
    
    os.makedirs(images_out_dir, exist_ok=True)
    os.makedirs(labels_out_dir, exist_ok=True)
    
    print(f"Processing data (DICOM Series -> 2D .npy slices)...")
    print(f"Source: {source_dir}")
    print(f"Output: {output_dir}")
    
    total_slices = 0
    processed_cases = 0
    
    for root, dirs, files in os.walk(source_dir):
        dcm_files = [f for f in files if f.endswith('.dcm') or f.endswith('.DCM')]
        
        if len(dcm_files) > 0:
            case_id = os.path.basename(root)
            print(f"Found DICOM Series: {case_id} ({len(dcm_files)} files)")

            try:
                reader = sitk.ImageSeriesReader()
                dicom_names = reader.GetGDCMSeriesFileNames(root)
                reader.SetFileNames(dicom_names)
                
                itk_img = reader.Execute()
                img_array = sitk.GetArrayFromImage(itk_img)  # Shape: (Depth, H, W)

                # Label processing is optional and commented out
                has_label = False
                label_array = None
                
                # if needed, load label here:
                # label_path = os.path.join(root, "mask.nii.gz") 
                # if os.path.exists(label_path):
                #     itk_label = sitk.ReadImage(label_path)
                #     label_array = sitk.GetArrayFromImage(itk_label)
                #     has_label = True

                depth = img_array.shape[0]
                
                for i in range(depth):
                    slice_img = img_array[i, :, :]
                    save_name = f"{case_id}_{i}.npy"
                    
                    np.save(os.path.join(images_out_dir, save_name), slice_img)
                    
                    if has_label and label_array is not None:
                        slice_label = label_array[i, :, :]
                        np.save(os.path.join(labels_out_dir, save_name), slice_label)
                    
                    total_slices += 1
                
                print(f"[Success] Case {case_id}: Split into {depth} slices.")
                processed_cases += 1
                
            except Exception as e:
                print(f"[Error] Failed to process {case_id}: {e}")

    print("-" * 30)
    print(f"Done!")
    print(f"Processed {processed_cases} cases, generated {total_slices} 2D npy files.")
    print(f"Saved to: {output_dir}")


if __name__ == "__main__":

    SOURCE_PATH = r"/mnt/sda/yx/knee/5t/data_unzipped/第二批/第二批5T" 
    
    OUTPUT_PATH = r"/mnt/sda/yx/knee/5t/npy_from_dicom_2"
    
    # ===========================================
    convert_dcm_to_2d_slices(SOURCE_PATH, OUTPUT_PATH)