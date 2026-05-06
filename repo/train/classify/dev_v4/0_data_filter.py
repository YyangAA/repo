#!/usr/bin/env python3
"""
文件夹过滤脚本
- 以路径1为基准，路径2和路径3中与路径1重名的文件夹会被过滤（不复制）
- 保留的文件夹复制到路径4
"""

import os
import shutil
import sys


def get_folder_names(path):
    """获取路径下所有文件夹的名称集合"""
    if not os.path.exists(path):
        print(f"错误: 路径不存在: {path}")
        sys.exit(1)
    
    folders = set()
    for item in os.listdir(path):
        item_path = os.path.join(path, item)
        if os.path.isdir(item_path):
            folders.add(item)
    return folders


def filter_and_copy(src_paths, filter_path, output_path):
    """
    过滤并复制文件夹
    
    Args:
        src_paths: 源路径列表（路径2和路径3）
        filter_path: 过滤基准路径（路径1）
        output_path: 输出路径（路径4）
    """
    
    # 1. 获取路径1中的文件夹名称（需要过滤掉的）
    filter_names = get_folder_names(filter_path)
    print(f"路径1 ({filter_path}) 中的文件夹: {filter_names}")
    print(f"共 {len(filter_names)} 个文件夹需要过滤")
    print("-" * 50)
    
    # 2. 创建输出目录
    os.makedirs(output_path, exist_ok=True)
    
    total_copied = 0
    total_filtered = 0
    
    # 3. 遍历路径2和路径3
    for src_path in src_paths:
        if not os.path.exists(src_path):
            print(f"警告: 源路径不存在，跳过: {src_path}")
            continue
            
        print(f"\n处理源路径: {src_path}")
        
        for item in os.listdir(src_path):
            item_path = os.path.join(src_path, item)
            
            # 只处理文件夹
            if not os.path.isdir(item_path):
                continue
            
            # 判断是否需要过滤
            if item in filter_names:
                print(f"  [过滤] {item} （与路径1重名）")
                total_filtered += 1
                continue
            
            # 复制到路径4
            dest_path = os.path.join(output_path, item)
            
            # 处理同名冲突（如果路径2和路径3中有同名文件夹）
            if os.path.exists(dest_path):
                # 添加后缀区分
                counter = 1
                while os.path.exists(dest_path):
                    new_name = f"{item}_from_{os.path.basename(src_path)}_{counter}"
                    dest_path = os.path.join(output_path, new_name)
                    counter += 1
            
            print(f"  [复制] {item} -> {dest_path}")
            shutil.copytree(item_path, dest_path)
            total_copied += 1
    
    print("-" * 50)
    print(f"\n处理完成!")
    print(f"过滤掉的文件夹: {total_filtered} 个")
    print(f"复制到路径4的文件夹: {total_copied} 个")
    print(f"输出路径: {output_path}")


if __name__ == "__main__":
    
    # ========== 请修改以下路径 ==========
    
    path1 = "/mnt/sda/yx/knee/nnUNet/repo/data/dcm"      # 路径1: 基准过滤文件夹
    path2 = "/mnt/sda/yx/knee/5t/data_unzipped/第二批/第二批5T"               # 路径2: 源文件夹1
    path3 = "/mnt/sda/yx/knee/5t/data_unzipped/第一批5T"               # 路径3: 源文件夹2
    path4 = "/mnt/sda/yx/knee/nnUNet/repo/data/classify_train_data"              # 路径4: 输出文件夹
    
    # ===================================
    
    # 路径2和路径3作为源路径列表
    source_paths = [path2, path3]
    
    filter_and_copy(source_paths, path1, path4)