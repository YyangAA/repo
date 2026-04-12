cd /mnt/sda/yx/knee/nnUNet/repo
conda activate knee_yx
rm -rf ./data/inference_output
rm -rf ./data/vis_results_old
rm -rf ./data/inference_results.csv
rm -rf ./data/npy
rm -rf /mnt/sda/yx/knee/nnUNet/nnUNet_raw/Dataset9999_Knee2D
rm -rf ./data/image_3d
rm -rf ./data/mask_3d
rm -rf ./data/report
python ./infer/segmentation/1_dcm2npy.py
python ./infer/segmentation/2_npy2nii.py
nnUNetv2_predict -i /mnt/sda/yx/knee/nnUNet/nnUNet_raw/Dataset9999_Knee2D/imagesTr -o ./data/inference_output -d 260410 -c 2d -f 0 -tr nnUNetTrainer_FreezeEncoder

# 可选 打印分割结果预测
# python ./infer/segmentation/4_vis.py

python ./infer/segmentation/5__nii23D.py
# 分类推理
# python ./infer/classify/SVM_RBF_inference_pipeline_0311.py     --image_folder "./data/image_3d"     --mask_folder "./data/mask_3d"     --model_dir "./checkpoint/results"     --output "./data/inference_results.csv"

# 分类推理（v2: 3D特征提取，与 dev_v2 训练对齐）
python ./infer/classify/SVM_RBF_inference_pipeline_v4.py \
    --image_folder "./data/image_3d" \
    --mask_folder "./data/mask_3d" \
    --model_dir "./checkpoint/results_260412" \
    --output "./data/inference_results.csv"

# 生成综合可视化诊断报告（分割结果图 + 损伤热力图 + 预测结果 + 真实标签对比）
# --excel 可选: 传入含真实标签的 Excel 文件可展示对比结果，不传则只展示预测
python ./infer/classify/visualize_report_v4.py \
    --image_folder "./data/image_3d" \
    --mask_folder  "./data/mask_3d" \
    --pred_csv     "./data/inference_results.csv" \
    --output_dir   "./data/report" \
    --rotate_90 \
    --excel        "/mnt/sda/yx/knee/5t/data_unzipped/第一批5T.xlsx"
