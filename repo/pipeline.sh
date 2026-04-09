cd /mnt/sda/yx/knee/nnUNet/repo
conda activate knee_yx
rm -rf ./data/inference_output
rm -rf ./data/vis_results_old
rm -rf ./data/inference_results.csv
rm -rf ./data/npy
rm -rf /mnt/sda/yx/knee/nnUNet/nnUNet_raw/Dataset9999_Knee2D
rm -rf ./data/image_3d
rm -rf ./data/mask_3d
python ./infer/segmentation/1_dcm2npy.py
python ./infer/segmentation/2_npy2nii.py
nnUNetv2_predict -i /mnt/sda/yx/knee/nnUNet/nnUNet_raw/Dataset9999_Knee2D/imagesTr -o ./data/inference_output -d 200 -c 2d -f 0 -tr nnUNetTrainer_FreezeEncoder

# 可选 打印分割结果预测
# python ./infer/segmentation/4_vis.py

python ./infer/segmentation/5__nii23D.py
# 分类推理
python ./infer/classify/SVM_RBF_inference_pipeline_0311.py     --image_folder "./data/image_3d"     --mask_folder "./data/mask_3d"     --model_dir "./checkpoint/results"     --output "./data/inference_results.csv"
