开发新的分类特征聚合方式，变成3D特征聚合，以前2d的要算均值和标准差两个特征，3D聚合后变成一个特征了，目前最低分类正确率0.86%，CKPT保持存在./checkpoint/results

复现路径：
python /mnt/sda/yx/knee/nnUNet/repo/train/classify/dev_v2/get_feature_v2.py
python /mnt/sda/yx/knee/nnUNet/repo/train/classify/dev_v2/lasso_v3.py
python /mnt/sda/yx/knee/nnUNet/repo/train/classify/dev_v2/SVM_RBF_save_model_v2.py