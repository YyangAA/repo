增加特征共享

python /mnt/sda/yx/knee/nnUNet/repo/train/classify/dev_v4/1_get_feature_v4.py

python /mnt/sda/yx/knee/nnUNet/repo/train/classify/dev_v4/add_cross_region_features.py  --input /mnt/sda/yx/knee/nnUNet/repo/train/classify/dev_v4/data_train/knee_radiomics_features_3d_integrated.csv  --output /mnt/sda/yx/knee/nnUNet/repo/train/classify/dev_v4/data_train/knee_radiomics_features_3d_integrated_cross.csv

python /mnt/sda/yx/knee/nnUNet/repo/train/classify/dev_v4/lasso_v4.py --input /mnt/sda/yx/knee/nnUNet/repo/train/classify/dev_v4/data_train/knee_radiomics_features_3d_integrated_cross.csv  --output /mnt/sda/yx/knee/nnUNet/repo/train/classify/dev_v4/data_train/feature_v4

python /mnt/sda/yx/knee/nnUNet/repo/train/classify/dev_v4/4_SVM_RBF_save_model_v4.py