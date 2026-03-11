step1 和 step2  都是数据处理的逻辑，如果复现训练过程的话需要跑一下
本模型依赖于nnunet进行训练
训练前的数据预处理命令：
nnUNetv2_plan_and_preprocess -d 600 -c 2d --verify_dataset_integrity
训练命令：
nnUNetv2_train 200 2d 0
模型推理：
nnUNetv2_predict -i ./nnUNet_raw/Dataset200_Knee2D/imagesTr -o inference_output -d 200 -c 2d -f 0