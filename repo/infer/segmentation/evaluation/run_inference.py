"""
run_inference.py
================
基于已训练的 nnU-Net 模型对测试集执行推理（复用 nnU-Net 官方 nnUNetPredictor）。

默认评估协议
------------
数据集 Dataset260426_Knee2D 无独立 imagesTs 测试集，因此默认对
**fold_0 的 held-out 验证集**（splits_final.json 中 fold 0 的 val 列表）做推理，
这是 nnU-Net 论文的标准评估协议。也可通过 --input_dir 指定任意输入目录
（例如将来提供的真实 imagesTs）。

输出
----
outputs/predictions/<case>.nii.gz          分割预测
outputs/predictions/<case>.npz + .pkl      概率图（--save_probabilities 时）
logs/inference_<timestamp>.log             运行日志

运行示例
--------
# 默认：对 fold_0 验证集推理，并保存概率图
python run_inference.py --save_probabilities

# 指定 checkpoint / fold
python run_inference.py --checkpoint checkpoint_final.pth --fold 0

# 对自定义输入目录推理（需为 nnU-Net 格式：<case>_0000.nii.gz）
python run_inference.py --input_dir /path/to/imagesTs
"""

import os
import sys
import shutil
import argparse
import datetime
import traceback

import torch

import utils


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="nnU-Net 推理（复用官方 nnUNetPredictor）")
    p.add_argument("--nnunet_raw", default=utils.DEFAULT_NNUNET_RAW,
                   help="nnUNet_raw 根目录")
    p.add_argument("--nnunet_preprocessed", default=utils.DEFAULT_NNUNET_PREPROCESSED,
                   help="nnUNet_preprocessed 根目录（读取 splits_final.json）")
    p.add_argument("--nnunet_results", default=utils.DEFAULT_NNUNET_RESULTS,
                   help="nnUNet_results 根目录（模型权重）")
    p.add_argument("--fold", type=int, default=utils.DEFAULT_FOLD,
                   help="使用的 fold（默认 0）")
    p.add_argument("--checkpoint", default=utils.DEFAULT_CHECKPOINT,
                   choices=["checkpoint_best.pth", "checkpoint_final.pth"],
                   help="使用的 checkpoint 文件名")
    p.add_argument("--input_dir", default=None,
                   help="自定义输入目录（nnU-Net 格式）。不指定则使用 fold 验证集样本")
    p.add_argument("--output_dir", default=utils.PREDICTIONS_DIR,
                   help="预测结果输出目录")
    p.add_argument("--save_probabilities", action="store_true",
                   help="是否保存概率图（.npz/.pkl）")
    p.add_argument("--device", default="cuda", choices=["cuda", "cpu"],
                   help="推理设备")
    p.add_argument("--disable_tta", action="store_true",
                   help="关闭测试时增强（镜像）以加速。默认开启 TTA")
    return p


def prepare_input_dir_from_val(args, val_cases, log) -> str:
    """
    将 fold 验证集对应的 imagesTr/<case>_0000.nii.gz 软链接/复制到一个临时输入目录，
    以便 nnUNetPredictor 按目录推理。
    """
    tmp_in = os.path.join(utils.OUTPUTS_DIR, "input_val_fold%d" % args.fold)
    if os.path.exists(tmp_in):
        shutil.rmtree(tmp_in)
    os.makedirs(tmp_in, exist_ok=True)

    missing = []
    for case in val_cases:
        src = utils.image_path(args.nnunet_raw, case, subdir="imagesTr")
        if not os.path.exists(src):
            missing.append(src)
            continue
        dst = os.path.join(tmp_in, f"{case}{utils.CHANNEL_SUFFIX}{utils.FILE_ENDING}")
        try:
            os.symlink(os.path.abspath(src), dst)
        except OSError:
            shutil.copy2(src, dst)
    if missing:
        log(f"[警告] 以下输入文件缺失，将跳过：\n" + "\n".join(missing))
    log(f"已准备验证集输入目录：{tmp_in}（{len(val_cases) - len(missing)} 例）")
    return tmp_in


def main():
    args = build_argparser().parse_args()

    utils.ensure_default_output_dirs()
    utils.ensure_dirs(args.output_dir)

    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(utils.LOGS_DIR, f"inference_{ts}.log")
    log_fp = open(log_path, "w", encoding="utf-8")

    def log(msg: str):
        line = f"[{datetime.datetime.now().strftime('%H:%M:%S')}] {msg}"
        print(line, flush=True)
        log_fp.write(line + "\n")
        log_fp.flush()

    # nnU-Net 需要这些环境变量
    os.environ["nnUNet_raw"] = args.nnunet_raw
    os.environ["nnUNet_preprocessed"] = args.nnunet_preprocessed
    os.environ["nnUNet_results"] = args.nnunet_results

    log("=" * 70)
    log("nnU-Net 推理开始")
    log(f"数据集   : {utils.DATASET_NAME}")
    log(f"配置     : {utils.CONFIGURATION} | Trainer: {utils.TRAINER} | Plans: {utils.PLANS}")
    log(f"fold     : {args.fold} | checkpoint: {args.checkpoint}")
    log(f"设备     : {args.device} | TTA: {not args.disable_tta}")
    log(f"结果目录 : {args.nnunet_results}")

    model_folder = utils.get_model_folder(args.nnunet_results)
    if not os.path.isdir(model_folder):
        log(f"[错误] 模型目录不存在: {model_folder}")
        sys.exit(1)
    log(f"模型目录 : {model_folder}")

    # 准备输入目录
    if args.input_dir:
        input_dir = args.input_dir
        log(f"使用自定义输入目录: {input_dir}")
    else:
        val_cases = utils.get_val_case_ids(args.nnunet_preprocessed, args.fold)
        log(f"未指定 input_dir，使用 fold_{args.fold} held-out 验证集 ({len(val_cases)} 例) 作为测试集：")
        log("  " + ", ".join(val_cases))
        input_dir = prepare_input_dir_from_val(args, val_cases, log)

    # 延迟导入，确保环境变量已设置
    from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor

    device = torch.device("cuda", 0) if args.device == "cuda" and torch.cuda.is_available() else torch.device("cpu")
    if args.device == "cuda" and device.type != "cuda":
        log("[警告] 请求 cuda 但不可用，回退到 CPU")

    predictor = nnUNetPredictor(
        tile_step_size=0.5,
        use_gaussian=True,
        use_mirroring=not args.disable_tta,
        perform_everything_on_device=(device.type == "cuda"),
        device=device,
        verbose=False,
        verbose_preprocessing=False,
        allow_tqdm=True,
    )

    log("加载模型 ...")
    predictor.initialize_from_trained_model_folder(
        model_folder,
        use_folds=(args.fold,),
        checkpoint_name=args.checkpoint,
    )

    log(f"开始推理 -> 输出目录: {args.output_dir}")
    try:
        predictor.predict_from_files(
            list_of_lists_or_source_folder=input_dir,
            output_folder_or_list_of_truncated_output_files=args.output_dir,
            save_probabilities=args.save_probabilities,
            overwrite=True,
            num_processes_preprocessing=2,
            num_processes_segmentation_export=2,
        )
    except Exception:
        log("[错误] 推理失败：\n" + traceback.format_exc())
        log_fp.close()
        sys.exit(1)

    # 统计输出
    preds = [f for f in os.listdir(args.output_dir) if f.endswith(utils.FILE_ENDING)]
    log(f"推理完成，共生成 {len(preds)} 个预测文件。")
    log(f"预测目录 : {args.output_dir}")
    if args.save_probabilities:
        npz = [f for f in os.listdir(args.output_dir) if f.endswith(".npz")]
        log(f"概率图   : {len(npz)} 个 .npz 文件")
    log(f"日志文件 : {log_path}")
    log("=" * 70)
    log_fp.close()


if __name__ == "__main__":
    main()
