"""
visualize.py
============
生成论文级可视化对比图：Input / Ground Truth / Prediction / Overlay / Difference Map。

默认输入：fold_0 validation 预测（训练已生成）。原图取自 imagesTr/<case>_0000.nii.gz。

对每个（随机抽样或全部）样本，输出一张多面板对比图：
  [1] Input (原始 MRI)
  [2] Ground Truth (GT 掩码叠加)
  [3] Prediction (预测掩码叠加)
  [4] Overlay (Prediction 填充 + GT 白色轮廓)
  [5] Difference Map (TP 绿 / FN 红 / FP 蓝)

输出目录：figures/
运行示例
--------
python visualize.py --num_samples 6 --seed 42
python visualize.py --all
python visualize.py --cases 成信元_0 祝昆_0
"""

import os
import argparse
import random
from typing import List

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.colors import ListedColormap, BoundaryNorm

import utils


def default_validation_pred_dir() -> str:
    return os.path.join(
        utils.get_model_folder(utils.DEFAULT_NNUNET_RESULTS),
        f"fold_{utils.DEFAULT_FOLD}", "validation",
    )


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="分割结果可视化")
    p.add_argument("--nnunet_raw", default=utils.DEFAULT_NNUNET_RAW)
    p.add_argument("--nnunet_preprocessed", default=utils.DEFAULT_NNUNET_PREPROCESSED)
    p.add_argument("--nnunet_results", default=utils.DEFAULT_NNUNET_RESULTS)
    p.add_argument("--pred_dir", default=None, help="默认使用 fold_0 validation 预测")
    p.add_argument("--gt_dir", default=None,
                   help="GT 目录，默认用 nnUNet_preprocessed/gt_segmentations")
    p.add_argument("--out_dir", default=utils.FIGURES_DIR)
    p.add_argument("--num_samples", type=int, default=6,
                   help="随机抽取的样本数（--all 时忽略）")
    p.add_argument("--all", action="store_true", help="可视化全部样本")
    p.add_argument("--cases", nargs="*", default=None, help="指定 case 列表")
    p.add_argument("--seed", type=int, default=42, help="随机种子（保证可复现）")
    p.add_argument("--dpi", type=int, default=150)
    return p


def make_class_cmap():
    colors = [(0, 0, 0, 0)]
    for c in utils.FOREGROUND_IDS:
        r, g, b = utils.CLASS_COLORS[c]
        colors.append((r, g, b, 1.0))
    cmap = ListedColormap(colors)
    bounds = [-0.5] + [i + 0.5 for i in range(len(utils.FOREGROUND_IDS) + 1)]
    norm = BoundaryNorm(bounds, cmap.N)
    return cmap, norm


def overlay_mask(ax, base_img, mask, alpha=0.5):
    cmap, norm = make_class_cmap()
    ax.imshow(base_img, cmap="gray")
    masked = np.ma.masked_where(mask == 0, mask)
    ax.imshow(masked, cmap=cmap, norm=norm, alpha=alpha, interpolation="nearest")


def difference_map(ax, base_img, pred, gt):
    """前景整体差异图：绿=TP，红=FN(漏检)，蓝=FP(误检)。"""
    pred_fg = pred > 0
    gt_fg = gt > 0
    tp = np.logical_and(pred_fg, gt_fg)
    fn = np.logical_and(np.logical_not(pred_fg), gt_fg)
    fp = np.logical_and(pred_fg, np.logical_not(gt_fg))

    ax.imshow(base_img, cmap="gray")
    overlay = np.zeros((*base_img.shape, 4), dtype=float)
    overlay[tp] = [0.0, 1.0, 0.0, 0.6]
    overlay[fn] = [1.0, 0.0, 0.0, 0.6]
    overlay[fp] = [0.0, 0.4, 1.0, 0.6]
    ax.imshow(overlay, interpolation="nearest")


def visualize_case(case: str, args, pred_dir) -> bool:
    img_p = utils.image_path(args.nnunet_raw, case, subdir="imagesTr")
    pred_p = utils.prediction_path(pred_dir, case)
    gt_p = (os.path.join(args.gt_dir, f"{case}{utils.FILE_ENDING}")
            if args.gt_dir else utils.gt_seg_path(args.nnunet_preprocessed, case))

    for p, name in [(img_p, "原图"), (pred_p, "预测"), (gt_p, "GT")]:
        if not os.path.exists(p):
            print(f"[跳过] {case}: 缺少{name} ({p})")
            return False

    img = utils.to_2d(utils.read_nii_array(img_p)[0]).astype(float)
    pred = utils.to_2d(utils.read_nii_array(pred_p)[0]).astype(int)
    gt = utils.to_2d(utils.read_nii_array(gt_p)[0]).astype(int)

    if img.max() > img.min():
        img_disp = (img - img.min()) / (img.max() - img.min())
    else:
        img_disp = img

    fig, axes = plt.subplots(1, 5, figsize=(25, 5.5))

    axes[0].imshow(img_disp, cmap="gray")
    axes[0].set_title("Input", fontsize=13)

    overlay_mask(axes[1], img_disp, gt, alpha=0.55)
    axes[1].set_title("Ground Truth", fontsize=13)

    overlay_mask(axes[2], img_disp, pred, alpha=0.55)
    axes[2].set_title("Prediction", fontsize=13)

    overlay_mask(axes[3], img_disp, pred, alpha=0.45)
    try:
        axes[3].contour(gt > 0, colors="white", linewidths=0.8)
    except Exception:
        pass
    axes[3].set_title("Overlay (Pred fill + GT contour)", fontsize=13)

    difference_map(axes[4], img_disp, pred, gt)
    axes[4].set_title("Difference Map", fontsize=13)

    for ax in axes:
        ax.axis("off")

    class_legend = [Patch(facecolor=utils.CLASS_COLORS[c], label=utils.ID_TO_NAME[c])
                    for c in utils.FOREGROUND_IDS]
    diff_legend = [
        Patch(facecolor=(0, 1, 0), label="TP (correct)"),
        Patch(facecolor=(1, 0, 0), label="FN (missed)"),
        Patch(facecolor=(0, 0.4, 1), label="FP (false)"),
    ]
    fig.legend(handles=class_legend + diff_legend, loc="lower center",
               ncol=7, fontsize=10, frameon=False, bbox_to_anchor=(0.5, -0.02))

    fig.suptitle(f"Case: {case}", fontsize=14)
    fig.tight_layout(rect=[0, 0.04, 1, 0.98])

    out_path = os.path.join(args.out_dir, f"vis_{case}.png")
    fig.savefig(out_path, dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"[保存] {out_path}")
    return True


def resolve_cases(args, pred_dir) -> List[str]:
    if args.cases:
        return args.cases
    preds = sorted([f[:-len(utils.FILE_ENDING)] for f in os.listdir(pred_dir)
                    if f.endswith(utils.FILE_ENDING)])
    if args.all or args.num_samples >= len(preds):
        return preds
    random.seed(args.seed)
    return sorted(random.sample(preds, args.num_samples))


def main():
    args = build_argparser().parse_args()
    utils.ensure_dirs(args.out_dir)

    pred_dir = args.pred_dir or default_validation_pred_dir()
    cases = resolve_cases(args, pred_dir)
    if not cases:
        print(f"[错误] {pred_dir} 中没有预测结果")
        return

    print(f"预测目录: {pred_dir}")
    print(f"将可视化 {len(cases)} 个病例：{', '.join(cases)}")
    ok = 0
    for case in cases:
        if visualize_case(case, args, pred_dir):
            ok += 1
    print(f"\n完成：成功生成 {ok}/{len(cases)} 张对比图，保存在 {args.out_dir}")


if __name__ == "__main__":
    main()
