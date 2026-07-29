"""
evaluate.py
===========
读取推理预测与 GT，计算每病例 / 每类别 / 整体的分割指标，并生成论文级结果表。

默认输入：fold_0 held-out 验证集的预测（训练时 nnU-Net 已生成，位于
nnUNet_results/.../fold_0/validation/），GT 为 nnUNet_preprocessed/.../gt_segmentations。
也可用 --pred_dir/--gt_dir 指向任意目录（如新推理输出或真实 imagesTs 标签）。

输出（默认写到 metrics/ 目录）
------------------------------
metrics/per_case_metrics.csv        每病例 × 每类别 的全部指标（长表）
metrics/per_case_foreground.csv     每病例的前景平均指标（论文 Case 级表）
metrics/summary_overall.csv         前景整体的 Mean/Std/Median/Min/Max
metrics/summary_per_class.csv       每类别的 Mean/Std/Median/Min/Max
metrics/paper_table.md              论文可直接粘贴的统计表
metrics/summary.json                所有汇总结果（机器可读）

运行示例
--------
python evaluate.py                                  # 用 fold_0 validation 预测
python evaluate.py --pred_dir outputs/predictions   # 用自定义推理输出
"""

import os
import json
import argparse
import datetime
from typing import Dict, List

import numpy as np
import pandas as pd

import utils
import metrics as M


def default_validation_pred_dir() -> str:
    """训练时生成的 fold_0 验证集预测目录。"""
    return os.path.join(
        utils.get_model_folder(utils.DEFAULT_NNUNET_RESULTS),
        f"fold_{utils.DEFAULT_FOLD}", "validation",
    )


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="nnU-Net 预测结果评估")
    p.add_argument("--nnunet_raw", default=utils.DEFAULT_NNUNET_RAW)
    p.add_argument("--nnunet_preprocessed", default=utils.DEFAULT_NNUNET_PREPROCESSED)
    p.add_argument("--nnunet_results", default=utils.DEFAULT_NNUNET_RESULTS)
    p.add_argument("--pred_dir", default=None,
                   help="预测结果目录。默认使用 fold_0 validation 预测")
    p.add_argument("--gt_dir", default=None,
                   help="GT 目录。默认使用 nnUNet_preprocessed 的 gt_segmentations")
    p.add_argument("--out_dir", default=utils.METRICS_DIR, help="指标输出目录")
    p.add_argument("--fold", type=int, default=utils.DEFAULT_FOLD)
    p.add_argument("--cases", nargs="*", default=None,
                   help="显式指定评估的 case 列表；默认自动匹配 pred_dir 中所有预测")
    return p


def resolve_cases(args, pred_dir) -> List[str]:
    if args.cases:
        return args.cases
    preds = [f[:-len(utils.FILE_ENDING)] for f in os.listdir(pred_dir)
             if f.endswith(utils.FILE_ENDING)]
    return sorted(preds)


def gt_path_for(args, case: str) -> str:
    if args.gt_dir:
        return os.path.join(args.gt_dir, f"{case}{utils.FILE_ENDING}")
    return utils.gt_seg_path(args.nnunet_preprocessed, case)


def main():
    args = build_argparser().parse_args()
    utils.ensure_dirs(args.out_dir)

    pred_dir = args.pred_dir or default_validation_pred_dir()
    class_ids = utils.FOREGROUND_IDS
    cases = resolve_cases(args, pred_dir)
    if not cases:
        print(f"[错误] 在 {pred_dir} 未找到任何预测结果 (.nii.gz)")
        return

    print(f"预测目录: {pred_dir}")
    print(f"评估 {len(cases)} 个病例，类别: {class_ids}")

    long_rows: List[dict] = []
    fg_rows: List[dict] = []
    collect: Dict[str, Dict[str, List[float]]] = {}

    def add_collect(scope: str, metric: str, value: float):
        collect.setdefault(scope, {}).setdefault(metric, []).append(value)

    skipped = []
    for case in cases:
        pred_p = utils.prediction_path(pred_dir, case)
        gt_p = gt_path_for(args, case)
        if not os.path.exists(pred_p):
            skipped.append((case, "缺预测")); continue
        if not os.path.exists(gt_p):
            skipped.append((case, "缺GT")); continue

        pred_arr, _ = utils.read_nii_array(pred_p)
        gt_arr, gt_spacing = utils.read_nii_array(gt_p)
        pred_arr = utils.to_2d(pred_arr)
        gt_arr = utils.to_2d(gt_arr)
        spacing2d = utils.get_2d_spacing(gt_spacing)

        per_class = M.evaluate_case(pred_arr, gt_arr, class_ids, spacing2d)
        fg = M.foreground_mean(per_class, class_ids)

        for c in class_ids:
            row = {"case": case, "class_id": c, "class_name": utils.ID_TO_NAME[c]}
            for metric in M.ALL_METRICS:
                v = per_class[c].get(metric, float("nan"))
                row[metric] = v
                add_collect(f"class_{c}", metric, v)
            long_rows.append(row)

        fg_row = {"case": case}
        for metric in M.ALL_METRICS:
            fg_row[metric] = fg[metric]
            add_collect("overall", metric, fg[metric])
        fg_rows.append(fg_row)

    if skipped:
        print("[警告] 跳过以下病例：")
        for c, reason in skipped:
            print(f"   - {c}: {reason}")

    if not fg_rows:
        print("[错误] 没有可评估的病例（预测/GT 均缺失）。")
        return

    df_long = pd.DataFrame(long_rows)
    df_fg = pd.DataFrame(fg_rows)
    per_case_metrics_csv = os.path.join(args.out_dir, "per_case_metrics.csv")
    per_case_fg_csv = os.path.join(args.out_dir, "per_case_foreground.csv")
    df_long.to_csv(per_case_metrics_csv, index=False, encoding="utf-8-sig")
    df_fg.to_csv(per_case_fg_csv, index=False, encoding="utf-8-sig")

    overall_summary = {m: M.summarize(collect["overall"][m]) for m in M.ALL_METRICS}
    per_class_summary = {
        c: {m: M.summarize(collect.get(f"class_{c}", {}).get(m, [])) for m in M.ALL_METRICS}
        for c in class_ids
    }

    df_overall = pd.DataFrame([{"Metric": m, **overall_summary[m]} for m in M.ALL_METRICS])
    overall_csv = os.path.join(args.out_dir, "summary_overall.csv")
    df_overall.to_csv(overall_csv, index=False, encoding="utf-8-sig")

    pc_rows = []
    for c in class_ids:
        for m in M.ALL_METRICS:
            pc_rows.append({"class_id": c, "class_name": utils.ID_TO_NAME[c],
                            "Metric": m, **per_class_summary[c][m]})
    df_per_class = pd.DataFrame(pc_rows)
    per_class_csv = os.path.join(args.out_dir, "summary_per_class.csv")
    df_per_class.to_csv(per_class_csv, index=False, encoding="utf-8-sig")

    paper_md = build_paper_markdown(overall_summary, per_class_summary, class_ids, len(fg_rows))
    paper_md_path = os.path.join(args.out_dir, "paper_table.md")
    with open(paper_md_path, "w", encoding="utf-8") as f:
        f.write(paper_md)

    summary_json = {
        "generated_at": datetime.datetime.now().isoformat(),
        "dataset": utils.DATASET_NAME,
        "configuration": utils.CONFIGURATION,
        "trainer": utils.TRAINER,
        "fold": args.fold,
        "pred_dir": pred_dir,
        "n_cases_evaluated": len(fg_rows),
        "cases": [r["case"] for r in fg_rows],
        "class_ids": class_ids,
        "class_names": {c: utils.ID_TO_NAME[c] for c in class_ids},
        "overall_summary": overall_summary,
        "per_class_summary": {str(c): per_class_summary[c] for c in class_ids},
        "skipped": skipped,
    }
    summary_json_path = os.path.join(args.out_dir, "summary.json")
    with open(summary_json_path, "w", encoding="utf-8") as f:
        json.dump(summary_json, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 60)
    print(f"评估完成：{len(fg_rows)} 个病例")
    print("=" * 60)
    print(paper_md)
    print("=" * 60)
    print("输出文件：")
    for p in [per_case_metrics_csv, per_case_fg_csv, overall_csv,
              per_class_csv, paper_md_path, summary_json_path]:
        print(f"  - {p}")


def _fmt(mean: float, std: float, pct: bool = False) -> str:
    if np.isnan(mean):
        return "N/A"
    if pct:
        return f"{mean*100:.2f} ± {std*100:.2f}"
    return f"{mean:.2f} ± {std:.2f}"


def build_paper_markdown(overall_summary, per_class_summary, class_ids, n_cases) -> str:
    lines = []
    lines.append(f"# nnU-Net 测试评估结果（{utils.DATASET_NAME}, {utils.CONFIGURATION}, fold_{utils.DEFAULT_FOLD}）")
    lines.append("")
    lines.append(f"评估样本数: **{n_cases}**（fold_0 held-out 验证集）")
    lines.append("")

    lines.append("## Table 1. Overall (foreground-averaged) performance")
    lines.append("")
    lines.append("| Metric | Mean ± Std | Median | Min | Max |")
    lines.append("|---|---|---|---|---|")
    pct_metrics = set(M.OVERLAP_METRICS)
    for m in M.ALL_METRICS:
        s = overall_summary[m]
        is_pct = m in pct_metrics
        mean_std = _fmt(s["Mean"], s["Std"], pct=is_pct)
        if np.isnan(s["Median"]):
            med = mn = mx = "N/A"
        elif is_pct:
            med = f"{s['Median']*100:.2f}"; mn = f"{s['Min']*100:.2f}"; mx = f"{s['Max']*100:.2f}"
        else:
            med = f"{s['Median']:.2f}"; mn = f"{s['Min']:.2f}"; mx = f"{s['Max']:.2f}"
        unit = " (%)" if is_pct else (" (px)" if m in M.BOUNDARY_METRICS else "")
        lines.append(f"| {m}{unit} | {mean_std} | {med} | {mn} | {mx} |")
    lines.append("")

    lines.append("## Table 2. Per-class performance (Mean ± Std)")
    lines.append("")
    lines.append("| Class | Dice (%) | IoU (%) | HD95 (px) | ASD (px) |")
    lines.append("|---|---|---|---|---|")
    for c in class_ids:
        d = per_class_summary[c]["Dice"]; i = per_class_summary[c]["IoU"]
        h = per_class_summary[c]["HD95"]; a = per_class_summary[c]["ASD"]
        lines.append(
            f"| {utils.ID_TO_NAME[c]} | {_fmt(d['Mean'], d['Std'], True)} | "
            f"{_fmt(i['Mean'], i['Std'], True)} | {_fmt(h['Mean'], h['Std'])} | "
            f"{_fmt(a['Mean'], a['Std'])} |"
        )
    lines.append("")

    lines.append("## Table 3. Per-class overlap metrics (Mean ± Std, %)")
    lines.append("")
    header = "| Class | " + " | ".join(M.OVERLAP_METRICS) + " |"
    sep = "|---|" + "|".join(["---"] * len(M.OVERLAP_METRICS)) + "|"
    lines.append(header); lines.append(sep)
    for c in class_ids:
        vals = [_fmt(per_class_summary[c][m]["Mean"], per_class_summary[c][m]["Std"], True)
                for m in M.OVERLAP_METRICS]
        lines.append(f"| {utils.ID_TO_NAME[c]} | " + " | ".join(vals) + " |")
    lines.append("")
    lines.append("> 注: 重叠类指标以百分比(%)显示；边界类指标(HD/HD95/ASD)以像素(px)为单位"
                 "(数据集 spacing=[1,1])。距离指标在某类别 GT 或预测为空时记为 N/A 并从统计中剔除。")
    lines.append("")
    return "\n".join(lines)


if __name__ == "__main__":
    main()
