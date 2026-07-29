# 复现 Table 3（每类别重叠指标）说明

本文件说明如何从零复现 `metrics/paper_table.md` 中的 **Table 3. Per-class overlap metrics (Mean ± Std, %)**。

Table 3 报告 4 个前景类别在 fold_0 held-out 验证集（12 例）上的 6 个重叠类指标：
**Dice、IoU、Precision、Recall、Specificity、Accuracy**（均以百分比 % 显示，Mean ± Std）。

目标结果：

| Class | Dice | IoU | Precision | Recall | Specificity | Accuracy |
|---|---|---|---|---|---|---|
| Femoral_Medial | 82.66 ± 13.43 | 72.11 ± 15.90 | 85.98 ± 12.24 | 80.00 ± 15.30 | 99.98 ± 0.01 | 99.95 ± 0.01 |
| Femoral_Lateral | 89.68 ± 3.78 | 81.48 ± 5.84 | 89.91 ± 7.13 | 89.94 ± 4.00 | 99.97 ± 0.02 | 99.95 ± 0.02 |
| Tibial_Medial | 83.47 ± 7.63 | 72.29 ± 10.99 | 81.91 ± 12.75 | 86.61 ± 7.78 | 99.97 ± 0.02 | 99.94 ± 0.02 |
| Tibial_Lateral | 92.08 ± 2.13 | 85.38 ± 3.62 | 91.35 ± 4.51 | 93.03 ± 2.92 | 99.97 ± 0.02 | 99.94 ± 0.02 |

---

## 1. 前置条件（输入数据，均为已有的持久文件）

| 输入 | 路径 |
|---|---|
| 预测（fold_0 验证集，训练时 nnU-Net 自动生成，12 个 `<case>.nii.gz`） | `nnUNet_results/Dataset260426_Knee2D/nnUNetTrainer_FreezeEncoder__nnUNetPlans__2d/fold_0/validation/` |
| Ground Truth（12 个 `<case>.nii.gz`） | `nnUNet_preprocessed/Dataset260426_Knee2D/gt_segmentations/` |
| 类别定义 | `nnUNet_raw/Dataset260426_Knee2D/dataset.json`（1=Femoral_Medial, 2=Femoral_Lateral, 3=Tibial_Medial, 4=Tibial_Lateral） |

> 评估直接复用训练已生成的验证集预测，**无需重新推理、无需 GPU/torch/nnU-Net**。

12 个评估病例：崔泽明_0、成信元_0、成信元_1、樊明利_0、樊明利_2、武小侠_2、王立安_0、王立安_2、祝昆_0、祝昆_2、程小英_2、范森宁_2。

---

## 2. 环境

Table 3 只用重叠类指标，仅需 `numpy / SimpleITK / pandas`（`scipy` 供其它边界指标使用）。
仓库自带的 `repo/venv310` 已满足，无需额外安装：

```bash
PY=/mnt/tidal-alsh-share2/dataset/askfollow/usr/yangxin/keyan/repo-dev_v4/repo/venv310/bin/python
```

---

## 3. 一条命令复现

```bash
cd /mnt/tidal-alsh-share2/dataset/askfollow/usr/yangxin/keyan/repo-dev_v4/repo/infer/segmentation/evaluation
/mnt/tidal-alsh-share2/dataset/askfollow/usr/yangxin/keyan/repo-dev_v4/repo/venv310/bin/python evaluate.py
```

运行后，Table 3 会：
1. 直接打印在终端（`paper_table.md` 内容的一部分）；
2. 写入文件 `metrics/paper_table.md`（Markdown 版 Table 3）；
3. 对应的机读数据写入：
   - `metrics/summary_per_class.csv` —— 每类别每指标的 Mean/Std/Median/Min/Max/N（**Table 3 的数据源**）
   - `metrics/per_case_metrics.csv` —— 每病例 × 每类别 的原始指标（用于自行核算）
   - `metrics/summary.json` —— 全部结果的 JSON

---

## 4. Table 3 的计算口径（逐字段对应代码）

指标实现见 [`metrics.py`](metrics.py)，聚合与成表见 [`evaluate.py`](evaluate.py)。

### 4.1 单病例、单类别的二值指标

对每个病例、每个类别 `c`，先取二值掩码 `pred==c`、`gt==c`，统计混淆矩阵 TP/FP/TN/FN，再计算：

| 指标 | 公式 |
|---|---|
| Dice (DSC) | 2·TP / (2·TP + FP + FN) |
| IoU (Jaccard) | TP / (TP + FP + FN) |
| Precision | TP / (TP + FP) |
| Recall (Sensitivity) | TP / (TP + FN) |
| Specificity | TN / (TN + FP) |
| Accuracy | (TP + TN) / (TP + TN + FP + FN) |

边界情况：某类别在该病例中 **GT 与预测都为空** → 该类别 6 个指标记为 `NaN`，并从统计中剔除（不计入均值）。

### 4.2 跨 12 个病例聚合（得到 Table 3 的 Mean ± Std）

对每个类别、每个指标，收集 12 个病例的值，用 **nanmean / 样本标准差(ddof=1)** 聚合，再乘 100 转百分比：

- `Mean = numpy.nanmean(values)`
- `Std  = numpy.nanstd(values, ddof=1)`（样本标准差）
- 表中显示：`{Mean*100:.2f} ± {Std*100:.2f}`

即 `summary_per_class.csv` 中 `Metric` 为 Dice/IoU/Precision/Recall/Specificity/Accuracy 的行，其 `Mean`、`Std` 列 ×100 即 Table 3 的数值。

---

## 5. 手动核对（可选）

若只想从 CSV 复算某个类别某指标，例如 Femoral_Lateral 的 Dice：

```bash
PY=/mnt/tidal-alsh-share2/dataset/askfollow/usr/yangxin/keyan/repo-dev_v4/repo/venv310/bin/python
$PY - <<'EOF'
import pandas as pd, numpy as np
df = pd.read_csv("metrics/per_case_metrics.csv")
sub = df[df.class_name == "Femoral_Lateral"]["Dice"].to_numpy()
print("Mean±Std = %.2f ± %.2f (%%), N=%d" %
      (np.nanmean(sub)*100, np.nanstd(sub, ddof=1)*100, np.sum(~np.isnan(sub))))
# 期望: 89.68 ± 3.78 (%)
EOF
```

也可直接查看聚合结果：

```bash
$PY - <<'EOF'
import pandas as pd
df = pd.read_csv("metrics/summary_per_class.csv")
overlap = ["Dice","IoU","Precision","Recall","Specificity","Accuracy"]
t3 = df[df.Metric.isin(overlap)].copy()
t3["Mean%"] = (t3["Mean"]*100).round(2)
t3["Std%"]  = (t3["Std"]*100).round(2)
print(t3[["class_name","Metric","Mean%","Std%"]].to_string(index=False))
EOF
```

---

## 6. 输出文件位置汇总

| 文件 | 说明 |
|---|---|
| `metrics/paper_table.md` | Table 1/2/3 的 Markdown（Table 3 在第三节） |
| `metrics/summary_per_class.csv` | **Table 3 的数据源**（每类别每指标 Mean/Std/Median/Min/Max） |
| `metrics/per_case_metrics.csv` | 每病例 × 每类别 原始指标，用于手动核算 |
| `metrics/summary.json` | 机读完整结果 |

---

## 7. 一致性校验

- 4 个类别 Dice 的平均（(82.66+89.68+83.47+92.08)/4 ≈ 86.97%）等于 Table 1 的 Overall Dice 86.97%；
- 该值与训练产物 `fold_0/validation/summary.json` 的 `foreground_mean.Dice`(0.8697) 完全一致，证明 Table 3 的重叠指标口径与 nnU-Net 官方一致、复现正确。
