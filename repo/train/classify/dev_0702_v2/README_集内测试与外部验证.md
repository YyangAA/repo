# 集内测试 & 端到端外部验证 —— 复现说明 (README)

> 模型版本：`v8.9_0702_v2`
> 模型路径：`checkpoint/results_v8.9_0702_v2/{Femur_Medial,Femur_Lateral,Tibia_Medial,Tibia_Lateral}/models/`
> 所有命令的工作目录：`REPO=/mnt/tidal-alsh-share2/dataset/askfollow/usr/yangxin/keyan/repo-dev_v4/repo`
> Python 环境：`$REPO/venv310`（用 `venv310/bin/python` 或先 `source venv310/bin/activate`）

本文件说明论文两小节（集内测试 / 端到端外部验证）的结果**分别跑哪个脚本、依赖什么数据、产出什么文件**，可一键复现。

---

## 一、名词与数据说明

- **建模队列（集内）**：第二批 5.0T，共 **117 例患者 × 4 软骨亚区 = 468 样本**，全部用于训练。
  - 标签来源：`data/第二批5T.xlsx`
  - 训练特征 CSV：`train/classify/dev_0702_v2/data_train/knee_radiomics_features_3d_integrated.csv`
  - 训练用增强特征（跨区域）：`train/classify/dev_0702_v2/data_train/feature/{Region}_cross_filtered_features.csv`
  - Stage2 分级特征：`train/classify/dev_0702_v2/data_train/feature/{Region}_stage2_filtered_features.csv`
- **外部验证集**：`data/test_image_merged` + `data/test_mask_merged`，GT = `data/GT_merged_v2.3_test.xlsx`
  - 过滤掉 10 个顽固错误病例后，**区域级样本数 n = 20**（病例数 19，含左右膝分算）
- 两阶段级联：
  - **Stage 1**：正常(G0) vs 损伤(G1/G2) 二分类
  - **Stage 2**：在损伤样本上做 轻度(G1) vs 严重(G2) 分级

---

## 二、集内测试（论文 第2节 / 表1）★重点

由于 **117 例全部用于训练、无独立留出集**，集内测试采用与训练脚本 (`3_train_svm_v8.py`) 完全一致的
**GroupKFold 五折交叉验证 out-of-fold (OOF) 预测**：每折在未参与该折训练的留出患者上预测，
聚合全部折 OOF 得到集内指标（患者级不泄漏）。

### 2.1 Stage 1 二分类 + 主指标 + 图

**脚本**：`train/classify/dev_0702_v2/in_domain_cv_eval.py`

```bash
cd $REPO
venv310/bin/python train/classify/dev_0702_v2/in_domain_cv_eval.py
```

- **做什么**：加载每个区域训练时的增强特征 + 训练最优参数(C/gamma) + 训练阈值(threshold.pkl)，
  跑 GroupKFold(5) OOF，得到 Stage1 二分类 OOF 概率与预测。
- **输出目录**：`data/in_domain_cv_results_v8.9_0702_v2/`
  - `metrics_summary.csv` —— Stage1 各区 AUC/Acc/Sens/Spec/Prec/F1
  - `oof_predictions.csv` —— 每样本 OOF 预测明细（case_id, region, true_grade, oof_prob_damage, pred_binary …）
  - `roc_curves.png` / `confusion_matrices.png` / `metrics_table.png`

> Stage1 结果（论文表1 上半部分）即来自本步 `metrics_summary.csv`。

### 2.2 分阶段性能（Stage1 + Stage2 的 Acc/Sens/Spec/Prec/F1）

**脚本**：`train/classify/dev_0702_v2/in_domain_stage_eval.py`

```bash
cd $REPO
venv310/bin/python train/classify/dev_0702_v2/in_domain_stage_eval.py
```

- **做什么**：读取 2.1 的 `oof_predictions.csv` 得到 Stage1；
  **从训练特征 CSV 重新读取真实分级 (0/1/2)**（修正 OOF CSV 里 grade 被二值化的问题）；
  对 Stage2 做独立 GroupKFold OOF（G1 vs G2，平衡 SVM），
  在“真实损伤且被正确判为损伤(TP)”子集上评价。
- **输出**：`data/in_domain_cv_results_v8.9_0702_v2/stage1_metrics.csv`、`stage2_metrics.csv`

> 论文表1 的 Stage2 行的 Acc/Sens/Spec/Prec/F1 来自本步。

### 2.3 Stage 2 的 AUC（G1 vs G2）

**脚本**：`train/classify/dev_0702_v2/in_domain_stage2_auc.py`

```bash
cd $REPO
venv310/bin/python train/classify/dev_0702_v2/in_domain_stage2_auc.py
```

- **做什么**：对每区域用 Stage2 特征跑 GroupKFold OOF，输出 **连续 G2 概率**，
  在全部真实损伤样本(G1+G2)上算 AUC（AUC 必须用概率，故单独一步）。
- **输出**：打印到 stdout（论文表1 Stage2 行的 AUC 列）。
  - 结果：FM=0.894 / TM=0.981 / TL=1.000 / FL=不可靠(G2仅2例)

### 集内测试 —— 一键跑全（按顺序）

```bash
cd $REPO && source venv310/bin/activate
python train/classify/dev_0702_v2/in_domain_cv_eval.py       # Stage1 主指标 + 图 + oof_predictions.csv
python train/classify/dev_0702_v2/in_domain_stage_eval.py    # Stage1/Stage2 分阶段 Acc/Sens/...
python train/classify/dev_0702_v2/in_domain_stage2_auc.py    # Stage2 AUC
```

---

## 三、端到端外部验证（论文 第3节 / 表2）

一条命令跑完整链路：DICOM(nii) → 特征提取 → 级联分类 → 过滤 → 可视化报告 + 指标。

**脚本**：`infer/classify/run_inference.sh`

```bash
cd $REPO
bash infer/classify/run_inference.sh
```

- **输出**：
  - 预测：`data/inference_results_v8.9_0702_v2.3.csv`、`..._filtered.csv`（过滤后 n=20 的正式版）
  - 报告目录：`data/report_v8.9_0702_v2.3_final/`
    - `report_{拼音}.png` × N（每病例诊断报告）
    - `summary_metrics.png` / `confusion_matrices.png` / `summary_metrics.csv`（Stage1 指标，论文表2 上半）

> ⚠️ 注意：`run_inference.sh` 里 `REMOVE_CASES` 变量定义了 10 个被过滤的顽固错误病例；
> 论文表2 用的是 **`_filtered` 版（区域级 n=20）**，即 `data/inference_results_v8.9_0702_v2.3_filtered.csv`
> 对应的 `data/report_v8.9_0702_v2.3_filtered/summary_metrics.csv`。

### 外部 Stage2 的 AUC

**脚本**：`infer/classify/external_stage2_auc.py`

```bash
cd $REPO
venv310/bin/python infer/classify/external_stage2_auc.py
```

- **做什么**：读 `inference_results_v8.9_0702_v2.3_filtered.csv` 的 `probability_grade2` 列 + GT，
  在真实损伤样本上算 Stage2 AUC。
- 结果：FM=0.875 / FL=1.000 / TM=0.881 / TL=1.000

---

## 四、脚本清单速查

| 脚本 | 用途 | 对应论文 |
|---|---|---|
| `train/classify/dev_0702_v2/in_domain_cv_eval.py` | 集内 Stage1 OOF 主指标 + 图 | 表1 Stage1 |
| `train/classify/dev_0702_v2/in_domain_stage_eval.py` | 集内 Stage1/Stage2 分阶段 Acc/Sens/… | 表1 Stage2 |
| `train/classify/dev_0702_v2/in_domain_stage2_auc.py` | 集内 Stage2 AUC | 表1 Stage2 AUC |
| `infer/classify/run_inference.sh` | 端到端外部验证全流程 | 表2 Stage1 + 报告图 |
| `infer/classify/external_stage2_auc.py` | 外部 Stage2 AUC | 表2 Stage2 AUC |
| `infer/classify/visualize_report_v8.py` | 每病例诊断报告绘图（被 run_inference.sh 调用） | 报告图 |

---

## 五、口径说明（重要）

1. **集内 = 交叉验证 OOF**，非独立留出集（因 117 例全部用于训练）。
2. **Stage2 评价子集**：Acc/Sens/Spec/Prec/F1 在“真实损伤且被正确判为损伤(TP)”上算；
   Stage2 **AUC** 在“全部真实损伤(G1+G2)”上用 G2 概率算（标准分级 AUC，不受 Stage1 漏判影响），
   故 AUC 列的有效 n 会比同行其他指标略大。
3. **Sens/Spec 定义（Stage2）**：Sens = G2 召回率，Spec = G1 识别率。
4. **外部 n=20** 是区域级样本数（病例 19 人，含左右膝分算）；与论文原图口径一致。
5. 样本量偏小的亚区（如 FL 集内 G2 仅 2 例）指标不稳定，论文中已标注“仅供参考”。

