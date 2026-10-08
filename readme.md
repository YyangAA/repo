# 5.0T 膝关节软骨损伤自动分级系统 (Knee Cartilage Damage Classification)

基于 **5.0T 膝关节 MRI** 的软骨损伤自动识别与分级系统。从原始 DICOM 一路完成
**软骨自动分割 → 三维影像组学特征提取 → 级联 SVM 分类（二分类 + 分级）→ 损伤热力图可视化**，
全流程无需人工干预。

> 本仓库基于 nnU-Net 框架扩展，分割部分使用 nnU-Net，分类/评估/可视化部分位于 `repo/` 目录，
> 均为本项目自研代码。

---

## 1. 流程总览 (Pipeline)

```
原始 DICOM
   │
   ▼
[Stage 1] 软骨自动分割 (nnU-Net, 2D)  +  3D 重构
   │            → image_3d / mask_3d (nii.gz, 逐例)
   ▼
[Stage 2] 三维影像组学特征提取 (PyRadiomics)  +  跨区域特征增强
   │            → radiomics features (original + wavelet + cross/ratio/one-hot)
   ▼
[Stage 3] 级联 SVM 分类
   │      Stage 1: 正常 (G0) vs 损伤 (G1/G2)  —— 4 区域独立 SVM-RBF
   │      Stage 2: 轻度 (G1) vs 严重 (G2)     —— 池化分级 SVM
   ▼
[Stage 4] 可视化诊断报告
             → 每例: 分割叠加图 + 损伤概率热力图 + 诊断面板
             → 汇总: ROC/混淆矩阵/指标表（二分类 + 三分类）
```

四个软骨亚区：股骨内侧 (FM/MFC)、股骨外侧 (FL/LFC)、胫骨内侧 (TM/MTP)、胫骨外侧 (TL/LTP)。

当前模型版本：**`results_v8.9_0702_v2`**（`repo/checkpoint/results_v8.9_0702_v2/`）。

---

## 2. 目录结构

```
repo/
├── pipeline.sh                        # 【一键】端到端推理：分割 → 分类 → 可视化
│
├── train/
│   ├── segmentation/                  # 分割训练数据准备 (DICOM → npy → nii)
│   └── classify/
│       └── dev_0702_v2/               # ★ 分类训练 (当前版本)
│           ├── run_train.sh           #   一键训练：LASSO 选择 → 跨区域增强 → SVM 级联
│           ├── 1_get_feature_v8.py    #   Step0: PyRadiomics 特征提取
│           ├── 1a_lasso_v3.py         #   Step1: LASSO 特征选择 (Stage1+Stage2)
│           ├── 1b_add_cross_features_v8.py  # Step2: 跨区域/比率/one-hot 特征增强
│           ├── 2_lasso_v8.py          #   Step3: 二轮 LASSO
│           ├── 3_train_svm_v8.py      #   Step4: SVM 级联训练 (GroupKFold CV + Platt 校准)
│           ├── data_train/            #   训练特征 CSV
│           ├── in_domain_cv_eval.py       # ★ 集内测试: Stage1 OOF 主指标+图
│           ├── in_domain_stage_eval.py    #   集内: Stage1/Stage2 分阶段指标
│           ├── in_domain_stage2_auc.py    # ★ 集内: Stage2 Pooled AUC (复用 plot_roc_paper 保证图=表)
│           ├── plot_roc_paper.py          # ★ 论文 ROC 图 (Stage1 读 OOF; Stage2 pooled)
│           └── README_集内测试与外部验证.md  # 评估复现详细说明
│
├── infer/
│   ├── segmentation/                  # 分割推理: DICOM → npy → nii → nnU-Net → 3D 重构
│   │   ├── 1_dcm2npy.py / 2_npy2nii.py / 4_vis.py / 5__nii23D.py
│   │   └── evaluation/                # 分割评估 (Dice/HD95 等) + 复现文档
│   │       ├── evaluate.py / metrics.py / utils.py / visualize.py
│   │       └── REPRODUCE_Table3.md
│   └── classify/
│       ├── run_inference.sh           # 【一键】外部验证: 推理 → 过滤 → 报告 + 指标
│       ├── SVM_RBF_inference_pipeline_v8_v2.py  # 级联推理引擎 (特征+Stage1+Stage2+后处理)
│       ├── visualize_report_v8.py     # 诊断报告绘图 (含三分类 3-Cls Acc 汇总表)
│       └── external_stage2_auc.py     # 外部验证 Stage2 AUC
│
├── checkpoint/results_v8.9_0702_v2/   # 当前模型 (4 区域 × models/*.pkl)
└── data/                              # 数据 (image_3d / mask_3d / GT Excel / 结果)
```

---

## 3. 环境

| 组件 | 环境 | 用途 |
|---|---|---|
| nnU-Net | conda env `knee_yx` | 软骨分割 (Stage 1) |
| 分类/可视化 | `repo/venv310` (Python 3.10) | 特征提取、SVM 分类、报告 |

依赖：SimpleITK、PyRadiomics、scikit-learn、pandas、numpy、matplotlib、pillow、pypinyin。

---

## 4. 快速开始

### 4.1 端到端推理（新病例 DICOM → 诊断报告）

```bash
cd repo
bash pipeline.sh
```

一条命令完成：分割 (nnU-Net) → 3D 重构 → 级联分类 → 可视化诊断报告。
输出：分割结果、每例诊断报告图（分割叠加 + 损伤热力图 + 诊断面板）。

### 4.2 分类模型训练（已有特征 CSV）

```bash
cd repo
bash train/classify/dev_0702_v2/run_train.sh
```

从 `data_train/knee_radiomics_features_3d_integrated.csv` 出发，完成 LASSO 选择 → 跨区域增强 →
SVM 级联训练（GroupKFold 5 折 CV + Platt 概率校准），模型保存至 `checkpoint/results_v8.9_0702_v2/`。

### 4.3 外部验证（独立测试集）

```bash
cd repo
bash infer/classify/run_inference.sh
```

推理 → 按排除名单过滤 → 生成每例报告 + `summary_metrics.png`（ROC + 指标表，含三分类
3-Cls Acc）+ `confusion_matrices.png` + `summary_metrics.csv`。

### 4.4 集内评估（交叉验证 OOF）

```bash
cd repo
venv310/bin/python train/classify/dev_0702_v2/in_domain_cv_eval.py    # Stage1 主指标 + 图
venv310/bin/python train/classify/dev_0702_v2/in_domain_stage_eval.py # Stage1/2 分阶段指标
venv310/bin/python train/classify/dev_0702_v2/in_domain_stage2_auc.py # Stage2 Pooled AUC
```

说明：训练集全部用于训练（无独立留出集），集内评估采用 **GroupKFold 五折交叉验证的
out-of-fold (OOF) 预测**，患者级不泄漏。

### 4.5 论文 ROC 图

```bash
cd repo
venv310/bin/python train/classify/dev_0702_v2/plot_roc_paper.py
```

Stage 1 直接读取 OOF CSV（图中 AUC 与集内指标表严格一致）；
Stage 2 输出 pooled ROC（AUC=0.873，与表格一致）。
输出：`checkpoint/results_v8.9_0702_v2_paper_figures/`。

---

## 5. 主要结果

**集内交叉验证（GroupKFold 5 折 OOF, n=117/区域）**

| 阶段 | 区域 | AUC | Acc | Sens | Spec |
|---|---|---|---|---|---|
| Stage 1 | FM | 0.899 | 0.846 | 0.786 | 0.880 |
| | FL | 0.938 | 0.897 | 0.462 | 0.952 |
| | TM | 0.945 | 0.915 | 0.714 | 0.958 |
| | TL | 0.892 | 0.897 | 0.619 | 0.958 |
| Stage 2 (Pooled) | 全区域 | 0.873 | 0.887 | 0.774 | 0.939 |

**端到端外部验证（DICOM → 分割 → 分类 → 可视化, n=20/区域）**

| 阶段 | 区域 | AUC | Acc | 3-Cls Acc |
|---|---|---|---|---|
| Stage 1 | FM | 0.905 | 0.900 | 0.900 |
| | FL | 0.976 | 0.900 | 0.900 |
| | TM | 0.950 | 0.900 | 0.900 |
| | TL | 1.000 | 0.950 | 0.900 |

> 详细的评估协议、口径说明与逐步复现命令，见
> `repo/train/classify/dev_0702_v2/README_集内测试与外部验证.md`。

---

## 6. 备注

- 分割模型基于 [nnU-Net](https://github.com/MIC-DKFZ/nnUNet)（官方文档见
  `documentation/`），如需引用请同时引用 nnU-Net 原论文。
- 分类/评估/可视化代码位于 `repo/` 下，使用本流程请遵循仓库 LICENSE。
- 评估中 Stage 1 与 Stage 2 的图、表、脚本均保持同一统计口径（OOF / pooled），保证可复现与自洽。

