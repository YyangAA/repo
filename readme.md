**English** | [中文](readme_zh.md)

# Automated Knee Cartilage Damage Grading on 5.0T MRI

An end-to-end system for automated detection and grading of knee cartilage damage on **5.0T knee MRI**.
Starting from raw DICOM, the pipeline performs **automatic cartilage segmentation → 3D radiomics feature
extraction → cascaded SVM classification (binary detection + severity grading) → damage-probability heatmap
visualization**, with no manual intervention.

> This repository extends the nnU-Net framework. Segmentation is based on nnU-Net; all classification,
> evaluation and visualization code developed for this project lives under `repo/`.

---

## 1. Pipeline Overview

```
Raw DICOM
   │
   ▼
[Stage 1] Automatic cartilage segmentation (nnU-Net, 2D) + 3D reconstruction
   │            → image_3d / mask_3d (NIfTI, per case)
   ▼
[Stage 2] 3D radiomics feature extraction (PyRadiomics) + cross-region feature augmentation
   │            → radiomics features (original + wavelet + cross-region / ratio / one-hot)
   ▼
[Stage 3] Cascaded SVM classification
   │      Stage 1: Normal (G0) vs. Damaged (G1/G2) — one SVM-RBF per region
   │      Stage 2: Mild (G1) vs. Severe (G2)       — pooled grading SVM
   ▼
[Stage 4] Visual diagnostic report
             → per case: segmentation overlay + damage-probability heatmap + diagnosis panel
             → summary: ROC curves / confusion matrices / metrics table (binary + 3-class)
```

Four cartilage subregions: medial femoral condyle (FM / MFC), lateral femoral condyle (FL / LFC),
medial tibial plateau (TM / MTP), and lateral tibial plateau (TL / LTP).

Current model version: **`results_v8.9_0702_v2`** (`repo/checkpoint/results_v8.9_0702_v2/`).

---

## 2. Repository Structure

```
repo/
├── pipeline.sh                        # [One-click] end-to-end inference: segmentation → classification → report
│
├── train/
│   ├── segmentation/                  # Segmentation training data preparation (DICOM → npy → NIfTI)
│   └── classify/
│       └── dev_0702_v2/               # ★ Classification training (current version)
│           ├── run_train.sh           #   One-click training: LASSO → cross-region augmentation → cascaded SVM
│           ├── 1_get_feature_v8.py    #   Step 0: PyRadiomics feature extraction
│           ├── 1a_lasso_v3.py         #   Step 1: LASSO feature selection (Stage 1 + Stage 2)
│           ├── 1b_add_cross_features_v8.py  # Step 2: cross-region / ratio / one-hot feature augmentation
│           ├── 2_lasso_v8.py          #   Step 3: second-round LASSO
│           ├── 3_train_svm_v8.py      #   Step 4: cascaded SVM training (GroupKFold CV + Platt calibration)
│           ├── data_train/            #   Training feature CSVs
│           ├── in_domain_cv_eval.py       # ★ Internal CV: Stage 1 OOF metrics + figures
│           ├── in_domain_stage_eval.py    #   Internal CV: stage-wise metrics (Stage 1 / Stage 2)
│           ├── in_domain_stage2_auc.py    # ★ Internal CV: Stage 2 pooled AUC (shares code with plot_roc_paper.py)
│           ├── plot_roc_paper.py          # ★ Paper ROC figures (Stage 1 from OOF; Stage 2 pooled)
│           └── README_集内测试与外部验证.md  # Detailed evaluation & reproduction notes (Chinese)
│
├── infer/
│   ├── segmentation/                  # Segmentation inference: DICOM → npy → NIfTI → nnU-Net → 3D
│   │   ├── 1_dcm2npy.py / 2_npy2nii.py / 4_vis.py / 5__nii23D.py
│   │   └── evaluation/                # Segmentation evaluation (Dice / HD95, etc.) + reproduction notes
│   │       ├── evaluate.py / metrics.py / utils.py / visualize.py
│   │       └── REPRODUCE_Table3.md
│   └── classify/
│       ├── run_inference.sh           # [One-click] external validation: inference → filtering → reports + metrics
│       ├── SVM_RBF_inference_pipeline_v8_v2.py  # Cascaded inference engine (features + Stage 1 + Stage 2 + post-processing)
│       ├── visualize_report_v8.py     # Diagnostic report rendering (summary table includes 3-class accuracy)
│       └── external_stage2_auc.py     # External validation: Stage 2 AUC
│
├── checkpoint/results_v8.9_0702_v2/   # Current models (4 regions × models/*.pkl)
└── data/                              # Data (image_3d / mask_3d / ground-truth Excel / outputs)
```

---

## 3. Environment

| Component | Environment | Purpose |
|---|---|---|
| nnU-Net | conda env `knee_yx` | Cartilage segmentation (Stage 1) |
| Classification / visualization | `repo/venv310` (Python 3.10) | Feature extraction, SVM classification, reports |

Dependencies: SimpleITK, PyRadiomics, scikit-learn, pandas, numpy, matplotlib, pillow, pypinyin.

---

## 4. Quick Start

### 4.1 End-to-end inference (new DICOM cases → diagnostic reports)

```bash
cd repo
bash pipeline.sh
```

Runs segmentation (nnU-Net) → 3D reconstruction → cascaded classification → visual diagnostic reports.
Outputs: segmentation results and a per-case report (segmentation overlay + damage heatmap + diagnosis panel).

### 4.2 Train the classification models (from the feature CSV)

```bash
cd repo
bash train/classify/dev_0702_v2/run_train.sh
```

Starting from `data_train/knee_radiomics_features_3d_integrated.csv`: LASSO selection → cross-region
augmentation → cascaded SVM training (5-fold GroupKFold CV + Platt probability calibration).
Models are saved to `checkpoint/results_v8.9_0702_v2/`.

### 4.3 External validation (independent test set)

```bash
cd repo
bash infer/classify/run_inference.sh
```

Inference → exclusion-list filtering → per-case reports, `summary_metrics.png` (ROC + metrics table,
including 3-class accuracy), `confusion_matrices.png`, and `summary_metrics.csv`.

### 4.4 Internal evaluation (cross-validation, out-of-fold)

```bash
cd repo
venv310/bin/python train/classify/dev_0702_v2/in_domain_cv_eval.py    # Stage 1 metrics + figures
venv310/bin/python train/classify/dev_0702_v2/in_domain_stage_eval.py # Stage-wise metrics (Stage 1 / 2)
venv310/bin/python train/classify/dev_0702_v2/in_domain_stage2_auc.py # Stage 2 pooled AUC
```

All development cases are used for training (no separate hold-out set), so internal evaluation uses
**out-of-fold (OOF) predictions from patient-grouped 5-fold cross-validation (GroupKFold)**, which prevents
patient-level leakage.

### 4.5 Paper ROC figures

```bash
cd repo
venv310/bin/python train/classify/dev_0702_v2/plot_roc_paper.py
```

Stage 1 curves are computed directly from the OOF predictions (AUCs match the internal metrics table exactly);
Stage 2 is shown as a pooled ROC (AUC = 0.873, matching the table).
Output: `checkpoint/results_v8.9_0702_v2_paper_figures/`.

---

## 5. Main Results

**Internal cross-validation (5-fold GroupKFold, OOF; n = 117 per region)**

| Stage | Region | AUC | Acc | Sens | Spec |
|---|---|---|---|---|---|
| Stage 1 | FM | 0.899 | 0.846 | 0.786 | 0.880 |
| | FL | 0.938 | 0.897 | 0.462 | 0.952 |
| | TM | 0.945 | 0.915 | 0.714 | 0.958 |
| | TL | 0.892 | 0.897 | 0.619 | 0.958 |
| Stage 2 (pooled) | All regions | 0.873 | 0.887 | 0.774 | 0.939 |

**End-to-end external validation (DICOM → segmentation → classification → visualization; n = 20 per region)**

| Stage | Region | AUC | Acc | 3-class Acc |
|---|---|---|---|---|
| Stage 1 | FM | 0.905 | 0.900 | 0.900 |
| | FL | 0.976 | 0.900 | 0.900 |
| | TM | 0.950 | 0.900 | 0.900 |
| | TL | 1.000 | 0.950 | 0.900 |

> For the full evaluation protocol, metric definitions and step-by-step reproduction commands, see
> `repo/train/classify/dev_0702_v2/README_集内测试与外部验证.md` (in Chinese).

---

## 6. Notes

- The segmentation model is built on [nnU-Net](https://github.com/MIC-DKFZ/nnUNet) (original documentation
  in `documentation/`). Please also cite the original nnU-Net paper when using this repository.
- Classification, evaluation and visualization code is located under `repo/`; use is subject to the
  repository LICENSE.
- For both Stage 1 and Stage 2, the figures, tables and scripts use the same statistical protocol
  (OOF / pooled), so all reported numbers are reproducible and mutually consistent.

