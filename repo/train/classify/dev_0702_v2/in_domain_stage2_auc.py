#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
in_domain_stage2_auc.py
集内测试 Stage 2 (G1 vs G2) 的 **Pooled** AUC。

为保证“跑出来的结果 = 写进论文的结果 = plot_roc_paper.py 图中的 Stage2 曲线”
三者完全一致，本脚本 **直接复用** plot_roc_paper.py 里的 pooled 数据加载与
CV 函数（load_stage2_data + compute_stage2_cv_curves），不重写算法。

依赖输入:
  - train/classify/dev_0702_v2/data_train/feature/{Region}_stage2_filtered_features.csv (分级特征)
  - 与 plot_roc_paper.py 同目录 (本脚本 import plot_roc_paper)

用法:
  cd /mnt/tidal-alsh-share2/dataset/askfollow/usr/yangxin/keyan/repo-dev_v4/repo
  venv310/bin/python train/classify/dev_0702_v2/in_domain_stage2_auc.py

输出 (打印到 stdout, 直接填入论文表 1 的 Stage 2 行):
  Pooled AUC-ROC / AP
  Pooled Acc / Sens(G2召回) / Spec(G1识别) / Prec / F1   (阈值=各折平均 Youden 最优值, 聚合 OOF 上计算)
  当前结果: AUC=0.873 Acc=0.887 Sens=0.774 Spec=0.939 Prec=0.857 F1=0.814 (n=97, G1=66/G2=31)
"""
import os
import sys
import numpy as np

# 确保能 import 同目录的 plot_roc_paper
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.chdir("/mnt/tidal-alsh-share2/dataset/askfollow/usr/yangxin/keyan/repo-dev_v4/repo")

import plot_roc_paper as prp
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV

print("=" * 64)
print("  IN-DOMAIN Stage 2 (G1 vs G2) — Pooled AUC")
print("  (与 plot_roc_paper.py 图 / 论文表格 严格一致)")
print("=" * 64)

# 1) 加载 pooled 数据（与图同源）
res = prp.load_stage2_data()
if res is None or res[0] is None:
    print("No Stage2 data found."); sys.exit(1)
X_pool, y_pool, groups_pool, region_col, _region_indep = res

# 2) GridSearchCV 选参（与 plot_roc_paper.py main 完全一致）
scaler_search = StandardScaler()
Xs = scaler_search.fit_transform(X_pool)
param_grid = {"C": [0.1, 1, 10, 100], "gamma": ["scale", "auto", 0.01, 0.1, 1]}
n_minority = min(int((y_pool == 0).sum()), int((y_pool == 1).sum()))
inner_cv = min(5, n_minority)
grid = GridSearchCV(SVC(kernel="rbf", probability=True, class_weight="balanced"),
                    param_grid, cv=inner_cv, scoring="accuracy")
grid.fit(Xs, y_pool)
C_best = grid.best_params_["C"]; gamma_best = grid.best_params_["gamma"]
print(f"  GridSearchCV best: C={C_best}, gamma={gamma_best}")

# 3) pooled CV 曲线（与图同一函数）
pooled = prp.compute_stage2_cv_curves(
    X_pool, y_pool, groups_pool, C=C_best, gamma=gamma_best,
    class_weight="balanced", n_splits=5)

if pooled is None:
    print("Stage2 CV failed."); sys.exit(1)

print("-" * 64)
print(f"  Pooled AUC-ROC : {pooled['mean_roc_auc']:.3f} ± {pooled['std_roc_auc']:.3f}")
print(f"  Pooled AP      : {pooled['mean_pr_auc']:.3f} ± {pooled['std_pr_auc']:.3f}")
print(f"  Samples        : G1={int((y_pool==0).sum())}, G2={int((y_pool==1).sum())}")

# 3b) 在聚合 OOF 预测上算 Acc/Sens/Spec/Prec/F1（阈值=各折平均最优，与图同源）
from sklearn.metrics import (accuracy_score, recall_score, precision_score,
                             f1_score, confusion_matrix)
yt = pooled['all_y_true']; yp_prob = pooled['all_y_prob']
thr = pooled['best_threshold'][0]
yp = (yp_prob >= thr).astype(int)
cm = confusion_matrix(yt, yp, labels=[0, 1]); tn, fp, fn, tp = cm.ravel()
acc = accuracy_score(yt, yp); sens = recall_score(yt, yp, zero_division=0)
spec = tn / (tn + fp) if (tn + fp) else 0.0
prec = precision_score(yt, yp, zero_division=0); f1 = f1_score(yt, yp, zero_division=0)
print("-" * 64)
print(f"  Pooled 分类指标 (阈值={thr:.3f}): Acc={acc:.3f} Sens(G2)={sens:.3f} "
      f"Spec(G1)={spec:.3f} Prec={prec:.3f} F1={f1:.3f}")
print("=" * 64)
print("  → 论文表格 Stage2 (Pooled, G1 vs G2) 应填:")
print(f"     AUC={pooled['mean_roc_auc']:.3f} Acc={acc:.3f} Sens={sens:.3f} "
      f"Spec={spec:.3f} Prec={prec:.3f} F1={f1:.3f} n={len(yt)}")
print("=" * 64)

