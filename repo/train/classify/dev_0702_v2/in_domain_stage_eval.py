#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
in_domain_stage_eval.py
集内测试 — 分阶段性能（Stage 1 二分类 + Stage 2 G1/G2 分级），GroupKFold OOF

Stage 1: 正常 vs 损伤（跨区域增强特征，复现训练 C/gamma）
Stage 2: 在真实损伤(grade>0)样本上，G1 vs G2 的 GroupKFold OOF 分级
         （复现训练 Stage2：stage2_filtered_features + 平衡 SVM）
输出 Stage2 指标以"真实损伤 & 预测损伤(TP)"子集为准，与端到端外部验证口径一致。
"""
import os, joblib
import numpy as np, pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.model_selection import GroupKFold
from sklearn.metrics import (accuracy_score, recall_score, precision_score,
                             f1_score, confusion_matrix, roc_curve, auc)

REPO = "/mnt/tidal-alsh-share2/dataset/askfollow/usr/yangxin/keyan/repo-dev_v4/repo"
FEAT = os.path.join(REPO, "train/classify/dev_0702_v2/data_train/feature")
MODEL = os.path.join(REPO, "checkpoint/results_v8.9_0702_v2")
OOF_CSV = os.path.join(REPO, "data/in_domain_cv_results_v8.9_0702_v2/oof_predictions.csv")
OUT = os.path.join(REPO, "data/in_domain_cv_results_v8.9_0702_v2")

REGIONS = ["Femur_Medial", "Femur_Lateral", "Tibia_Medial", "Tibia_Lateral"]
CN = {"Femur_Medial": "FM", "Femur_Lateral": "FL", "Tibia_Medial": "TM", "Tibia_Lateral": "TL"}
N_SPLITS = 5
RS = 42

TRUE_GRADE_CSV = os.path.join(REPO, "train/classify/dev_0702_v2/data_train/knee_radiomics_features_3d_integrated.csv")


def load_true_grades():
    """从训练特征 CSV 加载真实分级 (0/1/2)，按 (case_id, region)"""
    df = pd.read_csv(TRUE_GRADE_CSV)
    df = df.groupby(['case_id', 'region'])['grade'].first().reset_index()
    gmap = {}
    for _, r in df.iterrows():
        gmap[(str(r['case_id']), r['region'])] = int(r['grade'])
    return gmap


def load_stage2_feat(region):
    p = os.path.join(FEAT, f"{region}_stage2_filtered_features.csv")
    if not os.path.exists(p):
        p = os.path.join(FEAT, "pooled_stage2_FL_TL_filtered_features.csv")
        if not os.path.exists(p):
            return None
    df = pd.read_csv(p)
    if 'region' in df.columns:
        df = df[df['region'] == region]
    df = df.groupby(['case_id', 'region', 'grade']).mean().reset_index() if 'region' in df.columns \
        else df.groupby(['case_id', 'grade']).mean().reset_index()
    # 只保留损伤样本 (grade>0) 做 G1 vs G2
    df = df[df['grade'] > 0].copy()
    if len(df) < 4:
        return None
    drop = [c for c in ["case_id", "region", "grade", "cartilage_missing"] if c in df.columns]
    X = df.drop(columns=drop).fillna(0)
    y = (df['grade'] == 2).astype(int).values  # 1=G2, 0=G1
    groups = df['case_id'].values
    return X, y, groups, df['case_id'].values


def stage2_oof(region):
    """G1 vs G2 的 GroupKFold OOF，返回 {case_id: pred_g2(0/1), prob}"""
    data = load_stage2_feat(region)
    if data is None:
        return {}
    X, y, groups, cids = data
    n_minor = min((y == 0).sum(), (y == 1).sum())
    if n_minor < 2:
        return {}
    n_splits = min(N_SPLITS, n_minor, len(np.unique(groups)))
    if n_splits < 2:
        return {}
    cv = GroupKFold(n_splits=n_splits)
    pred = {}
    for tr, te in cv.split(X, y, groups=groups):
        if len(np.unique(y[tr])) < 2:
            continue
        sc = StandardScaler()
        Xtr = sc.fit_transform(X.iloc[tr]); Xte = sc.transform(X.iloc[te])
        m = SVC(kernel='rbf', C=1, gamma='scale', probability=True,
                class_weight='balanced', random_state=RS)
        m.fit(Xtr, y[tr])
        p = m.predict_proba(Xte)[:, 1]
        for i, idx in enumerate(te):
            pred[cids[idx]] = {'prob': p[i], 'pred': int(p[i] >= 0.5), 'true': int(y[idx])}
    return pred


def main():
    oof = pd.read_csv(OOF_CSV)  # Stage1 OOF：含 case_id, region, pred_binary, oof_prob_damage
    gmap = load_true_grades()   # 真实分级 0/1/2
    # 用真实分级覆盖 true_grade（修正原 OOF CSV 的二值化 bug）
    oof['true_grade'] = oof.apply(
        lambda r: gmap.get((str(r['case_id']), r['region']), int(r['true_grade'])), axis=1)
    print("=" * 74)
    print("  IN-DOMAIN CV — Stage-wise Performance (GroupKFold OOF)")
    print("=" * 74)

    rows1, rows2 = [], []
    for reg in REGIONS:
        sub = oof[oof['region'] == reg]
        # Stage1
        yt = (sub['true_grade'] > 0).astype(int).values
        yp = sub['pred_binary'].values
        # AUC 从原 CV 结果拿概率
        prob = sub['oof_prob_damage'].values
        fpr, tpr, _ = roc_curve(yt, prob); auc_v = auc(fpr, tpr)
        cm = confusion_matrix(yt, yp, labels=[0, 1]); tn, fp, fn, tp = cm.ravel()
        m1 = dict(auc=auc_v, acc=accuracy_score(yt, yp),
                  sens=recall_score(yt, yp, zero_division=0),
                  spec=tn / (tn + fp) if (tn + fp) else 0,
                  prec=precision_score(yt, yp, zero_division=0),
                  f1=f1_score(yt, yp, zero_division=0),
                  n=len(yt), pos=int(yt.sum()), neg=int((1 - yt).sum()))
        rows1.append((reg, m1))
        print(f"\n{CN[reg]} Stage1: n={m1['n']} pos/neg={m1['pos']}/{m1['neg']} "
              f"AUC={m1['auc']:.3f} Acc={m1['acc']:.3f} Sens={m1['sens']:.3f} "
              f"Spec={m1['spec']:.3f} Prec={m1['prec']:.3f} F1={m1['f1']:.3f}")

        # Stage2: 独立 OOF (G1 vs G2)
        s2pred = stage2_oof(reg)
        # 口径与外部一致：真实损伤 & 预测损伤(TP)
        s2t, s2p = [], []
        for _, r in sub.iterrows():
            cid = r['case_id']
            if r['true_grade'] > 0 and r['pred_binary'] == 1:
                # 真实 G2?
                true_g2 = 1 if r['true_grade'] == 2 else 0
                # 预测 G2? 用 stage2 OOF；若无则按 pred_grade 回退
                if cid in s2pred:
                    pred_g2 = s2pred[cid]['pred']
                else:
                    pred_g2 = 1 if r['pred_grade'] == 2 else 0
                s2t.append(true_g2); s2p.append(pred_g2)
        s2t = np.array(s2t); s2p = np.array(s2p)
        if len(s2t) >= 2 and len(np.unique(s2t)) >= 1:
            s2acc = accuracy_score(s2t, s2p)
            s2sens = recall_score(s2t, s2p, zero_division=0)  # G2 召回
            cm2 = confusion_matrix(s2t, s2p, labels=[0, 1])
            tn2, fp2, fn2, tp2 = cm2.ravel()
            s2spec = tn2 / (tn2 + fp2) if (tn2 + fp2) else 0
            s2prec = precision_score(s2t, s2p, zero_division=0)
            s2f1 = f1_score(s2t, s2p, zero_division=0)
            m2 = dict(acc=s2acc, sens=s2sens, spec=s2spec, prec=s2prec, f1=s2f1,
                      n=len(s2t), g1=int((s2t == 0).sum()), g2=int((s2t == 1).sum()))
            rows2.append((reg, m2))
            print(f"{CN[reg]} Stage2(G1/G2 on TP): n={m2['n']} G1/G2={m2['g1']}/{m2['g2']} "
                  f"Acc={m2['acc']:.3f} Sens(G2)={m2['sens']:.3f} Spec(G1)={m2['spec']:.3f} "
                  f"Prec={m2['prec']:.3f} F1={m2['f1']:.3f}")
        else:
            rows2.append((reg, None))
            print(f"{CN[reg]} Stage2: insufficient samples (n={len(s2t)})")

    # 保存 CSV
    d1 = pd.DataFrame([{'Region': CN[r], **{k: round(v, 3) if isinstance(v, float) else v
                                            for k, v in m.items()}} for r, m in rows1])
    d1.to_csv(os.path.join(OUT, "stage1_metrics.csv"), index=False)
    d2 = pd.DataFrame([{'Region': CN[r], **({k: round(v, 3) if isinstance(v, float) else v
                                             for k, v in m.items()} if m else {})}
                       for r, m in rows2])
    d2.to_csv(os.path.join(OUT, "stage2_metrics.csv"), index=False)
    print(f"\nSaved: {OUT}/stage1_metrics.csv, stage2_metrics.csv")


if __name__ == "__main__":
    main()

