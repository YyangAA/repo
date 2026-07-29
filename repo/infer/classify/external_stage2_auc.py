import pandas as pd, numpy as np
from sklearn.metrics import roc_auc_score
R='/mnt/tidal-alsh-share2/dataset/askfollow/usr/yangxin/keyan/repo-dev_v4/repo'
CN={'Femur_Medial':'FM','Femur_Lateral':'FL','Tibia_Medial':'TM','Tibia_Lateral':'TL'}
colmap={'Femur_Medial':'股骨内侧','Femur_Lateral':'股骨外侧','Tibia_Medial':'胫骨内侧','Tibia_Lateral':'胫骨外侧'}
pred=pd.read_csv(f'{R}/data/inference_results_v8.9_0702_v2.3_filtered.csv')
gt=pd.read_excel(f'{R}/data/GT_merged_v2.3_test.xlsx')
gt['key']=gt['患者姓名'].astype(str).str.replace('_knee','',regex=False).str.strip()
gtmap={}
for _,r in gt.iterrows():
    gtmap[r['key']]={reg:(int(r[c]) if pd.notna(r[c]) else -1) for reg,c in colmap.items()}
print("=== EXTERNAL (n=20) Stage2 AUC: G2 vs G1 on true-damaged ===")
for reg in ['Femur_Medial','Femur_Lateral','Tibia_Medial','Tibia_Lateral']:
    sub=pred[pred['region']==reg]
    yt,ys=[],[]
    for _,row in sub.iterrows():
        cid=str(row['case_id'])
        if cid not in gtmap:
            cand=[k for k in gtmap if k.split('-')[0]==cid.split('-')[0]]
            if not cand: continue
            cid=cand[0]
        g=gtmap[cid][reg]
        if g>0:  # 真实损伤才参与 G1/G2
            yt.append(1 if g==2 else 0); ys.append(float(row['probability_grade2']))
    yt=np.array(yt); ys=np.array(ys)
    if len(np.unique(yt))>=2:
        print(f"{CN[reg]}: AUC={roc_auc_score(yt,ys):.3f} n={len(yt)} G1/G2={int((yt==0).sum())}/{int((yt==1).sum())}")
    else:
        print(f"{CN[reg]}: AUC=N/A (single class) n={len(yt)} G1/G2={int((yt==0).sum())}/{int((yt==1).sum())}")

