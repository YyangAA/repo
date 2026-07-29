import os, numpy as np, pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score
R='/mnt/tidal-alsh-share2/dataset/askfollow/usr/yangxin/keyan/repo-dev_v4/repo'
FEAT=f'{R}/train/classify/dev_0702_v2/data_train/feature'
CN={'Femur_Medial':'FM','Femur_Lateral':'FL','Tibia_Medial':'TM','Tibia_Lateral':'TL'}
def load(region):
    p=f'{FEAT}/{region}_stage2_filtered_features.csv'
    if not os.path.exists(p):
        p=f'{FEAT}/pooled_stage2_FL_TL_filtered_features.csv'
        if not os.path.exists(p): return None
    df=pd.read_csv(p)
    if 'region' in df.columns: df=df[df['region']==region]
    keys=['case_id','region','grade'] if 'region' in df.columns else ['case_id','grade']
    df=df.groupby(keys).mean().reset_index()
    df=df[df['grade']>0].copy()
    if len(df)<4: return None
    drop=[c for c in ['case_id','region','grade','cartilage_missing'] if c in df.columns]
    X=df.drop(columns=drop).fillna(0); y=(df['grade']==2).astype(int).values; g=df['case_id'].values
    return X,y,g
print("=== IN-DOMAIN CV Stage2 AUC (G1 vs G2, GroupKFold OOF) ===")
for reg in ['Femur_Medial','Femur_Lateral','Tibia_Medial','Tibia_Lateral']:
    d=load(reg)
    if d is None: print(f"{CN[reg]}: no data"); continue
    X,y,g=d
    nm=min((y==0).sum(),(y==1).sum())
    if nm<2: print(f"{CN[reg]}: AUC=N/A minority<2 G1/G2={int((y==0).sum())}/{int((y==1).sum())}"); continue
    ns=min(5,nm,len(np.unique(g)))
    oof=np.full(len(y),np.nan)
    for tr,te in GroupKFold(n_splits=ns).split(X,y,groups=g):
        if len(np.unique(y[tr]))<2: continue
        sc=StandardScaler(); Xtr=sc.fit_transform(X.iloc[tr]); Xte=sc.transform(X.iloc[te])
        m=SVC(kernel='rbf',C=1,gamma='scale',probability=True,class_weight='balanced',random_state=42)
        m.fit(Xtr,y[tr]); oof[te]=m.predict_proba(Xte)[:,1]
    mask=~np.isnan(oof)
    if len(np.unique(y[mask]))>=2:
        print(f"{CN[reg]}: AUC={roc_auc_score(y[mask],oof[mask]):.3f} n={mask.sum()} G1/G2={int((y[mask]==0).sum())}/{int((y[mask]==1).sum())}")
    else:
        print(f"{CN[reg]}: AUC=N/A n={mask.sum()}")

