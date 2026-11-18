"""Compare six classifiers using trip-grouped CV on training trips only."""
import os, sys, time, json
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.base import clone
from sklearn.model_selection import GroupKFold
from sklearn.metrics import f1_score, log_loss
ROOT=Path(__file__).resolve().parent
DATA=Path(os.environ.get('PATHSNAP_DATA', ROOT/'data'))
sys.path.insert(0,str(ROOT/'src'))
from pathsnap import read_trip, make_candidates, make_models


def build_training_rows():
    Xs=[]; ys=[]; groups=[]
    folders=sorted(p for p in (DATA/'train').iterdir() if p.is_dir())
    if not folders: raise FileNotFoundError(f'No training trips found under {DATA}/train')
    for folder in folders:
        track,nodes,arcs,route=read_trip(folder)
        X,y,*_=make_candidates(track,nodes,arcs,route)
        Xs.append(X); ys.append(y); groups.extend([folder.name]*len(y))
    X=np.vstack(Xs); y=np.concatenate(ys); groups=np.asarray(groups)
    # Keep a deterministic, approximately trip-balanced subset for laptop-friendly CV.
    if len(y)>60000:
        rng=np.random.default_rng(42); keep=[]
        for tid in np.unique(groups):
            ix=np.flatnonzero(groups==tid)
            cap=max(1,round(60000*len(ix)/len(y)))
            keep.extend(ix if len(ix)<=cap else rng.choice(ix,cap,replace=False))
        keep=np.sort(keep); X,y,groups=X[keep],y[keep],groups[keep]
    return X,y,groups

X,y,groups=build_training_rows()
models=make_models(); splits=list(GroupKFold(n_splits=3).split(X,y,groups)); results=[]
print(f'CV dataset: {len(y):,} candidate rows, {len(np.unique(groups))} trips; 3-fold GroupKFold',flush=True)
for name,base in models.items():
    f1s=[]; losses=[]; elapsed=0; predict_elapsed=0
    for fold,(tr,va) in enumerate(splits,1):
        model=clone(base); t=time.perf_counter(); model.fit(X[tr],y[tr]); elapsed+=time.perf_counter()-t
        pt=time.perf_counter(); pred=model.predict(X[va]); predict_elapsed+=time.perf_counter()-pt; f1s.append(f1_score(y[va],pred,average='macro',zero_division=0))
        if hasattr(model,'predict_proba'):
            prob=model.predict_proba(X[va]); losses.append(log_loss(y[va],prob,labels=[0,1]))
    row={'model':name,'family':{'LogisticRegression':'linear/probabilistic','KNN':'instance-based','RandomForest':'ensemble','ExtraTrees':'ensemble','HistGradientBoosting':'boosting','AdaBoost':'boosting'}[name],
         'cv_candidate_macro_f1':float(np.mean(f1s)),'cv_macro_f1_std':float(np.std(f1s)),
         'cv_log_loss':float(np.mean(losses)) if losses else None,'cv_fit_seconds':float(elapsed),'cv_predict_seconds':float(predict_elapsed)}
    results.append(row); print(row,flush=True)

out=ROOT/'results'; out.mkdir(exist_ok=True)
df=pd.DataFrame(results).sort_values('cv_candidate_macro_f1',ascending=False)
df.to_csv(out/'model_comparison_cv.csv',index=False)
(out/'model_comparison_cv.json').write_text(json.dumps({'metric':'candidate-level binary macro F1','folds':3,'splitter':'GroupKFold by trip','rows':len(y),'trips':int(len(np.unique(groups))),'results':results},indent=2))
print('\nSaved results/model_comparison_cv.csv; top model:',df.iloc[0]['model'])
