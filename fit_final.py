"""Fit all six comparison models plus untuned/tuned RF deployment artifacts."""
import os,sys,json
from pathlib import Path
import numpy as np,joblib
from sklearn.ensemble import RandomForestClassifier
ROOT=Path(__file__).resolve().parent; DATA=Path(os.environ.get('PATHSNAP_DATA',ROOT/'data'))
sys.path.insert(0,str(ROOT/'src'))
from pathsnap import read_trip,make_candidates,make_models,FEATURES
Xs=[];ys=[];gs=[]
for folder in sorted(p for p in (DATA/'train').iterdir() if p.is_dir()):
    tr,n,a,r=read_trip(folder); x,y,*_=make_candidates(tr,n,a,r); Xs.append(x);ys.append(y);gs.extend([folder.name]*len(y))
X=np.vstack(Xs); y=np.concatenate(ys); g=np.asarray(gs)
# Deterministic stratified-by-trip row cap.
rng=np.random.default_rng(42); keep=[]
for tid in np.unique(g):
    ix=np.flatnonzero(g==tid); cap=max(1,round(60000*len(ix)/len(y))); keep.extend(ix if len(ix)<=cap else rng.choice(ix,cap,replace=False))
keep=np.sort(keep); X,y=X[keep],y[keep]
art=ROOT/'models'; art.mkdir(exist_ok=True)
for name,base in make_models().items():
    if name=='KNN':
        rng_knn=np.random.default_rng(73); ix=rng_knn.choice(len(y),min(15000,len(y)),replace=False); model=base.fit(X[ix],y[ix])
    else:
        model=base.fit(X,y)
    joblib.dump(model,art/f'model_{name}.joblib'); print('saved',name,flush=True)
bo=json.loads((ROOT/'results'/'bayesian_optimisation.json').read_text()); tuned=bo['best_trial']['params']
configs={'untuned':{'n_estimators':45,'max_depth':16,'min_samples_leaf':2},'tuned':tuned}
for label,params in configs.items():
    m=RandomForestClassifier(**params,class_weight='balanced_subsample',n_jobs=-1,random_state=42).fit(X,y)
    joblib.dump(m,art/f'model_{label}.joblib'); print('saved RF',label,params,flush=True)
# Random Forest led the six-model CV comparison. The tuned RF then improved
# the measured student-test development metrics and is used for the demo.
winner='RandomForest_BayesianTuned'
joblib.dump(joblib.load(art/'model_tuned.joblib'),art/'model.joblib')
info={'selected_model':winner,'selection_basis':'Random Forest led six-model grouped CV; Bayesian-tuned RF was then selected after improving student-test development metrics. Reserved evaluator data remains untouched.','features':FEATURES,'candidate_k':8,'seed':42,'training_rows':int(len(y)),'untuned_rf':configs['untuned'],'bayesian_tuned_rf':tuned}
(art/'model_info.json').write_text(json.dumps(info,indent=2)); print('Deployment model:',winner)
