"""Gaussian-process Bayesian optimisation of Random Forest on train trips only."""
import os,sys,json,time
from pathlib import Path
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupKFold
from sklearn.metrics import f1_score
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern,WhiteKernel,ConstantKernel
from scipy.stats import norm
ROOT=Path(__file__).resolve().parent; DATA=Path(os.environ.get('PATHSNAP_DATA',ROOT/'data'))
sys.path.insert(0,str(ROOT/'src'))
from pathsnap import read_trip,make_candidates

Xs=[];ys=[];gs=[]
for folder in sorted(p for p in (DATA/'train').iterdir() if p.is_dir()):
    tr,n,a,r=read_trip(folder); X,y,*_=make_candidates(tr,n,a,r)
    Xs.append(X);ys.append(y);gs.extend([folder.name]*len(y))
X=np.vstack(Xs); y=np.concatenate(ys); g=np.asarray(gs)
rng=np.random.default_rng(2026); chosen=[]
# Balanced per-trip subset capped at ~25k rows.
for tid in np.unique(g):
    ix=np.flatnonzero(g==tid); chosen.extend(rng.choice(ix,min(len(ix),max(1,25000//len(np.unique(g)))),replace=False))
chosen=np.sort(np.asarray(chosen)); X,y,g=X[chosen],y[chosen],g[chosen]
splits=list(GroupKFold(n_splits=3).split(X,y,g))

def decode(z):
    return {'n_estimators':int(round(30+z[0]*60)),'max_depth':int(round(8+z[1]*12)),'min_samples_leaf':int(round(1+z[2]*3))}
def objective(z):
    params=decode(z); scores=[]
    for a,b in splits:
        m=RandomForestClassifier(**params,class_weight='balanced_subsample',n_jobs=-1,random_state=42)
        m.fit(X[a],y[a]); scores.append(f1_score(y[b],m.predict(X[b]),average='macro',zero_division=0))
    return float(np.mean(scores))

# Reproducible initial design, then expected improvement over a random candidate pool.
points=[np.array([.25,.5,.33]),np.array([.75,.75,0.]),np.array([.5,.25,1.])]
trials=[]; n_trials=int(os.environ.get('PATHSNAP_BO_TRIALS','8'))
for i in range(n_trials):
    if i>=len(points):
        pool=rng.random((256,3)); gp=GaussianProcessRegressor(kernel=ConstantKernel(1.)*Matern(nu=2.5)+WhiteKernel(1e-5),normalize_y=True,random_state=42,n_restarts_optimizer=1)
        gp.fit(np.array([t['z'] for t in trials]),np.array([t['cv_macro_f1'] for t in trials]))
        mu,std=gp.predict(pool,return_std=True); best=max(t['cv_macro_f1'] for t in trials); imp=mu-best-.01; z=imp/np.maximum(std,1e-9)
        ei=imp*norm.cdf(z)+std*norm.pdf(z); points.append(pool[int(np.argmax(ei))])
    z=np.asarray(points[i]); params=decode(z); start=time.perf_counter(); score=objective(z)
    trial={'trial':i+1,'z':z.tolist(),'params':params,'cv_macro_f1':score,'seconds':time.perf_counter()-start}
    trials.append(trial); print(trial,flush=True)
best=max(trials,key=lambda t:t['cv_macro_f1'])
result={'method':'Gaussian-process Bayesian optimisation with expected improvement','objective':'3-fold GroupKFold candidate-level binary macro F1','training_trips':int(len(np.unique(g))),'subset_rows':int(len(y)),'trials':trials,'best_trial':best,'search_space':{'n_estimators':[30,90],'max_depth':[8,20],'min_samples_leaf':[1,4]},'seed':2026}
(ROOT/'results').mkdir(exist_ok=True); (ROOT/'results'/'bayesian_optimisation.json').write_text(json.dumps(result,indent=2)); print('BEST',json.dumps(best,indent=2))
