"""Evaluate six classifiers and tuned-vs-untuned RF on the 10 student-test trips."""
import os,sys,json
from pathlib import Path
import numpy as np,joblib
from sklearn.metrics import f1_score,precision_score,recall_score,log_loss,accuracy_score
ROOT=Path(__file__).resolve().parent; DATA=Path(os.environ.get('PATHSNAP_DATA',ROOT/'data'))
sys.path.insert(0,str(ROOT/'src'))
from pathsnap import read_trip,make_candidates,viterbi_decode

folders=sorted(p for p in (DATA/'student_test').iterdir() if p.is_dir())
if not folders: raise FileNotFoundError(f'No student_test trips found under {DATA}')
model_names=['LogisticRegression','KNN','RandomForest','ExtraTrees','HistGradientBoosting','AdaBoost']
models={name:joblib.load(ROOT/'models'/f'model_{name}.joblib') for name in model_names}
models['RandomForest_untuned']=joblib.load(ROOT/'models'/'model_untuned.joblib')
models['RandomForest_BayesianTuned']=joblib.load(ROOT/'models'/'model_tuned.joblib')

def continuity(seq,arcs):
    if len(seq)<2:return 1.0
    return float(np.mean([(q==s or (0<=q<len(arcs) and 0<=s<len(arcs) and arcs[q,1]==arcs[s,0])) for q,s in zip(seq[:-1],seq[1:])]))

# Build candidate features exactly once; re-use them across all model comparisons.
trip_data=[]
for folder in folders:
    track,nodes,arcs,route=read_trip(folder)
    X,y,obs,cids,truth,missed,*_=make_candidates(track,nodes,arcs,route)
    trip_data.append((folder.name,track,arcs,X,y,obs,cids,truth,missed))

def score_model(model,include_hmm=False):
    all_y=[]; all_pred=[]; all_prob=[]; top1=[]; hmm_top1=[]; cont=[]; hmm_cont=[]; miss=0; total=0
    for tid,track,arcs,X,y,obs,cids,truth,missed in trip_data:
        prob=model.predict_proba(X)[:,1]; pred=model.predict(X)
        all_y.extend(y.tolist()); all_pred.extend(pred.tolist()); all_prob.extend(prob.tolist())
        miss+=missed; total+=len(track)
        best={}
        for oi,aid,score in zip(obs,cids,prob):
            if int(oi) not in best or score>best[int(oi)][0]: best[int(oi)]=(float(score),int(aid))
        obs_ids=sorted(best); classifier=np.asarray([best[i][1] for i in obs_ids])
        decoded=np.asarray(viterbi_decode(model,X,obs,cids,arcs,timestamps=track[:,2])) if include_hmm else np.asarray([])
        valid=np.asarray([i for i in obs_ids if i<len(truth) and truth[i]>=0],dtype=int)
        top1.append(float(np.mean(classifier[valid]==truth[valid])) if len(valid) else float('nan'))
        n=min(len(decoded),len(truth)); hmm_top1.append(float(np.mean(decoded[:n]==truth[:n])) if n else float('nan'))
        cont.append(continuity(classifier,arcs)); hmm_cont.append(continuity(decoded,arcs) if include_hmm else float('nan'))
    y=np.asarray(all_y); pred=np.asarray(all_pred); prob=np.clip(np.asarray(all_prob),1e-6,1-1e-6)
    return {'candidate_macro_f1':float(f1_score(y,pred,average='macro',zero_division=0)),
            'candidate_precision_positive':float(precision_score(y,pred,zero_division=0)),
            'candidate_recall_positive':float(recall_score(y,pred,zero_division=0)),
            'candidate_log_loss':float(log_loss(y,prob,labels=[0,1])),
            'observation_top1_accuracy_mean_per_trip':float(np.nanmean(top1)),
            'hmm_top1_accuracy_mean_per_trip':float(np.nanmean(hmm_top1)) if include_hmm else None,
            'classifier_route_continuity_mean':float(np.mean(cont)),
            'hmm_route_continuity_mean':float(np.nanmean(hmm_cont)) if include_hmm else None,
            'candidate_miss_rate':float(miss/total),'student_test_trips':len(folders),'gps_observations':total}

results={}
for name,model in models.items():
    print('Evaluating',name,flush=True); results[name]=score_model(model,include_hmm=name in ['RandomForest','RandomForest_untuned','RandomForest_BayesianTuned']); print(results[name],flush=True)
(ROOT/'results').mkdir(exist_ok=True)
(ROOT/'results'/'student_test_model_comparison.json').write_text(json.dumps(results,indent=2))
# Keep a backwards-compatible summary for report generation.
rf={'untuned':results['RandomForest_untuned'],'tuned':results['RandomForest_BayesianTuned']}
(ROOT/'results'/'tuned_vs_untuned_student_test.json').write_text(json.dumps(rf,indent=2))
selected='RandomForest_BayesianTuned'
(ROOT/'results'/'student_test_metrics.json').write_text(json.dumps({'selected_model':selected,**results[selected]},indent=2))
print('Saved all student-test metrics.')
