"""PathSnap baseline: GPS-to-road candidate generation, classifier and Viterbi decoder."""
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree
from sklearn.base import clone
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier, HistGradientBoostingClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score, f1_score, log_loss, precision_score, recall_score
import joblib, json, time

FEATURES = ['distance_m','midpoint_distance_m','segment_length_m','heading_difference','gps_speed_mps','candidate_rank','start_distance_m','end_distance_m','time_gap_s']
EARTH = 6371000.0

def read_trip(folder, include_route=True):
    tid=folder.name
    track=np.loadtxt(folder/f'{tid}.track',dtype=float)
    nodes=np.loadtxt(folder/f'{tid}.nodes',dtype=float)
    arcs=np.loadtxt(folder/f'{tid}.arcs',dtype=int).reshape(-1,2)
    route=np.loadtxt(folder/f'{tid}.route',dtype=int).reshape(-1) if include_route else np.array([], dtype=int)
    if track.ndim==1: track=track.reshape(1,-1)
    if nodes.ndim==1: nodes=nodes.reshape(1,-1)
    return track,nodes,arcs,route

def local_xy(lon,lat,lon0,lat0):
    return np.column_stack(((np.asarray(lon)-lon0)*np.cos(np.deg2rad(lat0))*np.pi/180*EARTH,
                            (np.asarray(lat)-lat0)*np.pi/180*EARTH))

def angle_diff(a,b):
    return np.abs((a-b+180.0)%360.0-180.0)

def bearing(p1,p2):
    dx=p2[...,0]-p1[...,0]; dy=p2[...,1]-p1[...,1]
    return (np.degrees(np.arctan2(dx,dy))+360)%360

def make_candidates(track,nodes,arcs,route,k=8):
    """Return candidate feature rows, labels, observation indices and candidate arc ids."""
    n=len(track); lon0=float(np.median(track[:,0])); lat0=float(np.median(track[:,1]))
    xy=local_xy(track[:,0],track[:,1],lon0,lat0); nxy=local_xy(nodes[:,0],nodes[:,1],lon0,lat0)
    valid=(arcs[:,0]>=0)&(arcs[:,1]>=0)&(arcs[:,0]<len(nodes))&(arcs[:,1]<len(nodes))
    arc_ids=np.flatnonzero(valid); aa=arcs[valid]
    p1=nxy[aa[:,0]]; p2=nxy[aa[:,1]]; mid=(p1+p2)/2
    vec=p2-p1; lens=np.linalg.norm(vec,axis=1); headings=bearing(p1,p2)
    tree=cKDTree(mid); kk=min(k,len(mid)); md,ix=tree.query(xy,k=kk)
    if kk==1: md=md[:,None]; ix=ix[:,None]
    # Reference label for each observation: nearest supplied route segment, via a spatial index.
    route=route[(route>=0)&(route<len(arcs))]
    route=route[np.isin(route,arc_ids)]
    if len(route):
        pos=np.searchsorted(arc_ids,route); rp1=p1[pos]; rp2=p2[pos]; rmid=(rp1+rp2)/2
        rtree=cKDTree(rmid); _,rix=rtree.query(xy,k=min(4,len(route)))
        if rix.ndim==1: rix=rix[:,None]
        q1=rp1[rix]; q2=rp2[rix]; vv=q2-q1; den=np.sum(vv*vv,axis=2)
        tt=np.divide(np.sum((xy[:,None,:]-q1)*vv,axis=2),den,out=np.zeros_like(den),where=den>0); tt=np.clip(tt,0,1)
        proj=q1+tt[:,:,None]*vv; ds=np.linalg.norm(proj-xy[:,None,:],axis=2)
        ref_labels=route[rix[np.arange(n),np.argmin(ds,axis=1)]]
    else: ref_labels=np.full(n,-1,dtype=int)
    gps_heading=np.zeros(n); gps_speed=np.zeros(n); gaps=np.ones(n)
    if n>1:
        delta=np.diff(xy,axis=0); gps_heading[1:]=bearing(xy[:-1],xy[1:]); gaps[1:]=np.maximum(np.diff(track[:,2]),1e-3); gps_speed[1:]=np.linalg.norm(delta,axis=1)/gaps[1:]
    # Vectorised feature construction for all observations and their K nearest road segments.
    cand_p1=p1[ix]; cand_p2=p2[ix]; vv=cand_p2-cand_p1; den=np.sum(vv*vv,axis=2)
    tt=np.divide(np.sum((xy[:,None,:]-cand_p1)*vv,axis=2),den,out=np.zeros_like(den),where=den>0); tt=np.clip(tt,0,1)
    proj=cand_p1+tt[:,:,None]*vv; dist=np.linalg.norm(xy[:,None,:]-proj,axis=2)
    ch=bearing(cand_p1,cand_p2); ad=angle_diff(gps_heading[:,None],ch)
    mdist=np.asarray(md); lengths=np.broadcast_to(lens[ix],dist.shape)
    starts=np.linalg.norm(xy[:,None,:]-cand_p1,axis=2); ends=np.linalg.norm(xy[:,None,:]-cand_p2,axis=2)
    ranks=np.broadcast_to(np.arange(kk),(n,kk)); speeds=np.broadcast_to(gps_speed[:,None],(n,kk)); gapmat=np.broadcast_to(gaps[:,None],(n,kk))
    feat=np.stack([dist,mdist,lengths,ad,speeds,ranks,starts,ends,gapmat],axis=2).reshape(-1,9).astype(np.float32)
    candidate_ids=arc_ids[ix].reshape(-1); labels=(candidate_ids==np.repeat(ref_labels,kk)).astype(np.int8)
    obs=np.repeat(np.arange(n),kk); missed=int(np.sum(~np.any(arc_ids[ix]==ref_labels[:,None],axis=1)))
    rows=feat; ys=labels; obs=obs; cids=candidate_ids
    return np.asarray(rows,dtype=np.float32),np.asarray(ys,dtype=np.int8),np.asarray(obs),np.asarray(cids),ref_labels,missed,xy,nxy,arcs

def make_models(seed=42):
    return {
      'LogisticRegression':make_pipeline(StandardScaler(),LogisticRegression(max_iter=300,class_weight='balanced',random_state=seed)),
      'KNN':make_pipeline(StandardScaler(),KNeighborsClassifier(n_neighbors=7,weights='distance',algorithm='kd_tree',leaf_size=40,n_jobs=-1)),
      'RandomForest':RandomForestClassifier(n_estimators=45,max_depth=16,min_samples_leaf=2,class_weight='balanced_subsample',n_jobs=1,random_state=seed),
      'ExtraTrees':ExtraTreesClassifier(n_estimators=50,max_depth=18,min_samples_leaf=2,class_weight='balanced',n_jobs=1,random_state=seed),
      'HistGradientBoosting':HistGradientBoostingClassifier(max_iter=60,max_leaf_nodes=15,l2_regularization=1.0,random_state=seed),
      'AdaBoost':AdaBoostClassifier(n_estimators=40,learning_rate=.5,random_state=seed)
    }

def top1_from_rows(model,X,obs,cids,truth):
    p=model.predict_proba(X)[:,1] if hasattr(model,'predict_proba') else model.decision_function(X)
    pred={}
    for i,s,a in zip(obs,p,cids):
        if i not in pred or s>pred[i][0]: pred[i]=(float(s),int(a))
    valid=[i for i in pred if truth[i]>=0]
    return np.mean([pred[i][1]==truth[i] for i in valid]) if valid else float('nan')

def viterbi_decode(model,X,obs,cids,arcs,transition_penalty=3.0,timestamps=None,reset_gap_s=30.0):
    """Connectivity-aware Viterbi decoding; restart after long GPS recording gaps."""
    probs=model.predict_proba(X)[:,1]
    grouped={}
    for i,a,p in zip(obs,cids,probs):
        grouped.setdefault(int(i),{})[int(a)]=max(float(p),1e-8)
    if not grouped:return []
    steps=sorted(grouped)
    outgoing={}
    for arc_id,(u,v) in enumerate(arcs): outgoing.setdefault(int(u),[]).append(arc_id)
    next_arcs={arc_id:set(outgoing.get(int(v),[])) for arc_id,(u,v) in enumerate(arcs)}
    # Split into independent sequences after a recording interruption. A gap is
    # not interpreted as evidence that the vehicle remained stationary.
    chunks=[]; current=[steps[0]]
    for step in steps[1:]:
        gap=(float(timestamps[step])-float(timestamps[current[-1]])) if timestamps is not None else 0.0
        if timestamps is not None and gap>reset_gap_s:
            chunks.append(current); current=[step]
        else: current.append(step)
    chunks.append(current)
    full_path=[]
    for chunk in chunks:
        states=[list(grouped[i]) for i in chunk]
        dp={state:np.log(grouped[chunk[0]][state]) for state in states[0]}; back=[]
        for t in range(1,len(chunk)):
            cur={}; bp={}
            for state in states[t]:
                best=-1e30; prev_best=None
                for prev,val in dp.items():
                    connected=(prev==state or state in next_arcs.get(prev,set()))
                    score=val+np.log(grouped[chunk[t]][state])-(0.0 if connected else transition_penalty)
                    if score>best: best=score; prev_best=prev
                cur[state]=best; bp[state]=prev_best
            dp=cur; back.append(bp)
        state=max(dp,key=dp.get); path=[state]
        for bp in reversed(back): state=bp[state]; path.append(state)
        full_path.extend(reversed(path))
    return full_path
