from pathlib import Path
import sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from pathsnap import read_trip,make_candidates,make_models,viterbi_decode

def test_training_trip_candidate_pipeline():
    folder=ROOT/'data'/'train'/'00000005'
    track,nodes,arcs,route=read_trip(folder)
    X,y,obs,cids,truth,miss,*_=make_candidates(track,nodes,arcs,route)
    assert X.ndim==2 and X.shape[1]==9
    assert len(X)==len(y)==len(obs)==len(cids)
    assert set(np.unique(y)).issubset({0,1})
    assert len(truth)==len(track)
    assert len(make_models())==6

def test_gap_aware_decoder_returns_one_state_per_observation():
    folder=ROOT/'data'/'train'/'00000005'
    track,nodes,arcs,route=read_trip(folder)
    X,y,obs,cids,truth,miss,*_=make_candidates(track,nodes,arcs,route)
    from sklearn.ensemble import RandomForestClassifier
    model=RandomForestClassifier(n_estimators=5,random_state=42).fit(X,y)
    path=viterbi_decode(model,X,obs,cids,arcs,timestamps=track[:,2])
    assert len(path)==len(track)
    assert all(0<=int(a)<len(arcs) for a in path)
