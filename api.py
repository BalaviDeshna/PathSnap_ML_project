from pathlib import Path
import sys, json
import numpy as np, joblib
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
sys.path.insert(0,str(Path(__file__).parent/'src'))
from pathsnap import read_trip, make_candidates, viterbi_decode
ROOT=Path(__file__).parent
import os
DATA=Path(os.environ.get('PATHSNAP_DATA',str(ROOT/'data')))
PARTITION=os.environ.get('PATHSNAP_PARTITION','student_test')
MODEL_PATH=ROOT/'models'/'model.joblib'
app=FastAPI(title='PathSnap API',description='GPS map-matching and route reconstruction')
app.add_middleware(CORSMiddleware,allow_origins=['*'],allow_methods=['*'],allow_headers=['*'])
model=joblib.load(MODEL_PATH) if MODEL_PATH.exists() else None
@app.get('/',response_class=HTMLResponse)
def home(): return (ROOT/'frontend'/'index.html').read_text()
@app.get('/health')
def health(): return {'status':'ok','model_loaded':model is not None}
@app.get('/trips')
def trips():
    p=DATA/PARTITION
    return {'trips':sorted(x.name for x in p.iterdir() if x.is_dir())}
@app.get('/predict/{trip_id}')
def predict(trip_id:str):
    if model is None: raise HTTPException(503,'Train the model first: python train.py')
    folder=DATA/PARTITION/trip_id
    if not folder.exists(): raise HTTPException(404,'Unknown trip in configured partition')
    track,nodes,arcs,route=read_trip(folder, include_route=False)
    X,y,obs,cids,truth,miss,xy,nxy,all_arcs=make_candidates(track,nodes,arcs,route)
    probs=model.predict_proba(X)[:,1]
    # Classifier-only top candidate per GPS point
    chosen={}
    for i,a,p in zip(obs,cids,probs):
        if int(i) not in chosen or p>chosen[int(i)][0]: chosen[int(i)]=(float(p),int(a))
    classifier=[chosen[i][1] for i in sorted(chosen)]
    decoded=viterbi_decode(model,X,obs,cids,arcs,timestamps=track[:,2])
    # Build road geometry in GPS coordinates.
    def geometry(ids):
        out=[]
        for aid in ids:
            if 0<=int(aid)<len(arcs):
                u,v=arcs[int(aid)]
                if u<len(nodes) and v<len(nodes): out.append([[float(nodes[u,1]),float(nodes[u,0])],[float(nodes[v,1]),float(nodes[v,0])]])
        return out
    # Never return reference labels or metrics derived from them through the demo API.
    return {'trip_id':trip_id,'gps':[[float(b),float(a)] for a,b in track[:,:2]],'classifier_route':geometry(classifier),'predicted_route':geometry(decoded),'observation_count':len(track),'decoded_observations':len(decoded)}
