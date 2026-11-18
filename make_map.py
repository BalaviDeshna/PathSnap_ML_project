from pathlib import Path
import os
import sys
import numpy as np,joblib,folium
sys.path.insert(0,str(Path(__file__).parent/'src'))
from pathsnap import read_trip,make_candidates,viterbi_decode
ROOT=Path(__file__).parent; DATA=Path(os.environ.get('PATHSNAP_DATA',str(ROOT/'data')));tid='00000001'
f=DATA/'train'/tid;track,nodes,arcs,route=read_trip(f);X,y,obs,cids,truth,miss,xy,nxy,_=make_candidates(track,nodes,arcs,route);model=joblib.load(ROOT/'models'/'model.joblib');p=model.predict_proba(X)[:,1]
best={}
for i,a,pr in zip(obs,cids,p):
 if int(i) not in best or pr>best[int(i)][0]:best[int(i)]=(pr,int(a))
cls=[best[i][1] for i in sorted(best)];hmm=viterbi_decode(model,X,obs,cids,arcs,timestamps=track[:,2])
m=folium.Map(location=[float(track[:,1].mean()),float(track[:,0].mean())],zoom_start=13,control_scale=True)
folium.PolyLine(track[:,[1,0]].tolist(),color='#2674d8',weight=3,tooltip='Observed GPS').add_to(m)
def addroute(ids,name,color):
 fg=folium.FeatureGroup(name=name,show=True);pts=[]
 for aid in ids:
  if 0<=int(aid)<len(arcs):
   u,v=arcs[int(aid)]
   if u<len(nodes) and v<len(nodes):pts.extend([[float(nodes[u,1]),float(nodes[u,0])],[float(nodes[v,1]),float(nodes[v,0])]])
 if pts:folium.PolyLine(pts,color=color,weight=4,opacity=.85,tooltip=name).add_to(fg)
 fg.add_to(m)
addroute(route,'Reference route','#15945b');addroute(cls,'Classifier route','#e58a22');addroute(hmm,'HMM / Viterbi route','#9a45b8')
folium.LayerControl(collapsed=False).add_to(m);m.save(ROOT/'visualizations'/'training_trip_00000001_full_map.html')
