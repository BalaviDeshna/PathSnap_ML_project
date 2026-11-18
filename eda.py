"""Dataset EDA for the supplied train and student_test partitions."""
import os,sys
from pathlib import Path
import numpy as np,pandas as pd,matplotlib.pyplot as plt,folium
ROOT=Path(__file__).resolve().parent; DATA=Path(os.environ.get('PATHSNAP_DATA',ROOT/'data'))
sys.path.insert(0,str(ROOT/'src'))
from pathsnap import read_trip
for d in [ROOT/'results',ROOT/'visualizations']: d.mkdir(exist_ok=True)
rows=[]; gap_rows=[]; gps_rows=[]
for part in ['train','student_test']:
    for folder in sorted(p for p in (DATA/part).iterdir() if p.is_dir()):
        track,nodes,arcs,route=read_trip(folder); gaps=np.diff(track[:,2]) if len(track)>1 else np.array([])
        finite=np.isfinite(track).all(axis=1)
        rows.append({'partition':part,'trip_id':folder.name,'gps_points':len(track),'nodes':len(nodes),'arcs':len(arcs),'route_arcs':len(route),'missing_or_nonfinite_gps_rows':int((~finite).sum()),'nonpositive_time_gaps':int((gaps<=0).sum()),'gaps_over_2s':int((gaps>2).sum()),'gaps_over_30s':int((gaps>30).sum()),'median_gap_s':float(np.median(gaps)) if len(gaps) else None,'max_gap_s':float(np.max(gaps)) if len(gaps) else None,'lon_min':float(np.nanmin(track[:,0])),'lon_max':float(np.nanmax(track[:,0])),'lat_min':float(np.nanmin(track[:,1])),'lat_max':float(np.nanmax(track[:,1]))})
        for g in gaps:
            gap_rows.append({'partition':part,'trip_id':folder.name,'gap_s':float(g)})
        gps_rows.append({'partition':part,'trip_id':folder.name,'observations':len(track),'first_timestamp':float(track[0,2]),'last_timestamp':float(track[-1,2])})
df=pd.DataFrame(rows); df.to_csv(ROOT/'results'/'dataset_inventory.csv',index=False)
pd.DataFrame(gap_rows).to_csv(ROOT/'results'/'timestamp_gaps.csv',index=False)
pd.DataFrame(gps_rows).to_csv(ROOT/'results'/'trip_timestamp_ranges.csv',index=False)
summary=df.groupby('partition').agg(trips=('trip_id','count'),gps_points=('gps_points','sum'),median_gps_points=('gps_points','median'),median_nodes=('nodes','median'),median_arcs=('arcs','median'),median_route_arcs=('route_arcs','median'),gps_rows_invalid=('missing_or_nonfinite_gps_rows','sum'),nonpositive_gaps=('nonpositive_time_gaps','sum'),gaps_over_2s=('gaps_over_2s','sum'),gaps_over_30s=('gaps_over_30s','sum')).reset_index()
summary.to_csv(ROOT/'results'/'dataset_summary.csv',index=False)
train=df[df.partition=='train']
plt.figure(figsize=(8,4));plt.hist(train.gps_points,bins=18);plt.xlabel('GPS observations per trip');plt.ylabel('Trip count');plt.title('Training trajectory lengths');plt.tight_layout();plt.savefig(ROOT/'visualizations'/'trajectory_lengths.png',dpi=160);plt.close()
plt.figure(figsize=(8,4));plt.scatter(train.lon_min,train.lat_min,s=18);plt.xlabel('Minimum longitude');plt.ylabel('Minimum latitude');plt.title('Training trip geographic extents');plt.tight_layout();plt.savefig(ROOT/'visualizations'/'geographic_extent.png',dpi=160);plt.close()
plt.figure(figsize=(8,4));plt.hist([np.log10(np.maximum(np.array([r['gap_s'] for r in gap_rows if r['partition']==p]),1e-3)) for p in ['train','student_test']],bins=30,label=['Train','Student test'],alpha=.7);plt.xlabel('log10(timestamp gap in seconds)');plt.ylabel('Interval count');plt.title('GPS sampling interval distribution');plt.legend();plt.tight_layout();plt.savefig(ROOT/'visualizations'/'timestamp_gap_distribution.png',dpi=160);plt.close()
# Representative training trip map: GPS, road-network geometry and reference route.
f=DATA/'train'/'00000001'; tr,n,a,r=read_trip(f); center=[float(np.mean(tr[:,1])),float(np.mean(tr[:,0]))]
m=folium.Map(location=center,zoom_start=13,control_scale=True)
folium.PolyLine(tr[:,[1,0]].tolist(),color='#2878c8',weight=3,tooltip='Observed GPS trajectory').add_to(m)
road=folium.FeatureGroup(name='Road network (local sample)',show=False)
# Restrict to the trip's local bounding box and cap rendered segments for a responsive HTML map.
near=np.flatnonzero((n[:,0]>=tr[:,0].min()-.003)&(n[:,0]<=tr[:,0].max()+.003)&(n[:,1]>=tr[:,1].min()-.003)&(n[:,1]<=tr[:,1].max()+.003))
near_set=set(near.tolist()); selected=[(int(u),int(v)) for u,v in a if int(u) in near_set and int(v) in near_set]
if len(selected)>2500: selected=[selected[i] for i in np.linspace(0,len(selected)-1,2500,dtype=int)]
for u,v in selected:
    folium.PolyLine([[float(n[u,1]),float(n[u,0])],[float(n[v,1]),float(n[v,0])]],color='#888888',weight=1,opacity=.35).add_to(road)
road.add_to(m)
ref=folium.FeatureGroup(name='Reference route',show=True)
for aid in r:
    if 0<=aid<len(a):
        u,v=a[aid]
        if 0<=u<len(n) and 0<=v<len(n): folium.PolyLine([[float(n[u,1]),float(n[u,0])],[float(n[v,1]),float(n[v,0])]],color='#169b62',weight=4).add_to(ref)
ref.add_to(m); folium.LayerControl(collapsed=False).add_to(m); m.save(ROOT/'visualizations'/'training_trip_00000001_map.html')
print(summary.to_string(index=False)); print('EDA outputs saved in results/ and visualizations/')
