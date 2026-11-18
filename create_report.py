"""Generate a measured, data-driven PathSnap technical report PDF."""
from pathlib import Path
import os,sys,json
import numpy as np,pandas as pd,matplotlib.pyplot as plt,joblib
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet,ParagraphStyle
from reportlab.lib.units import cm
from reportlab.platypus import SimpleDocTemplate,Paragraph,Spacer,Table,TableStyle,Image,PageBreak
ROOT=Path(__file__).resolve().parent; DATA=Path(os.environ.get('PATHSNAP_DATA',ROOT/'data')); R=ROOT/'visualizations'; R.mkdir(exist_ok=True)
sys.path.insert(0,str(ROOT/'src'))
from pathsnap import read_trip,make_candidates,viterbi_decode
cv=pd.read_csv(ROOT/'results'/'model_comparison_cv.csv')
student=json.loads((ROOT/'results'/'student_test_model_comparison.json').read_text())
tv=json.loads((ROOT/'results'/'tuned_vs_untuned_student_test.json').read_text())
bo=json.loads((ROOT/'results'/'bayesian_optimisation.json').read_text())
summary=pd.read_csv(ROOT/'results'/'dataset_summary.csv')
info=json.loads((ROOT/'models'/'model_info.json').read_text())
# Model comparison plots
plt.figure(figsize=(9,4.5)); plt.bar(cv.model,cv.cv_candidate_macro_f1); plt.xticks(rotation=25,ha='right'); plt.ylabel('3-fold CV candidate macro F1'); plt.title('Six-model comparison (80 training trips)'); plt.tight_layout(); plt.savefig(R/'model_cv_comparison.png',dpi=180); plt.close()
order=['LogisticRegression','KNN','RandomForest','ExtraTrees','HistGradientBoosting','AdaBoost']
plt.figure(figsize=(9,4.5)); plt.bar(order,[student[k]['observation_top1_accuracy_mean_per_trip'] for k in order]); plt.xticks(rotation=25,ha='right'); plt.ylabel('Mean observation-level Top-1 accuracy'); plt.title('Six-model student-test comparison (10 trips)'); plt.tight_layout(); plt.savefig(R/'student_test_top1_comparison.png',dpi=180); plt.close()
labels=['Candidate macro F1','Candidate log loss','Observation Top-1','HMM Top-1']
unt=tv['untuned']; tun=tv['tuned']; x=np.arange(len(labels)); w=.36
plt.figure(figsize=(9,4.5)); plt.bar(x-w/2,[unt['candidate_macro_f1'],unt['candidate_log_loss'],unt['observation_top1_accuracy_mean_per_trip'],unt['hmm_top1_accuracy_mean_per_trip']],w,label='Untuned RF'); plt.bar(x+w/2,[tun['candidate_macro_f1'],tun['candidate_log_loss'],tun['observation_top1_accuracy_mean_per_trip'],tun['hmm_top1_accuracy_mean_per_trip']],w,label='Bayesian-tuned RF'); plt.xticks(x,labels,rotation=12); plt.ylabel('Metric value'); plt.title('Untuned vs Bayesian-tuned Random Forest'); plt.legend(); plt.tight_layout(); plt.savefig(R/'tuned_vs_untuned.png',dpi=180); plt.close()
# Static route reconstruction figure for a real training trip
folder=DATA/'train'/'00000001'; track,nodes,arcs,route=read_trip(folder); X,y,obs,cids,truth,miss,*_=make_candidates(track,nodes,arcs,route); model=joblib.load(ROOT/'models'/'model.joblib'); prob=model.predict_proba(X)[:,1]
best={}
for oi,aid,p in zip(obs,cids,prob):
    if int(oi) not in best or p>best[int(oi)][0]: best[int(oi)]=(float(p),int(aid))
classifier=[best[i][1] for i in sorted(best)]; decoded=viterbi_decode(model,X,obs,cids,arcs,timestamps=track[:,2])
def route_xy(ids):
    xx=[]; yy=[]
    for aid in ids:
        if 0<=int(aid)<len(arcs):
            u,v=arcs[int(aid)]
            if 0<=u<len(nodes) and 0<=v<len(nodes): xx.extend([nodes[u,0],nodes[v,0],np.nan]); yy.extend([nodes[u,1],nodes[v,1],np.nan])
    return xx,yy
plt.figure(figsize=(8,6)); plt.plot(track[:,0],track[:,1],'.',ms=2,label='GPS observations');
for ids,label in [(route,'Reference route'),(classifier,'Classifier route'),(decoded,'HMM / Viterbi route')]:
    xx,yy=route_xy(ids); plt.plot(xx,yy,lw=1.4,label=label)
plt.xlabel('Longitude'); plt.ylabel('Latitude'); plt.title('Training trip 00000001: route reconstruction'); plt.legend(); plt.tight_layout(); plt.savefig(R/'route_reconstruction_snapshot.png',dpi=180); plt.close()

styles=getSampleStyleSheet(); styles.add(ParagraphStyle(name='SmallX',parent=styles['BodyText'],fontSize=8,leading=10)); styles.add(ParagraphStyle(name='SectionX',parent=styles['Heading2'],textColor=colors.HexColor('#243b70'),spaceBefore=8,spaceAfter=5))
story=[]
story += [Paragraph('PathSnap: GPS Map-Matching and Route Reconstruction',styles['Title']),Paragraph('Technical report — reproducible development experiments',styles['Heading2']),Paragraph('This report is generated from the scripts and measured outputs in this project package. All supervised experiments use the supplied 80 training trips; development evaluation uses the 10 student_test trips. The separate organiser evaluator partition was not included in the supplied data.',styles['BodyText']),Spacer(1,10)]
story += [Paragraph('1. Dataset and exploratory analysis',styles['SectionX']),Paragraph('The supplied dataset contains GPS tracks, road-network nodes, directed arcs and reference route arc sequences. The EDA script inventories trip sizes, coordinate extents, missing/non-finite GPS rows, timestamp order and sampling gaps. It exports dataset_inventory.csv, dataset_summary.csv, timestamp_gaps.csv and trip_timestamp_ranges.csv, plus distribution plots and an interactive Folium map.',styles['BodyText'])]
eda=[['Partition','Trips','GPS observations','Median GPS/trip','Median nodes','Median arcs']]
for _,r in summary.iterrows(): eda.append([r['partition'],int(r['trips']),int(r['gps_points']),int(r['median_gps_points']),int(r['median_nodes']),int(r['median_arcs'])])
t=Table(eda,colWidths=[2.5*cm,1.5*cm,2.8*cm,2.8*cm,2.3*cm,2.3*cm]); t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#243b70')),('TEXTCOLOR',(0,0),(-1,0),colors.white),('GRID',(0,0),(-1,-1),.4,colors.grey),('FONTSIZE',(0,0),(-1,-1),8),('PADDING',(0,0),(-1,-1),5)])); story += [t,Spacer(1,6)]
trainrow=summary[summary.partition=='train'].iloc[0]; testrow=summary[summary.partition=='student_test'].iloc[0]
story += [Paragraph(f"The EDA found {int(trainrow.gps_rows_invalid)} invalid GPS rows and {int(trainrow.nonpositive_gaps)} non-positive timestamp gaps in training; corresponding student-test counts were {int(testrow.gps_rows_invalid)} and {int(testrow.nonpositive_gaps)}. Training had {int(trainrow.gaps_over_2s)} intervals over 2 seconds and {int(trainrow.gaps_over_30s)} over 30 seconds. These interruptions are treated as missing recording intervals, not as evidence that a vehicle stopped.",styles['BodyText']),Image(str(R/'trajectory_lengths.png'),width=15*cm,height=7.5*cm),Image(str(R/'timestamp_gap_distribution.png'),width=15*cm,height=7.5*cm),PageBreak()]
story += [Paragraph('2. Candidate generation and feature engineering',styles['SectionX']),Paragraph('For each GPS observation, a KD-tree over road-segment midpoints retrieves up to eight nearest directed road-segment candidates. Nine features are used: point-to-segment distance, midpoint distance, segment length, heading difference, GPS speed, candidate rank, distance to start node, distance to end node, and timestamp gap. The binary candidate target is positive when the candidate arc equals the nearest supplied reference-route arc to that GPS observation.',styles['BodyText']),Paragraph('Important label assumption: the .route file is a chronological sequence of arc IDs, not a pointwise label for each GPS observation. The nearest-reference-route-segment rule is therefore an approximation used to construct candidate labels. Candidate miss rate measures how often the nearest reference segment is absent from the eight generated candidates; a classifier or decoder cannot recover a segment that candidate generation omitted.',styles['BodyText']),Paragraph('3. Six-model comparative study',styles['SectionX']),Paragraph('Six classifiers from distinct algorithm families were compared using 3-fold GroupKFold. Trip IDs, rather than individual candidate rows, define the folds to reduce leakage between observations from the same trip. The reported candidate Macro F1 is the unweighted mean of the F1 scores for binary classes 0 and 1.',styles['BodyText'])]
rows=[['Model','Family','CV Macro F1','CV Log Loss','Fit (s)','Predict (s)']]
for _,r in cv.sort_values('cv_candidate_macro_f1',ascending=False).iterrows(): rows.append([r.model,r.family,f"{r.cv_candidate_macro_f1:.4f} ± {r.cv_macro_f1_std:.4f}",f"{r.cv_log_loss:.4f}",f"{r.cv_fit_seconds:.2f}",f"{r.cv_predict_seconds:.2f}"])
t=Table(rows,colWidths=[2.8*cm,2.9*cm,3.0*cm,2.3*cm,2.1*cm,2.1*cm]); t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#243b70')),('TEXTCOLOR',(0,0),(-1,0),colors.white),('GRID',(0,0),(-1,-1),.4,colors.grey),('FONTSIZE',(0,0),(-1,-1),7.5),('PADDING',(0,0),(-1,-1),4)])); story += [t,Spacer(1,6),Image(str(R/'model_cv_comparison.png'),width=15*cm,height=7.5*cm)]
winner=cv.sort_values('cv_candidate_macro_f1',ascending=False).iloc[0]
story += [Paragraph(f"Random Forest was the highest-scoring model in grouped CV (candidate Macro F1 {winner.cv_candidate_macro_f1:.4f}). Bayesian tuning was then evaluated on the permitted 10-trip student-test development partition; the tuned RF improved candidate Macro F1, Log Loss and observation Top-1 slightly, so it is used for the demo. All six models were compared on the same student-test partition; see results/student_test_model_comparison.json.",styles['BodyText']),Image(str(R/'student_test_top1_comparison.png'),width=15*cm,height=7.5*cm),PageBreak()]
story += [Paragraph('4. Bayesian hyperparameter optimisation',styles['SectionX']),Paragraph('Gaussian-process Bayesian optimisation with an expected-improvement acquisition function searched Random Forest n_estimators (30–90), max_depth (8–20) and min_samples_leaf (1–4). The objective was 3-fold GroupKFold candidate-level binary Macro F1 on a reproducible subset of training candidate rows. The search ran eight trials; the best trial is reported below.',styles['BodyText'])]
best=bo['best_trial']; bp=best['params']; story += [Paragraph(f"Best trial: {best['trial']}; n_estimators={bp['n_estimators']}; max_depth={bp['max_depth']}; min_samples_leaf={bp['min_samples_leaf']}; CV Macro F1={best['cv_macro_f1']:.4f}.",styles['BodyText'])]
brows=[['Trial','Trees','Depth','Min leaf','CV Macro F1']]
for tr in bo['trials']:
    p=tr['params']; brows.append([tr['trial'],p['n_estimators'],p['max_depth'],p['min_samples_leaf'],f"{tr['cv_macro_f1']:.4f}"])
t=Table(brows,colWidths=[2.2*cm,2.5*cm,2.5*cm,2.5*cm,3.5*cm]); t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#243b70')),('TEXTCOLOR',(0,0),(-1,0),colors.white),('GRID',(0,0),(-1,-1),.4,colors.grey),('FONTSIZE',(0,0),(-1,-1),8),('PADDING',(0,0),(-1,-1),4)])); story += [t,Spacer(1,6)]
for k,label in [('untuned','Untuned Random Forest'),('tuned','Bayesian-tuned Random Forest')]:
    m=tv[k]; story.append(Paragraph(f"{label}: student-test candidate Macro F1 {m['candidate_macro_f1']:.4f}; candidate Log Loss {m['candidate_log_loss']:.4f}; observation Top-1 {m['observation_top1_accuracy_mean_per_trip']:.4f}; HMM Top-1 {m['hmm_top1_accuracy_mean_per_trip']:.4f}.",styles['BodyText']))
story += [Image(str(R/'tuned_vs_untuned.png'),width=15*cm,height=7.5*cm),Paragraph('The tuned and untuned models are compared on the same student-test partition. The student-test metrics are development results, not evaluator results. The tuned RF artifact is selected for the demo after the documented student-test development comparison; the reserved evaluator partition remains untouched.',styles['BodyText']),PageBreak()]
story += [Paragraph('5. HMM / Viterbi route reconstruction',styles['SectionX']),Paragraph('Each observation’s hidden state is one of its road-segment candidates. The classifier probability is used as the emission score. A transition is favoured when the next segment is the same segment or starts at the previous segment’s end node; disconnected transitions receive a fixed penalty. Dynamic programming and backtracking select the best candidate sequence. The decoder restarts after timestamp gaps greater than 30 seconds so long recording interruptions are not treated as continuous movement.',styles['BodyText'])]
rf=student['RandomForest_BayesianTuned']; story += [Paragraph(f"On the student-test trips, the Random Forest classifier-only mean per-trip Top-1 accuracy was {rf['observation_top1_accuracy_mean_per_trip']:.3%}; after Viterbi decoding it was {rf['hmm_top1_accuracy_mean_per_trip']:.3%}. Mean route continuity increased from {rf['classifier_route_continuity_mean']:.3%} to {rf['hmm_route_continuity_mean']:.3%}. Candidate miss rate was {rf['candidate_miss_rate']:.3%}.",styles['BodyText']),Image(str(R/'route_reconstruction_snapshot.png'),width=15*cm,height=10.5*cm),Paragraph('Interactive map: visualizations/training_trip_00000001_full_map.html. The Folium map contains GPS observations, local road-network geometry, reference route, classifier route and Viterbi route as selectable layers.',styles['SmallX']),PageBreak()]
story += [Paragraph('6. Limitations and reproducibility',styles['SectionX']),Paragraph('The implementation is a baseline, not a claim of optimal map matching. Limitations include the approximate point labels derived from route arc sequences, fixed K=8 candidate generation, a local equirectangular coordinate approximation, a fixed disconnected-transition penalty, and simplified treatment of transitions. The Viterbi decoder restarts after large timestamp gaps but does not yet learn transition probabilities from data or use travel-time constraints.',styles['BodyText']),Paragraph('Reproducibility: run eda.py, train.py, bayes_tune.py, fit_final.py, evaluate.py, make_map.py and create_report.py in that order. The scripts save raw experiment outputs under results/, trained models under models/, and plots/maps/PDF under visualizations/. The separate evaluator partition is not bundled; do not claim these student-test metrics as final evaluator performance. The team should review and understand all code and disclose AI assistance according to the hackathon policy.',styles['BodyText'])]
doc=SimpleDocTemplate(str(R/'PathSnap_Technical_Report.pdf'),pagesize=A4,rightMargin=1.5*cm,leftMargin=1.5*cm,topMargin=1.4*cm,bottomMargin=1.4*cm)
doc.build(story)
print('Created',R/'PathSnap_Technical_Report.pdf')
