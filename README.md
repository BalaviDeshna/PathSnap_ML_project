# PathSnap — GPS Map Matching & Route Reconstruction

This project follows the requested folder structure and provides a reproducible workflow for the PES UE24CS352A PathSnap hackathon. It includes 80 training trips and 10 student-test trips, six classifiers, trip-grouped cross-validation, Bayesian optimisation, candidate generation, feature engineering, Viterbi route reconstruction, FastAPI and a browser UI.

> **Data boundary:** the included `student_test` data is for development only. The organiser's reserved evaluator partition is not included. Never describe student-test results as evaluator results.

## Folder structure

```text
PathSnap/
├── data/
│   ├── train/
│   └── student_test/
├── notebooks/
│   ├── 01_dataset_exploration.ipynb
│   ├── 02_preprocessing.ipynb
│   ├── 03_feature_engineering.ipynb
│   ├── 04_model_experiments.ipynb
│   ├── 05_hyperparameter_tuning.ipynb
│   └── 06_hmm_route_reconstruction.ipynb
├── src/
│   ├── data_loader.py
│   ├── preprocessing.py
│   ├── candidate_generation.py
│   ├── feature_engineering.py
│   ├── train.py
│   ├── evaluate.py
│   ├── hmm.py
│   └── api.py
├── models/
├── results/
├── visualizations/
├── frontend/
├── requirements.txt
└── README.md
```

Root-level orchestration scripts (`eda.py`, `train.py`, `bayes_tune.py`, `fit_final.py`, `evaluate.py`, `make_map.py`, `create_report.py`) are retained to make the pipeline easy to run.

## Setup (Windows / VS Code)

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If PowerShell blocks activation, run `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` and activate again.

## Run everything

```powershell
.\run_all.bat
```

This runs EDA → six-model GroupKFold comparison → Bayesian tuning → final model fitting → student-test evaluation → map generation → PDF report. Outputs are saved under `results/`, `models/` and `visualizations/`.

Run individual stages with `python eda.py`, `python train.py`, `python bayes_tune.py`, `python fit_final.py`, `python evaluate.py`, `python make_map.py` and `python create_report.py`.

## API and frontend

Start API (Laptop 1):

```powershell
uvicorn api:app --host 0.0.0.0 --port 8000
```

Serve frontend (Laptop 2 or same machine):

```powershell
python -m http.server 5500 --directory frontend
```

Open `http://localhost:5500`; set the API URL to `http://localhost:8000` (or Laptop 1's LAN address). API defaults to `student_test` for demonstration. For organiser evaluation, configure the exact supplied partition server-side and keep all reference labels hidden.

## Methods and limitations

- Candidate generation: up to 8 nearest road segments using a midpoint KD-tree.
- Nine features: point-to-segment distance, midpoint distance, segment length, heading difference, GPS speed, candidate rank, start/end node distances and timestamp gap.
- Models: Logistic Regression, KNN, Random Forest, Extra Trees, HistGradientBoosting and AdaBoost.
- Model comparison: 3-fold GroupKFold by trip; candidate-level binary Macro F1 and log loss.
- Bayesian optimisation: Gaussian-process surrogate with expected improvement; Random Forest search uses training trips only.
- Decoder: classifier probabilities as emissions; directed graph connectivity as a transition preference; resets after long GPS recording gaps.
- Labels are approximated by assigning GPS observations to their nearest supplied reference-route arc because `.route` is a sequence of arc IDs, not pointwise labels. This approximation must be disclosed.
- Fixed K=8 means a reference road segment can be absent from the candidate set; report candidate miss rate.
- The decoder uses a fixed disconnected-transition penalty, not a learned travel-time model.
- Student-test metrics are development metrics only.

Read `VIVA_QA.md` for presentation preparation. Verify generated metrics and disclose AI assistance according to the course policy.
