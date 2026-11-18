@echo off
setlocal
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  echo ERROR: Virtual environment not found.
  echo First run: py -3.11 -m venv .venv
  echo Then: .venv\Scripts\activate ^& pip install -r requirements.txt
  exit /b 1
)
set "PATHSNAP_DATA=%CD%\data"
if "%PATHSNAP_BO_TRIALS%"=="" set "PATHSNAP_BO_TRIALS=8"
set PY="%CD%\.venv\Scripts\python.exe"
echo [1/7] EDA
%PY% eda.py || goto :error
echo [2/7] Six-model grouped cross-validation
%PY% train.py || goto :error
echo [3/7] Bayesian hyperparameter optimisation
%PY% bayes_tune.py || goto :error
echo [4/7] Fit final model artifacts
%PY% fit_final.py || goto :error
echo [5/7] Evaluate on student_test
%PY% evaluate.py || goto :error
echo [6/7] Generate interactive map
%PY% make_map.py || goto :error
echo [7/7] Generate technical report
%PY% create_report.py || goto :error
echo.
echo SUCCESS. Outputs are in results\, models\ and visualizations\.
exit /b 0
:error
echo.
echo ERROR: A pipeline stage failed. Read the error above and rerun that stage manually.
exit /b 1
