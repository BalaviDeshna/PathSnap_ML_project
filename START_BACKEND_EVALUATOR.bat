@echo off
cd /d "%~dp0"
call .venv\Scripts\activate
set PATHSNAP_PARTITION=evaluator_test
set PATHSNAP_DATA=%CD%\data
echo Starting PathSnap evaluator backend on all network interfaces...
echo Keep this window open during evaluation.
uvicorn api:app --host 0.0.0.0 --port 8000
