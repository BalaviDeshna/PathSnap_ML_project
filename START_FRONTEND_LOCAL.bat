@echo off
cd /d "%~dp0"
echo Starting frontend static server on port 5500...
echo Open http://localhost:5500 on this laptop.
py -3.11 -m http.server 5500 --directory frontend
