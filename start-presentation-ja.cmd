@echo off
cd /d "%~dp0"
".venv\Scripts\python.exe" -m nod live --language ja --response production --open-browser
pause
