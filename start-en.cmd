@echo off
cd /d "%~dp0"
".venv\Scripts\python.exe" -m nod live --language en --open-browser
pause
