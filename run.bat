@echo off
setlocal

REM Go to project folder (the folder this .bat file is in)
cd /d "%~dp0"

REM Activate venv and run the script
call .venv\Scripts\activate

REM Run and log output
python daily_brief.py >> output\run.log 2>&1

endlocal
