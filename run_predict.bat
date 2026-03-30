@echo off
cd /d %~dp0
if "%~1"=="" (
    echo Usage: run_predict.bat image_path
    pause
    exit /b 1
)
python predict_one_neu.py "%~1"
pause
