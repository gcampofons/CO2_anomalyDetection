@echo off
REM Automatic setup script for Windows
REM CO2 Anomaly Detection Project

echo ========================================
echo  CO2 Anomaly Detection - Setup
echo ========================================
echo.

REM Verify Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python is not installed or not in PATH
    echo Please install Python 3.13
    pause
    exit /b 1
)

echo [1/5] Verifying Python version...
for /f "tokens=2" %%i in ('python --version') do set PYTHON_VERSION=%%i
echo Found Python %PYTHON_VERSION%

REM Check if Python 3.13 is installed
echo %PYTHON_VERSION% | findstr /r "^3\.13" >nul
if errorlevel 1 (
    echo [ERROR] Python 3.13 is required, but found %PYTHON_VERSION%
    echo Please install Python 3.13 from https://www.python.org/downloads/
    pause
    exit /b 1
)

echo [OK] Python 3.13 detected
echo.

REM Create virtual environment if it doesn't exist
if not exist "venv" (
    echo [2/5] Creating virtual environment...
    python -m venv venv
    echo Virtual environment created successfully.
) else (
    echo [2/5] Virtual environment already exists.
)
echo.

REM Activate virtual environment and upgrade pip
echo [3/5] Activating virtual environment and upgrading pip...
call venv\Scripts\activate.bat
python -m pip install --upgrade pip --quiet
echo.

REM Install dependencies
echo [4/5] Installing dependencies...
pip install -r requirements.txt
echo.

REM Verify installation
echo [5/5] Verifying installation...
python -c "import tensorflow, numpy, pandas, sklearn; print('All packages installed successfully!')"
echo.
echo ========================================
echo  Installation completed!
echo ========================================
echo.
echo To run the project:
echo   1. Activate virtual environment: venv\Scripts\activate
echo   2. Run the script: python AE_LSTM_anomaly_detection.py
echo.
echo IMPORTANT: Download the dataset CEIP_Albea_ValldAlba.csv
echo from https://doi.org/10.5281/zenodo.5036228
echo.

pause
