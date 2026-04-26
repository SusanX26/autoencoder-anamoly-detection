@echo off
title FraudSense AI - Setup & Launch
color 0A

echo.
echo  ****************************************************
echo  *                                                  *
echo  *          FraudSense AI - Auto Setup              *
echo  *    Credit Card Fraud Detection Dashboard         *
echo  *                                                  *
echo  ****************************************************
echo.

:: -----------------------------------------------
:: STEP 1: Check Python
:: -----------------------------------------------
echo [1/6] Checking Python installation...
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Python is not installed or not in PATH.
    echo         Please download and install Python 3.9+ from https://python.org
    echo         Make sure to check "Add Python to PATH" during installation!
    pause
    exit /b 1
)
echo [OK] Python found.

:: -----------------------------------------------
:: STEP 2: Check Node.js
:: -----------------------------------------------
echo [2/6] Checking Node.js installation...
node --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Node.js is not installed or not in PATH.
    echo         Please download and install Node.js from https://nodejs.org
    pause
    exit /b 1
)
echo [OK] Node.js found.

:: -----------------------------------------------
:: STEP 3: Setup Python Virtual Environment
:: -----------------------------------------------
echo [3/6] Setting up Python virtual environment...
if not exist "venv" (
    python -m venv venv
    echo [OK] Virtual environment created.
) else (
    echo [OK] Virtual environment already exists.
)

:: Activate it
call venv\Scripts\activate.bat

:: Install Python requirements
echo Installing Python dependencies (this may take a few minutes)...
pip install -q -r requirements.txt
if %errorlevel% neq 0 (
    echo [ERROR] Failed to install Python dependencies.
    pause
    exit /b 1
)
echo [OK] Python dependencies installed.

:: -----------------------------------------------
:: STEP 4: Check for model files
:: -----------------------------------------------
echo [4/6] Checking for trained model files...
if not exist "models\standard_ae.onnx" (
    echo [INFO] Models not found. Checking for dataset...
    if not exist "temp_data.csv" (
        if not exist "creditcard_2023.csv\creditcard_2023.csv" (
            echo [WARNING] Dataset file not found!
            pause
        )
    )
    echo [INFO] Training both Standard and Sparse models (this takes 3-5 minutes)...
    python fraud_detector_engine.py
    if %errorlevel% neq 0 (
        echo [ERROR] Model training failed.
        pause
        exit /b 1
    )
    echo [OK] Both models trained and exported.
) else (
    echo [OK] Multi-model system found. Skipping training.
)

:: -----------------------------------------------
:: STEP 5: Install Dashboard (Frontend) Dependencies
:: -----------------------------------------------
echo [5/6] Installing dashboard dependencies...
cd dashboard
call npm install --silent
if %errorlevel% neq 0 (
    echo [ERROR] npm install failed.
    cd ..
    pause
    exit /b 1
)
cd ..
echo [OK] Dashboard dependencies installed.

:: -----------------------------------------------
:: STEP 6: Launch Both Servers
:: -----------------------------------------------
echo [6/6] Launching FraudSense AI...
echo.
echo  Starting API server on http://localhost:8000
echo  Starting Dashboard on  http://localhost:5173
echo.
echo  Press CTRL+C in any window to stop.
echo.

:: Start API server in a new window
start "FraudSense - API Server" cmd /k "call venv\Scripts\activate.bat && python api_server.py"

:: Wait 3 seconds for API to start
timeout /t 3 /nobreak >nul

:: Start Dashboard dev server in a new window
start "FraudSense - Dashboard" cmd /k "cd dashboard && npm run dev"

:: Wait 3 more seconds then open browser
timeout /t 3 /nobreak >nul
start "" "http://localhost:5173"

echo.
echo  ****************************************************
echo  *   FraudSense AI is running!                     *
echo  *   Dashboard: http://localhost:5173              *
echo  *   API Docs:  http://localhost:8000/docs         *
echo  ****************************************************
echo.
pause
