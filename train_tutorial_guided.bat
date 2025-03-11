@echo off
echo ====================================================
echo Cities: Skylines 2 Tutorial-Guided RL Agent Training
echo ====================================================
echo.

REM Check if Python is installed
where python >nul 2>nul
if %errorlevel% neq 0 (
    echo Python not found. Please install Python 3.8 or later.
    exit /b
)

REM Check if virtual environment exists, create if not
if not exist venv (
    echo Creating virtual environment...
    python -m venv venv
    if %errorlevel% neq 0 (
        echo Failed to create virtual environment.
        exit /b
    )
)

REM Activate virtual environment
echo Activating virtual environment...
call venv\Scripts\activate.bat

REM Install dependencies if needed
if not exist venv\Lib\site-packages\stable_baselines3 (
    echo Installing dependencies...
    pip install -r requirements.txt
    
    if %errorlevel% neq 0 (
        echo Failed to install dependencies.
        exit /b
    )
    
    echo Installing Ollama client...
    pip install requests pillow
)

REM Check if Ollama is running
echo Checking if Ollama is running...
curl -s http://localhost:11434/api/version >nul
if %errorlevel% neq 0 (
    echo WARNING: Ollama server doesn't seem to be running at http://localhost:11434
    echo Please start Ollama and ensure the llava:7b-v1.6-vision model is available.
    echo You can download it with: ollama pull llava:7b-v1.6-vision
    
    choice /C YN /M "Do you want to continue anyway"
    if %errorlevel% equ 2 exit /b
)

REM Run the tutorial-guided training script
echo Starting tutorial-guided training...
python train_tutorial_guided.py --config config/tutorial_guided_config.yaml

REM Deactivate virtual environment
call venv\Scripts\deactivate.bat

echo Training complete!
pause 