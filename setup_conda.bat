@echo off
echo Setting up Anaconda environment for Cities: Skylines 2 RL Agent
echo ---------------------------------------------------

REM Check if Conda is installed
call conda --version > nul 2>&1
if errorlevel 1 (
    echo Anaconda/Miniconda is not installed or not in PATH
    echo Please install Anaconda or Miniconda from: https://www.anaconda.com/download/
    pause
    exit /b 1
)

REM Create conda environment if it doesn't exist
call conda env list | findstr "cs2_agent" > nul
if errorlevel 1 (
    echo Creating conda environment 'cs2_agent'...
    call conda create -y -n cs2_agent python=3.10
) else (
    echo Conda environment 'cs2_agent' already exists
)

REM Activate conda environment
call conda activate cs2_agent

REM Upgrade pip
call python -m pip install --upgrade pip

REM Install core ML libraries without version constraints
echo Installing required packages without version constraints...
call conda install -y -c conda-forge "pytorch" "tensorflow" "gymnasium" "optuna" "numpy" "pandas" "matplotlib" "seaborn" "tensorboard" "pyyaml" "tqdm" "pytest" "jupyter" "requests" "pillow"

REM Install pip packages that might not be available in conda
call pip install stable-baselines3 pytesseract pywin32 pyautogui mss keyboard pygame pydirectinput pynput opencv-python ray

REM Check Tesseract installation
echo Checking Tesseract OCR installation...
tesseract --version > nul 2>&1
if errorlevel 1 (
    echo WARNING: Tesseract OCR is not installed or not in PATH
    echo Please install Tesseract OCR from: https://github.com/UB-Mannheim/tesseract/wiki
    echo After installation, add it to your PATH environment variable
    echo Recommended installation path: C:\Program Files\Tesseract-OCR
)

echo.
echo Setup complete! You can now run the training scripts.
echo Remember to activate the conda environment with 'conda activate cs2_agent' before running any scripts manually.
pause 