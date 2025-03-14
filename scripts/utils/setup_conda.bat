@echo off
setlocal EnableDelayedExpansion

echo Setting up Anaconda environment for CS2 RL Agent
echo ---------------------------------------------------

REM Set working directory to script location
cd /d "%~dp0"

REM Include common functions library if it exists, otherwise continue with standalone functionality
if exist "%~dp0common_functions.bat" (
    call "%~dp0common_functions.bat"
    set USING_COMMON=1
) else (
    set USING_COMMON=0
)

REM Check if Conda is installed
call conda --version > nul 2>&1
if errorlevel 1 (
    echo ERROR: Anaconda/Miniconda is not installed or not in PATH
    echo Please install Anaconda or Miniconda from: https://www.anaconda.com/download/
    pause
    exit /b 1
)
echo Anaconda is installed.

REM Create conda environment if it doesn't exist
call conda env list | findstr "cs2_agent" > nul
if errorlevel 1 (
    echo Creating conda environment 'cs2_agent'...
    set CREATE_CMD=call conda create -y -n cs2_agent python=3.10
    
    REM Use error handler if available, otherwise use direct command
    if "%USING_COMMON%"=="1" (
        call :error_handler "%CREATE_CMD%" 3
    ) else (
        %CREATE_CMD%
        if errorlevel 1 (
            echo Failed to create conda environment.
            pause
            exit /b 1
        )
    )
) else (
    echo Conda environment 'cs2_agent' already exists.
)

REM Activate conda environment
if "%USING_COMMON%"=="1" (
    call :activate_conda
) else (
    echo Activating conda environment...
    call conda activate cs2_agent
    if errorlevel 1 (
        echo Failed to activate conda environment.
        pause
        exit /b 1
    )
)

REM Upgrade pip
echo Upgrading pip...
call python -m pip install --upgrade pip

REM Install ML libraries using caching when possible
if "%USING_COMMON%"=="1" (
    call :check_dependencies "requirements.txt"
) else (
    if not exist ".deps_installed" (
        echo Installing required packages...
        
        REM Install core ML libraries
        call conda install -y -c conda-forge "pytorch" "tensorflow" "gymnasium" "optuna" "numpy" "pandas" "matplotlib" "seaborn" "tensorboard" "pyyaml" "tqdm" "pytest" "jupyter" "requests" "pillow"
        if errorlevel 1 (
            echo WARNING: Some conda packages failed to install.
        )
        
        REM Install pip packages
        call pip install stable-baselines3 pytesseract pywin32 pyautogui mss keyboard pygame pydirectinput pynput opencv-python ray
        if errorlevel 1 (
            echo WARNING: Some pip packages failed to install.
        )
        
        echo Installed on %date% %time% > .deps_installed
    ) else (
        echo Dependencies already installed (using cached info).
        echo Delete .deps_installed file to force reinstallation.
    )
)

REM Check Tesseract installation
echo Checking Tesseract OCR installation...
tesseract --version > nul 2>&1
if errorlevel 1 (
    echo WARNING: Tesseract OCR is not installed or not in PATH
    echo Please install Tesseract OCR from: https://github.com/UB-Mannheim/tesseract/wiki
    echo After installation, add it to your PATH environment variable
    echo Recommended installation path: C:\Program Files\Tesseract-OCR
)

REM Setup GPU optimization if using common functions
if "%USING_COMMON%"=="1" (
    call :setup_gpu
    call :detect_gpu
    
    if "%GPU_AVAILABLE%"=="1" (
        echo NVIDIA GPU detected with CUDA version %CUDA_VERSION%.
        echo GPU acceleration is available.
    ) else (
        echo No GPU detected. Machine learning will use CPU only.
    )
)

echo.
echo Setup complete! You can now run the training scripts.
echo Remember to activate the conda environment with 'conda activate cs2_agent' before running any scripts manually.
pause
endlocal 