@echo off
setlocal EnableDelayedExpansion

echo ======================================================
echo All-in-One Setup and Training for CS2 RL Adaptive Agent
echo ======================================================

REM Set working directory to script location
cd /d "%~dp0"

REM Include common functions library
call ..\utils\common_functions.bat

REM Check for help command
if /I "%~1"=="help" (
    echo Usage: all_in_one_setup_and_train.bat [timesteps] [starting_mode] [focus] [load_path]
    echo.
    echo Parameters:
    echo   timesteps     - Number of training timesteps (default: 20000)
    echo   starting_mode - Starting training mode (default: discovery)
    echo                  Valid options: discovery, tutorial, vision, autonomous, strategic
    echo   focus         - Whether to auto-focus the game window (default: true)
    echo                  Valid options: true, false
    echo   load_path     - Path to load a saved adaptive agent from (optional)
    echo.
    echo Examples:
    echo   all_in_one_setup_and_train.bat
    echo   all_in_one_setup_and_train.bat 30000
    echo   all_in_one_setup_and_train.bat 20000 vision
    echo   all_in_one_setup_and_train.bat 20000 discovery false
    echo   all_in_one_setup_and_train.bat 50000 strategic
    echo   all_in_one_setup_and_train.bat 20000 discovery true "models/adaptive/run_20250313/agent"
    echo.
    exit /b 0
)

REM Parse command line arguments
set TIMESTEPS=20000
set STARTING_MODE=discovery
set FOCUS=true
set LOAD_PATH=

if not "%~1"=="" set TIMESTEPS=%~1
if not "%~2"=="" set STARTING_MODE=%~2
if not "%~3"=="" set FOCUS=%~3
if not "%~4"=="" set LOAD_PATH=%~4

REM Configure focus flag
if /I "%FOCUS%"=="true" (
    set FOCUS_FLAG=--focus
) else (
    set FOCUS_FLAG=
)

REM Configure load path
if not "%LOAD_PATH%"=="" (
    set LOAD_FLAG=--load "%LOAD_PATH%"
) else (
    set LOAD_FLAG=
)

echo.
echo === Configuration ===
echo Timesteps: %TIMESTEPS%
echo Starting Mode: %STARTING_MODE%
echo Auto-focus: %FOCUS%
if not "%LOAD_PATH%"=="" (
    echo Loading from: %LOAD_PATH%
)
echo.

REM ======== Environment Setup =========

echo === Setting up environment ===

REM Create required directories
mkdir ..\..\logs 2>nul
mkdir ..\..\logs\tensorboard 2>nul
mkdir ..\..\models 2>nul
mkdir ..\..\models\adaptive 2>nul

REM Check for Python and Conda
where python >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo Python not found in PATH. Checking for Conda...
    call :activate_conda
)

REM Check for pip
where pip >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo ERROR: pip not found in PATH. Please install Python or activate the correct environment.
    exit /b 1
)

REM Install requirements if needed
if not exist ..\..\requirements_installed.txt (
    echo Installing requirements...
    pip install -r ..\..\requirements.txt
    if %ERRORLEVEL%==0 (
        echo Requirements installed successfully > ..\..\requirements_installed.txt
    ) else (
        echo ERROR: Failed to install requirements.
        exit /b 1
    )
) else (
    echo Requirements already installed.
)

REM Check if TensorFlow needs patching
python -c "import tensorflow" >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo TensorFlow not found, skipping compatibility check.
) else (
    echo Checking TensorFlow compatibility...
    python ..\..\src\utils\patch_tensorflow.py --check
    if %ERRORLEVEL% NEQ 0 (
        echo Applying TensorFlow patch...
        python ..\..\src\utils\patch_tensorflow.py --apply
    )
)

REM Check if Ollama is running
call :check_ollama

REM Setup GPU environment
call :setup_gpu

REM ======== Training =========

echo.
echo === Starting Adaptive Agent Training ===
echo.
echo IMPORTANT: Make sure Cities: Skylines 2 is running and visible on your screen.
echo The training will begin in 5 seconds.
timeout /t 5

REM Run the training with error handling
set CMD=python ..\..\training\train_adaptive.py --timesteps %TIMESTEPS% %FOCUS_FLAG% --starting-mode %STARTING_MODE% %LOAD_FLAG%
call :error_handler "%CMD%" 3

REM Set high priority for the Python process
call :set_high_priority "python.exe"

echo.
if %ERRORLEVEL%==0 (
    echo Training completed successfully!
) else (
    echo Training completed with errors. Check error_log.txt for details.
)

REM Clean up temporary files
del /q temp_*.txt 2>nul

echo.
echo ======================================================
echo All-in-One Setup and Training completed
echo ======================================================

pause
endlocal
exit /b 0 