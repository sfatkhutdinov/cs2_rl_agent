@echo off
setlocal EnableDelayedExpansion

echo ======================================================
echo Vision-Guided RL Agent Training for CS2
echo ======================================================

REM Set working directory to script location
cd /d "%~dp0"

REM Include common functions library
call %~dp0..\utils\common_functions.bat

REM Check for help command
if /I "%~1"=="help" (
    echo Usage: train_vision_guided.bat [timesteps] [focus]
    echo.
    echo Parameters:
    echo   timesteps - Number of training timesteps (default: 1000000)
    echo   focus     - Whether to auto-focus the game window (default: true)
    echo              Valid options: true, false
    echo.
    echo Examples:
    echo   train_vision_guided.bat
    echo   train_vision_guided.bat 2000000
    echo   train_vision_guided.bat 500000 false
    echo.
    exit /b 0
)

REM Process command line arguments
set TIMESTEPS=1000000
set FOCUS=true

if not "%~1"=="" set TIMESTEPS=%~1
if not "%~2"=="" set FOCUS=%~2

echo Vision-Guided Agent Configuration:
echo - Training timesteps: %TIMESTEPS%
echo - Auto-focus game window: %FOCUS%

REM ======== STEP 1: Check Python Environment ========
echo.
echo Checking Python environment...

REM Activate conda environment (only if needed)
call :activate_conda

REM ======== STEP 2: Check Ollama ========
echo.
echo Checking if Ollama is running...
call :check_ollama

echo Checking if required model is available...
curl -s http://localhost:11434/api/tags | findstr "granite3.2-vision" > nul
if errorlevel 1 (
    echo Downloading Granite 3.2 Vision model...
    set PULL_CMD=curl -X POST http://localhost:11434/api/pull -d "{\"name\":\"granite3.2-vision:latest\"}"
    call :error_handler "%PULL_CMD%" 3
) else (
    echo Granite vision model already installed.
)

REM Warm up the vision model
echo Warming up vision model...
curl -s -X POST http://localhost:11434/api/generate -d "{\"model\":\"granite3.2-vision:latest\",\"prompt\":\"Hello, are you ready to analyze game screens?\",\"stream\":false}" > nul

REM Setup GPU environment
call :setup_gpu

REM Check for GPU with caching
call :detect_gpu

if "%GPU_AVAILABLE%"=="1" (
    echo NVIDIA GPU detected with CUDA version %CUDA_VERSION%
    echo Training will use GPU acceleration.
) else (
    echo WARNING: No GPU detected or CUDA drivers not properly installed.
    echo Training will proceed with CPU only (slower).
    echo For GPU support, install NVIDIA drivers from: https://www.nvidia.com/Download/index.aspx
)

REM ======== STEP 3: Create Required Directories ========
echo.
echo Creating required directories...
if not exist logs mkdir logs
if not exist models mkdir models
if not exist models\vision_guided mkdir models\vision_guided
if not exist tensorboard mkdir tensorboard
if not exist tensorboard\vision_guided mkdir tensorboard\vision_guided

REM ======== STEP 4: Start Training ========
echo.
echo IMPORTANT: Please make sure Cities: Skylines 2 is running and visible on your screen.
echo The script will proceed to training in 5 seconds.
echo.
timeout /t 5

REM Build command with appropriate flags
set TRAIN_CMD=python train_vision_guided.py --config config/vision_guided_config.yaml --timesteps %TIMESTEPS%

if /I "%FOCUS%"=="true" (
    set TRAIN_CMD=!TRAIN_CMD! --focus
)

REM Run the training with error handling
echo Starting vision-guided training...
echo Running command: !TRAIN_CMD!
call :error_handler "!TRAIN_CMD!" 3

REM Set high priority for Python processes
call :set_high_priority "python.exe"

REM Clean up temporary files
call :cleanup_temp "temp_vision_*.txt"

echo.
if %ERRORLEVEL% equ 0 (
    echo Vision-guided agent training completed successfully!
) else (
    echo Vision-guided agent training encountered an error.
    echo Please check error_log.txt for more information.
)

pause
endlocal 