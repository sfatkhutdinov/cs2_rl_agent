@echo off
setlocal EnableDelayedExpansion

echo ======================================================
echo Strategic Agent Training for Cities: Skylines 2
echo ======================================================

REM Set working directory to script location
cd /d "%~dp0"

REM Include common functions library
call %~dp0..\utils\common_functions.bat

REM Check for help command
if /I "%~1"=="help" (
    echo Usage: train_strategic_agent.bat [timesteps] [bootstrap] [use_adaptive] [load_checkpoint]
    echo.
    echo Parameters:
    echo   timesteps      - Number of training timesteps (default: 5000000)
    echo   bootstrap      - Whether to use knowledge bootstrapping (default: true)
    echo                    Valid options: true, false
    echo   use_adaptive   - Whether to use the adaptive agent wrapper (default: false)
    echo                    Valid options: true, false
    echo   load_checkpoint - Path to load a checkpoint from (optional)
    echo.
    echo Examples:
    echo   train_strategic_agent.bat
    echo   train_strategic_agent.bat 10000000
    echo   train_strategic_agent.bat 5000000 false
    echo   train_strategic_agent.bat 10000000 true true
    echo   train_strategic_agent.bat 5000000 true false "models/strategic/checkpoint_500000"
    echo.
    exit /b 0
)

REM Process command line arguments
set TIMESTEPS=5000000
set BOOTSTRAP=true
set USE_ADAPTIVE=false
set CHECKPOINT=

if not "%~1"=="" set TIMESTEPS=%~1
if not "%~2"=="" set BOOTSTRAP=%~2
if not "%~3"=="" set USE_ADAPTIVE=%~3
if not "%~4"=="" set CHECKPOINT=%~4

echo Strategic Agent Configuration:
echo - Training timesteps: %TIMESTEPS%
echo - Knowledge bootstrapping: %BOOTSTRAP%
echo - Using adaptive agent: %USE_ADAPTIVE%
if not "%CHECKPOINT%"=="" echo - Loading checkpoint: %CHECKPOINT%

REM ======== STEP 1: Check Python Environment ========
echo.
echo Checking Python environment...

REM Activate conda environment (only if needed)
call :activate_conda

REM ======== STEP 2: Check Ollama ========
echo.
echo Checking if Ollama is running...
call :check_ollama

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

echo Checking if required model is available...
curl -s http://localhost:11434/api/tags | findstr "granite3.2-vision" > nul
if errorlevel 1 (
    echo Downloading Granite 3.2 Vision model...
    set PULL_CMD=curl -X POST http://localhost:11434/api/pull -d "{\"name\":\"granite3.2-vision:latest\"}"
    call :error_handler "%PULL_CMD%" 3
) else (
    echo Granite vision model already installed.
)

REM ======== STEP 3: Create Required Directories ========
echo.
echo Creating required directories...
if not exist logs mkdir logs
if not exist models mkdir models
if not exist models\strategic mkdir models\strategic
if not exist tensorboard mkdir tensorboard
if not exist tensorboard\strategic mkdir tensorboard\strategic

REM ======== STEP 4: Start Training ========
echo.
echo IMPORTANT: Please make sure Cities: Skylines 2 is running and visible on your screen.
echo The script will proceed to training in 5 seconds.
echo.
timeout /t 5

REM Build command with appropriate flags
set TRAIN_CMD=python train_strategic.py --timesteps %TIMESTEPS%

if /I "%BOOTSTRAP%"=="true" (
    set TRAIN_CMD=!TRAIN_CMD! --knowledge-bootstrap
)

if /I "%USE_ADAPTIVE%"=="true" (
    set TRAIN_CMD=!TRAIN_CMD! --use-adaptive
)

if not "%CHECKPOINT%"=="" (
    set TRAIN_CMD=!TRAIN_CMD! --load-checkpoint "%CHECKPOINT%"
)

REM Run the training with error handling
echo Starting strategic agent training...
echo Running command: !TRAIN_CMD!
call :error_handler "!TRAIN_CMD!" 3

REM Set high priority for Python processes
call :set_high_priority "python.exe"

REM Clean up temporary files
call :cleanup_temp "temp_strategic_*.txt"

echo.
if %ERRORLEVEL% equ 0 (
    echo Strategic agent training completed successfully!
) else (
    echo Strategic agent training encountered an error.
    echo Please check error_log.txt for more information.
)

pause
endlocal 