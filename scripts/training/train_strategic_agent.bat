@echo off
setlocal EnableDelayedExpansion

echo ======================================================
echo Strategic Agent Training for Cities: Skylines 2
echo ======================================================

REM Set working directory to script location
cd /d "%~dp0"

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
call check_conda.bat
if %ERRORLEVEL% neq 0 (
    echo ERROR: Conda environment not properly set up
    echo Please run all_in_one_setup_and_train.bat first to set up the environment
    pause
    exit /b 1
)

REM ======== STEP 2: Check Ollama ========
echo.
echo Checking if Ollama is running...
curl -s http://localhost:11434/api/tags > nul
if errorlevel 1 (
    echo ERROR: Ollama is not running. Please start Ollama first.
    echo You can download Ollama from: https://ollama.ai/
    echo After installation, run: ollama serve
    pause
    exit /b 1
)

echo Checking if required model is available...
curl -s http://localhost:11434/api/tags | findstr "granite3.2-vision" > nul
if errorlevel 1 (
    echo Downloading Granite 3.2 Vision model...
    curl -X POST http://localhost:11434/api/pull -d "{\"name\":\"granite3.2-vision:latest\"}"
    if errorlevel 1 (
        echo ERROR: Failed to download vision model.
        pause
        exit /b 1
    )
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
set CMD=python train_strategic.py --timesteps %TIMESTEPS%

if /I "%BOOTSTRAP%"=="true" (
    set CMD=!CMD! --knowledge-bootstrap
)

if /I "%USE_ADAPTIVE%"=="true" (
    set CMD=!CMD! --use-adaptive
)

if not "%CHECKPOINT%"=="" (
    set CMD=!CMD! --load-checkpoint "%CHECKPOINT%"
)

REM Run the training
echo Starting strategic agent training...
echo Running command: !CMD!
!CMD!

echo.
if %ERRORLEVEL% equ 0 (
    echo Strategic agent training completed successfully!
) else (
    echo Strategic agent training encountered an error.
    echo Please check the logs for more information.
)

pause
endlocal 