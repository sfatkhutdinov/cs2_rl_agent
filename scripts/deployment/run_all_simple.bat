@echo off
setlocal EnableDelayedExpansion

echo ======================================================
echo Simplified CS2 Agent Training with GPU Support
echo ======================================================

REM Set working directory to script location
cd /d "%~dp0"

REM Include common functions library
call %~dp0..\utils\common_functions.bat

REM Check for help command
if /I "%~1"=="help" (
    echo Usage: run_all_simple.bat [timesteps] [mode]
    echo.
    echo Parameters:
    echo   timesteps - Number of training timesteps (default: 1000)
    echo   mode      - Training mode (default: goal)
    echo              Valid options: goal, explore
    echo.
    echo Examples:
    echo   run_all_simple.bat
    echo   run_all_simple.bat 5000
    echo   run_all_simple.bat 2000 explore
    echo.
    exit /b 0
)

REM Process command line arguments
set TIMESTEPS=1000
set MODE=goal

if not "%~1"=="" set TIMESTEPS=%~1
if not "%~2"=="" set MODE=%~2

echo Simple Agent Configuration:
echo - Training timesteps: %TIMESTEPS%
echo - Training mode: %MODE%

REM ======== STEP 1: Check Python Environment ========
echo.
echo Checking Python environment...

REM Setup configuration path
set CONFIG_FILE=config/discovery_config.yaml
set PYTHONPATH=%~dp0

REM Activate conda environment (only if needed)
call :activate_conda

REM ======== STEP 2: Create Required Directories ========
echo.
echo Creating required directories...

REM Create logs directory if it doesn't exist
mkdir logs 2>nul
mkdir logs\vision_debug 2>nul
mkdir models 2>nul
mkdir models\discovery 2>nul
mkdir tensorboard 2>nul
mkdir tensorboard\discovery 2>nul

REM ======== STEP 3: Install Essential Dependencies ========
echo.
echo Checking dependencies...

REM Check dependencies with caching
call :check_dependencies "requirements.txt"

REM ======== STEP 4: Setup GPU Environment ========
echo.
echo Setting up GPU environment...

REM Setup GPU environment variables
call :setup_gpu

REM Check for GPU with caching
call :detect_gpu

if "%GPU_AVAILABLE%"=="1" (
    echo NVIDIA GPU detected with CUDA version %CUDA_VERSION%
    echo Training will use GPU acceleration.
    
    REM Simple GPU setup
    set GPU_CMD=python setup_gpu.py
    call :error_handler "%GPU_CMD%" 2
) else (
    echo WARNING: No GPU detected or CUDA drivers not properly installed.
    echo Training will proceed with CPU only (slower).
    echo For GPU support, install NVIDIA drivers from: https://www.nvidia.com/Download/index.aspx
)

REM ======== STEP 5: Check Ollama ========
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

REM ======== STEP 6: Focus Game Window ========
echo.
echo Focusing game window...

REM Attempt to focus the game window
echo Attempting to focus game window...
python enhanced_focus.py "Cities: Skylines II"
if %ERRORLEVEL% NEQ 0 (
    echo Falling back to simpler focus method...
    powershell -command "(New-Object -ComObject WScript.Shell).AppActivate('Cities: Skylines II')"
)

echo Waiting for the game to stabilize...
echo TIP: If agent button presses don't register, try clicking manually in the game window once.
timeout /t 5 > nul

REM ======== STEP 7: Start Training ========
echo.
echo Starting agent training...

REM Build command with appropriate flags
if /i "%MODE%"=="explore" (
    echo Running in exploration-focused mode
    set TRAIN_CMD=python train_discovery.py --config %CONFIG_FILE% --timesteps %TIMESTEPS% --exploration-focus
) else (
    echo Running in city-building goal-oriented mode
    set TRAIN_CMD=python train_discovery.py --config %CONFIG_FILE% --timesteps %TIMESTEPS% --goal-focus
)

REM Run the training with error handling
echo Running command: !TRAIN_CMD!
call :error_handler "!TRAIN_CMD!" 3

REM Set high priority for Python processes
call :set_high_priority "python.exe"

REM Clean up temporary files
call :cleanup_temp "temp_discovery_*.txt"

echo.
if %ERRORLEVEL% equ 0 (
    echo Training completed successfully!
) else (
    echo Training encountered an error.
    echo Please check error_log.txt for more information.
)

echo ======================================================
echo Training complete! Press any key to exit...
echo ======================================================
pause
endlocal 