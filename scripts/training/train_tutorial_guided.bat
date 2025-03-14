@echo off
setlocal EnableDelayedExpansion

echo ======================================================
echo Tutorial-Guided RL Agent Training for CS2
echo ======================================================

REM Set working directory to script location
cd /d "%~dp0"

REM Include common functions library
call %~dp0..\utils\common_functions.bat

REM Check for help command
if /I "%~1"=="help" (
    echo Usage: train_tutorial_guided.bat [timesteps] [focus]
    echo.
    echo Parameters:
    echo   timesteps - Number of training timesteps (default: 1000000)
    echo   focus     - Whether to auto-focus the game window (default: true)
    echo              Valid options: true, false
    echo.
    echo Examples:
    echo   train_tutorial_guided.bat
    echo   train_tutorial_guided.bat 2000000
    echo   train_tutorial_guided.bat 500000 false
    echo.
    exit /b 0
)

REM Process command line arguments
set TIMESTEPS=1000000
set FOCUS=true

if not "%~1"=="" set TIMESTEPS=%~1
if not "%~2"=="" set FOCUS=%~2

echo Tutorial-Guided Agent Configuration:
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
curl -s http://localhost:11434/api/tags | findstr "llava:7b-v1.6-vision" > nul
if errorlevel 1 (
    echo Downloading Llava vision model...
    set PULL_CMD=curl -X POST http://localhost:11434/api/pull -d "{\"name\":\"llava:7b-v1.6-vision\"}"
    call :error_handler "%PULL_CMD%" 3
) else (
    echo Llava vision model already installed.
)

REM Warm up the vision model
echo Warming up vision model...
curl -s -X POST http://localhost:11434/api/generate -d "{\"model\":\"llava:7b-v1.6-vision\",\"prompt\":\"Hello, are you ready to analyze game screens?\",\"stream\":false}" > nul

REM ======== STEP 3: Create Required Directories ========
echo.
echo Creating required directories...
if not exist logs mkdir logs
if not exist models mkdir models
if not exist models\tutorial mkdir models\tutorial
if not exist tensorboard mkdir tensorboard
if not exist tensorboard\tutorial mkdir tensorboard\tutorial

REM ======== STEP 4: Start Training ========
echo.
echo IMPORTANT: Please make sure Cities: Skylines 2 is running and visible on your screen.
echo The script will proceed to training in 5 seconds.
echo.
timeout /t 5

REM Build command with appropriate flags
set TRAIN_CMD=python train_tutorial_guided.py --config config/tutorial_guided_config.yaml --timesteps %TIMESTEPS%

if /I "%FOCUS%"=="true" (
    set TRAIN_CMD=!TRAIN_CMD! --focus
)

REM Run the training with error handling
echo Starting tutorial-guided training...
echo Running command: !TRAIN_CMD!
call :error_handler "!TRAIN_CMD!" 3

REM Set high priority for Python processes
call :set_high_priority "python.exe"

REM Clean up temporary files
call :cleanup_temp "temp_tutorial_*.txt"

echo.
if %ERRORLEVEL% equ 0 (
    echo Tutorial-guided agent training completed successfully!
) else (
    echo Tutorial-guided agent training encountered an error.
    echo Please check error_log.txt for more information.
)

pause
endlocal 