@echo off
setlocal EnableDelayedExpansion

echo ======================================================
echo Discovery Agent Training with Window Focus
echo ======================================================

REM Set working directory to script location
cd /d "%~dp0"

REM Include common functions library
call %~dp0..\utils\common_functions.bat

REM Check for help command
if /I "%~1"=="help" (
    echo Usage: train_discovery_with_focus.bat [timesteps]
    echo.
    echo Parameters:
    echo   timesteps - Number of training timesteps (default: 2000000)
    echo.
    echo Examples:
    echo   train_discovery_with_focus.bat
    echo   train_discovery_with_focus.bat 5000000
    echo.
    exit /b 0
)

REM Process command line arguments
set TIMESTEPS=2000000
if not "%~1"=="" set TIMESTEPS=%~1

echo Discovery Agent Configuration:
echo - Training timesteps: %TIMESTEPS%
echo - Auto-focus game window: Enabled

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

REM ======== STEP 3: Create Required Directories ========
echo.
echo Creating required directories...
if not exist logs mkdir logs
if not exist models mkdir models
if not exist models\discovery mkdir models\discovery
if not exist tensorboard mkdir tensorboard
if not exist tensorboard\discovery mkdir tensorboard\discovery

REM ======== STEP 4: Check Directory Structure ========
echo.
echo Checking directory structure...
set DIR_CHECK_CMD=python check_directories.py
call :error_handler "%DIR_CHECK_CMD%" 2

REM ======== STEP 5: Start Training ========
echo.
echo IMPORTANT: Please make sure Cities: Skylines 2 is running and visible on your screen.
echo The script will focus on the game window in 5 seconds.
echo.
timeout /t 5

REM Run the training with error handling
echo Starting discovery-based training with window focus...
set TRAIN_CMD=python -m src.train --config config/discovery_config.yaml --timesteps %TIMESTEPS% --focus
call :error_handler "%TRAIN_CMD%" 3

REM Set high priority for Python processes
call :set_high_priority "python.exe"

REM Clean up temporary files
call :cleanup_temp "temp_discovery_*.txt"

echo.
if %ERRORLEVEL% equ 0 (
    echo Discovery agent training completed successfully!
) else (
    echo Discovery agent training encountered an error.
    echo Please check error_log.txt for more information.
)

pause
endlocal 