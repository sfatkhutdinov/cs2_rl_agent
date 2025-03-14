@echo off
setlocal EnableDelayedExpansion

echo ======================================================
echo Ollama Vision Models Setup for CS2 RL Agent
echo ======================================================

REM Set working directory to script location
cd /d "%~dp0"

REM Include common functions library
call %~dp0common_functions.bat

REM Check for help command
if /I "%~1"=="help" (
    echo Usage: setup_ollama.bat [model]
    echo.
    echo Parameters:
    echo   model - Vision model to install (default: granite3.2-vision)
    echo           Valid options: granite3.2-vision, llava:7b-v1.6-vision, llama3.2-vision
    echo.
    echo Examples:
    echo   setup_ollama.bat
    echo   setup_ollama.bat granite3.2-vision
    echo   setup_ollama.bat llava:7b-v1.6-vision
    echo.
    exit /b 0
)

REM Process command line arguments
set MODEL=granite3.2-vision
if not "%~1"=="" set MODEL=%~1

echo Ollama Setup Configuration:
echo - Vision model to install: %MODEL%

REM ======== STEP 1: Check if Ollama is Running ========
echo.
echo Checking if Ollama is running...
call :check_ollama

REM ======== STEP 2: Check if Model is Installed ========
echo.
echo Checking if %MODEL% model is installed...
curl -s http://localhost:11434/api/tags | findstr "%MODEL%" > nul
if errorlevel 1 (
    echo %MODEL% model not found. Installing...
    set PULL_CMD=curl -X POST http://localhost:11434/api/pull -d "{\"name\":\"%MODEL%\"}"
    call :error_handler "%PULL_CMD%" 3
    
    if %ERRORLEVEL% equ 0 (
        echo %MODEL% model installed successfully.
    ) else (
        echo ERROR: Failed to install %MODEL% model.
        pause
        exit /b 1
    )
) else (
    echo %MODEL% model is already installed.
)

REM ======== STEP 3: Warm Up Model ========
echo.
echo Warming up the %MODEL% model with a test query...
set WARMUP_CMD=curl -s -X POST http://localhost:11434/api/generate -d "{\"model\":\"%MODEL%\",\"prompt\":\"Hello, are you ready to analyze game screens?\",\"stream\":false}"
call :error_handler "%WARMUP_CMD%" 2

REM ======== STEP 4: Verify Model Works ========
echo.
echo Testing model functionality...
set TEST_CMD=curl -s -X POST http://localhost:11434/api/generate -d "{\"model\":\"%MODEL%\",\"prompt\":\"Confirm that you can process text inputs correctly.\",\"stream\":false}"
call :error_handler "%TEST_CMD%" 2

REM ======== STEP 5: Final Status ========
echo.
echo Ollama setup completed successfully.
echo The %MODEL% model is ready for use.
echo.
echo Available models:
curl -s http://localhost:11434/api/tags | findstr "\"name\"" | findstr "vision"
echo.
echo Next steps:
echo 1. Make sure Cities: Skylines 2 is running and visible
echo 2. Run one of the training scripts (e.g., train_discovery_with_focus.bat)
echo.
pause
endlocal 