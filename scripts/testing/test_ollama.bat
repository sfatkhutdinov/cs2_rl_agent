@echo off
setlocal EnableDelayedExpansion

echo ======================================================
echo Ollama Integration Test Runner
echo ======================================================

REM Set working directory to script location
cd /d "%~dp0"

REM Include common functions library
call %~dp0..\utils\common_functions.bat

REM Check for help command
if /I "%~1"=="help" (
    echo Usage: test_ollama.bat [model] [verbose]
    echo.
    echo Parameters:
    echo   model   - Model to test (default: granite3.2-vision)
    echo             Valid options: granite3.2-vision, llava:7b-v1.6-vision, llama3.2-vision
    echo   verbose - Enable verbose output (default: false)
    echo             Valid options: true, false
    echo.
    echo Examples:
    echo   test_ollama.bat
    echo   test_ollama.bat granite3.2-vision
    echo   test_ollama.bat llava:7b-v1.6-vision true
    echo.
    exit /b 0
)

REM Process command line arguments
set MODEL=granite3.2-vision
set VERBOSE=false

if not "%~1"=="" set MODEL=%~1
if not "%~2"=="" set VERBOSE=%~2

echo Ollama Test Configuration:
echo - Model to test: %MODEL%
echo - Verbose output: %VERBOSE%

REM ======== STEP 1: Check Python Environment ========
echo.
echo Checking Python environment...

REM Activate conda environment (only if needed)
call :activate_conda

REM ======== STEP 2: Check Ollama Service ========
echo.
echo Checking if Ollama is running...
call :check_ollama

echo Checking if required model is available...
curl -s http://localhost:11434/api/tags | findstr "%MODEL%" > nul
if errorlevel 1 (
    echo %MODEL% model not found. Installing...
    set PULL_CMD=curl -X POST http://localhost:11434/api/pull -d "{\"name\":\"%MODEL%\"}"
    call :error_handler "%PULL_CMD%" 3
) else (
    echo %MODEL% model already installed.
)

REM ======== STEP 3: Create Test Environment ========
echo.
echo Creating test environment...

if not exist logs mkdir logs
if not exist logs\tests mkdir logs\tests
if not exist logs\tests\ollama mkdir logs\tests\ollama

REM ======== STEP 4: Run Test ========
echo.
echo Running Ollama integration test...

REM Build test command with appropriate flags
set TEST_CMD=python test_ollama.py --model %MODEL%

if /I "%VERBOSE%"=="true" (
    set TEST_CMD=!TEST_CMD! --verbose
)

REM Run the test with error handling
echo Running command: !TEST_CMD!
call :error_handler "!TEST_CMD!" 2

echo.
if %ERRORLEVEL% equ 0 (
    echo Ollama integration test completed successfully!
) else (
    echo Ollama integration test failed.
    echo Please check logs\tests\ollama\test_log.txt for more details.
)

pause
endlocal 