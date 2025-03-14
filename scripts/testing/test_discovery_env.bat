@echo off
setlocal EnableDelayedExpansion

echo ======================================================
echo DiscoveryEnvironment Test Runner
echo ======================================================

REM Set working directory to script location
cd /d "%~dp0"

REM Include common functions library
call %~dp0..\utils\common_functions.bat

REM Check for help command
if /I "%~1"=="help" (
    echo Usage: test_discovery_env.bat [verbose] [focus]
    echo.
    echo Parameters:
    echo   verbose - Enable verbose output (default: false)
    echo             Valid options: true, false
    echo   focus   - Focus game window during test (default: false)
    echo             Valid options: true, false
    echo.
    echo Examples:
    echo   test_discovery_env.bat
    echo   test_discovery_env.bat true
    echo   test_discovery_env.bat true true
    echo.
    exit /b 0
)

REM Process command line arguments
set VERBOSE=false
set FOCUS=false

if not "%~1"=="" set VERBOSE=%~1
if not "%~2"=="" set FOCUS=%~2

echo DiscoveryEnvironment Test Configuration:
echo - Verbose output: %VERBOSE%
echo - Focus game window: %FOCUS%

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
curl -s http://localhost:11434/api/tags | findstr "granite3.2-vision" > nul
if errorlevel 1 (
    echo Granite 3.2 Vision model not found. Installing...
    set PULL_CMD=curl -X POST http://localhost:11434/api/pull -d "{\"name\":\"granite3.2-vision:latest\"}"
    call :error_handler "%PULL_CMD%" 3
) else (
    echo Granite vision model already installed.
)

REM ======== STEP 3: Create Test Environment ========
echo.
echo Creating test environment...

if not exist logs mkdir logs
if not exist logs\tests mkdir logs\tests
if not exist logs\tests\discovery mkdir logs\tests\discovery

REM ======== STEP 4: Run Test ========
echo.
echo Running DiscoveryEnvironment import test...

REM Build test command with appropriate flags
set TEST_CMD=python test_discovery_env.py

if /I "%VERBOSE%"=="true" (
    set TEST_CMD=!TEST_CMD! --verbose
)

if /I "%FOCUS%"=="true" (
    set TEST_CMD=!TEST_CMD! --focus
)

REM Run the test with error handling
echo Running command: !TEST_CMD!
call :error_handler "!TEST_CMD!" 2

echo.
if %ERRORLEVEL% equ 0 (
    echo DiscoveryEnvironment test passed successfully!
) else (
    echo DiscoveryEnvironment test failed.
    echo Please check logs\tests\discovery\test_log.txt for more details.
)

pause
endlocal 