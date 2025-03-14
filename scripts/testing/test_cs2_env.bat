@echo off
setlocal EnableDelayedExpansion

echo ======================================================
echo CS2Environment Test Runner
echo ======================================================

REM Set working directory to script location
cd /d "%~dp0"

REM Include common functions library
call %~dp0..\utils\common_functions.bat

REM Check for help command
if /I "%~1"=="help" (
    echo Usage: test_cs2_env.bat [verbose]
    echo.
    echo Parameters:
    echo   verbose - Enable verbose output (default: false)
    echo             Valid options: true, false
    echo.
    echo Examples:
    echo   test_cs2_env.bat
    echo   test_cs2_env.bat true
    echo.
    exit /b 0
)

REM Process command line arguments
set VERBOSE=false
if not "%~1"=="" set VERBOSE=%~1

echo Test Configuration:
echo - Verbose output: %VERBOSE%

REM ======== STEP 1: Check Python Environment ========
echo.
echo Checking Python environment...

REM Activate conda environment (only if needed)
call :activate_conda

REM ======== STEP 2: Create Test Environment ========
echo.
echo Creating test environment...

if not exist ..\..\testing mkdir ..\..\testing
if not exist logs mkdir logs
if not exist logs\tests mkdir logs\tests

REM ======== STEP 3: Run Test ========
echo.
echo Testing CS2Environment...

REM Build test command with appropriate flags
set TEST_CMD=python ..\..\testing\test_cs2_env.py

if /I "%VERBOSE%"=="true" (
    set TEST_CMD=!TEST_CMD! --verbose
)

REM Run the test with error handling
echo Running command: !TEST_CMD!
call :error_handler "!TEST_CMD!" 2

echo.
if %ERRORLEVEL% equ 0 (
    echo CS2Environment test passed successfully!
) else (
    echo CS2Environment test failed.
    echo Please check logs\tests\cs2_env_test_log.txt for more details.
)

pause
endlocal 