@echo off
setlocal

echo ======================================================
echo Testing Adaptive Agent Modes
echo ======================================================

REM Check Python environment
call check_conda.bat
if %ERRORLEVEL% neq 0 (
    echo Error: Conda environment not properly set up
    exit /b 1
)

REM Set PYTHONPATH to include the current directory
set PYTHONPATH=%CD%;%PYTHONPATH%

echo Running adaptive mode test...
python test_adaptive_modes.py

if %ERRORLEVEL% equ 0 (
    echo.
    echo ======================================================
    echo Test Successful! Adaptive agent can access all modes.
    echo You can safely run the adaptive agent with:
    echo .\all_in_one_setup_and_train.bat [timesteps] adaptive
    echo ======================================================
) else (
    echo.
    echo ======================================================
    echo Test Failed! Some modes could not be initialized.
    echo Please check the logs for more information.
    echo ======================================================
)

pause
endlocal 