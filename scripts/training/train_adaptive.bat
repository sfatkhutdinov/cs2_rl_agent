@echo off
setlocal EnableDelayedExpansion

echo Training Adaptive CS2 RL Agent
echo --------------------------------

REM Set Python path to include the project root
set PYTHONPATH=%~dp0..\..

REM Include common functions library
call %~dp0..\utils\common_functions.bat

REM Activate conda environment (only if needed)
call :activate_conda

REM Check if Ollama is running
call :check_ollama

REM Setup GPU environment
call :setup_gpu

REM Process command line arguments
set TIMESTEPS=10000
set FOCUS=--focus
set STARTING_MODE=discovery
set EXTRA_ARGS=

if not "%~1"=="" set TIMESTEPS=%~1
if not "%~2"=="" set STARTING_MODE=%~2
if /I "%~3"=="nofocus" set FOCUS=

echo Will train for %TIMESTEPS% timesteps starting with %STARTING_MODE% mode

echo.
echo IMPORTANT: Make sure Cities: Skylines 2 is running and visible on your screen.
echo The training will begin in 5 seconds.
timeout /t 5

REM Run the adaptive training script with error handling
set TRAINING_CMD=python ..\..\training\train_adaptive.py --timesteps %TIMESTEPS% %FOCUS% --starting-mode %STARTING_MODE% %EXTRA_ARGS%
call :error_handler "%TRAINING_CMD%" 3

REM Set high priority for the Python process
call :set_high_priority "python.exe" 

echo.
if %ERRORLEVEL%==0 (
    echo Training completed successfully!
) else (
    echo Training completed with errors. Check error_log.txt for details.
)

REM Clean up temporary files
call :cleanup_temp "temp_adaptive_*.txt"

pause
endlocal 