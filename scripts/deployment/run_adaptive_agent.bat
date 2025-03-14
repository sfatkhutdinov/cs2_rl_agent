@echo off
setlocal EnableDelayedExpansion

echo Adaptive CS2 RL Agent - All-in-One Deployment
echo ----------------------------------------------

REM Set Python path to include the project root
set PYTHONPATH=%~dp0..\..

REM Include common functions library (assuming it exists)
if exist %~dp0..\utils\common_functions.bat (
    call %~dp0..\utils\common_functions.bat
) else (
    echo ERROR: Common functions library not found!
    exit /b 1
)

REM Activate conda environment (only if needed)
call :activate_conda

REM Check if Ollama is running
call :check_ollama

REM Setup GPU environment
call :setup_gpu

REM Process command line arguments
set TIMESTEPS=20000
set STARTING_MODE=discovery
set FOCUS=--focus
set CONFIG=config/adaptive_config.yaml
set LOAD_PATH=

:parse_args
if "%~1"=="" goto :end_parse_args
if /I "%~1"=="--timesteps" (
    set TIMESTEPS=%~2
    shift
    shift
    goto :parse_args
)
if /I "%~1"=="--mode" (
    set STARTING_MODE=%~2
    shift
    shift
    goto :parse_args
)
if /I "%~1"=="--nofocus" (
    set FOCUS=
    shift
    goto :parse_args
)
if /I "%~1"=="--config" (
    set CONFIG=%~2
    shift
    shift
    goto :parse_args
)
if /I "%~1"=="--load" (
    set LOAD_PATH=%~2
    shift
    shift
    goto :parse_args
)
shift
goto :parse_args
:end_parse_args

echo.
echo === Adaptive Agent Configuration ===
echo Timesteps: %TIMESTEPS%
echo Starting Mode: %STARTING_MODE%
echo Auto-focus: %FOCUS%
echo Config: %CONFIG%
if not "%LOAD_PATH%"=="" (
    echo Loading from: %LOAD_PATH%
)

echo.
echo IMPORTANT: Make sure Cities: Skylines 2 is running and visible on your screen.
echo The AI agent will begin in 5 seconds.
timeout /t 5

REM Run the adaptive agent training script with error handling
set CMD=python ..\..\training\train_adaptive.py --config %CONFIG% --timesteps %TIMESTEPS% %FOCUS% --starting-mode %STARTING_MODE%
if not "%LOAD_PATH%"=="" (
    set CMD=%CMD% --load "%LOAD_PATH%"
)

call :error_handler "%CMD%" 3

REM Set high priority for the Python process
call :set_high_priority "python.exe" 

echo.
if %ERRORLEVEL%==0 (
    echo Agent operation completed successfully!
) else (
    echo Agent operation completed with errors. Check error_log.txt for details.
)

REM Clean up temporary files
call :cleanup_temp "temp_adaptive_*.txt"

pause
endlocal
exit /b 0 