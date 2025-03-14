@echo off
setlocal EnableDelayedExpansion

echo Running Discovery Agent for CS2 RL Agent
echo ----------------------------------------

REM Set working directory to script location
cd /d "%~dp0"

REM Include common functions library
call ..\utils\common_functions.bat

REM Setup environment variables
set PYTHONPATH=%~dp0..\..
set CONFIG_FILE=config/discovery_config.yaml
set TIMESTEPS=1000
set MODE=goal

REM Process command line arguments
if not "%~1"=="" set TIMESTEPS=%~1
if not "%~2"=="" set MODE=%~2
echo Training for %TIMESTEPS% timesteps in %MODE% mode

REM Activate conda environment (only if needed)
call :activate_conda

REM Check if dependencies are installed (using cached results)
call :check_dependencies "requirements.txt"

REM Setup GPU environment
call :setup_gpu

REM Detect GPU capabilities with caching
call :detect_gpu

if "%GPU_AVAILABLE%"=="1" (
    echo NVIDIA GPU detected with CUDA version %CUDA_VERSION%
    echo Running in GPU mode
) else (
    echo WARNING: No GPU detected or CUDA drivers not properly installed
    echo Running in CPU mode (slower performance)
    echo For TensorFlow/PyTorch GPU support, you need CUDA Toolkit 11.8 and cuDNN
    echo You can download drivers from: https://www.nvidia.com/Download/index.aspx
)

REM Check if Ollama is running (for vision capabilities)
call :check_ollama

echo.
echo IMPORTANT: Make sure Cities: Skylines 2 is running and visible on your screen.
echo The agent will begin in 5 seconds.
timeout /t 5

REM Run the discovery agent with error handling
set AGENT_CMD=python ..\..\training\train_discovery.py --mode %MODE% --timesteps %TIMESTEPS% --config %CONFIG_FILE% --deployment
call :error_handler "%AGENT_CMD%" 3

REM Set high priority for Python processes
call :set_high_priority "python.exe"

echo.
if %ERRORLEVEL%==0 (
    echo Discovery agent completed successfully!
) else (
    echo Discovery agent encountered errors. Check error_log.txt for details.
)

REM Clean up temporary files
call :cleanup_temp "temp_discovery_*.txt"

pause
endlocal 