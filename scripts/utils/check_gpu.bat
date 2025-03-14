@echo off
setlocal EnableDelayedExpansion

echo Running GPU Checker to diagnose GPU availability for ML...
echo.

REM Include common functions library
call %~dp0common_functions.bat

REM Activate conda environment (only if needed)
call :activate_conda

REM Setup GPU environment variables
call :setup_gpu

REM Detect GPU with caching
call :detect_gpu

if "%GPU_AVAILABLE%"=="1" (
    echo [SUCCESS] NVIDIA GPU detected with CUDA version %CUDA_VERSION%
    
    REM Run the Python GPU check script for more detailed information
    set GPU_CHECK_CMD=python ..\..\utils\check_gpu.py
    
    REM Run with error handling
    call :error_handler "%GPU_CHECK_CMD%" 2
) else (
    echo [WARNING] No compatible GPU detected for machine learning tasks.
    echo Check your NVIDIA drivers or GPU compatibility.
    echo.
    echo You can download drivers from: https://www.nvidia.com/Download/index.aspx
)

echo.
echo GPU check complete. Press any key to exit...
pause > nul 
endlocal 