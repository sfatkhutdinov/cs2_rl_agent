@echo off
REM Common Functions Library for CS2 RL Agent Batch Scripts
REM Created: March 13, 2025 20:40
REM This file contains common functions used across batch scripts to reduce duplication

REM Usage: call common_functions.bat
REM Then call functions with: call :function_name [parameters]

REM Prevent this script from running directly
if "%~0"=="%~f0" (
    echo This is a library file and should not be run directly.
    echo Use 'call common_functions.bat' from another script.
    exit /b 1
)

goto :eof

:activate_conda
REM Activates conda environment only if not already activated
REM Usage: call :activate_conda
if not "%CONDA_DEFAULT_ENV%"=="cs2_agent" (
    echo Activating conda environment cs2_agent...
    call conda activate cs2_agent
) else (
    echo Already in cs2_agent environment
)
goto :eof

:setup_gpu
REM Sets up environment variables for GPU optimization
REM Usage: call :setup_gpu
echo Setting environment variables for GPU...
set CUDA_VISIBLE_DEVICES=0
set TF_FORCE_GPU_ALLOW_GROWTH=true
set TF_GPU_ALLOCATOR=cuda_malloc_async
set PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
goto :eof

:check_dependencies
REM Checks and installs dependencies if needed
REM Usage: call :check_dependencies [requirements_file]
if "%~1"=="" set REQ_FILE=requirements.txt
if not "%~1"=="" set REQ_FILE=%~1

if not exist ".deps_installed" (
    echo Installing dependencies from %REQ_FILE%...
    pip install -r %REQ_FILE%
    echo Installed on %date% %time% > .deps_installed
) else (
    echo Dependencies already installed
)
goto :eof

:check_ollama
REM Checks if Ollama is running
REM Usage: call :check_ollama
echo Checking if Ollama is running...
curl -s http://localhost:11434/api/tags > nul
if %ERRORLEVEL% NEQ 0 (
    echo ERROR: Ollama is not running. Please start Ollama first.
    echo You can download Ollama from: https://ollama.ai/
    echo After installation, run: ollama serve
    exit /b 1
) else (
    echo Ollama is running
)
goto :eof

:detect_gpu
REM Detects GPU capabilities and caches results
REM Usage: call :detect_gpu
REM After call, GPU_AVAILABLE will be 1 if GPU is available, 0 otherwise
REM CUDA_VERSION will contain the CUDA version if GPU is available
if exist ".gpu_config" (
    echo Loading cached GPU configuration...
    for /f "tokens=*" %%a in (.gpu_config) do set %%a
) else (
    echo Detecting GPU capabilities...
    nvidia-smi >nul 2>&1
    if %ERRORLEVEL%==0 (
        set GPU_AVAILABLE=1
        for /f "tokens=3" %%i in ('nvidia-smi --query-driver-version --format=csv,noheader') do set CUDA_VERSION=%%i
        echo GPU_AVAILABLE=1 > .gpu_config
        echo CUDA_VERSION=%CUDA_VERSION% >> .gpu_config
    ) else (
        set GPU_AVAILABLE=0
        echo GPU_AVAILABLE=0 > .gpu_config
    )
)
goto :eof

:error_handler
REM Handles errors with retry logic
REM Usage: call :error_handler [command] [max_retries]
REM Example: call :error_handler "python train.py" 3
set COMMAND=%~1
set MAX_RETRIES=%~2
if "%MAX_RETRIES%"=="" set MAX_RETRIES=3
set ERROR_COUNT=0

:retry
echo Attempt !ERROR_COUNT!: Running %COMMAND%...
%COMMAND%
if %ERRORLEVEL% NEQ 0 (
    set /a ERROR_COUNT+=1
    if !ERROR_COUNT! LSS %MAX_RETRIES% (
        echo Encountered error, retrying (!ERROR_COUNT!/%MAX_RETRIES%)...
        timeout /t 5
        goto retry
    ) else (
        echo Failed after %MAX_RETRIES% attempts.
        echo Error occurred at %date% %time% > error_log.txt
        exit /b 1
    )
)
goto :eof

:cleanup_temp
REM Cleans up temporary files
REM Usage: call :cleanup_temp [file_pattern]
if "%~1"=="" set TEMP_PATTERN=temp_*.*
if not "%~1"=="" set TEMP_PATTERN=%~1
echo Cleaning up temporary files matching %TEMP_PATTERN%...
if exist %TEMP_PATTERN% del /q %TEMP_PATTERN%
goto :eof

:set_high_priority
REM Sets high priority for a process
REM Usage: call :set_high_priority [process_name]
if "%~1"=="" (
    echo Error: Process name required
    exit /b 1
)
echo Setting high priority for %~1...
wmic process where name='%~1' CALL setpriority "high priority" >nul 2>&1
goto :eof 