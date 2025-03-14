@echo off
setlocal EnableDelayedExpansion

echo ======================================================
echo GPU Setup and Optimization Tool for CS2 RL Agent
echo ======================================================

REM Set working directory to script location
cd /d "%~dp0"

REM Include common functions library
call %~dp0common_functions.bat

REM Check for help command
if /I "%~1"=="help" (
    echo Usage: enable_gpu.bat [force_reinstall]
    echo.
    echo Parameters:
    echo   force_reinstall - Force reinstallation of GPU packages (default: false)
    echo                    Valid options: true, false
    echo.
    echo Examples:
    echo   enable_gpu.bat
    echo   enable_gpu.bat true
    echo.
    exit /b 0
)

REM Process command line arguments
set FORCE_REINSTALL=false
if not "%~1"=="" set FORCE_REINSTALL=%~1

echo GPU Setup Configuration:
echo - Force reinstallation: %FORCE_REINSTALL%

REM ======== STEP 1: Check Python Environment ========
echo.
echo Checking Python environment...

REM Activate conda environment (only if needed)
call :activate_conda

REM ======== STEP 2: Setup GPU Environment Variables ========
echo.
echo Setting up GPU environment variables...
call :setup_gpu

REM ======== STEP 3: Check GPU Availability ========
echo.
echo Checking for NVIDIA GPU...
call :detect_gpu

if "%GPU_AVAILABLE%"=="1" (
    echo NVIDIA GPU detected with CUDA version %CUDA_VERSION%
    echo Proceeding with GPU optimization.
) else (
    echo WARNING: NVIDIA driver or GPU not detected!
    echo Please install the latest NVIDIA drivers from:
    echo https://www.nvidia.com/Download/index.aspx
    echo.
    echo The script will continue with CPU-only setup.
)

REM ======== STEP 4: Install Required Dependencies ========
echo.
echo Installing/updating required GPU dependencies...

REM Only install if forced or not already installed
if /I "%FORCE_REINSTALL%"=="true" (
    echo Forcing reinstallation of GPU packages...
    set INSTALL_DEPS=1
) else (
    set INSTALL_DEPS=0
    pip show torch | findstr "Version: 2.0.1" > nul
    if errorlevel 1 set INSTALL_DEPS=1
)

if "%INSTALL_DEPS%"=="1" (
    echo Installing required dependencies...
    set DEPS_CMD=pip install numpy opencv-python
    call :error_handler "%DEPS_CMD%" 3
    
    REM Uninstall existing PyTorch
    echo Uninstalling existing PyTorch...
    pip uninstall -y torch torchvision torchaudio
    
    REM Install specific PyTorch version with CUDA
    echo Installing PyTorch with CUDA support...
    set TORCH_CMD=pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu118
    call :error_handler "%TORCH_CMD%" 3
    
    REM Uninstall existing TensorFlow
    echo Uninstalling existing TensorFlow...
    pip uninstall -y tensorflow tensorflow-gpu
    
    REM Install specific TensorFlow version
    echo Installing TensorFlow...
    set TF_CMD=pip install tensorflow==2.13.0
    call :error_handler "%TF_CMD%" 3
    
    REM Install other ML dependencies
    echo Installing other ML dependencies...
    set ML_CMD=pip install stable-baselines3==2.1.0 tensorboard==2.13.0
    call :error_handler "%ML_CMD%" 3
) else (
    echo GPU dependencies are already installed.
    echo Use 'enable_gpu.bat true' to force reinstallation.
)

REM ======== STEP 5: Run CUDA Fix Tool ========
echo.
echo Running CUDA fix tool...
set CUDA_FIX_CMD=python fix_cuda.py
call :error_handler "%CUDA_FIX_CMD%" 2

REM ======== STEP 6: Verify GPU Detection ========
echo.
echo Verifying GPU detection...

REM Verify PyTorch GPU
echo Testing PyTorch CUDA...
set PYTORCH_CMD=python -c "import torch; print(f'PyTorch CUDA available: {torch.cuda.is_available()}'); print(f'GPU Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"None\"}')"
call :error_handler "%PYTORCH_CMD%" 2

REM Verify TensorFlow GPU
echo Testing TensorFlow GPU...
set TF_CMD=python -c "import tensorflow as tf; print(f'TensorFlow GPU devices: {tf.config.list_physical_devices(\"GPU\")}')"
call :error_handler "%TF_CMD%" 2

REM ======== STEP 7: Run Comprehensive GPU Setup ========
echo.
echo Running comprehensive GPU setup...
set SETUP_CMD=python setup_gpu.py
call :error_handler "%SETUP_CMD%" 2

if %ERRORLEVEL% equ 0 (
    echo.
    echo GPU setup completed successfully.
    if "%GPU_AVAILABLE%"=="1" (
        echo Your system is now configured to use GPU acceleration.
    ) else (
        echo WARNING: No GPU was detected. Training will use CPU.
        echo For GPU support, please install NVIDIA drivers.
    )
) else (
    echo.
    echo GPU setup encountered issues.
    echo Please run fix_cuda.py directly for more detailed diagnostics.
)

echo.
echo If you encounter issues, try:
echo 1. Running 'enable_gpu.bat true' to force reinstall
echo 2. Updating your GPU drivers
echo 3. Checking compatibility with the installed CUDA version
echo.
pause
endlocal 