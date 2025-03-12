@echo off
echo =====================================
echo GPU Setup and Fix Tool for ML Training
echo =====================================

REM Activate virtual environment if it exists
if exist venv\Scripts\activate.bat (
    echo Activating virtual environment...
    call venv\Scripts\activate.bat
)

REM Set environment variables for GPU
echo Setting environment variables for GPU...
set CUDA_VISIBLE_DEVICES=0
set TF_FORCE_GPU_ALLOW_GROWTH=true
set TF_GPU_ALLOCATOR=cuda_malloc_async
set PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

REM Install required dependencies
echo Installing/updating required dependencies...
pip install numpy opencv-python

REM Check NVIDIA drivers
echo Checking for NVIDIA GPU...
nvidia-smi >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo WARNING: NVIDIA driver or GPU not detected!
    echo Please install the latest NVIDIA drivers from:
    echo https://www.nvidia.com/Download/index.aspx
    echo.
    pause
) else (
    echo NVIDIA GPU detected. Running nvidia-smi for info:
    nvidia-smi
)

REM Execute the comprehensive CUDA fix tool
echo Running CUDA fix tool...
python fix_cuda.py
if %ERRORLEVEL% NEQ 0 (
    echo WARNING: CUDA fix tool reported issues.
    echo.
    pause
) else (
    echo CUDA fix tool completed successfully.
)

REM Direct installation of PyTorch with CUDA
echo Installing PyTorch with CUDA support directly...
pip uninstall -y torch torchvision torchaudio
pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu118

REM Install TensorFlow
echo Installing TensorFlow...
pip uninstall -y tensorflow tensorflow-gpu
pip install tensorflow==2.13.0

REM Install other ML dependencies
echo Installing other ML dependencies...
pip install stable-baselines3==2.1.0 tensorboard==2.13.0

REM Verify GPU detection
echo Verifying GPU detection...
python -c "import torch; print(f'PyTorch CUDA available: {torch.cuda.is_available()}'); print(f'GPU Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"None\"}')"
python -c "import tensorflow as tf; print(f'TensorFlow GPU devices: {tf.config.list_physical_devices(\"GPU\")}')"

REM Run the full GPU setup script
echo Running comprehensive GPU setup...
python setup_gpu.py

echo GPU setup completed.
echo If GPU is still not detected, please run fix_cuda.py directly and follow its instructions.
echo.
echo Press any key to exit...
pause > nul 