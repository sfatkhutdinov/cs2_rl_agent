@echo off
echo Checking Anaconda setup for CS2 RL Agent
echo ---------------------------------------------------

REM Check if Conda is installed
call conda --version > nul 2>&1
if errorlevel 1 (
    echo ERROR: Anaconda/Miniconda is not installed or not in PATH
    echo Please install from: https://www.anaconda.com/download/
    pause
    exit /b 1
)

REM Check if cs2_agent environment exists
call conda env list | findstr "cs2_agent" > nul
if errorlevel 1 (
    echo ERROR: cs2_agent environment not found
    echo Please run setup_conda.bat to create it
    pause
    exit /b 1
)

REM Activate the conda environment
call conda activate cs2_agent

REM Check required packages
echo Checking required packages...
echo.

REM Check PyTorch
python -c "import torch; print(f'PyTorch {torch.__version__} - CUDA available: {torch.cuda.is_available()}')" || echo ERROR: PyTorch not installed correctly

REM Check TensorFlow
python -c "import tensorflow as tf; print(f'TensorFlow {tf.__version__} - GPU available: {len(tf.config.list_physical_devices(\"GPU\")) > 0}')" || echo ERROR: TensorFlow not installed correctly

REM Check Gymnasium
python -c "import gymnasium; print(f'Gymnasium {gymnasium.__version__}')" || echo ERROR: Gymnasium not installed correctly

REM Check Stable-Baselines3
python -c "import stable_baselines3; print(f'Stable-Baselines3 {stable_baselines3.__version__}')" || echo ERROR: Stable-Baselines3 not installed correctly

REM Check OpenCV
python -c "import cv2; print(f'OpenCV {cv2.__version__}')" || echo ERROR: OpenCV not installed correctly

echo.
echo Conda environment check complete!
echo If you see any errors above, run setup_conda.bat to fix them.
pause 