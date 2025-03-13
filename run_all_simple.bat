@echo off
setlocal EnableDelayedExpansion

echo ======================================================
echo Simplified CS2 Agent Training with GPU Support
echo ======================================================

REM Set working directory to script location
cd /d "%~dp0"

REM Setup environment variables
set PYTHONPATH=%~dp0
set CONFIG_FILE=config/discovery_config.yaml
set TIMESTEPS=1000
set MODE=goal

REM Process command line arguments
if not "%~1"=="" set TIMESTEPS=%~1
if not "%~2"=="" set MODE=%~2
echo Will train for %TIMESTEPS% timesteps in %MODE% mode

REM Create logs directory if it doesn't exist
mkdir logs 2>nul
mkdir logs\vision_debug 2>nul

REM Check if virtual environment exists and activate it
if exist venv\Scripts\activate.bat (
    echo Activating virtual environment...
    call conda activate cs2_agent
)

echo ======================================================
echo STEP 1: Installing Essential Dependencies
echo ======================================================

REM Install required dependencies
echo Installing/updating required dependencies...
pip install numpy opencv-python pyautogui pywin32 psutil pillow pytesseract mss --quiet
pip install --upgrade Pillow --quiet

REM Direct installation of PyTorch with CUDA support
echo Installing PyTorch with CUDA support...
pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu118 --quiet

REM Install TensorFlow
echo Installing TensorFlow...
pip install tensorflow==2.13.0 --quiet

REM Install machine learning dependencies
echo Installing machine learning dependencies...
pip install stable-baselines3==2.1.0 tensorboard==2.13.0 gymnasium --quiet

echo ======================================================
echo STEP 2: Setting Up GPU Environment
echo ======================================================

REM Set environment variables for CUDA
echo Setting environment variables for GPU...
set CUDA_VISIBLE_DEVICES=0
set TF_FORCE_GPU_ALLOW_GROWTH=true
set TF_GPU_ALLOCATOR=cuda_malloc_async
set PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

REM Check for CUDA installation
echo Checking for CUDA installation...
nvidia-smi >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo WARNING: NVIDIA driver or CUDA installation issue detected.
    echo Will continue with CPU mode.
    echo.
) else (
    echo NVIDIA drivers found: 
    nvidia-smi | findstr "NVIDIA-SMI" 
    nvidia-smi | findstr "Driver Version"
)

REM Create simple enable_cuda.py if it doesn't exist
if not exist "enable_cuda.py" (
    echo Creating simplified CUDA enabler...
    echo import os > enable_cuda.py
    echo import sys >> enable_cuda.py
    echo # Set environment variables >> enable_cuda.py
    echo os.environ["CUDA_VISIBLE_DEVICES"] = "0" >> enable_cuda.py
    echo os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true" >> enable_cuda.py
    echo try: >> enable_cuda.py
    echo     import torch >> enable_cuda.py
    echo     print(f"PyTorch CUDA available: {torch.cuda.is_available()}") >> enable_cuda.py
    echo     if torch.cuda.is_available(): >> enable_cuda.py
    echo         print(f"GPU Device: {torch.cuda.get_device_name(0)}") >> enable_cuda.py
    echo except ImportError: >> enable_cuda.py
    echo     print("PyTorch not installed") >> enable_cuda.py
    echo try: >> enable_cuda.py
    echo     import tensorflow as tf >> enable_cuda.py
    echo     print(f"TensorFlow GPU devices: {tf.config.list_physical_devices('GPU')}") >> enable_cuda.py
    echo except ImportError: >> enable_cuda.py
    echo     print("TensorFlow not installed") >> enable_cuda.py
)

REM Verify GPU detection with PyTorch and TensorFlow
echo Verifying GPU detection...
python -c "import torch; print(f'PyTorch CUDA available: {torch.cuda.is_available()}'); print(f'GPU Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"None\"}')"
python -c "import tensorflow as tf; print(f'TensorFlow GPU devices: {tf.config.list_physical_devices(\"GPU\")}')"

echo ======================================================
echo STEP 3: Checking Ollama Service
echo ======================================================

REM Check if Ollama is running and start it if needed
echo Checking if Ollama is running...
powershell -command "if (!(Get-Process -Name ollama -ErrorAction SilentlyContinue)) { echo 'Starting Ollama service...'; Start-Process 'ollama' -WindowStyle Minimized }"

REM Wait for Ollama to initialize
echo Waiting for Ollama to initialize...
timeout /t 5 > nul

REM Pull the required model if it doesn't exist
echo Checking for granite3.2-vision model...
powershell -command "$result = (ollama list 2>&1 | Select-String 'granite3.2-vision'); if(-not $result) { echo 'Pulling granite3.2-vision model...'; ollama pull granite3.2-vision:latest }"

echo ======================================================
echo STEP 4: Focusing Game Window
echo ======================================================

echo Attempting to focus game window...
python enhanced_focus.py "Cities: Skylines II"
if %ERRORLEVEL% NEQ 0 (
    echo Falling back to simpler focus method...
    powershell -command "(New-Object -ComObject WScript.Shell).AppActivate('Cities: Skylines II')"
)

echo Waiting for the game to stabilize...
echo TIP: If agent button presses don't register, try clicking manually in the game window once.
timeout /t 5 > nul

echo ======================================================
echo STEP 5: Starting Agent Training
echo ======================================================

REM Run the discovery agent training with appropriate mode
echo Starting Discovery Agent Training...
if /i "%MODE%"=="explore" (
    echo Running in exploration-focused mode
    python train_discovery.py --config %CONFIG_FILE% --timesteps %TIMESTEPS% --exploration-focus
) else (
    echo Running in city-building goal-oriented mode
    python train_discovery.py --config %CONFIG_FILE% --timesteps %TIMESTEPS% --goal-focus
)

echo ======================================================
echo Training complete! Press any key to exit...
echo ======================================================
pause > nul 