@echo off
REM Setup environment variables
set PYTHONPATH=%~dp0
set CONFIG_FILE=config/discovery_config.yaml
set TIMESTEPS=1000
set MODE=goal

REM Process command line arguments
if not "%~1"=="" set TIMESTEPS=%~1
if not "%~2"=="" set MODE=%~2
echo Training for %TIMESTEPS% timesteps in %MODE% mode

REM Check if virtual environment exists and activate it
if exist venv\Scripts\activate.bat (
    echo Activating virtual environment...
    call venv\Scripts\activate.bat
)

REM Install required dependencies
echo Installing/updating required dependencies...
pip install numpy opencv-python pyautogui pywin32 psutil pillow pytesseract mss

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
    echo Please make sure you have the latest NVIDIA drivers installed.
    echo You can download them from: https://www.nvidia.com/Download/index.aspx
    echo.
    echo For TensorFlow/PyTorch GPU support, you need CUDA Toolkit 11.8 and cuDNN.
    echo Proceeding with CPU mode for now.
    echo.
    pause
) else (
    echo NVIDIA drivers found. Checking GPU capability...
)

REM Run the GPU setup script
echo Running GPU setup script...
python setup_gpu.py
if %ERRORLEVEL% NEQ 0 (
    echo WARNING: GPU setup failed. Continuing with CPU mode.
    echo.
    pause
) else (
    echo GPU setup successful!
)

REM Check if Tesseract OCR is installed
echo Checking for Tesseract OCR...
where tesseract >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo WARNING: Tesseract OCR not found in PATH.
    echo Please install Tesseract OCR from https://github.com/UB-Mannheim/tesseract/wiki
    echo and add it to your PATH.
    echo The training will continue but OCR-based UI detection will be disabled.
    echo.
    echo Continuing in 5 seconds...
    timeout /t 5 > nul
)

REM Create a more robust window focus script
echo Creating enhanced window focus helper...
REM Ensure we're writing to the correct directory
cd /d "%~dp0"

REM Check if Ollama is running and start it if needed
echo Checking if Ollama is running...
powershell -command "if (!(Get-Process -Name ollama -ErrorAction SilentlyContinue)) { echo 'Starting Ollama service...'; Start-Process 'ollama' -WindowStyle Minimized }"

REM Wait for Ollama to initialize
echo Waiting for Ollama to initialize...
timeout /t 5 > nul

REM Pull the required model if it doesn't exist
echo Checking for granite3.2-vision model...
powershell -command "$result = (ollama list 2>&1 | Select-String 'granite3.2-vision'); if(-not $result) { echo 'Pulling granite3.2-vision model...'; ollama pull granite3.2-vision:latest }"

REM Run a quick test to verify dependencies and methods
echo Running preliminary tests...
python test_focus.py
if %ERRORLEVEL% NEQ 0 (
    echo WARNING: Preliminary tests failed. Continuing anyway, but expect issues.
    echo Consider fixing the issues before continuing.
    echo.
    pause
)

echo Activating the new enhanced input simulation system...
echo This will make button presses register more reliably in the game.
echo.

REM Apply common fixes for Python modules
echo Checking and applying fixes for common Python issues...
pip install --upgrade Pillow
pip install --upgrade opencv-python

REM Ensure the correct directories for screenshots exist
mkdir logs 2>nul
mkdir logs\vision_debug 2>nul

echo Attempting to focus game window with enhanced methods...
python enhanced_focus.py "Cities: Skylines II"
if %ERRORLEVEL% NEQ 0 (
    echo WARNING: Failed to focus game window with enhanced methods.
    echo Falling back to simpler focus method...
    
    REM Simple fallback focus method
    powershell -command "(New-Object -ComObject WScript.Shell).AppActivate('Cities: Skylines II')"
    timeout /t 3 > nul
)

echo Waiting for the game to stabilize and recognize inputs...
echo TIP: If agent button presses don't register, try clicking manually in the game window once.
timeout /t 5 > nul

REM Run the discovery agent training with appropriate mode
echo Starting Discovery Agent Training with Enhanced Window Focus...
if /i "%MODE%"=="explore" (
    echo Running in exploration-focused mode
    python train_discovery.py --config %CONFIG_FILE% --timesteps %TIMESTEPS% --exploration-focus
) else (
    echo Running in city-building goal-oriented mode
    python train_discovery.py --config %CONFIG_FILE% --timesteps %TIMESTEPS% --goal-focus
)

echo Training complete. Press any key to exit...
pause > nul 