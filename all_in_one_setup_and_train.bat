@echo off
setlocal EnableDelayedExpansion

echo ======================================================
echo All-in-One Setup and Training for CS2 RL Agent
echo ======================================================

REM Set working directory to script location
cd /d "%~dp0"

REM Check for help command
if /I "%~1"=="help" (
    echo Usage: all_in_one_setup_and_train.bat [timesteps] [mode] [options]
    echo.
    echo Parameters:
    echo   timesteps - Number of training timesteps (default: 1000)
    echo   mode      - Training mode (default: discovery)
    echo              Valid options: discovery, tutorial, vision, autonomous, adaptive, strategic
    echo   options   - Additional options based on the selected mode
    echo.
    echo For basic modes (discovery, tutorial, vision, autonomous, adaptive):
    echo   all_in_one_setup_and_train.bat [timesteps] [mode] [focus]
    echo   focus     - Whether to auto-focus the game window (default: true)
    echo              Valid options: true, false
    echo.
    echo For strategic mode:
    echo   all_in_one_setup_and_train.bat [timesteps] strategic [bootstrap] [use_adaptive] [checkpoint]
    echo   bootstrap     - Whether to use knowledge bootstrapping (default: true)
    echo                   Valid options: true, false
    echo   use_adaptive  - Whether to use the adaptive agent wrapper (default: false)
    echo                   Valid options: true, false
    echo   checkpoint    - Path to load a checkpoint from (optional)
    echo.
    echo Examples:
    echo   all_in_one_setup_and_train.bat
    echo   all_in_one_setup_and_train.bat 2000
    echo   all_in_one_setup_and_train.bat 5000 vision
    echo   all_in_one_setup_and_train.bat 2000 discovery false
    echo   all_in_one_setup_and_train.bat 5000 adaptive
    echo   all_in_one_setup_and_train.bat 10000 strategic
    echo   all_in_one_setup_and_train.bat 10000 strategic false
    echo   all_in_one_setup_and_train.bat 10000 strategic true true
    echo   all_in_one_setup_and_train.bat 10000 strategic true false "models/strategic/checkpoint_500000"
    echo.
    @REM exit /b 0
)

REM Process command line arguments
set TIMESTEPS=1000
set MODE=discovery

REM Set defaults for basic mode options
set FOCUS=true

REM Set defaults for strategic mode options
set BOOTSTRAP=true
set USE_ADAPTIVE=false
set CHECKPOINT=

REM Process basic parameters
if not "%~1"=="" set TIMESTEPS=%~1
if not "%~2"=="" set MODE=%~2

REM Process additional parameters based on mode
if /I "%MODE%"=="strategic" (
    if not "%~3"=="" set BOOTSTRAP=%~3
    if not "%~4"=="" set USE_ADAPTIVE=%~4
    if not "%~5"=="" set CHECKPOINT=%~5
) else (
    if not "%~3"=="" set FOCUS=%~3
)

REM Display configuration
echo Will train for %TIMESTEPS% timesteps in %MODE% mode
if /I "%MODE%"=="strategic" (
    echo Strategic training configuration:
    echo - Knowledge bootstrapping: %BOOTSTRAP%
    echo - Using adaptive agent: %USE_ADAPTIVE%
    if not "%CHECKPOINT%"=="" echo - Loading from checkpoint: %CHECKPOINT%
) else (
    echo Focus game window: %FOCUS%
)

REM ======== STEP 1: Check and Setup Anaconda ========
echo.
echo Step 1: Checking Anaconda installation...
call conda --version > nul 2>&1
if errorlevel 1 (
    echo ERROR: Anaconda/Miniconda is not installed or not in PATH
    echo Please install Anaconda or Miniconda from: https://www.anaconda.com/download/
    echo Add Anaconda to your PATH during installation
    pause
    exit /b 1
)
echo Anaconda is installed.

REM ======== STEP 2: Create and Setup Conda Environment ========
echo.
echo Step 2: Setting up Conda environment...
REM Check if environment exists
call conda env list | findstr "cs2_agent" > nul
if errorlevel 1 (
    echo Creating new cs2_agent conda environment...
    call conda create -y -n cs2_agent python=3.10
    if errorlevel 1 (
        echo ERROR: Failed to create conda environment
        pause
        exit /b 1
    )
) else (
    echo Conda environment 'cs2_agent' already exists, will use it.
)

REM Activate the environment
echo Activating conda environment...
call conda activate cs2_agent
if errorlevel 1 (
    echo ERROR: Failed to activate conda environment
    pause
    exit /b 1
)

REM ======== STEP 3: Install Required Packages ========
echo.
echo Step 3: Installing required packages...
REM Upgrade pip
call python -m pip install --upgrade pip

REM Install core ML libraries without version constraints using conda
echo Installing conda packages...
call conda install -y -c conda-forge "pytorch" "tensorflow" "gymnasium" "optuna" "numpy" "pandas" "matplotlib" "seaborn" "tensorboard" "pyyaml" "tqdm" "pytest" "jupyter" "requests" "pillow"
if errorlevel 1 (
    echo WARNING: Some conda packages may have failed to install.
    choice /c YN /m "Continue with installation? (Y/N)"
    if errorlevel 2 exit /b 1
)

REM Install pip packages that might not be available in conda
echo Installing pip packages...
call pip install stable-baselines3>=2.0.0 pytesseract pywin32 pyautogui mss keyboard pygame pydirectinput pynput opencv-python ray
if errorlevel 1 (
    echo WARNING: Some pip packages may have failed to install.
    choice /c YN /m "Continue with installation? (Y/N)"
    if errorlevel 2 exit /b 1
)

REM Install PyTorch with CUDA support
echo Installing PyTorch with CUDA support...
call pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
if errorlevel 1 (
    echo WARNING: Failed to install PyTorch with CUDA support.
    choice /c YN /m "Continue with installation? (Y/N)"
    if errorlevel 2 exit /b 1
)

REM Install TensorFlow with compatible version
echo Installing TensorFlow...
call pip install tensorflow==2.13.0
if errorlevel 1 (
    echo WARNING: Failed to install TensorFlow.
    choice /c YN /m "Continue with installation? (Y/N)"
    if errorlevel 2 exit /b 1
)

REM ======== STEP 4: Check GPU Support ========
echo.
echo Step 4: Setting up GPU support...
REM Check NVIDIA GPU
nvidia-smi > nul 2>&1
if errorlevel 1 (
    echo WARNING: NVIDIA GPU not detected or driver issue.
    echo Training will use CPU only, which will be much slower.
    echo If you have an NVIDIA GPU, please install the latest drivers.
) else (
    echo NVIDIA GPU detected, configuring for GPU acceleration...
    python setup_gpu.py
)

REM ======== STEP 5: Check Ollama ========
echo.
echo Step 5: Checking if Ollama is running...
curl -s http://localhost:11434/api/tags > nul
if errorlevel 1 (
    echo ERROR: Ollama is not running. Please start Ollama first.
    echo You can download Ollama from: https://ollama.ai/
    echo After installation, run: ollama serve
    pause
    exit /b 1
)

echo Checking if required model is available...
curl -s http://localhost:11434/api/tags | findstr "llama3.2-vision" > nul
if errorlevel 1 (
    echo Downloading Llama3.2 Vision model...
    curl -X POST http://localhost:11434/api/pull -d "{\"name\":\"llama3.2-vision:latest\"}"
    if errorlevel 1 (
        echo ERROR: Failed to download vision model.
        pause
        exit /b 1
    )
) else (
    echo Llama vision model already installed.
)

REM Warm up the vision model
echo Warming up vision model...
curl -s -X POST http://localhost:11434/api/generate -d "{\"model\":\"llama3.2-vision:latest\",\"prompt\":\"Hello, are you ready to analyze game screens?\",\"stream\":false}" > nul

REM ======== STEP 6: Check Directory Structure ========
echo.
echo Step 6: Checking directory structure...
python check_directories.py
if errorlevel 1 (
    echo ERROR: Directory structure check failed.
    pause
    exit /b 1
)

REM ======== Create Required Directories ========
echo.
echo Creating required directories...
if not exist logs mkdir logs
if not exist models mkdir models
if not exist models\strategic mkdir models\strategic
if not exist tensorboard mkdir tensorboard
if not exist tensorboard\strategic mkdir tensorboard\strategic

REM ======== STEP 7: Run Tests ========
echo.
echo Step 7: Running environment tests...

echo Testing configuration...
python test_config.py config/discovery_config.yaml
if errorlevel 1 (
    echo Configuration test failed. Please fix the issues before running training.
    pause
    exit /b 1
)

echo Testing CS2Environment class...
python test_cs2_env.py
if errorlevel 1 (
    echo CS2Environment test failed. Please fix the errors before running training.
    pause
    exit /b 1
)

echo Testing DiscoveryEnvironment class...
python test_discovery_env.py
if errorlevel 1 (
    echo DiscoveryEnvironment test failed. Please fix the errors before running training.
    pause
    exit /b 1
)

REM ======== STEP 8: Start Training ========
echo.
echo Step 8: Starting training...
echo.
echo IMPORTANT: Please make sure Cities: Skylines 2 is running and visible on your screen.
echo The script will focus on the game window in 5 seconds.
echo.
timeout /t 5

echo Starting %MODE% training...

REM Handle different training modes
if /I "%MODE%"=="discovery" (
    if /I "%FOCUS%"=="true" (
        python train_discovery.py --config config/discovery_config.yaml --timesteps %TIMESTEPS% --focus
    ) else (
        python train_discovery.py --config config/discovery_config.yaml --timesteps %TIMESTEPS%
    )
) else if /I "%MODE%"=="tutorial" (
    python train_tutorial_guided.py --config config/tutorial_guided_config.yaml --timesteps %TIMESTEPS%
) else if /I "%MODE%"=="vision" (
    python train_vision_guided.py --config config/vision_guided_config.yaml --timesteps %TIMESTEPS%
) else if /I "%MODE%"=="autonomous" (
    python train_autonomous.py --config config/autonomous_config.yaml --timesteps %TIMESTEPS%
) else if /I "%MODE%"=="adaptive" (
    if /I "%FOCUS%"=="true" (
        python train_adaptive.py --config config/adaptive_config.yaml --timesteps %TIMESTEPS% --focus
    ) else (
        python train_adaptive.py --config config/adaptive_config.yaml --timesteps %TIMESTEPS%
    )
) else if /I "%MODE%"=="strategic" (
    REM Build command with options for strategic mode
    set CMD=python train_strategic.py --timesteps %TIMESTEPS%
    
    REM Add bootstrap flag if specified
    if /I "%BOOTSTRAP%"=="true" (
        set CMD=!CMD! --knowledge-bootstrap
    )
    
    REM Add adaptive flag if specified
    if /I "%USE_ADAPTIVE%"=="true" (
        set CMD=!CMD! --use-adaptive
    )
    
    REM Add checkpoint if specified
    if not "%CHECKPOINT%"=="" (
        set CMD=!CMD! --load-checkpoint "%CHECKPOINT%"
    )
    
    REM Execute the command
    echo Running command: !CMD!
    !CMD!
) else (
    echo Unknown mode: %MODE%
    echo Valid modes are: discovery, tutorial, vision, autonomous, adaptive, strategic
    pause
    exit /b 1
)

echo.
if %ERRORLEVEL% equ 0 (
    echo Training completed successfully!
) else (
    echo Training encountered an error. Please check the logs for more information.
)

echo.
echo ======================================================
echo All-in-One Setup and Training Completed
echo ======================================================
pause 