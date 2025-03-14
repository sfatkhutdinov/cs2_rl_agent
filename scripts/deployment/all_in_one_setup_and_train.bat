@echo off
setlocal EnableDelayedExpansion

echo ======================================================
echo All-in-One Setup and Training for CS2 RL Agent
echo ======================================================

REM Set working directory to script location
cd /d "%~dp0"

REM Include common functions library
call ..\utils\common_functions.bat

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
    exit /b 0
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
REM Use the activate_conda function which handles environment checking and activation
call :activate_conda

REM ======== STEP 3: Install Required Packages ========
echo.
echo Step 3: Installing required packages...
REM Use the check_dependencies function to avoid redundant installations
call :check_dependencies "requirements.txt"

REM ======== STEP 4: Check GPU Support ========
echo.
echo Step 4: Setting up GPU support...
REM Setup GPU environment variables
call :setup_gpu

REM Detect GPU with caching
call :detect_gpu

if "%GPU_AVAILABLE%"=="1" (
    echo NVIDIA GPU detected with CUDA version %CUDA_VERSION%
    echo Configuring for GPU acceleration...
    set GPU_CMD=python setup_gpu.py
    call :error_handler "%GPU_CMD%" 2
) else (
    echo WARNING: NVIDIA GPU not detected or driver issue.
    echo Training will use CPU only, which will be much slower.
    echo If you have an NVIDIA GPU, please install the latest drivers.
)

REM ======== STEP 5: Check Ollama ========
echo.
echo Step 5: Checking if Ollama is running...
call :check_ollama

echo Checking if required model is available...
curl -s http://localhost:11434/api/tags | findstr "llama3.2-vision" > nul
if errorlevel 1 (
    echo Downloading Llama3.2 Vision model...
    set PULL_CMD=curl -X POST http://localhost:11434/api/pull -d "{\"name\":\"llama3.2-vision:latest\"}"
    call :error_handler "%PULL_CMD%" 3
) else (
    echo Llama vision model already installed.
)

REM Warm up the vision model
echo Warming up vision model...
curl -s -X POST http://localhost:11434/api/generate -d "{\"model\":\"llama3.2-vision:latest\",\"prompt\":\"Hello, are you ready to analyze game screens?\",\"stream\":false}" > nul

REM ======== STEP 6: Check Directory Structure ========
echo.
echo Step 6: Checking directory structure...
set DIR_CHECK_CMD=python check_directories.py
call :error_handler "%DIR_CHECK_CMD%" 2

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
set CONFIG_TEST_CMD=python test_config.py config/discovery_config.yaml
call :error_handler "%CONFIG_TEST_CMD%" 2

echo Testing CS2Environment class...
set ENV_TEST_CMD=python test_cs2_env.py
call :error_handler "%ENV_TEST_CMD%" 2

echo Testing DiscoveryEnvironment class...
set DISC_ENV_TEST_CMD=python test_discovery_env.py
call :error_handler "%DISC_ENV_TEST_CMD%" 2

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
set TRAIN_CMD=

if /I "%MODE%"=="discovery" (
    if /I "%FOCUS%"=="true" (
        set TRAIN_CMD=python train_discovery.py --config config/discovery_config.yaml --timesteps %TIMESTEPS% --focus
    ) else (
        set TRAIN_CMD=python train_discovery.py --config config/discovery_config.yaml --timesteps %TIMESTEPS%
    )
) else if /I "%MODE%"=="tutorial" (
    set TRAIN_CMD=python train_tutorial_guided.py --config config/tutorial_guided_config.yaml --timesteps %TIMESTEPS%
) else if /I "%MODE%"=="vision" (
    set TRAIN_CMD=python train_vision_guided.py --config config/vision_guided_config.yaml --timesteps %TIMESTEPS%
) else if /I "%MODE%"=="autonomous" (
    set TRAIN_CMD=python train_autonomous.py --config config/autonomous_config.yaml --timesteps %TIMESTEPS%
) else if /I "%MODE%"=="adaptive" (
    if /I "%FOCUS%"=="true" (
        set TRAIN_CMD=python train_adaptive.py --config config/adaptive_config.yaml --timesteps %TIMESTEPS% --focus
    ) else (
        set TRAIN_CMD=python train_adaptive.py --config config/adaptive_config.yaml --timesteps %TIMESTEPS%
    )
) else if /I "%MODE%"=="strategic" (
    REM Build command with options for strategic mode
    set TRAIN_CMD=python train_strategic.py --timesteps %TIMESTEPS%
    
    REM Add bootstrap flag if specified
    if /I "%BOOTSTRAP%"=="true" (
        set TRAIN_CMD=!TRAIN_CMD! --knowledge-bootstrap
    )
    
    REM Add adaptive flag if specified
    if /I "%USE_ADAPTIVE%"=="true" (
        set TRAIN_CMD=!TRAIN_CMD! --use-adaptive
    )
    
    REM Add checkpoint if specified
    if not "%CHECKPOINT%"=="" (
        set TRAIN_CMD=!TRAIN_CMD! --load-checkpoint "%CHECKPOINT%"
    )
) else (
    echo Unknown mode: %MODE%
    echo Valid modes are: discovery, tutorial, vision, autonomous, adaptive, strategic
    pause
    exit /b 1
)

REM Execute the training command with error handling
echo Running command: !TRAIN_CMD!
call :error_handler "!TRAIN_CMD!" 3

REM Set high priority for Python processes
call :set_high_priority "python.exe"

REM Clean up temporary files
call :cleanup_temp "temp_*.*"

echo.
if %ERRORLEVEL% equ 0 (
    echo Training completed successfully!
) else (
    echo Training encountered an error. Please check error_log.txt for more information.
)

echo.
echo ======================================================
echo All-in-One Setup and Training Completed
echo ======================================================
pause
endlocal 