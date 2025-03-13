@echo off
echo Training Adaptive CS2 RL Agent
echo --------------------------------

REM Activate conda environment
call conda activate cs2_agent

REM Check if Ollama is running
echo Checking if Ollama is running...
curl -s http://localhost:11434/api/tags > nul
if %ERRORLEVEL% NEQ 0 (
    echo ERROR: Ollama is not running. Please start Ollama first.
    echo You can download Ollama from: https://ollama.ai/
    echo After installation, run: ollama serve
    pause
    exit /b 1
)

REM Process command line arguments
set TIMESTEPS=10000
set FOCUS=--focus
set STARTING_MODE=discovery
set EXTRA_ARGS=

if not "%~1"=="" set TIMESTEPS=%~1
if not "%~2"=="" set STARTING_MODE=%~2
if /I "%~3"=="nofocus" set FOCUS=

echo Will train for %TIMESTEPS% timesteps starting with %STARTING_MODE% mode

echo.
echo IMPORTANT: Make sure Cities: Skylines 2 is running and visible on your screen.
echo The training will begin in 5 seconds.
timeout /t 5

REM Run the adaptive training script
python train_adaptive.py --timesteps %TIMESTEPS% %FOCUS% --starting-mode %STARTING_MODE% %EXTRA_ARGS%

echo.
echo Training complete!
pause 