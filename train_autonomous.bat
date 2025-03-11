@echo off
echo Starting Autonomous Agent Training for Cities: Skylines 2
echo ---------------------------------------------------
echo This will launch a fully autonomous agent that will learn to play
echo Cities: Skylines 2 from scratch through exploration.
echo.
echo Please ensure:
echo 1. Cities: Skylines 2 is installed and working
echo 2. You have set up the environment as per WINDOWS_SETUP.md
echo 3. The game window is visible and not minimized
echo 4. You've started with a new, empty city
echo.
echo The agent will start by exploring the UI and gradually learn to play.
echo This process may take many hours or days for significant learning.
echo.
echo Press Ctrl+C to stop training at any time. Models will be saved periodically.
echo.
pause

echo Setting up Python environment...
REM Try multiple possible virtual environment paths
if exist venv\Scripts\activate.bat (
    call venv\Scripts\activate.bat
) else if exist .venv\Scripts\activate.bat (
    call .venv\Scripts\activate.bat
) else if exist env\Scripts\activate.bat (
    call env\Scripts\activate.bat
) else (
    echo Virtual environment not found. Using system Python.
)

echo Creating necessary directories...
if not exist logs mkdir logs
if not exist logs\autonomous mkdir logs\autonomous
if not exist models mkdir models
if not exist models\autonomous mkdir models\autonomous
if not exist tensorboard mkdir tensorboard
if not exist tensorboard\autonomous mkdir tensorboard\autonomous

echo Launching autonomous agent training...
python train_autonomous.py --config config/autonomous_agent.yaml

echo.
echo Training complete or interrupted.
echo Check the logs and models folders for results.
pause 