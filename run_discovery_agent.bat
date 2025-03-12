@echo off
REM Setup environment variables
set PYTHONPATH=%~dp0
set CONFIG_FILE=config/discovery_config.yaml

REM Check if virtual environment exists and activate it
if exist venv\Scripts\activate.bat (
    echo Activating virtual environment...
    call venv\Scripts\activate.bat
)

REM Install required dependencies
echo Installing/updating required dependencies...
pip install numpy opencv-python pyautogui winsound

REM Run the discovery agent training
echo Starting Discovery Agent Training with Enhanced Keyboard Capabilities...
python train_discovery.py --config %CONFIG_FILE%

echo Training complete. Press any key to exit...
pause > nul 