@echo off
echo Cities: Skylines 2 RL Agent - Vision Interface Calibration
echo.
echo This tool will help you calibrate the vision interface for your screen.
echo Ensure Cities: Skylines 2 is running and a city is loaded!
echo.

REM Navigate to the project root directory
cd /d "%~dp0"

REM Run the calibration mode
python src\test_vision_windows.py --calibrate

echo.
echo Calibration completed. Press any key to exit.
pause > nul 