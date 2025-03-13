@echo off
echo Cities: Skylines 2 RL Agent - Auto-Detection Tool
echo.
echo This tool will automatically detect UI elements without manual coordinates
echo Ensure Cities: Skylines 2 is running and a city is loaded!
echo.

REM Navigate to the project root directory
cd /d "%~dp0"

REM Run the auto-detection script
python auto_detect.py

echo.
echo Test completed. Press any key to exit.
pause > nul 