@echo off
echo Cities: Skylines 2 RL Agent - UI Template Capture
echo.
echo This tool will help you capture UI templates for automatic detection
echo Ensure Cities: Skylines 2 is running and a city is loaded!
echo.

REM Navigate to the project root directory
cd /d "%~dp0"

REM Run the auto-detection script in template capture mode
python auto_detect.py --capture-templates

echo.
echo Template capture completed. Press any key to exit.
pause > nul 