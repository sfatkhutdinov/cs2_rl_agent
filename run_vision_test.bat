@echo off
echo Running Cities: Skylines 2 RL Agent Vision Interface Test
echo.
echo Ensure Cities: Skylines 2 is running and a city is loaded!
echo.

REM Navigate to the project root directory
cd /d "%~dp0"

REM Run the test script
python src\test_vision_windows.py

echo.
echo Test completed. Press any key to exit.
pause > nul 