@echo off
REM This script runs the vision calibration tool
REM It will help you set up the correct window detection parameters

REM Set the path to the CS2 RL agent directory
set BASE_DIR=%~dp0..\..
cd %BASE_DIR%

echo Running vision calibration tool...
python testing\test_vision_windows.py --calibrate

echo Done.
pause 