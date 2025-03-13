@echo off
REM This script runs the vision test to verify that the vision system works correctly
REM It will take screenshots and show them with detected elements

REM Set the path to the CS2 RL agent directory
set BASE_DIR=%~dp0..\..
cd %BASE_DIR%

echo Running vision test...
python testing\test_vision_windows.py

echo Done.
pause 