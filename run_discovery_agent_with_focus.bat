@echo off
REM Setup environment variables
set PYTHONPATH=%~dp0
set CONFIG_FILE=config/discovery_config.yaml
set TIMESTEPS=1000
set MODE=goal

REM Process command line arguments
if not "%~1"=="" set TIMESTEPS=%~1
if not "%~2"=="" set MODE=%~2
echo Training for %TIMESTEPS% timesteps in %MODE% mode

REM Check if virtual environment exists and activate it
if exist venv\Scripts\activate.bat (
    echo Activating virtual environment...
    call venv\Scripts\activate.bat
)

REM Install required dependencies
echo Installing/updating required dependencies...
pip install numpy opencv-python pyautogui win32gui psutil

REM Add window focus fix - Create temporary Python script
echo Creating window focus helper script...
(
echo import time
echo import win32gui
echo import win32con
echo import sys
echo import os
echo import logging
echo.
echo def setup_logging():
echo     logging.basicConfig(
echo         level=logging.INFO,
echo         format="%%^(asctime^)s - %%^(name^)s - %%^(levelname^)s - %%^(message^)s",
echo         handlers=[logging.StreamHandler()]
echo     ^)
echo     return logging.getLogger("window_focus_helper"^)
echo.
echo def ensure_window_focus(window_name="Cities: Skylines II", max_attempts=5^):
echo     logger = setup_logging(^)
echo     logger.info("Starting window focus helper..."^)
echo     
echo     for attempt in range(max_attempts^):
echo         logger.info(f"Attempt {attempt+1}/{max_attempts} to find and focus game window"^)
echo         
echo         try:
echo             # Find the window by name
echo             window_handle = win32gui.FindWindow(None, window_name^)
echo             
echo             if window_handle:
echo                 logger.info(f"Found game window with handle {window_handle}"^)
echo                 
echo                 # Check if the window is minimized
echo                 if win32gui.IsIconic(window_handle^):
echo                     logger.info("Window is minimized, restoring..."^)
echo                     win32gui.ShowWindow(window_handle, win32con.SW_RESTORE^)
echo                     time.sleep(1.0^)  # Increased wait time for window to restore
echo                 
echo                 # Bring the window to the foreground
echo                 logger.info("Setting foreground window..."^)
echo                 win32gui.SetForegroundWindow(window_handle^)
echo                 
echo                 # Force window to be active and on top
echo                 logger.info("Setting window position to be on top..."^)
echo                 win32gui.SetWindowPos(
echo                     window_handle,
echo                     win32con.HWND_TOPMOST,
echo                     0, 0, 0, 0,
echo                     win32con.SWP_NOMOVE | win32con.SWP_NOSIZE
echo                 ^)
echo                 
echo                 # After a short delay, allow other windows to go on top again
echo                 time.sleep(0.5^)
echo                 win32gui.SetWindowPos(
echo                     window_handle,
echo                     win32con.HWND_NOTOPMOST,
echo                     0, 0, 0, 0,
echo                     win32con.SWP_NOMOVE | win32con.SWP_NOSIZE
echo                 ^)
echo                 
echo                 # Wait to ensure the window is focused
echo                 time.sleep(1.0^)
echo                 
echo                 # Verify focus was achieved
echo                 active_window = win32gui.GetForegroundWindow(^)
echo                 if active_window == window_handle:
echo                     logger.info("Successfully focused game window!"^)
echo                     return True
echo                 else:
echo                     logger.warning("Failed to verify window focus, will retry..."^)
echo             else:
echo                 logger.warning(f"Could not find window with name '{window_name}'"^)
echo                 
echo             # Wait before trying again
echo             time.sleep(2.0^)
echo                 
echo         except Exception as e:
echo             logger.error(f"Error focusing window: {str(e^)}"^)
echo             time.sleep(1.0^)
echo     
echo     logger.error(f"Failed to focus game window after {max_attempts} attempts"^)
echo     return False
echo.
echo if __name__ == "__main__":
echo     window_name = "Cities: Skylines II"
echo     if len(sys.argv^) > 1:
echo         window_name = sys.argv[1]
echo     
echo     success = ensure_window_focus(window_name^)
echo     sys.exit(0 if success else 1^)
) > window_focus_helper.py

REM Try to focus the game window before starting training
echo Attempting to focus game window...
python window_focus_helper.py "Cities: Skylines II"

REM Run the discovery agent training with appropriate mode
echo Starting Discovery Agent Training with Enhanced Feedback...
if /i "%MODE%"=="explore" (
    echo Running in exploration-focused mode
    python train_discovery.py --config %CONFIG_FILE% --timesteps %TIMESTEPS% --exploration-focus
) else (
    echo Running in city-building goal-oriented mode
    python train_discovery.py --config %CONFIG_FILE% --timesteps %TIMESTEPS% --goal-focus
)

echo Training complete. Press any key to exit...
pause > nul 