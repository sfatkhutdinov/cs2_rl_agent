import sys 
import time 
import win32gui 
import win32con 
import win32api 
import win32process 
import logging 
import traceback 
from PIL import Image, ImageGrab 
import numpy as np 
 
# Configure logging 
logging.basicConfig( 
    level=logging.INFO, 
    format="%(asctime)s - %(levelname)s - %(message)s", 
    handlers=[logging.StreamHandler()] 
) 
logger = logging.getLogger("focus_helper") 
 
def force_game_window_focus(window_name="Cities: Skylines II", max_attempts=3): 
    """Aggressively try to force focus on the game window using multiple methods""" 
    logger.info(f"Attempting to focus game window: {window_name}") 
 
    # Try each focus method multiple times 
    for attempt in range(max_attempts): 
        logger.info(f"Focus attempt {attempt+1}/{max_attempts}") 
 
        try: 
            # Find the window first 
            window_handle = win32gui.FindWindow(None, window_name) 
 
            if not window_handle: 
                logger.warning(f"Could not find window with name '{window_name}'") 
                time.sleep(1.0) 
                continue 
 
            logger.info(f"Found window with handle {window_handle}") 
 
            # Method 1: Restore if minimized 
            if win32gui.IsIconic(window_handle): 
                logger.info("Window is minimized, restoring...") 
                win32gui.ShowWindow(window_handle, win32con.SW_RESTORE) 
                time.sleep(1.0) 
            # Method 2: Show window 
            win32gui.ShowWindow(window_handle, win32con.SW_SHOW) 
            time.sleep(0.5) 
            # Method 3: Get current window position 
            rect = win32gui.GetWindowRect(window_handle) 
            x, y, right, bottom = rect 
            width, height = right - x, bottom - y 
            logger.info(f"Window position: {x}, {y}, {width}x{height}") 
            # Method 4: Set foreground window 
            win32gui.SetForegroundWindow(window_handle) 
            time.sleep(0.5) 
            # Method 5: Make topmost 
            win32gui.SetWindowPos( 
                window_handle, 
                win32con.HWND_TOPMOST, 
                x, y, width, height, 
                win32con.SWP_SHOWWINDOW 
            ) 
            time.sleep(0.5) 
            # Method 6: Back to normal z-order but keep focus 
            win32gui.SetWindowPos( 
                window_handle, 
                win32con.HWND_NOTOPMOST, 
                x, y, width, height, 
                win32con.SWP_SHOWWINDOW 
            ) 
            # Method 7: Thread attachment technique 
            try: 
                # Get thread info to help bypass focus stealing prevention 
                current_thread = win32api.GetCurrentThreadId() 
                remote_thread, remote_process = win32process.GetWindowThreadProcessId(window_handle) 
                # Attach threads 
                win32process.AttachThreadInput(current_thread, remote_thread, True) 
                # Set focus in multiple ways 
                win32gui.SetFocus(window_handle) 
                win32gui.SetActiveWindow(window_handle) 
                win32gui.SetForegroundWindow(window_handle) 
                # Detach threads 
                win32process.AttachThreadInput(current_thread, remote_thread, False) 
            except Exception as thread_err: 
                logger.warning(f"Thread attachment technique failed: {thread_err}") 
            
            # Method 8: Send system commands to ensure active state
            # SC_SCREENSAVER doesn't exist in win32con, use alternative commands
            try:
                # Send a null system command to wake up the window
                win32api.SendMessage(window_handle, win32con.WM_SYSCOMMAND, win32con.SC_HOTKEY, 0)
                # Alternative activation commands
                win32api.SendMessage(window_handle, win32con.WM_ACTIVATE, win32con.WA_ACTIVE, 0)
                win32api.SendMessage(window_handle, win32con.WM_NCACTIVATE, 1, 0)
            except Exception as cmd_err:
                logger.warning(f"System command failed: {cmd_err}")
            
            # Final check 
            time.sleep(1.0)  # Give Windows time to process everything 
            active_window = win32gui.GetForegroundWindow() 
            if active_window == window_handle: 
                logger.info("Successfully focused window!") 
                # Take a screenshot to verify visual state 
                try: 
                    # Use PIL for screenshot 
                    screen = ImageGrab.grab() 
                    screen_np = np.array(screen) 
                    # Convert to PIL Image and save 
                    Image.fromarray(screen_np).save("focused_window_screenshot.png") 
                    logger.info("Saved screenshot of focused window") 
                except Exception as ss_err: 
                    logger.warning(f"Failed to save screenshot: {ss_err}") 
                return True 
            else: 
                logger.warning("Failed to verify focus (focus stealing prevention may be active)") 
        except Exception as e: 
            logger.error(f"Error during focus attempt: {e}") 
            logger.error(traceback.format_exc()) 
        # Wait before trying again 
        time.sleep(2.0) 
    logger.error("All focus attempts failed") 
    return False 
 
if __name__ == "__main__": 
    # Get window name from command line if provided 
    window_name = "Cities: Skylines II" 
    if len(sys.argv) > 1: 
        window_name = sys.argv[1] 
 
    # Try to focus the window 
    success = force_game_window_focus(window_name) 
 
    # Exit with status code 
    sys.exit(0 if success else 1) 
