"""
Focus helper module for maintaining window focus during agent training.
"""

import time
import logging
import threading
import win32gui
import win32con
import win32process
from typing import Optional, Callable

logger = logging.getLogger(__name__)

class FocusHelper:
    """
    Helper class to maintain focus on the game window during agent training.
    """
    
    def __init__(self, window_name: str = "Cities: Skylines II", check_interval: float = 2.0):
        """
        Initialize the focus helper.
        
        Args:
            window_name: Name of the game window to maintain focus on
            check_interval: How often to check if focus is maintained (in seconds)
        """
        self.window_name = window_name
        self.check_interval = check_interval
        self.window_handle = None
        self.running = False
        self.focus_thread = None
        self.last_focus_time = 0
        self.focus_attempts = 0
        self.focus_successful = False
        self.on_focus_lost_callback = None
        self.on_focus_restored_callback = None
        
    def find_window(self) -> bool:
        """
        Find the game window by name.
        
        Returns:
            True if window was found, False otherwise
        """
        try:
            # Find the window by name
            self.window_handle = win32gui.FindWindow(None, self.window_name)
            
            if self.window_handle:
                logger.info(f"Found game window with handle {self.window_handle}")
                return True
            else:
                logger.warning(f"Could not find window with name '{self.window_name}'")
                return False
        except Exception as e:
            logger.error(f"Error finding window: {str(e)}")
            return False
            
    def set_focus_callbacks(self, 
                           on_focus_lost: Optional[Callable] = None, 
                           on_focus_restored: Optional[Callable] = None):
        """
        Set callback functions for focus events.
        
        Args:
            on_focus_lost: Function to call when focus is lost
            on_focus_restored: Function to call when focus is restored
        """
        self.on_focus_lost_callback = on_focus_lost
        self.on_focus_restored_callback = on_focus_restored
            
    def ensure_focus(self, force_topmost: bool = True) -> bool:
        """
        Ensure the game window has focus.
        
        Args:
            force_topmost: Whether to force the window to be topmost
            
        Returns:
            True if successful, False otherwise
        """
        if not self.window_handle:
            if not self.find_window():
                return False
        
        try:
            # Record focus attempt
            self.focus_attempts += 1
            current_time = time.time()
            
            # Try to activate the window in multiple ways for maximum reliability
            
            # Method 1: Check if the window is minimized and restore it
            if win32gui.IsIconic(self.window_handle):
                logger.info("Window is minimized, restoring...")
                win32gui.ShowWindow(self.window_handle, win32con.SW_RESTORE)
                time.sleep(0.5)  # Give time for the window to restore
            
            # Method 2: Make sure window is visible
            win32gui.ShowWindow(self.window_handle, win32con.SW_SHOW)
            time.sleep(0.1)
            
            # Method 3: Bring the window to the foreground
            logger.debug("Setting foreground window...")
            win32gui.SetForegroundWindow(self.window_handle)
            time.sleep(0.2)
            
            # Method 4: Alt+Tab simulation to focus the window
            try:
                import pyautogui
                # Press Alt
                pyautogui.keyDown('alt')
                # Press Tab
                pyautogui.press('tab')
                # Release Alt
                pyautogui.keyUp('alt')
                time.sleep(0.2)
            except Exception as e:
                logger.warning(f"Alt+Tab simulation failed: {str(e)}")
            
            if force_topmost:
                # Method 5: Force window to be active and on top
                logger.debug("Setting window position to be on top...")
                # First, get the current window position
                rect = win32gui.GetWindowRect(self.window_handle)
                x, y, right, bottom = rect
                width = right - x
                height = bottom - y
                
                # Force window to be topmost
                win32gui.SetWindowPos(
                    self.window_handle,
                    win32con.HWND_TOPMOST,
                    x, y, width, height,  # Keep same position and size
                    win32con.SWP_SHOWWINDOW
                )
                
                # After a short delay, allow other windows to go on top again (but keep focus)
                time.sleep(0.2)
                win32gui.SetWindowPos(
                    self.window_handle,
                    win32con.HWND_NOTOPMOST,
                    x, y, width, height,  # Keep same position and size
                    win32con.SWP_SHOWWINDOW
                )
            
            # Method 6: Another variant of SetForegroundWindow which sometimes works better
            try:
                # Importing here to avoid dependency issues
                import win32api
                import win32process
                
                # Get thread/process info to use with this technique
                current_thread = win32api.GetCurrentThreadId()
                remote_thread, remote_process = win32process.GetWindowThreadProcessId(self.window_handle)
                
                # Attach threads to help with focus stealing prevention
                win32process.AttachThreadInput(current_thread, remote_thread, True)
                # Set focus and activate
                win32gui.SetFocus(self.window_handle)
                win32gui.SetActiveWindow(self.window_handle)
                win32gui.SetForegroundWindow(self.window_handle)
                # Detach threads
                win32process.AttachThreadInput(current_thread, remote_thread, False)
            except Exception as e:
                logger.warning(f"Advanced focus technique failed: {str(e)}")
            
            # Wait longer to ensure the window is focused
            time.sleep(0.5)
            
            # Verify focus was achieved
            was_focused = self.is_window_focused()
            if was_focused:
                self.last_focus_time = current_time
                self.focus_successful = True
                
                # Call the callback if focus was restored after being lost
                if self.on_focus_restored_callback and not self.is_window_focused():
                    self.on_focus_restored_callback()
                    
                logger.debug("Successfully focused game window")
            else:
                logger.warning("Failed to ensure window focus")
                
                # Call the callback if focus was lost
                if self.on_focus_lost_callback and self.focus_successful:
                    self.on_focus_lost_callback()
                    
            return was_focused
        except Exception as e:
            logger.error(f"Error ensuring focus: {str(e)}")
            return False
            
    def is_window_focused(self) -> bool:
        """
        Check if the game window currently has focus.
        
        Returns:
            True if the window has focus, False otherwise
        """
        try:
            # Get the handle of the currently active window
            active_window = win32gui.GetForegroundWindow()
            
            # If we haven't found our window yet, try to find it
            if not self.window_handle:
                if not self.find_window():
                    return False
            
            # Check if the active window is our game window
            return active_window == self.window_handle
        except Exception as e:
            logger.error(f"Error checking window focus: {str(e)}")
            return False
            
    def _focus_monitor_thread(self):
        """
        Thread that monitors and maintains window focus.
        """
        logger.info("Starting focus monitor thread")
        self.running = True
        last_focus_status = False
        
        while self.running:
            try:
                # Check if window is focused
                is_focused = self.is_window_focused()
                
                # If focus changed, log it
                if is_focused != last_focus_status:
                    if is_focused:
                        logger.info("Game window gained focus")
                        if self.on_focus_restored_callback:
                            self.on_focus_restored_callback()
                    else:
                        logger.info("Game window lost focus")
                        if self.on_focus_lost_callback:
                            self.on_focus_lost_callback()
                
                # If not focused, attempt to refocus
                if not is_focused:
                    logger.debug("Window not focused, attempting to regain focus")
                    self.ensure_focus()
                
                last_focus_status = is_focused
                
                # Sleep for the check interval
                time.sleep(self.check_interval)
            except Exception as e:
                logger.error(f"Error in focus monitor thread: {str(e)}")
                time.sleep(1.0)  # Sleep a bit longer on error
                
        logger.info("Focus monitor thread stopped")
            
    def start_focus_monitoring(self):
        """
        Start a background thread to monitor and maintain window focus.
        """
        if self.focus_thread and self.focus_thread.is_alive():
            logger.warning("Focus monitoring already running")
            return
            
        # Find the window first
        if not self.window_handle:
            self.find_window()
            
        # Start the monitoring thread
        self.focus_thread = threading.Thread(target=self._focus_monitor_thread, daemon=True)
        self.focus_thread.start()
        logger.info("Started focus monitoring")
            
    def stop_focus_monitoring(self):
        """
        Stop the focus monitoring thread.
        """
        self.running = False
        if self.focus_thread:
            self.focus_thread.join(timeout=1.0)
            logger.info("Stopped focus monitoring")
            
    def get_focus_stats(self):
        """
        Get statistics about focus maintenance.
        
        Returns:
            Dictionary with focus statistics
        """
        return {
            "focus_attempts": self.focus_attempts,
            "last_focus_time": self.last_focus_time,
            "focus_successful": self.focus_successful,
            "is_focused": self.is_window_focused() if self.window_handle else False,
        } 