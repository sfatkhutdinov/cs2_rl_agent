"""
Window management utilities for interacting with the game window.
"""

import logging
import time
import pyautogui
import win32gui
import win32con
import win32process
import psutil
import os
from typing import Tuple, Optional, List

logger = logging.getLogger(__name__)


class WindowManager:
    """
    Manages window focus and interactions with the game window.
    """
    
    def __init__(self, window_name: str = "Cities: Skylines II"):
        """
        Initialize the window manager.
        
        Args:
            window_name: Name of the game window to manage
        """
        self.window_name = window_name
        self.window_handle = None
        self.game_pid = None
        self.screen_resolution = self._get_screen_resolution()
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Try to find the window at initialization
        self.find_window()
        
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
                # Get the process ID associated with the window
                _, self.game_pid = win32process.GetWindowThreadProcessId(self.window_handle)
                self.logger.info(f"Found game window with handle {self.window_handle} and PID {self.game_pid}")
                return True
            else:
                self.logger.warning(f"Could not find window with name '{self.window_name}'")
                return False
        except Exception as e:
            self.logger.error(f"Error finding window: {str(e)}")
            return False
    
    def focus_window(self) -> bool:
        """
        Bring the game window into focus.
        
        Returns:
            True if successful, False otherwise
        """
        if not self.window_handle:
            if not self.find_window():
                return False
        
        try:
            # Check if the window is minimized
            if win32gui.IsIconic(self.window_handle):
                # Restore the window if it's minimized
                win32gui.ShowWindow(self.window_handle, win32con.SW_RESTORE)
                time.sleep(0.5)  # Give time for the window to restore
            
            # Bring the window to the foreground
            win32gui.SetForegroundWindow(self.window_handle)
            
            # Wait a bit to ensure the window is focused
            time.sleep(0.2)
            
            # Verify focus was achieved
            return self.is_window_focused()
        except Exception as e:
            self.logger.error(f"Error focusing window: {str(e)}")
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
            self.logger.error(f"Error checking window focus: {str(e)}")
            return False
    
    def get_window_rect(self) -> Optional[Tuple[int, int, int, int]]:
        """
        Get the rectangle coordinates of the game window.
        
        Returns:
            Tuple of (left, top, right, bottom) or None if window not found
        """
        if not self.window_handle:
            if not self.find_window():
                return None
        
        try:
            # Get the window rectangle
            return win32gui.GetWindowRect(self.window_handle)
        except Exception as e:
            self.logger.error(f"Error getting window rectangle: {str(e)}")
            return None
    
    def get_client_rect(self) -> Optional[Tuple[int, int, int, int]]:
        """
        Get the client area rectangle coordinates of the game window.
        
        Returns:
            Tuple of (left, top, right, bottom) or None if window not found
        """
        if not self.window_handle:
            if not self.find_window():
                return None
        
        try:
            # Get the client rectangle
            client_rect = win32gui.GetClientRect(self.window_handle)
            left, top = win32gui.ClientToScreen(self.window_handle, (0, 0))
            right, bottom = win32gui.ClientToScreen(self.window_handle, (client_rect[2], client_rect[3]))
            return (left, top, right, bottom)
        except Exception as e:
            self.logger.error(f"Error getting client rectangle: {str(e)}")
            return None
    
    def _get_screen_resolution(self) -> Tuple[int, int]:
        """
        Get the screen resolution.
        
        Returns:
            Tuple of (width, height)
        """
        try:
            width, height = pyautogui.size()
            return (width, height)
        except Exception as e:
            logger.error(f"Error getting screen resolution: {str(e)}")
            return (1920, 1080)  # Default fallback resolution
    
    def is_game_running(self) -> bool:
        """
        Check if the game process is running.
        
        Returns:
            True if the game is running, False otherwise
        """
        if not self.game_pid:
            if not self.find_window():
                return False
        
        try:
            # Check if the process is running
            return psutil.pid_exists(self.game_pid)
        except Exception as e:
            self.logger.error(f"Error checking if game is running: {str(e)}")
            return False
    
    def get_window_position(self) -> Optional[Tuple[int, int]]:
        """
        Get the position of the top-left corner of the window.
        
        Returns:
            Tuple of (x, y) or None if window not found
        """
        rect = self.get_window_rect()
        if rect:
            return (rect[0], rect[1])
        return None
    
    def get_window_size(self) -> Optional[Tuple[int, int]]:
        """
        Get the size of the window.
        
        Returns:
            Tuple of (width, height) or None if window not found
        """
        rect = self.get_window_rect()
        if rect:
            return (rect[2] - rect[0], rect[3] - rect[1])
        return None
    
    def move_window(self, x: int, y: int, width: int, height: int) -> bool:
        """
        Move and resize the window.
        
        Args:
            x: Left position
            y: Top position
            width: Window width
            height: Window height
            
        Returns:
            True if successful, False otherwise
        """
        if not self.window_handle:
            if not self.find_window():
                return False
        
        try:
            # Move and resize the window
            win32gui.MoveWindow(self.window_handle, x, y, width, height, True)
            return True
        except Exception as e:
            self.logger.error(f"Error moving window: {str(e)}")
            return False
    
    def close(self):
        """
        Release any resources and clean up.
        """
        self.window_handle = None
        self.game_pid = None 