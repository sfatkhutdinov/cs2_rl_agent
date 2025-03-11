import os
import time
import logging
import ctypes
from typing import Optional

# Import Windows-specific modules only on Windows
try:
    import win32gui
    import win32con
    import win32process
    WINDOWS_AVAILABLE = True
except ImportError:
    WINDOWS_AVAILABLE = False

def find_window_by_title(title_substring: str) -> Optional[int]:
    """
    Find a window by a substring in its title.
    
    Args:
        title_substring: Substring to search for in window titles
        
    Returns:
        Window handle if found, None otherwise
    """
    if not WINDOWS_AVAILABLE:
        return None
        
    result = []
    
    def callback(hwnd, _):
        if win32gui.IsWindowVisible(hwnd):
            window_title = win32gui.GetWindowText(hwnd)
            if title_substring.lower() in window_title.lower():
                result.append(hwnd)
                
    win32gui.EnumWindows(callback, None)
    return result[0] if result else None

def focus_window(window_handle: int) -> bool:
    """
    Focus on a window by its handle.
    
    Args:
        window_handle: Window handle to focus on
        
    Returns:
        True if successful, False otherwise
    """
    if not WINDOWS_AVAILABLE or not window_handle:
        return False
        
    try:
        # Bring window to foreground
        win32gui.ShowWindow(window_handle, win32con.SW_RESTORE)
        win32gui.SetForegroundWindow(window_handle)
        
        # Simulate Alt key press/release to ensure focus
        ctypes.windll.user32.keybd_event(0x12, 0, 0, 0)  # Alt press
        ctypes.windll.user32.keybd_event(0x12, 0, 2, 0)  # Alt release
        
        # Wait a moment for the window to gain focus
        time.sleep(0.5)
        return True
    except Exception as e:
        logging.error(f"Error focusing window: {str(e)}")
        return False

def focus_game_window(game_name: str = "Cities: Skylines") -> bool:
    """
    Find and focus on the game window.
    
    Args:
        game_name: Name of the game window to search for
        
    Returns:
        True if successful, False otherwise
    """
    window_handle = find_window_by_title(game_name)
    if window_handle:
        return focus_window(window_handle)
    
    # If not found with the primary name, try alternatives
    alternative_names = ["Cities", "Skylines", "Colossal Order"]
    for alt_name in alternative_names:
        window_handle = find_window_by_title(alt_name)
        if window_handle:
            return focus_window(window_handle)
    
    logging.warning(f"Game window '{game_name}' not found")
    return False

def is_game_window_focused(game_name: str = "Cities: Skylines") -> bool:
    """
    Check if the game window is currently focused.
    
    Args:
        game_name: Name of the game window to search for
        
    Returns:
        True if the game window is focused, False otherwise
    """
    if not WINDOWS_AVAILABLE:
        return False
        
    try:
        # Get the currently active window
        active_window = win32gui.GetForegroundWindow()
        if active_window == 0:
            return False
            
        # Check if it's the game window
        window_title = win32gui.GetWindowText(active_window)
        return game_name.lower() in window_title.lower() or any(
            alt.lower() in window_title.lower() 
            for alt in ["Cities", "Skylines", "Colossal Order"]
        )
    except:
        return False

def refocus_if_needed(game_name: str = "Cities: Skylines") -> bool:
    """
    Refocus on the game window if it's not currently focused.
    
    Args:
        game_name: Name of the game window to search for
        
    Returns:
        True if the game window is now focused, False otherwise
    """
    if not is_game_window_focused(game_name):
        return focus_game_window(game_name)
    return True 