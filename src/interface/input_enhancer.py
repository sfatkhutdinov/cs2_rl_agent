"""
Enhanced input methods that more closely mimic human input patterns.
This helps with games that may detect and ignore automated input methods.
"""

import time
import random
import logging
import win32api
import win32con
import pyautogui
import numpy as np
from typing import Tuple, Optional, Union, List

# Configure logging
logger = logging.getLogger("InputEnhancer")

class InputEnhancer:
    """
    Enhanced input methods to better mimic human input and ensure game responsiveness.
    """
    
    def __init__(self, min_delay: float = 0.05, max_delay: float = 0.15):
        """
        Initialize the input enhancer.
        
        Args:
            min_delay: Minimum delay between actions (seconds)
            max_delay: Maximum delay between actions (seconds)
        """
        self.min_delay = min_delay
        self.max_delay = max_delay
        self.last_action_time = 0
        
        # Configure pyautogui for safety
        pyautogui.PAUSE = 0.1  # Default pause between pyautogui commands
        pyautogui.FAILSAFE = True  # Enable failsafe
        
    def _human_delay(self):
        """Add a small, random delay to simulate human reaction time."""
        delay = random.uniform(self.min_delay, self.max_delay)
        time.sleep(delay)
        
    def _get_random_offset(self, radius: int = 3) -> Tuple[int, int]:
        """
        Generate a small random offset to make clicks look more human.
        
        Args:
            radius: Maximum pixel radius for the offset
            
        Returns:
            (x_offset, y_offset) tuple
        """
        x_offset = random.randint(-radius, radius)
        y_offset = random.randint(-radius, radius)
        return (x_offset, y_offset)
        
    def _enhance_movement(self, 
                         start_x: int, start_y: int, 
                         end_x: int, end_y: int, 
                         duration: float = 0.5):
        """
        Move the mouse in a more human-like pattern.
        
        Args:
            start_x: Starting X position
            start_y: Starting Y position
            end_x: Ending X position
            end_y: Ending Y position
            duration: Total duration of the movement
        """
        # Calculate distance
        distance = ((end_x - start_x) ** 2 + (end_y - start_y) ** 2) ** 0.5
        
        # For short distances, use direct movement
        if distance < 100:
            pyautogui.moveTo(end_x, end_y, duration=duration * 0.8)
            return
            
        # For longer distances, use a slightly curved path with variable speed
        # Number of steps depending on distance
        steps = max(10, min(20, int(distance / 20)))
        
        # Create a slightly curved path
        # Calculate a midpoint with a slight offset
        midx = (start_x + end_x) / 2
        midy = (start_y + end_y) / 2
        
        # Add a random offset to create a curve
        curve_offset = random.uniform(0.1, 0.3) * distance / 5
        if random.random() > 0.5:
            # Curve to the right
            midx += curve_offset
        else:
            # Curve to the left
            midx -= curve_offset
            
        # Create a quadratic bezier curve
        t_values = np.linspace(0, 1, steps)
        x_values = [(1-t)**2 * start_x + 2*(1-t)*t * midx + t**2 * end_x for t in t_values]
        y_values = [(1-t)**2 * start_y + 2*(1-t)*t * midy + t**2 * end_y for t in t_values]
        
        # Move along the curve with variable speed (slower at start and end)
        step_durations = []
        for i in range(len(t_values) - 1):
            # Slow at start and end, faster in the middle
            if i < steps / 4:
                # Acceleration phase
                factor = 0.7 + 0.6 * (i / (steps / 4))
            elif i > 3 * steps / 4:
                # Deceleration phase
                factor = 0.7 + 0.6 * ((steps - i) / (steps / 4))
            else:
                # Constant speed phase
                factor = 1.3
                
            step_durations.append(duration / steps * factor)
            
        # Execute the movement
        current_time = time.time()
        for i in range(len(x_values) - 1):
            x, y = x_values[i], y_values[i]
            next_x, next_y = x_values[i+1], y_values[i+1]
            step_duration = step_durations[i]
            
            pyautogui.moveTo(next_x, next_y, duration=step_duration)
            
        # Ensure we end exactly at the target position
        pyautogui.moveTo(end_x, end_y, duration=0.05)
        
    def move_to(self, x: int, y: int, duration: Optional[float] = None):
        """
        Move the mouse to a position with human-like movement.
        
        Args:
            x: Target X position
            y: Target Y position
            duration: Duration of the movement (if None, calculated based on distance)
        """
        # Get current position
        current_x, current_y = pyautogui.position()
        
        # Calculate distance
        distance = ((x - current_x) ** 2 + (y - current_y) ** 2) ** 0.5
        
        # Calculate appropriate duration if not specified
        if duration is None:
            if distance < 100:
                duration = random.uniform(0.1, 0.2)
            elif distance < 300:
                duration = random.uniform(0.2, 0.3)
            else:
                duration = random.uniform(0.3, 0.5)
        
        # Add small human delay before moving
        self._human_delay()
        
        # Add a small random offset to the target
        offset_x, offset_y = self._get_random_offset(2)
        target_x = x + offset_x
        target_y = y + offset_y
        
        # Perform the enhanced movement
        self._enhance_movement(current_x, current_y, target_x, target_y, duration)
        
        # Record last action time
        self.last_action_time = time.time()
        
    def click(self, 
             x: Optional[int] = None, 
             y: Optional[int] = None, 
             button: str = 'left', 
             clicks: int = 1,
             interval: float = 0.1):
        """
        Perform a click with human-like behavior.
        
        Args:
            x: X position (if None, use current position)
            y: Y position (if None, use current position)
            button: Mouse button ('left', 'right', 'middle')
            clicks: Number of clicks
            interval: Interval between clicks
        """
        # First, move to the position if specified
        if x is not None and y is not None:
            self.move_to(x, y)
            
        # Get current position
        current_x, current_y = pyautogui.position()
        
        # Map button string to win32 constant
        if button == 'left':
            down_event = win32con.MOUSEEVENTF_LEFTDOWN
            up_event = win32con.MOUSEEVENTF_LEFTUP
        elif button == 'right':
            down_event = win32con.MOUSEEVENTF_RIGHTDOWN
            up_event = win32con.MOUSEEVENTF_RIGHTUP
        elif button == 'middle':
            down_event = win32con.MOUSEEVENTF_MIDDLEDOWN
            up_event = win32con.MOUSEEVENTF_MIDDLEUP
        else:
            logger.error(f"Invalid button: {button}")
            return
            
        # Perform the clicks
        for i in range(clicks):
            # Down event
            win32api.mouse_event(down_event, 0, 0, 0, 0)
            
            # Random delay between down and up
            time.sleep(random.uniform(0.01, 0.03))
            
            # Up event
            win32api.mouse_event(up_event, 0, 0, 0, 0)
            
            # Wait interval between clicks if more than one
            if i < clicks - 1:
                time.sleep(interval)
                
        # Record last action time
        self.last_action_time = time.time()
        
        # Add a small delay after clicking
        self._human_delay()
        
    def double_click(self, x: Optional[int] = None, y: Optional[int] = None):
        """
        Perform a double click with proper timing.
        
        Args:
            x: X position (if None, use current position)
            y: Y position (if None, use current position)
        """
        self.click(x, y, clicks=2, interval=0.1)
        
    def right_click(self, x: Optional[int] = None, y: Optional[int] = None):
        """
        Perform a right click.
        
        Args:
            x: X position (if None, use current position)
            y: Y position (if None, use current position)
        """
        self.click(x, y, button='right')
        
    def press_key(self, key: str, presses: int = 1, interval: float = 0.1):
        """
        Press a keyboard key with human-like timing.
        
        Args:
            key: Key to press
            presses: Number of presses
            interval: Interval between presses
        """
        for i in range(presses):
            # Add small human delay before pressing
            self._human_delay()
            
            # Press the key
            pyautogui.keyDown(key)
            time.sleep(random.uniform(0.01, 0.03))  # Random hold time
            pyautogui.keyUp(key)
            
            # Wait interval between presses if more than one
            if i < presses - 1:
                time.sleep(interval)
                
        # Record last action time
        self.last_action_time = time.time()
        
    def press_keys(self, keys: List[str]):
        """
        Press multiple keys in sequence.
        
        Args:
            keys: List of keys to press
        """
        for key in keys:
            self.press_key(key)
            time.sleep(random.uniform(0.05, 0.1))
            
    def hotkey(self, *args):
        """
        Press a hotkey combination.
        
        Args:
            *args: Keys to press in combination
        """
        # Add small human delay before pressing
        self._human_delay()
        
        # Use pyautogui's hotkey function which handles key combinations correctly
        pyautogui.hotkey(*args)
        
        # Record last action time
        self.last_action_time = time.time()
        
    def drag(self, start_x: int, start_y: int, end_x: int, end_y: int, duration: float = 0.5, button: str = 'left'):
        """
        Perform a drag operation with human-like movement.
        
        Args:
            start_x: Starting X position
            start_y: Starting Y position
            end_x: Ending X position
            end_y: Ending Y position
            duration: Duration of the drag
            button: Mouse button to use ('left', 'right', 'middle')
        """
        # Move to start position
        self.move_to(start_x, start_y)
        
        # Map button string to win32 constant
        if button == 'left':
            down_event = win32con.MOUSEEVENTF_LEFTDOWN
            up_event = win32con.MOUSEEVENTF_LEFTUP
        elif button == 'right':
            down_event = win32con.MOUSEEVENTF_RIGHTDOWN
            up_event = win32con.MOUSEEVENTF_RIGHTUP
        elif button == 'middle':
            down_event = win32con.MOUSEEVENTF_MIDDLEDOWN
            up_event = win32con.MOUSEEVENTF_MIDDLEUP
        else:
            logger.error(f"Invalid button: {button}")
            return
            
        # Press the mouse button
        win32api.mouse_event(down_event, 0, 0, 0, 0)
        
        # Wait a small amount of time before starting to drag
        time.sleep(random.uniform(0.05, 0.1))
        
        # Move to end position with human-like movement
        self._enhance_movement(start_x, start_y, end_x, end_y, duration)
        
        # Wait before releasing
        time.sleep(random.uniform(0.05, 0.1))
        
        # Release the mouse button
        win32api.mouse_event(up_event, 0, 0, 0, 0)
        
        # Record last action time
        self.last_action_time = time.time()
        
    def scroll(self, clicks: int):
        """
        Scroll the mouse wheel.
        
        Args:
            clicks: Number of clicks (positive for up, negative for down)
        """
        # Add small human delay before scrolling
        self._human_delay()
        
        # Perform the scroll in small increments for more natural motion
        increment = 1 if clicks > 0 else -1
        remaining = abs(clicks)
        
        while remaining > 0:
            # Decide how many clicks to do in this batch
            batch = min(remaining, random.randint(1, 3))
            remaining -= batch
            
            # Scroll the batch
            pyautogui.scroll(batch * increment)
            
            # Small random delay between batches
            if remaining > 0:
                time.sleep(random.uniform(0.05, 0.1))
                
        # Record last action time
        self.last_action_time = time.time()
        
    def wait_after_action(self, min_wait: float = 0.3, max_wait: float = 0.7):
        """
        Wait a random amount of time after an action.
        This helps ensure the game has processed the input.
        
        Args:
            min_wait: Minimum wait time
            max_wait: Maximum wait time
        """
        time.sleep(random.uniform(min_wait, max_wait)) 