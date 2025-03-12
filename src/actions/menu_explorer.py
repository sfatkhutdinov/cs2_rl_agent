"""
Menu exploration for the CS2 reinforcement learning agent.
"""

import logging
import random
import time
from typing import Dict, Any, List, Tuple, Optional

import pyautogui

from src.actions.action_handler import ActionHandler, Action, ActionType


class MenuExplorer:
    """
    Explores game menus and UI elements to discover functionality.
    """
    
    def __init__(self, window_manager=None, vision_interface=None):
        """
        Initialize the menu explorer.
        
        Args:
            window_manager: Window manager for handling window focus
            vision_interface: Vision interface for analyzing UI
        """
        self.logger = logging.getLogger(self.__class__.__name__)
        self.window_manager = window_manager
        self.vision_interface = vision_interface
        
        # Tracking discovered UI elements
        self.discovered_elements = {}
        self.exploration_attempts = 0
        self.successful_explorations = 0
        
    def explore_random_menu(self) -> Dict[str, Any]:
        """
        Explore a random menu in the game.
        
        Returns:
            Dict with exploration results
        """
        self.logger.info("Exploring random menu")
        self.exploration_attempts += 1
        
        try:
            # Ensure game window is focused
            if self.window_manager and not self.window_manager.is_window_focused():
                self.window_manager.focus_window()
                time.sleep(0.5)
            
            # Get current screen state before exploration
            initial_screen = None
            if self.vision_interface:
                initial_screen = self.vision_interface.capture_screen()
            
            # Perform a sequence of random menu actions
            actions_performed = []
            elements_found = []
            
            # First, try to open a menu with ESC key
            pyautogui.press('escape')
            time.sleep(0.8)  # Wait for menu to appear
            actions_performed.append("open_menu")
            
            # Try a few random interactions
            for _ in range(3):
                # Choose a random spot to click
                screen_width, screen_height = pyautogui.size()
                x = random.randint(100, screen_width - 100)
                y = random.randint(100, screen_height - 100)
                
                # Move to the spot and click
                pyautogui.moveTo(x, y, duration=0.3)
                time.sleep(0.2)
                pyautogui.click()
                time.sleep(0.8)  # Wait to see what happens
                
                actions_performed.append(f"click_at_{x}_{y}")
                
                # If we have vision, try to detect UI elements
                if self.vision_interface:
                    ui_elements = self._detect_ui_elements()
                    for element in ui_elements:
                        if element not in elements_found:
                            elements_found.append(element)
            
            # Close menu with ESC key
            pyautogui.press('escape')
            time.sleep(0.5)
            actions_performed.append("close_menu")
            
            # Check if we found anything new
            success = len(elements_found) > 0
            if success:
                self.successful_explorations += 1
            
            # Record result
            result = {
                "success": success,
                "actions": actions_performed,
                "elements": elements_found,
                "attempt_number": self.exploration_attempts
            }
            
            self.logger.info(f"Menu exploration complete. Success: {success}, Elements found: {len(elements_found)}")
            return result
            
        except Exception as e:
            self.logger.error(f"Error exploring menu: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "attempt_number": self.exploration_attempts
            }
    
    def explore_specific_menu(self, menu_name: str) -> Dict[str, Any]:
        """
        Explore a specific menu by name.
        
        Args:
            menu_name: Name of the menu to explore
            
        Returns:
            Dict with exploration results
        """
        self.logger.info(f"Exploring specific menu: {menu_name}")
        self.exploration_attempts += 1
        
        try:
            # Ensure game window is focused
            if self.window_manager and not self.window_manager.is_window_focused():
                self.window_manager.focus_window()
                time.sleep(0.5)
            
            # Map common menu names to key shortcuts
            menu_shortcuts = {
                "main": "escape",
                "budget": "b",
                "roads": "r",
                "zoning": "z",
                "services": "s",
                "water": "w",
                "electricity": "e",
                "info": "i"
            }
            
            # Try to open the menu using shortcut or general approach
            menu_opened = False
            if menu_name.lower() in menu_shortcuts:
                pyautogui.press(menu_shortcuts[menu_name.lower()])
                time.sleep(0.8)
                menu_opened = True
            else:
                # Try to open main menu and then navigate
                pyautogui.press('escape')
                time.sleep(0.8)
                menu_opened = True
            
            if not menu_opened:
                return {
                    "success": False,
                    "error": f"Could not open menu: {menu_name}",
                    "attempt_number": self.exploration_attempts
                }
            
            # Perform a few exploratory actions in the menu
            actions_performed = [f"open_menu_{menu_name}"]
            elements_found = []
            
            # Try a few interactions in the menu
            for _ in range(3):
                # If we have vision, try to find clickable elements
                clickable_elements = self._find_clickable_elements()
                
                if clickable_elements:
                    # Click on a random clickable element
                    element = random.choice(clickable_elements)
                    x, y = element["position"]
                    
                    # Move to the element and click
                    pyautogui.moveTo(x, y, duration=0.3)
                    time.sleep(0.2)
                    pyautogui.click()
                    time.sleep(0.8)
                    
                    actions_performed.append(f"click_element_{element['name']}")
                    
                    # Add the element to our found list if not already there
                    if element not in elements_found:
                        elements_found.append(element)
                else:
                    # If no elements found, just click a random spot
                    screen_width, screen_height = pyautogui.size()
                    x = random.randint(100, screen_width - 100)
                    y = random.randint(100, screen_height - 100)
                    
                    # Move to the spot and click
                    pyautogui.moveTo(x, y, duration=0.3)
                    time.sleep(0.2)
                    pyautogui.click()
                    time.sleep(0.8)
                    
                    actions_performed.append(f"click_at_{x}_{y}")
            
            # Close menu with ESC key
            pyautogui.press('escape')
            time.sleep(0.5)
            actions_performed.append("close_menu")
            
            # Check if we found anything new
            success = len(elements_found) > 0
            if success:
                self.successful_explorations += 1
            
            # Record result
            result = {
                "success": success,
                "actions": actions_performed,
                "elements": elements_found,
                "attempt_number": self.exploration_attempts,
                "menu_name": menu_name
            }
            
            self.logger.info(f"Menu exploration complete. Menu: {menu_name}, Success: {success}, Elements found: {len(elements_found)}")
            return result
            
        except Exception as e:
            self.logger.error(f"Error exploring menu {menu_name}: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "attempt_number": self.exploration_attempts,
                "menu_name": menu_name
            }
    
    def _detect_ui_elements(self) -> List[Dict[str, Any]]:
        """
        Detect UI elements in the current screen using vision.
        
        Returns:
            List of detected UI elements
        """
        if not self.vision_interface:
            return []
            
        try:
            # Capture current screen
            screen = self.vision_interface.capture_screen()
            if screen is None:
                return []
                
            # Use vision to identify UI elements
            if hasattr(self.vision_interface, 'detect_ui_elements_with_vision'):
                ui_elements = self.vision_interface.detect_ui_elements_with_vision()
                return [
                    {
                        "name": element.get("name", "unknown"),
                        "type": element.get("type", "button"),
                        "position": element.get("position", (0, 0)),
                        "confidence": element.get("confidence", 0.0)
                    }
                    for element in ui_elements.values()
                    if element.get("confidence", 0.0) > 0.5  # Filter by confidence
                ]
            
            return []
            
        except Exception as e:
            self.logger.error(f"Error detecting UI elements: {str(e)}")
            return []
    
    def _find_clickable_elements(self) -> List[Dict[str, Any]]:
        """
        Find clickable elements in the current screen.
        
        Returns:
            List of clickable elements with positions
        """
        # If we have vision, use it to find elements
        if self.vision_interface:
            return self._detect_ui_elements()
            
        # Otherwise, just return some standard positions to try
        screen_width, screen_height = pyautogui.size()
        
        # Common UI locations in strategy games like Cities: Skylines
        common_positions = [
            {"name": "top_left", "position": (50, 50), "type": "button"},
            {"name": "top_right", "position": (screen_width - 50, 50), "type": "button"},
            {"name": "bottom_left", "position": (50, screen_height - 50), "type": "button"},
            {"name": "bottom_right", "position": (screen_width - 50, screen_height - 50), "type": "button"},
            {"name": "center_top", "position": (screen_width // 2, 100), "type": "button"},
            {"name": "center", "position": (screen_width // 2, screen_height // 2), "type": "button"}
        ]
        
        return common_positions
    
    def save_discovered_elements(self, filepath: str) -> bool:
        """
        Save discovered UI elements to a file.
        
        Args:
            filepath: Path to save the elements to
            
        Returns:
            True if successful, False otherwise
        """
        try:
            import json
            
            with open(filepath, 'w') as f:
                json.dump({
                    "discovered_elements": self.discovered_elements,
                    "exploration_attempts": self.exploration_attempts,
                    "successful_explorations": self.successful_explorations,
                    "timestamp": time.time()
                }, f, indent=2)
                
            self.logger.info(f"Saved discovered elements to {filepath}")
            return True
        except Exception as e:
            self.logger.error(f"Error saving discovered elements: {str(e)}")
            return False
    
    def load_discovered_elements(self, filepath: str) -> bool:
        """
        Load discovered UI elements from a file.
        
        Args:
            filepath: Path to load the elements from
            
        Returns:
            True if successful, False otherwise
        """
        try:
            import json
            import os
            
            if not os.path.exists(filepath):
                self.logger.warning(f"File not found: {filepath}")
                return False
                
            with open(filepath, 'r') as f:
                data = json.load(f)
                
                if "discovered_elements" in data:
                    self.discovered_elements = data["discovered_elements"]
                if "exploration_attempts" in data:
                    self.exploration_attempts = data["exploration_attempts"]
                if "successful_explorations" in data:
                    self.successful_explorations = data["successful_explorations"]
                    
            self.logger.info(f"Loaded discovered elements from {filepath}")
            return True
        except Exception as e:
            self.logger.error(f"Error loading discovered elements: {str(e)}")
            return False 