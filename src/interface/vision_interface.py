import time
import numpy as np
import cv2
import pytesseract
import pyautogui
import mss
from typing import Dict, Any, Tuple, List, Optional
import logging
from .base_interface import BaseInterface
import os


class VisionInterface(BaseInterface):
    """
    Vision-based interface for interacting with Cities: Skylines 2.
    
    This interface uses computer vision techniques (screen capture, OCR) to interact
    with the game when no direct API is available.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the vision interface.
        
        Args:
            config: Configuration dictionary
        """
        super().__init__(config)
        self.logger = logging.getLogger("VisionInterface")
        
        # Windows-specific setup for Tesseract
        if os.name == 'nt':  # Windows
            try:
                # Try to use Tesseract from PATH
                pytesseract.pytesseract.tesseract_cmd = 'tesseract'
            except Exception:
                # If not in PATH, try common installation locations
                possible_paths = [
                    r'C:\Program Files\Tesseract-OCR\tesseract.exe',
                    r'C:\Program Files (x86)\Tesseract-OCR\tesseract.exe'
                ]
                for path in possible_paths:
                    if os.path.exists(path):
                        pytesseract.pytesseract.tesseract_cmd = path
                        self.logger.info(f"Found Tesseract at: {path}")
                        break
                else:
                    self.logger.warning("Tesseract not found in common locations. OCR will not work.")
        
        # Set up screen capture
        self.sct = mss.mss()
        self.screen_region = tuple(config["interface"]["vision"]["screen_region"])
        self.monitor = {"top": self.screen_region[0], 
                       "left": self.screen_region[1], 
                       "width": self.screen_region[2], 
                       "height": self.screen_region[3]}
        
        # OCR settings
        self.ocr_confidence = config["interface"]["vision"]["ocr_confidence"]
        
        # UI element locations - Default values for 1920x1080 resolution
        # These should be calibrated for your specific setup
        self.ui_elements = {
            # Top info bar elements
            "population": {"region": (100, 50, 200, 80)},
            "happiness": {"region": (300, 50, 400, 80)},
            "budget": {"region": (500, 50, 600, 80)},
            "traffic": {"region": (700, 50, 800, 80)},
            
            # Control panels - adjust these based on your UI layout
            "residential_zone": {"region": (100, 700, 150, 750)},
            "commercial_zone": {"region": (160, 700, 210, 750)},
            "industrial_zone": {"region": (220, 700, 270, 750)},
            "roads": {"region": (280, 700, 330, 750)},
            "power": {"region": (340, 700, 390, 750)},
            "water": {"region": (400, 700, 450, 750)},
            
            # Speed controls
            "speed_1": {"region": (800, 30, 820, 50)},
            "speed_2": {"region": (830, 30, 850, 50)},
            "speed_3": {"region": (860, 30, 880, 50)},
            
            # Menu buttons
            "menu": {"region": (20, 20, 50, 50)},
            "new_game": {"region": (100, 200, 300, 230)},
        }
        
        # Game state tracking
        self.last_metrics = {}
        self.last_screenshot = None
        self.game_speed = 1
    
    def connect(self) -> bool:
        """
        Connect to the game via screen capture.
        
        Returns:
            True if connection was successful, False otherwise
        """
        try:
            # Take a screenshot to verify we can capture the screen
            screenshot = self.sct.grab(self.monitor)
            if screenshot is None:
                self.logger.error("Failed to capture screen.")
                return False
            
            self.last_screenshot = np.array(screenshot)
            
            # TODO: Verify that Cities: Skylines 2 is actually running
            # This would involve looking for specific UI elements or the game logo
            
            self.connected = True
            self.logger.info("Successfully connected to Cities: Skylines 2 via screen capture.")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to connect: {str(e)}")
            return False
    
    def disconnect(self) -> None:
        """Disconnect from the game."""
        self.sct.close()
        self.connected = False
        self.logger.info("Disconnected from the game.")
    
    def get_game_state(self) -> Dict[str, Any]:
        """
        Get the current game state.
        
        Returns:
            Dictionary containing the current game state
        """
        if not self.connected:
            self.logger.warning("Not connected to the game.")
            return {}
        
        # Capture visual observation
        visual_obs = self.get_visual_observation()
        
        # Get metrics
        metrics = self.get_metrics()
        
        return {
            "visual_observation": visual_obs,
            "metrics": metrics,
            "game_speed": self.game_speed,
            "timestamp": time.time()
        }
    
    def get_visual_observation(self) -> np.ndarray:
        """
        Get visual observation (screenshot) from the game.
        
        Returns:
            NumPy array containing the screenshot
        """
        if not self.connected:
            self.logger.warning("Not connected to the game.")
            return np.zeros((84, 84, 3), dtype=np.uint8)
        
        try:
            screenshot = self.sct.grab(self.monitor)
            self.last_screenshot = np.array(screenshot)
            
            # Resize to the specified dimensions in the config
            image_size = self.config["environment"]["observation_space"]["image_size"]
            resized_image = cv2.resize(self.last_screenshot, (image_size[1], image_size[0]))
            
            # Convert to grayscale if specified
            if self.config["environment"]["observation_space"]["grayscale"]:
                resized_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)
                resized_image = np.expand_dims(resized_image, axis=2)
            
            return resized_image
            
        except Exception as e:
            self.logger.error(f"Failed to get visual observation: {str(e)}")
            return np.zeros((84, 84, 3), dtype=np.uint8)
    
    def get_metrics(self) -> Dict[str, float]:
        """
        Get current game metrics (population, happiness, etc.) using OCR.
        
        Returns:
            Dictionary containing game metrics
        """
        if not self.connected:
            self.logger.warning("Not connected to the game.")
            return {}
        
        metrics = {}
        
        try:
            # Extract regions for each metric and apply OCR
            for metric_name, metadata in self.ui_elements.items():
                if metric_name in ["population", "happiness", "budget", "traffic"]:
                    region = metadata["region"]
                    sub_img = self.last_screenshot[region[1]:region[3], region[0]:region[2]]
                    
                    # Apply OCR
                    text = pytesseract.image_to_string(
                        sub_img, 
                        config=f'--psm 7 --oem 3 -c tessedit_char_whitelist=0123456789,.'
                    )
                    
                    # Parse the number
                    try:
                        # Remove non-numeric characters
                        text = ''.join(c for c in text if c.isdigit() or c == '.')
                        value = float(text) if text else 0.0
                        metrics[metric_name] = value
                    except ValueError:
                        self.logger.warning(f"Failed to parse value for {metric_name}: '{text}'")
                        metrics[metric_name] = 0.0
            
            # Store the metrics for later comparison
            self.last_metrics = metrics
            return metrics
            
        except Exception as e:
            self.logger.error(f"Failed to get metrics: {str(e)}")
            return {}
    
    def perform_action(self, action: Dict[str, Any]) -> bool:
        """
        Perform an action in the game by simulating mouse clicks and keyboard inputs.
        
        Args:
            action: Dictionary describing the action to perform
            
        Returns:
            True if the action was performed successfully, False otherwise
        """
        if not self.connected:
            self.logger.warning("Not connected to the game.")
            return False
        
        try:
            action_type = action.get("type", "")
            
            if action_type == "zone":
                # Handle zoning actions
                zone_type = action.get("zone_type", "")
                position = action.get("position", (0, 0))
                
                # Click on the zone tool
                zone_ui = self.ui_elements.get(f"{zone_type}_zone", None)
                if zone_ui is None:
                    self.logger.warning(f"Unknown zone type: {zone_type}")
                    return False
                
                region = zone_ui["region"]
                pyautogui.click(region[0] + (region[2] - region[0]) // 2, 
                               region[1] + (region[3] - region[1]) // 2)
                
                # Click on the map position
                map_x, map_y = position
                pyautogui.click(map_x, map_y)
                pyautogui.click(map_x, map_y)  # Double-click to confirm
                
            elif action_type == "infrastructure":
                # Handle infrastructure actions (roads, power, water, etc.)
                infra_type = action.get("infra_type", "")
                position = action.get("position", (0, 0))
                
                # Click on the infrastructure tool
                infra_ui = self.ui_elements.get(infra_type, None)
                if infra_ui is None:
                    self.logger.warning(f"Unknown infrastructure type: {infra_type}")
                    return False
                
                region = infra_ui["region"]
                pyautogui.click(region[0] + (region[2] - region[0]) // 2, 
                               region[1] + (region[3] - region[1]) // 2)
                
                # Click on the map position
                map_x, map_y = position
                pyautogui.click(map_x, map_y)
                
                # If it's a road or pipe, we need two points
                if infra_type in ["roads", "power", "water"]:
                    end_position = action.get("end_position", None)
                    if end_position:
                        end_x, end_y = end_position
                        pyautogui.click(end_x, end_y)
                
            elif action_type == "budget":
                # Handle budget adjustments - this would depend on the specific UI
                # This is a placeholder and would need to be adapted to the actual game UI
                budget_action = action.get("budget_action", "")
                # TODO: Implement budget adjustment actions
                
            else:
                self.logger.warning(f"Unknown action type: {action_type}")
                return False
            
            # Sleep briefly to allow the game to process the action
            time.sleep(0.5)
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to perform action: {str(e)}")
            return False
    
    def set_game_speed(self, speed: int) -> bool:
        """
        Set the game speed.
        
        Args:
            speed: Game speed (1-3)
            
        Returns:
            True if the speed was set successfully, False otherwise
        """
        if not self.connected:
            self.logger.warning("Not connected to the game.")
            return False
        
        if speed < 1 or speed > 3:
            self.logger.warning(f"Invalid game speed: {speed}. Must be between 1 and 3.")
            return False
        
        try:
            # Click on the speed button
            speed_ui = self.ui_elements.get(f"speed_{speed}", None)
            if speed_ui is None:
                self.logger.warning(f"No UI element defined for speed {speed}")
                return False
            
            region = speed_ui["region"]
            pyautogui.click(region[0] + (region[2] - region[0]) // 2, 
                           region[1] + (region[3] - region[1]) // 2)
            
            self.game_speed = speed
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to set game speed: {str(e)}")
            return False
    
    def is_game_over(self) -> bool:
        """
        Check if the game is over (e.g., city went bankrupt).
        
        Returns:
            True if the game is over, False otherwise
        """
        if not self.connected:
            self.logger.warning("Not connected to the game.")
            return True
        
        # Check for bankruptcy - this is a simple implementation that assumes
        # bankruptcy is indicated by a negative budget value
        metrics = self.get_metrics()
        if metrics.get("budget", 0) < 0:
            return True
        
        # TODO: Add more sophisticated game over detection
        
        return False
    
    def reset_game(self) -> bool:
        """
        Reset the game to its initial state by starting a new game.
        
        Returns:
            True if the game was reset successfully, False otherwise
        """
        if not self.connected:
            self.logger.warning("Not connected to the game.")
            return False
        
        try:
            # Click on the menu button
            menu_region = self.ui_elements["menu"]["region"]
            pyautogui.click(menu_region[0] + (menu_region[2] - menu_region[0]) // 2, 
                           menu_region[1] + (menu_region[3] - menu_region[1]) // 2)
            
            time.sleep(1)  # Wait for menu to open
            
            # Click on new game
            new_game_region = self.ui_elements["new_game"]["region"]
            pyautogui.click(new_game_region[0] + (new_game_region[2] - new_game_region[0]) // 2, 
                           new_game_region[1] + (new_game_region[3] - new_game_region[1]) // 2)
            
            time.sleep(5)  # Wait for the game to load
            
            # Reset internal state
            self.last_metrics = {}
            self.game_speed = 1
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to reset game: {str(e)}")
            return False 