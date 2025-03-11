import time
import numpy as np
import cv2
import pytesseract
import pyautogui
import mss
from typing import Dict, Any, Tuple, List, Optional
import logging
import os
from .base_interface import BaseInterface


class AutoVisionInterface(BaseInterface):
    """
    Automated vision-based interface for interacting with Cities: Skylines 2.
    
    This interface uses advanced computer vision techniques to automatically
    detect UI elements without requiring manual coordinate input.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the automated vision interface.
        
        Args:
            config: Configuration dictionary
        """
        super().__init__(config)
        self.logger = logging.getLogger("AutoVisionInterface")
        
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
        
        # Auto-detect screen resolution if not specified
        try:
            if "screen_region" in config["interface"]["vision"] and config["interface"]["vision"]["screen_region"]:
                # Ensure screen_region is a tuple or list with 4 elements
                screen_region = config["interface"]["vision"]["screen_region"]
                if isinstance(screen_region, (list, tuple)) and len(screen_region) == 4:
                    self.screen_region = tuple(screen_region)
                    self.logger.info(f"Using configured screen region: {self.screen_region}")
                else:
                    raise ValueError(f"Invalid screen_region format: {screen_region}")
            else:
                # Get primary monitor dimensions
                monitor_info = mss.mss().monitors[1]  # Primary monitor
                self.screen_region = (0, 0, monitor_info["width"], monitor_info["height"])
                self.logger.info(f"Auto-detected screen resolution: {self.screen_region[2]}x{self.screen_region[3]}")
        except (KeyError, TypeError) as e:
            # Fallback to auto-detection
            monitor_info = mss.mss().monitors[1]  # Primary monitor
            self.screen_region = (0, 0, monitor_info["width"], monitor_info["height"])
            self.logger.warning(f"Error configuring screen region ({str(e)}), auto-detected: {self.screen_region[2]}x{self.screen_region[3]}")
        
        self.monitor = {
            "top": self.screen_region[0], 
            "left": self.screen_region[1], 
            "width": self.screen_region[2], 
            "height": self.screen_region[3]
        }
        
        # OCR settings
        self.ocr_confidence = config["interface"]["vision"].get("ocr_confidence", 0.7)
        
        # UI detection settings
        self.detection_method = config["interface"]["vision"].get("detection_method", "ocr")
        
        # Create template directory if it doesn't exist
        self.template_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
            "data", "templates"
        )
        os.makedirs(self.template_dir, exist_ok=True)
        
        # UI element templates
        self.templates = {}
        self.load_templates()
        
        # UI element cache (detected regions)
        self.ui_element_cache = {}
        self.cache_timestamp = 0
        self.cache_validity = 60  # Cache valid for 60 seconds
        
        # Game state tracking
        self.last_metrics = {}
        self.last_screenshot = None
        self.game_speed = 1
    
    def load_templates(self):
        """Load template images for UI elements if they exist."""
        template_files = {
            "population": "population_label.png",
            "happiness": "happiness_label.png",
            "budget": "budget_label.png",
            "traffic": "traffic_label.png",
            "speed_button": "speed_button.png",
            "menu_button": "menu_button.png"
        }
        
        for element, filename in template_files.items():
            path = os.path.join(self.template_dir, filename)
            if os.path.exists(path):
                self.templates[element] = cv2.imread(path, cv2.IMREAD_COLOR)
                self.logger.info(f"Loaded template for {element}")
            else:
                self.logger.warning(f"Template for {element} not found: {path}")
    
    def capture_templates(self):
        """
        Capture templates for UI elements.
        
        This function will guide the user to select regions of the screen
        for each UI element and save them as template images.
        """
        self.logger.info("Starting template capture mode...")
        self.logger.info("This will help create templates for UI elements")
        
        # Take a screenshot
        screenshot = np.array(self.sct.grab(self.monitor))
        
        elements_to_capture = [
            "population", "happiness", "budget", "traffic",
            "speed_button", "menu_button"
        ]
        
        for element in elements_to_capture:
            self.logger.info(f"Capturing template for {element}...")
            self.logger.info("Position your mouse at the top-left corner of the element and press Enter")
            input("Press Enter when ready...")
            
            x1, y1 = pyautogui.position()
            
            self.logger.info("Now position your mouse at the bottom-right corner of the element and press Enter")
            input("Press Enter when ready...")
            
            x2, y2 = pyautogui.position()
            
            # Extract the template
            template = screenshot[y1:y2, x1:x2]
            
            # Save the template
            template_path = os.path.join(self.template_dir, f"{element}_label.png")
            cv2.imwrite(template_path, template)
            
            self.logger.info(f"Template for {element} saved to {template_path}")
            
            # Add to templates
            self.templates[element] = template
        
        self.logger.info("Template capture completed!")
    
    def connect(self) -> bool:
        """
        Connect to the game by verifying screen capture works.
        
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
            
            # Try to detect UI elements to verify game is running
            if self.detect_ui_elements():
                self.connected = True
                self.logger.info("Successfully connected to Cities: Skylines 2 via auto vision interface.")
                return True
            else:
                self.logger.error("Failed to detect any UI elements. Make sure the game is running and visible.")
                return False
            
        except Exception as e:
            self.logger.error(f"Failed to connect: {str(e)}")
            return False
    
    def disconnect(self) -> None:
        """Disconnect from the game."""
        if hasattr(self, 'sct'):
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
    
    def capture_screen(self) -> np.ndarray:
        """
        Capture the current screen without any processing.
        
        Returns:
            NumPy array containing the raw screenshot
        """
        if not self.connected:
            self.logger.warning("Not connected to the game.")
            return np.zeros((self.screen_region[3], self.screen_region[2], 3), dtype=np.uint8)
        
        try:
            screenshot = self.sct.grab(self.monitor)
            return np.array(screenshot)
        except Exception as e:
            self.logger.error(f"Failed to capture screen: {str(e)}")
            return np.zeros((self.screen_region[3], self.screen_region[2], 3), dtype=np.uint8)
    
    def detect_ui_elements(self) -> bool:
        """
        Detect UI elements in the current screenshot.
        
        Returns:
            True if at least one UI element was detected, False otherwise
        """
        # Check if cache is valid
        current_time = time.time()
        if current_time - self.cache_timestamp < self.cache_validity and self.ui_element_cache:
            return True
        
        if self.last_screenshot is None:
            self.last_screenshot = np.array(self.sct.grab(self.monitor))
        
        # Clear the cache
        self.ui_element_cache = {}
        
        # Choose detection method
        if self.detection_method == "template":
            success = self._detect_ui_elements_template()
        else:  # "ocr" (default)
            success = self._detect_ui_elements_ocr()
        
        if success:
            self.cache_timestamp = current_time
            return True
        
        return False
    
    def _detect_ui_elements_template(self) -> bool:
        """Detect UI elements using template matching."""
        if not self.templates:
            self.logger.warning("No templates available for detection. Run capture_templates() first.")
            return False
        
        detected = False
        
        for element, template in self.templates.items():
            try:
                # Template matching
                result = cv2.matchTemplate(
                    self.last_screenshot, 
                    template, 
                    cv2.TM_CCOEFF_NORMED
                )
                
                # Get the best match
                min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
                
                # If good match found
                if max_val > 0.7:  # Threshold
                    h, w = template.shape[:2]
                    top_left = max_loc
                    bottom_right = (top_left[0] + w, top_left[1] + h)
                    
                    # Store the element region
                    self.ui_element_cache[element] = {
                        "region": (top_left[0], top_left[1], bottom_right[0], bottom_right[1]),
                        "confidence": max_val
                    }
                    detected = True
                    
                    # For metrics, also detect the value area (to the right of the label)
                    if element in ["population", "happiness", "budget", "traffic"]:
                        value_region = (
                            bottom_right[0], top_left[1],
                            bottom_right[0] + 100, bottom_right[1]  # Assume value is within 100px to the right
                        )
                        self.ui_element_cache[f"{element}_value"] = {
                            "region": value_region,
                            "confidence": max_val
                        }
            except Exception as e:
                self.logger.warning(f"Error detecting {element}: {str(e)}")
        
        return detected
    
    def _detect_ui_elements_ocr(self) -> bool:
        """Detect UI elements using OCR."""
        # Convert to grayscale for better OCR
        gray = cv2.cvtColor(self.last_screenshot, cv2.COLOR_BGR2GRAY)
        
        # Run OCR on the entire screen
        try:
            data = pytesseract.image_to_data(gray, output_type=pytesseract.Output.DICT)
            
            # Look for key UI labels
            ui_labels = {
                "population": ["population", "pop", "residents"],
                "happiness": ["happiness", "happy", "satisfaction"],
                "budget": ["budget", "money", "cash", "funds"],
                "traffic": ["traffic", "flow"]
            }
            
            detected = False
            
            for i, text in enumerate(data["text"]):
                if not text.strip():
                    continue
                
                # Check if this text matches any UI labels
                for element, possible_labels in ui_labels.items():
                    if any(label.lower() in text.lower() for label in possible_labels):
                        # Found a UI element
                        x = data["left"][i]
                        y = data["top"][i]
                        w = data["width"][i]
                        h = data["height"][i]
                        
                        # Store the element region
                        self.ui_element_cache[element] = {
                            "region": (x, y, x + w, y + h),
                            "confidence": data["conf"][i] / 100
                        }
                        
                        # For metrics, also detect the value area (to the right of the label)
                        value_region = (
                            x + w, y,
                            x + w + 100, y + h  # Assume value is within 100px to the right
                        )
                        self.ui_element_cache[f"{element}_value"] = {
                            "region": value_region,
                            "confidence": data["conf"][i] / 100
                        }
                        
                        detected = True
                        break
            
            # Look for common UI buttons
            button_labels = {
                "play": ["play", "resume"],
                "pause": ["pause"],
                "speed": ["speed", "1x", "2x", "3x"]
            }
            
            for i, text in enumerate(data["text"]):
                if not text.strip():
                    continue
                
                # Check if this text matches any button labels
                for button, possible_labels in button_labels.items():
                    if any(label.lower() in text.lower() for label in possible_labels):
                        # Found a button
                        x = data["left"][i]
                        y = data["top"][i]
                        w = data["width"][i]
                        h = data["height"][i]
                        
                        # Store the button region
                        self.ui_element_cache[f"{button}_button"] = {
                            "region": (x, y, x + w, y + h),
                            "confidence": data["conf"][i] / 100
                        }
                        
                        detected = True
                        break
            
            return detected
            
        except Exception as e:
            self.logger.error(f"OCR failed: {str(e)}")
            return False
    
    def extract_metrics_from_regions(self) -> Dict[str, float]:
        """Extract metric values from detected regions."""
        metrics = {}
        
        if not self.ui_element_cache:
            self.detect_ui_elements()
            if not self.ui_element_cache:
                return metrics
        
        for metric in ["population", "happiness", "budget", "traffic"]:
            value_key = f"{metric}_value"
            if value_key in self.ui_element_cache:
                region = self.ui_element_cache[value_key]["region"]
                x, y, w, h = region
                
                # Extract sub-image
                sub_img = self.last_screenshot[y:h, x:w]
                
                # Apply OCR to get the value
                try:
                    text = pytesseract.image_to_string(
                        sub_img, 
                        config=f'--psm 7 --oem 3 -c tessedit_char_whitelist=0123456789,.'
                    )
                    
                    # Parse the number
                    text = ''.join(c for c in text if c.isdigit() or c == '.')
                    if text:
                        try:
                            value = float(text)
                            metrics[metric] = value
                        except ValueError:
                            self.logger.warning(f"Failed to parse value for {metric}: '{text}'")
                except Exception as e:
                    self.logger.warning(f"Error extracting {metric}: {str(e)}")
        
        return metrics
    
    def get_metrics(self):
        """
        Get current game metrics from the UI.
        
        Returns:
            Dictionary of metrics (population, happiness, budget, traffic)
        """
        # Take a screenshot and extract metrics
        try:
            # Modified to be more robust in the absence of templates
            # This reduces the dependency on perfect template matching
            metrics = {}
            if self.is_connected:
                screenshot = self.capture_screen()
                
                # Try to get values using templates, but don't fail if templates aren't found
                try:
                    population = self._read_metric_value(screenshot, "population") 
                    metrics["population"] = population if population is not None else 0
                except Exception as e:
                    self.logger.debug(f"Failed to read population: {str(e)}")
                    metrics["population"] = 0
                    
                try:
                    happiness = self._read_metric_value(screenshot, "happiness")
                    metrics["happiness"] = happiness if happiness is not None else 50  # Default to neutral happiness
                except Exception as e:
                    self.logger.debug(f"Failed to read happiness: {str(e)}")
                    metrics["happiness"] = 50
                    
                try:
                    budget = self._read_metric_value(screenshot, "budget")
                    metrics["budget_balance"] = budget if budget is not None else 10000  # Default starting budget
                except Exception as e:
                    self.logger.debug(f"Failed to read budget: {str(e)}")
                    metrics["budget_balance"] = 10000
                    
                try:
                    traffic = self._read_metric_value(screenshot, "traffic")
                    metrics["traffic"] = traffic if traffic is not None else 0
                except Exception as e:
                    self.logger.debug(f"Failed to read traffic: {str(e)}")
                    metrics["traffic"] = 0
                    
            return metrics
        except Exception as e:
            self.logger.warning(f"Failed to get metrics: {str(e)}")
            # Return default values when metrics cannot be read
            return {
                "population": 0,
                "happiness": 50,
                "budget_balance": 10000,
                "traffic": 0
            }
    
    def perform_action(self, action: Dict[str, Any]) -> bool:
        """
        Perform an action in the game.
        
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
                
                # Need to implement UI detection for zone buttons
                # For now, use fixed positions based on common UI layouts
                self._click_zone_tool(zone_type)
                
                # Click on the map position
                map_x, map_y = position
                self._safe_click(map_x, map_y)
                
            elif action_type == "infrastructure":
                # Handle infrastructure actions
                infra_type = action.get("infra_type", "")
                position = action.get("position", (0, 0))
                
                # Need to implement UI detection for infrastructure buttons
                # For now, use fixed positions based on common UI layouts
                self._click_infrastructure_tool(infra_type)
                
                # Click on the map position
                map_x, map_y = position
                self._safe_click(map_x, map_y)
                
                # If it's a road or pipe, we need two points
                if infra_type in ["roads", "power", "water"]:
                    end_position = action.get("end_position", None)
                    if end_position:
                        end_x, end_y = end_position
                        self._safe_click(end_x, end_y)
                
            elif action_type == "budget":
                # Handle budget actions
                budget_action = action.get("budget_action", "")
                return self._handle_budget_action(budget_action)
                
            elif action_type == "game_control":
                # Handle game control actions
                control_type = action.get("control_type", "")
                
                if control_type == "speed":
                    value = action.get("value", 1)
                    return self.set_game_speed(value)
                elif control_type == "pause":
                    self._click_pause_button()
                    return True
                
            else:
                self.logger.warning(f"Unknown action type: {action_type}")
                return False
            
            # Sleep briefly to allow the game to process the action
            time.sleep(0.5)
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to perform action: {str(e)}")
            return False
    
    def _safe_click(self, x, y, button='left', clicks=1, interval=0.1):
        """
        Performs a click operation with safety measures to prevent triggering PyAutoGUI failsafe.
        
        Args:
            x: X coordinate
            y: Y coordinate
            button: Mouse button to click ('left', 'right', 'middle')
            clicks: Number of clicks
            interval: Interval between clicks
        """
        try:
            # Safety check to ensure coordinates are within screen boundaries
            screen_width = self.screen_region[2]
            screen_height = self.screen_region[3]
            
            # Ensure x and y are within safe boundaries (5% buffer from edges)
            safe_x = max(screen_width * 0.05, min(x, screen_width * 0.95))
            safe_y = max(screen_height * 0.05, min(y, screen_height * 0.95))
            
            # Calculate maximum movement speed to avoid triggering failsafe
            current_x, current_y = pyautogui.position()
            distance = ((safe_x - current_x) ** 2 + (safe_y - current_y) ** 2) ** 0.5
            
            # For extra safety, use intermediate points for very long distances
            if distance > 500:
                # Move in two steps to avoid diagonal movements across the screen
                mid_x = current_x + (safe_x - current_x) * 0.5
                mid_y = current_y + (safe_y - current_y) * 0.5
                
                try:
                    # Move to intermediate point first
                    pyautogui.moveTo(mid_x, mid_y, duration=0.2)
                    time.sleep(0.05)  # Short pause
                    # Then move to final destination
                    pyautogui.moveTo(safe_x, safe_y, duration=0.2)
                except Exception as move_error:
                    self.logger.error(f"Mouse movement failed: {move_error}")
                    # Try a direct move as fallback with longer duration
                    pyautogui.moveTo(safe_x, safe_y, duration=0.5)
            elif distance > 200:
                pyautogui.moveTo(safe_x, safe_y, duration=0.2)  # Medium speed for medium distances
            else:
                pyautogui.moveTo(safe_x, safe_y, duration=0.1)  # Faster for short distances
            
            # Add a small pause before clicking to ensure mouse has settled
            time.sleep(0.05)
                
            # Perform the click
            pyautogui.click(x=safe_x, y=safe_y, button=button, clicks=clicks, interval=interval)
            return True
        except Exception as e:
            self.logger.error(f"Click failed: {e}")
            # Ensure mouse is moved to a safe position if something goes wrong
            try:
                # Move to center of screen as a safe position
                pyautogui.moveTo(screen_width/2, screen_height/2, duration=0.5)
            except:
                pass
            return False
    
    def _click_zone_tool(self, zone_type: str) -> bool:
        """Click on a zone tool button."""
        # Use relative positions on screen for now
        # Could be enhanced with UI element detection
        screen_width = self.screen_region[2]
        screen_height = self.screen_region[3]
        
        # Positions are approximate and would work on most layouts
        positions = {
            "residential": (screen_width * 0.1, screen_height * 0.9),
            "commercial": (screen_width * 0.15, screen_height * 0.9),
            "industrial": (screen_width * 0.2, screen_height * 0.9),
            "office": (screen_width * 0.25, screen_height * 0.9),
            "delete_zone": (screen_width * 0.27, screen_height * 0.9)
        }
        
        if zone_type in positions:
            self._safe_click(positions[zone_type][0], positions[zone_type][1])
            return True
        else:
            self.logger.warning(f"Unknown zone type: {zone_type}")
            return False
    
    def _click_infrastructure_tool(self, infra_type: str) -> bool:
        """Click on an infrastructure tool button."""
        # Use relative positions on screen for now
        screen_width = self.screen_region[2]
        screen_height = self.screen_region[3]
        
        # Positions are approximate and would work on most layouts
        positions = {
            "road": (screen_width * 0.3, screen_height * 0.9),
            "road_straight": (screen_width * 0.3, screen_height * 0.9),
            "power_line": (screen_width * 0.35, screen_height * 0.9),
            "power": (screen_width * 0.35, screen_height * 0.9),
            "water_pipe": (screen_width * 0.4, screen_height * 0.9),
            "water": (screen_width * 0.4, screen_height * 0.9),
            "park": (screen_width * 0.45, screen_height * 0.9),
            "service_building": (screen_width * 0.5, screen_height * 0.9),
            "delete_infrastructure": (screen_width * 0.55, screen_height * 0.9)
        }
        
        if infra_type in positions:
            self._safe_click(positions[infra_type][0], positions[infra_type][1])
            return True
        else:
            self.logger.warning(f"Unknown infrastructure type: {infra_type}")
            return False
    
    def _handle_budget_action(self, budget_action: str) -> bool:
        """
        Handle budget-related actions.
        
        Args:
            budget_action: The budget action to perform
            
        Returns:
            True if successful, False otherwise
        """
        # First, open the budget panel
        self._open_budget_panel()
        time.sleep(0.5)
        
        # Get screen dimensions for positioning
        screen_width = self.screen_region[2]
        screen_height = self.screen_region[3]
        
        # Define positions for different budget actions
        # These are approximate and would need to be adjusted for actual game UI
        action_positions = {
            # Tax rates
            "increase_residential_tax": (screen_width * 0.6, screen_height * 0.3),
            "decrease_residential_tax": (screen_width * 0.4, screen_height * 0.3),
            "increase_commercial_tax": (screen_width * 0.6, screen_height * 0.4),
            "decrease_commercial_tax": (screen_width * 0.4, screen_height * 0.4),
            "increase_industrial_tax": (screen_width * 0.6, screen_height * 0.5),
            "decrease_industrial_tax": (screen_width * 0.4, screen_height * 0.5),
            
            # Service budgets
            "increase_service_budget": (screen_width * 0.6, screen_height * 0.6),
            "decrease_service_budget": (screen_width * 0.4, screen_height * 0.6)
        }
        
        # Click on the appropriate button if it exists
        if budget_action in action_positions:
            pos_x, pos_y = action_positions[budget_action]
            self._safe_click(pos_x, pos_y)
            time.sleep(0.3)
            
            # Close the budget panel
            self._close_budget_panel()
            return True
        else:
            self.logger.warning(f"Unknown budget action: {budget_action}")
            # Close the budget panel even if action failed
            self._close_budget_panel()
            return False
    
    def _open_budget_panel(self):
        """Open the budget panel in the game."""
        # This might be a button in the UI, a keyboard shortcut, etc.
        # For simplicity, let's assume it's a button in the bottom-right corner
        screen_width = self.screen_region[2]
        screen_height = self.screen_region[3]
        
        # Click on the budget button (approximate position)
        budget_button_x = screen_width * 0.9
        budget_button_y = screen_height * 0.9
        self._safe_click(budget_button_x, budget_button_y)
    
    def _close_budget_panel(self):
        """Close the budget panel."""
        # This might be a close button, ESC key, etc.
        # For simplicity, press ESC
        pyautogui.press('escape')
        # Reduced wait time
        time.sleep(0.1)
    
    def _click_speed_button(self, speed: int) -> bool:
        """Click on a speed button."""
        if "speed_button" in self.ui_element_cache:
            region = self.ui_element_cache["speed_button"]["region"]
            center_x = (region[0] + region[2]) // 2
            center_y = (region[1] + region[3]) // 2
            
            # Depending on game UI, we might need to adjust click position
            # based on which speed button we want
            if speed == 1:
                pyautogui.click(center_x - 20, center_y)  # Leftmost button
            elif speed == 2:
                pyautogui.click(center_x, center_y)  # Middle button
            elif speed == 3:
                pyautogui.click(center_x + 20, center_y)  # Rightmost button
            
            return True
        else:
            # Fallback to relative screen position
            screen_width = self.screen_region[2]
            screen_height = self.screen_region[3]
            
            # Top right corner usually has speed controls
            base_x = screen_width * 0.9
            base_y = screen_height * 0.1
            
            if speed == 1:
                pyautogui.click(base_x - 40, base_y)
            elif speed == 2:
                pyautogui.click(base_x - 20, base_y)
            elif speed == 3:
                pyautogui.click(base_x, base_y)
            
            return True
    
    def _click_pause_button(self) -> bool:
        """Click on the pause button."""
        if "pause_button" in self.ui_element_cache:
            region = self.ui_element_cache["pause_button"]["region"]
            center_x = (region[0] + region[2]) // 2
            center_y = (region[1] + region[3]) // 2
            pyautogui.click(center_x, center_y)
            return True
        elif "play_button" in self.ui_element_cache:
            region = self.ui_element_cache["play_button"]["region"]
            center_x = (region[0] + region[2]) // 2
            center_y = (region[1] + region[3]) // 2
            pyautogui.click(center_x, center_y)
            return True
        else:
            # Fallback to relative screen position
            screen_width = self.screen_region[2]
            screen_height = self.screen_region[3]
            
            # Top right corner usually has pause control
            pyautogui.click(screen_width * 0.85, screen_height * 0.1)
            return True
    
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
            success = self._click_speed_button(speed)
            if success:
                self.game_speed = speed
                return True
            return False
            
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
        if metrics.get("budget_balance", 0) < 0:
            return True
        
        # TODO: Add OCR-based detection of game over dialogs
        
        return False
    
    def reset_game(self) -> bool:
        """
        Reset the game to its initial state.
        
        Returns:
            True if the game was reset successfully, False otherwise
        """
        if not self.connected:
            self.logger.warning("Not connected to the game.")
            return False
        
        try:
            # Try to pause the game first
            self._click_pause_button()
            time.sleep(0.5)
            
            # First try to find menu button through detection
            menu_clicked = False
            
            # Try OCR detection first
            self.detect_ui_elements()
            if "menu_button" in self.ui_element_cache:
                region = self.ui_element_cache["menu_button"]["region"]
                center_x = (region[0] + region[2]) // 2
                center_y = (region[1] + region[3]) // 2
                pyautogui.click(center_x, center_y)
                menu_clicked = True
            
            if not menu_clicked:
                # Try ESC key first
                pyautogui.press('esc')
                time.sleep(0.5)
                
                # Then try clicking top-left corner as fallback
                pyautogui.click(20, 20)
            
            time.sleep(1)  # Wait for menu to open
            
            # Look for "New Game" button using OCR
            screenshot = self.capture_screen()
            text = pytesseract.image_to_string(screenshot)
            
            new_game_clicked = False
            if "new game" in text.lower():
                # Try to find and click "New Game" text location
                data = pytesseract.image_to_data(screenshot, output_type=pytesseract.Output.DICT)
                for i, word in enumerate(data['text']):
                    if 'new game' in word.lower():
                        x = data['left'][i]
                        y = data['top'][i]
                        pyautogui.click(x + self.screen_region[0], y + self.screen_region[1])
                        new_game_clicked = True
                        break
            
            if not new_game_clicked:
                # Fallback to common positions
                screen_width = self.screen_region[2]
                screen_height = self.screen_region[3]
                
                # Try a few common positions for the "New Game" button
                positions = [
                    (0.2, 0.3),  # Common position 1
                    (0.2, 0.4),  # Common position 2
                    (0.3, 0.3)   # Common position 3
                ]
                
                for x_ratio, y_ratio in positions:
                    pyautogui.click(int(screen_width * x_ratio), int(screen_height * y_ratio))
                    time.sleep(0.5)
            
            # Wait for the game to load and stabilize
            time.sleep(7)  # Increased wait time for game load
            
            # Reset internal state
            self.last_metrics = {}
            self.game_speed = 1
            self.ui_element_cache = {}
            self.cache_timestamp = 0
            
            # Try to detect UI elements to confirm we're in game
            if self.detect_ui_elements():
                self.logger.info("Game reset successful - UI elements detected")
                return True
            else:
                self.logger.warning("Game reset may have failed - no UI elements detected")
                return False
            
        except Exception as e:
            self.logger.error(f"Failed to reset game: {str(e)}")
            return False
    
    def reset(self) -> bool:
        """
        Reset the interface state and game to initial conditions.
        
        Returns:
            bool: True if reset was successful, False otherwise
        """
        try:
            # Reset internal state
            self.last_screenshot = None
            self.last_metrics = {}
            self.cached_ui_elements = {}
            
            # Reset game state
            success = self.reset_game()
            if not success:
                self.logger.warning("Failed to reset game state")
                return False
            
            # Wait for game to stabilize
            time.sleep(1.0)
            
            # Re-detect UI elements
            self.detect_ui_elements()
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error during interface reset: {str(e)}")
            return False
    
    def click_at_coordinates(self, x, y):
        """
        Click at the specified screen coordinates safely.
        
        Args:
            x: X coordinate
            y: Y coordinate
        """
        self._safe_click(x, y)
    
    def press_key(self, key):
        """
        Presses a keyboard key.
        
        Args:
            key: Key to press
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Disallow ESC and Function keys for safety
            if key in ['esc', 'f1', 'f2', 'f3', 'f4', 'f5', 'f6', 'f7', 'f8', 'f9', 'f10', 'f11', 'f12']:
                self.logger.warning(f"Blocked pressing of restricted key: {key}")
                return False
                
            pyautogui.press(key)
            self.logger.debug(f"Pressed key: {key}")
            return True
        except Exception as e:
            self.logger.error(f"Failed to press key {key}: {e}")
            return False
            
    def press_hotkey(self, modifier, key):
        """
        Presses a hotkey combination (modifier + key).
        
        Args:
            modifier: Modifier key ('ctrl', 'shift', 'alt')
            key: Regular key
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Block dangerous combinations
            if modifier == 'alt' and key == 'f4':
                self.logger.warning("Blocked pressing of restricted hotkey: alt+f4")
                return False
                
            pyautogui.hotkey(modifier, key)
            self.logger.debug(f"Pressed hotkey: {modifier}+{key}")
            return True
        except Exception as e:
            self.logger.error(f"Failed to press hotkey {modifier}+{key}: {e}")
            return False
            
    def rotate_camera_left(self):
        """
        Rotates the camera view counterclockwise.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            # Most city builders use Q for counterclockwise rotation
            pyautogui.press('q')
            self.logger.debug("Rotated camera left")
            return True
        except Exception as e:
            self.logger.error(f"Failed to rotate camera left: {e}")
            return False
            
    def rotate_camera_right(self):
        """
        Rotates the camera view clockwise.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            # Most city builders use E for clockwise rotation
            pyautogui.press('e')
            self.logger.debug("Rotated camera right")
            return True
        except Exception as e:
            self.logger.error(f"Failed to rotate camera right: {e}")
            return False
            
    def reset_camera_rotation(self):
        """
        Resets the camera rotation to the default north-facing orientation.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            # Press backspace which resets camera in CS2
            pyautogui.press('backspace')
            self.logger.debug("Reset camera rotation")
            return True
        except Exception as e:
            self.logger.error(f"Failed to reset camera rotation: {e}")
            return False
            
    def tilt_camera_up(self):
        """
        Tilts the camera view upward (more top-down view).
        
        Returns:
            True if successful, False otherwise
        """
        try:
            # Home key often tilts up in city builders
            pyautogui.press('home')
            self.logger.debug("Tilted camera up")
            return True
        except Exception as e:
            self.logger.error(f"Failed to tilt camera up: {e}")
            return False
            
    def tilt_camera_down(self):
        """
        Tilts the camera view downward (more horizontal view).
        
        Returns:
            True if successful, False otherwise
        """
        try:
            # End key often tilts down in city builders
            pyautogui.press('end')
            self.logger.debug("Tilted camera down")
            return True
        except Exception as e:
            self.logger.error(f"Failed to tilt camera down: {e}")
            return False
            
    def zoom_with_wheel(self, clicks):
        """
        Zooms in or out using the mouse wheel.
        
        Args:
            clicks: Number of wheel clicks (positive = zoom in, negative = zoom out)
            
        Returns:
            True if successful, False otherwise
        """
        try:
            pyautogui.scroll(clicks)
            direction = "in" if clicks > 0 else "out"
            self.logger.debug(f"Zoomed {direction} using mouse wheel")
            return True
        except Exception as e:
            self.logger.error(f"Failed to zoom with mouse wheel: {e}")
            return False
    
    def drag_mouse(self, start_x, start_y, end_x, end_y, duration=0.2):
        """
        Drag the mouse from one position to another (pan the view).
        
        Args:
            start_x, start_y: Starting coordinates
            end_x, end_y: Ending coordinates
            duration: How long the drag should take
        """
        # Ensure coordinates are within safe bounds
        screen_width = self.screen_region[2]
        screen_height = self.screen_region[3]
        safety_margin = 5
        
        start_x = max(safety_margin, min(start_x, screen_width - safety_margin))
        start_y = max(safety_margin, min(start_y, screen_height - safety_margin))
        end_x = max(safety_margin, min(end_x, screen_width - safety_margin))
        end_y = max(safety_margin, min(end_y, screen_height - safety_margin))
        
        # Perform the drag
        pyautogui.moveTo(start_x, start_y)
        pyautogui.dragTo(end_x, end_y, duration=duration)
        
        # Reduced wait time after drag
        time.sleep(0.1)
    
    def pan_view(self, direction, distance=100):
        """
        Pan the view in the specified direction.
        
        Args:
            direction: 'left', 'right', 'up', or 'down'
            distance: How far to pan in pixels
        """
        # Get screen center
        screen_width = self.screen_region[2]
        screen_height = self.screen_region[3]
        center_x = screen_width // 2
        center_y = screen_height // 2
        
        # Calculate start and end coordinates based on direction
        start_x, start_y = center_x, center_y
        end_x, end_y = center_x, center_y
        
        if direction == 'left':
            end_x = center_x + distance  # Dragging right moves view left
        elif direction == 'right':
            end_x = center_x - distance  # Dragging left moves view right
        elif direction == 'up':
            end_y = center_y + distance  # Dragging down moves view up
        elif direction == 'down':
            end_y = center_y - distance  # Dragging up moves view down
        else:
            self.logger.warning(f"Unknown pan direction: {direction}")
            return
        
        # Perform the drag
        self.drag_mouse(start_x, start_y, end_x, end_y)
    
    def middle_click(self, x=None, y=None):
        """
        Middle-click at the specified coordinates or at the current mouse position.
        Some games use this for camera control.
        
        Args:
            x, y: Optional coordinates to middle-click at
        """
        if x is not None and y is not None:
            # Move to the specified coordinates first
            screen_width = self.screen_region[2]
            screen_height = self.screen_region[3]
            safety_margin = 5
            
            safe_x = max(safety_margin, min(x, screen_width - safety_margin))
            safe_y = max(safety_margin, min(y, screen_height - safety_margin))
            
            pyautogui.moveTo(safe_x, safe_y)
        
        # Perform middle click
        pyautogui.middleClick()
        time.sleep(0.1) 