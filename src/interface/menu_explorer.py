import time
import random
import numpy as np
import cv2
import pytesseract
import pyautogui
from typing import Dict, Any, List, Tuple, Optional
import logging
import os


class MenuExplorer:
    """
    Menu exploration system that helps the agent discover and navigate
    through the complex menu system of Cities: Skylines 2.
    """
    
    def __init__(self, logger=None):
        """Initialize the menu explorer."""
        self.logger = logger or logging.getLogger("MenuExplorer")
        
        # Known menu positions (will be discovered through exploration)
        self.menu_positions = {}
        
        # Menu hierarchy knowledge (built through exploration)
        self.menu_hierarchy = {
            "main": [],  # Top-level menus
            "submenu": {},  # Submenus under each top menu
            "actions": {}  # Actions available in each submenu
        }
        
        # Exploration state
        self.exploration_phase = "initial"  # initial, main_menus, submenus, advanced
        self.current_menu = None
        self.current_submenu = None
        self.clicks_since_discovery = 0
        self.exploration_regions = []
        self.exploration_counter = 0
        
        # Define screen regions to explore (will be refined during exploration)
        self._initialize_exploration_regions()
    
    def _initialize_exploration_regions(self):
        """Initialize regions of the screen to explore for UI elements."""
        # Get screen dimensions
        screen_width, screen_height = pyautogui.size()
        
        # Define regions to explore (normalized coordinates)
        regions = [
            # Top menu bar (typical location for main menus)
            (0.0, 0.0, 1.0, 0.1),
            
            # Left sidebar (common for tool palettes)
            (0.0, 0.1, 0.2, 0.9),
            
            # Bottom bar (common for control panels)
            (0.0, 0.9, 1.0, 1.0),
            
            # Right sidebar (often for stats and details)
            (0.8, 0.1, 1.0, 0.9),
            
            # Center screen (for dialogs and popups)
            (0.3, 0.3, 0.7, 0.7)
        ]
        
        # Convert to pixel coordinates
        self.exploration_regions = [
            (
                int(r[0] * screen_width),
                int(r[1] * screen_height),
                int(r[2] * screen_width),
                int(r[3] * screen_height)
            )
            for r in regions
        ]
    
    def explore_screen(self, screenshot: np.ndarray) -> Dict[str, Any]:
        """
        Explore the screen for UI elements to interact with.
        
        Args:
            screenshot: Current screenshot of the game
            
        Returns:
            Dictionary with exploration results and recommended actions
        """
        self.exploration_counter += 1
        
        # If we haven't found any menus yet, explore aggressively
        if not self.menu_positions and self.exploration_phase == "initial":
            return self._initial_exploration(screenshot)
        
        # If we're exploring main menus
        if self.exploration_phase == "main_menus":
            return self._main_menu_exploration(screenshot)
        
        # If we're exploring submenus
        if self.exploration_phase == "submenus":
            return self._submenu_exploration(screenshot)
        
        # Default exploration - try random known menus or discover new elements
        if random.random() < 0.7 and self.menu_positions:  # 70% use known menus
            menu_name = random.choice(list(self.menu_positions.keys()))
            return {
                "action": "click_menu",
                "menu_name": menu_name,
                "position": self.menu_positions[menu_name]["position"],
                "confidence": self.menu_positions[menu_name]["confidence"]
            }
        else:  # 30% explore new areas
            return self._discover_new_elements(screenshot)
    
    def _initial_exploration(self, screenshot: np.ndarray) -> Dict[str, Any]:
        """
        Initial exploration phase - systematically check regions for buttons.
        
        Args:
            screenshot: Current screenshot of the game
            
        Returns:
            Dictionary with exploration results
        """
        self.logger.info("Initial exploration phase - looking for main UI elements")
        
        # Select a region to explore
        region_idx = self.exploration_counter % len(self.exploration_regions)
        region = self.exploration_regions[region_idx]
        
        # Extract the region from the screenshot
        region_img = screenshot[region[1]:region[3], region[0]:region[2]]
        
        # Try to find buttons and text in this region
        menu_elements = self._find_interactive_elements(region_img, region)
        
        if menu_elements:
            # Found something to interact with
            element = random.choice(menu_elements)
            self.logger.info(f"Found potential menu element: {element['text']}")
            
            # Store this as a potential menu
            self.menu_positions[element['text']] = {
                "position": element['position'],
                "confidence": element['confidence']
            }
            
            # If we've found several menu items, move to main menu exploration
            if len(self.menu_positions) >= 3:
                self.exploration_phase = "main_menus"
                self.logger.info("Moving to main menu exploration phase")
            
            return {
                "action": "click_menu",
                "menu_name": element['text'],
                "position": element['position'],
                "confidence": element['confidence']
            }
        
        # If nothing found, return a random click in the region
        x = random.randint(region[0], region[2])
        y = random.randint(region[1], region[3])
        
        return {
            "action": "explore_click",
            "position": (x, y),
            "confidence": 0.1
        }
    
    def _main_menu_exploration(self, screenshot: np.ndarray) -> Dict[str, Any]:
        """
        Explore main menus by systematically clicking on them and observing changes.
        
        Args:
            screenshot: Current screenshot of the game
            
        Returns:
            Dictionary with exploration results
        """
        self.logger.info("Main menu exploration phase")
        
        # Either click a known menu or find a new one
        if random.random() < 0.8 and self.menu_positions:  # 80% use known menus
            menu_names = list(self.menu_positions.keys())
            
            # If we have a current menu, try to pick a different one
            if self.current_menu and len(menu_names) > 1:
                menu_names.remove(self.current_menu)
            
            menu_name = random.choice(menu_names)
            self.current_menu = menu_name
            
            self.clicks_since_discovery += 1
            # If we've clicked a lot without finding new elements, move to submenu phase
            if self.clicks_since_discovery > 10:
                self.exploration_phase = "submenus"
                self.clicks_since_discovery = 0
                self.logger.info("Moving to submenu exploration phase")
            
            return {
                "action": "click_menu",
                "menu_name": menu_name,
                "position": self.menu_positions[menu_name]["position"],
                "confidence": self.menu_positions[menu_name]["confidence"]
            }
        else:  # 20% look for new menus
            return self._discover_new_elements(screenshot)
    
    def _submenu_exploration(self, screenshot: np.ndarray) -> Dict[str, Any]:
        """
        Explore submenus that appear after clicking main menus.
        
        Args:
            screenshot: Current screenshot of the game
            
        Returns:
            Dictionary with exploration results
        """
        self.logger.info("Submenu exploration phase")
        
        # First click a main menu if we don't have a current one
        if not self.current_menu and self.menu_positions:
            menu_name = random.choice(list(self.menu_positions.keys()))
            self.current_menu = menu_name
            
            return {
                "action": "click_menu",
                "menu_name": menu_name,
                "position": self.menu_positions[menu_name]["position"],
                "confidence": self.menu_positions[menu_name]["confidence"]
            }
        
        # Then look for submenus in specific regions (usually below or beside main menus)
        if self.current_menu:
            menu_pos = self.menu_positions[self.current_menu]["position"]
            
            # Define regions to look for submenus
            submenu_regions = [
                # Below the menu
                (menu_pos[0] - 100, menu_pos[1] + 20, menu_pos[0] + 100, menu_pos[1] + 200),
                # To the right of the menu
                (menu_pos[0] + 20, menu_pos[1] - 50, menu_pos[0] + 300, menu_pos[1] + 50)
            ]
            
            # Check each region for potential submenu elements
            for region in submenu_regions:
                # Keep region within screen bounds
                region = (
                    max(0, region[0]),
                    max(0, region[1]),
                    min(screenshot.shape[1], region[2]),
                    min(screenshot.shape[0], region[3])
                )
                
                region_img = screenshot[region[1]:region[3], region[0]:region[2]]
                submenu_elements = self._find_interactive_elements(region_img, region)
                
                if submenu_elements:
                    element = random.choice(submenu_elements)
                    self.logger.info(f"Found potential submenu element: {element['text']}")
                    
                    # Store this as a submenu under the current main menu
                    if self.current_menu not in self.menu_hierarchy["submenu"]:
                        self.menu_hierarchy["submenu"][self.current_menu] = []
                    
                    if element['text'] not in self.menu_hierarchy["submenu"][self.current_menu]:
                        self.menu_hierarchy["submenu"][self.current_menu].append(element['text'])
                        self.menu_positions[f"{self.current_menu}_{element['text']}"] = {
                            "position": element['position'],
                            "confidence": element['confidence']
                        }
                    
                    return {
                        "action": "click_submenu",
                        "menu_name": self.current_menu,
                        "submenu_name": element['text'],
                        "position": element['position'],
                        "confidence": element['confidence']
                    }
        
        # If we can't find submenus, go back to exploring main menus
        self.current_menu = None
        self.exploration_phase = "main_menus"
        return self._main_menu_exploration(screenshot)
    
    def _discover_new_elements(self, screenshot: np.ndarray) -> Dict[str, Any]:
        """
        Try to discover new UI elements.
        
        Args:
            screenshot: Current screenshot of the game
            
        Returns:
            Dictionary with exploration results
        """
        self.logger.info("Exploring for new UI elements")
        
        # Choose a random region to explore
        region_idx = random.randint(0, len(self.exploration_regions) - 1)
        region = self.exploration_regions[region_idx]
        
        # Extract the region from the screenshot
        region_img = screenshot[region[1]:region[3], region[0]:region[2]]
        
        # Try to find buttons and text in this region
        elements = self._find_interactive_elements(region_img, region)
        
        if elements:
            # Found something to interact with
            element = random.choice(elements)
            
            # Check if this is a new element
            is_new = True
            for menu_name, menu_data in self.menu_positions.items():
                if self._positions_close(menu_data["position"], element["position"]):
                    is_new = False
                    break
            
            if is_new:
                self.logger.info(f"Discovered new UI element: {element['text']}")
                self.menu_positions[element['text']] = {
                    "position": element['position'],
                    "confidence": element['confidence']
                }
                self.clicks_since_discovery = 0
            
            return {
                "action": "click_discovered",
                "element_name": element['text'],
                "position": element['position'],
                "confidence": element['confidence']
            }
        
        # If nothing found, return a random click
        x = random.randint(region[0], region[2])
        y = random.randint(region[1], region[3])
        
        return {
            "action": "random_click",
            "position": (x, y),
            "confidence": 0.1
        }
    
    def _find_interactive_elements(self, image: np.ndarray, region: Tuple[int, int, int, int]) -> List[Dict[str, Any]]:
        """
        Find potential interactive elements in an image.
        
        Args:
            image: Image to analyze
            region: Region coordinates of the image (x1, y1, x2, y2)
            
        Returns:
            List of dictionaries with information about found elements
        """
        elements = []
        
        # Skip if image is too small
        if image.shape[0] < 10 or image.shape[1] < 10:
            return elements
        
        try:
            # Convert to grayscale for better OCR
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Run OCR to find text
            ocr_data = pytesseract.image_to_data(gray, output_type=pytesseract.Output.DICT)
            
            for i, text in enumerate(ocr_data["text"]):
                if not text.strip():
                    continue
                
                # Get text position
                x = ocr_data["left"][i] + region[0]
                y = ocr_data["top"][i] + region[1]
                w = ocr_data["width"][i]
                h = ocr_data["height"][i]
                conf = float(ocr_data["conf"][i]) / 100.0
                
                # Skip low confidence detections
                if conf < 0.5:
                    continue
                
                elements.append({
                    "text": text,
                    "position": (x + w // 2, y + h // 2),  # Center point
                    "region": (x, y, x + w, y + h),
                    "confidence": conf
                })
            
            # Also try to find buttons using edge detection and contour finding
            edges = cv2.Canny(gray, 50, 150)
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for contour in contours:
                # Get bounding rectangle
                x, y, w, h = cv2.boundingRect(contour)
                
                # Skip very small or very large contours
                if w < 20 or h < 20 or w > image.shape[1] * 0.8 or h > image.shape[0] * 0.8:
                    continue
                
                # See if this looks like a button (rectangles with approximately 3:1 or 4:1 ratio)
                aspect_ratio = float(w) / h
                if 1.5 <= aspect_ratio <= 5.0:
                    # Absolute coordinates
                    abs_x = x + region[0]
                    abs_y = y + region[1]
                    
                    # Look for text inside this potential button
                    button_text = "button"
                    for element in elements:
                        ex, ey = element["position"]
                        if abs_x <= ex <= abs_x + w and abs_y <= ey <= abs_y + h:
                            button_text = element["text"]
                            break
                    
                    elements.append({
                        "text": button_text,
                        "position": (abs_x + w // 2, abs_y + h // 2),  # Center point
                        "region": (abs_x, abs_y, abs_x + w, abs_y + h),
                        "confidence": 0.7  # Moderate confidence for shape-based detection
                    })
        
        except Exception as e:
            logging.warning(f"Error finding interactive elements: {str(e)}")
        
        return elements
    
    def _positions_close(self, pos1: Tuple[int, int], pos2: Tuple[int, int], threshold: int = 20) -> bool:
        """
        Check if two positions are close to each other.
        
        Args:
            pos1: First position (x, y)
            pos2: Second position (x, y)
            threshold: Distance threshold in pixels
            
        Returns:
            True if positions are within threshold distance of each other
        """
        x1, y1 = pos1
        x2, y2 = pos2
        distance = ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5
        return distance <= threshold 