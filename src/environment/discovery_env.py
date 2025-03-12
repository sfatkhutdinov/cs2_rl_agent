import logging
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import random
import datetime
import json
import os
import time
import threading
from typing import Dict, Any, Tuple, List, Optional, Union
import pyautogui  # Add this import to handle direct actions

from src.environment.vision_guided_env import VisionGuidedCS2Environment
from src.interface.ollama_vision_interface import OllamaVisionInterface
from src.interface.window_manager import WindowManager
from src.actions.action_handler import ActionHandler
from src.actions.menu_explorer import MenuExplorer


class DiscoveryEnvironment(VisionGuidedCS2Environment):
    """
    An environment that focuses on discovering game mechanics through exploration,
    guided by vision intelligence, and learns from tutorials with minimal predefined structure.
    """
    
    def __init__(self, 
                 base_env_config: Dict[str, Any] = None,
                 observation_config: Dict[str, Any] = None,
                 vision_config: Dict[str, Any] = None,
                 use_fallback_mode: bool = True,
                 discovery_frequency: float = 0.3,
                 tutorial_frequency: float = 0.3, 
                 random_action_frequency: float = 0.2,
                 exploration_randomness: float = 0.5,
                 logger: Optional[logging.Logger] = None):
        """
        Initialize the discovery-based environment.
        
        Args:
            base_env_config: Configuration for the base environment
            observation_config: Configuration for observations
            vision_config: Configuration for vision guidance
            use_fallback_mode: Whether to use fallback mode if game connection fails
            discovery_frequency: How often to try discovering new UI elements
            tutorial_frequency: How often to look for tutorials
            random_action_frequency: How often to perform completely random actions
            exploration_randomness: How random exploration should be (0=focused, 1=random)
            logger: Logger instance
        """
        # Set up debug directory first to avoid attribute errors
        self.debug_dir = os.path.join("logs", "discovery_debug")
        os.makedirs(self.debug_dir, exist_ok=True)
        
        # Initialize tracking variables
        self.total_steps = 0  # Track total steps taken in the environment
        self.discovery_frequency = discovery_frequency
        self.tutorial_frequency = tutorial_frequency
        self.random_action_frequency = random_action_frequency
        self.exploration_randomness = exploration_randomness
        
        # Track discovered elements
        self.discovered_ui_elements = []
        self.discovered_actions = []
        self.discovered_tutorials = []
        self.current_tutorial = None
        self.tutorial_progress = 0
        self.discovery_phase = "exploration"  # Can be "exploration", "tutorial", "targeted"
        self.current_action_sequence = []
        self.successful_action_sequences = []
        
        # Statistics tracking
        self.stats = {
            "discoveries_made": 0,
            "tutorials_started": 0,
            "tutorials_completed": 0,
            "successful_sequences": 0,
            "total_rewards": 0,
            "total_steps": 0,
            "menus_explored": 0,
            "ui_elements_discovered": 0,
            "successful_interactions": 0,
            "failed_interactions": 0,
        }
        
        # Simplify the action space in the base config to focus on generic actions
        base_env_config = base_env_config or {}
        simplified_action_space = {
            "type": "discrete",
            "actions": [
                # Basic mouse actions
                {"name": "mouse_click", "type": "mouse", "action": "click", "coords": "current"},
                {"name": "mouse_right_click", "type": "mouse", "action": "right_click", "coords": "current"},
                {"name": "mouse_move", "type": "mouse", "action": "move", "coords": "random"},
                
                # Basic camera controls
                {"name": "camera_pan_left", "type": "camera", "action": "pan_left"},
                {"name": "camera_pan_right", "type": "camera", "action": "pan_right"},
                {"name": "camera_pan_up", "type": "camera", "action": "pan_up"},
                {"name": "camera_pan_down", "type": "camera", "action": "pan_down"},
                {"name": "camera_zoom_in", "type": "camera", "action": "zoom_in"},
                {"name": "camera_zoom_out", "type": "camera", "action": "zoom_out"},
                {"name": "camera_rotate_left", "type": "camera", "action": "rotate_left"},
                {"name": "camera_rotate_right", "type": "camera", "action": "rotate_right"},
                
                # Common keyboard actions
                {"name": "key_escape", "type": "keyboard", "key": "escape"},
                {"name": "key_enter", "type": "keyboard", "key": "enter"},
                {"name": "key_space", "type": "keyboard", "key": "space"},
                {"name": "key_tab", "type": "keyboard", "key": "tab"},
            ]
        }
        
        # Add all letter keys (a-z)
        for key_code in range(ord('a'), ord('z')+1):
            key = chr(key_code)
            simplified_action_space["actions"].append({
                "name": f"key_{key}",
                "type": "keyboard",
                "key": key
            })
        
        # Add number keys (0-9)
        for i in range(10):
            simplified_action_space["actions"].append({
                "name": f"key_{i}",
                "type": "keyboard",
                "key": str(i)
            })
        
        # Add function keys (F1-F12)
        for i in range(1, 13):
            simplified_action_space["actions"].append({
                "name": f"key_f{i}",
                "type": "keyboard",
                "key": f"f{i}"
            })
        
        # Add arrow keys
        for arrow in ["up", "down", "left", "right"]:
            simplified_action_space["actions"].append({
                "name": f"key_arrow_{arrow}",
                "type": "keyboard",
                "key": arrow
            })
            
        # Add required action configuration sections that CS2Environment expects
        # Even though we're using simplified actions, the parent class still checks for these
        simplified_action_space["zone"] = [
            {"name": "residential", "key": "z"},
            {"name": "commercial", "key": "x"},
            {"name": "industrial", "key": "c"},
            {"name": "office", "key": "v"}
        ]
        
        simplified_action_space["infrastructure"] = [
            {
                "name": "road",
                "key": "r",
                "types": [
                    {"name": "two_lane", "key": "1"},
                    {"name": "four_lane", "key": "2"},
                    {"name": "highway", "key": "3"}
                ]
            },
            {
                "name": "water",
                "key": "w",
                "types": [
                    {"name": "water_pipe", "key": "1"},
                    {"name": "water_tower", "key": "2"},
                    {"name": "water_treatment", "key": "3"}
                ]
            },
            {
                "name": "electricity",
                "key": "e",
                "types": [
                    {"name": "power_line", "key": "1"},
                    {"name": "wind_turbine", "key": "2"},
                    {"name": "solar_plant", "key": "3"}
                ]
            },
            {
                "name": "services",
                "key": "u",
                "types": [
                    {"name": "police", "key": "1"},
                    {"name": "fire", "key": "2"},
                    {"name": "healthcare", "key": "3"},
                    {"name": "education", "key": "4"}
                ]
            }
        ]
        
        simplified_action_space["camera"] = {
            "sensitivity": 0.5,
            "rotation_speed": 0.2,
            "zoom_speed": 0.3
        }
        
        simplified_action_space["build"] = [
            {"name": "road", "key": "r"},
            {"name": "water", "key": "w"},
            {"name": "electricity", "key": "e"},
            {"name": "services", "key": "u"}
        ]
        
        simplified_action_space["demolish"] = {"key": "b"}
        
        simplified_action_space["ui"] = [
            {"name": "open_menu", "key": "escape"},
            {"name": "confirm", "key": "enter"},
            {"name": "cancel", "key": "escape"},
            {"name": "tab", "key": "tab"}
        ]
        
        simplified_action_space["budget"] = [
            {"name": "open_budget", "key": "f1"},
            {"name": "increase_tax", "key": "+"},
            {"name": "decrease_tax", "key": "-"},
            {"name": "balance_budget", "key": "b"},
            {"name": "close_budget", "key": "escape"}
        ]
        
        # Set the simplified action space
        if "action_space" in base_env_config:
            base_env_config["action_space"] = simplified_action_space
        else:
            base_env_config["action_space"] = simplified_action_space
        
        # Ensure observation_space is properly configured
        if "observation_space" not in base_env_config:
            # Create default observation space based on observation config
            obs_config = observation_config or {}
            screenshot_width = obs_config.get("screenshot_width", 224)
            screenshot_height = obs_config.get("screenshot_height", 224)
            grayscale = obs_config.get("grayscale", False)
            minimap_width = obs_config.get("minimap_width", 84)
            minimap_height = obs_config.get("minimap_height", 84)
            
            channels = 1 if grayscale else 3
            
            observation_space = {
                "type": "dict",
                "spaces": {
                    "metrics": {
                        "type": "box",
                        "shape": [10],  # Default metrics size
                        "low": -1.0,
                        "high": 1.0
                    }
                }
            }
            
            if obs_config.get("include_screenshot", True):
                observation_space["spaces"]["screenshot"] = {
                    "type": "box",
                    "shape": [screenshot_height, screenshot_width, channels],
                    "low": 0,
                    "high": 255
                }
            
            if obs_config.get("include_minimap", True):
                observation_space["spaces"]["minimap"] = {
                    "type": "box",
                    "shape": [minimap_height, minimap_width, channels],
                    "low": 0,
                    "high": 255
                }
                
            base_env_config["observation_space"] = observation_space
        
        # Call parent initialization with our configs
        super().__init__(
            base_env_config=base_env_config,
            observation_config=observation_config,
            vision_config=vision_config,
            use_fallback_mode=use_fallback_mode,
            exploration_frequency=discovery_frequency,
            random_action_frequency=random_action_frequency,
            logger=logger
        )
        
        self.logger = logger or logging.getLogger("DiscoveryEnv")
        
        self.logger.info("Discovery-based environment initialized")
        
        # Initialize action statistics
        self.action_stats = {}
        
        # Initialize window manager
        self.window_manager = WindowManager(window_name=base_env_config.get("window_name", "Cities: Skylines II"))
        
        # Focus on the game window at startup
        if self.window_manager:
            self.window_manager.focus_window()
        
        # Create action log file
        self.action_log_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 
            "logs", 
            f"action_log_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        )
        os.makedirs(os.path.dirname(self.action_log_path), exist_ok=True)
        with open(self.action_log_path, "w") as f:
            f.write(f"Action Log - Started at {datetime.datetime.now()}\n")
            f.write("=" * 80 + "\n")
        
        # Configure action feedback
        self.show_action_feedback = base_env_config.get("show_action_feedback", True)
        self.action_delay = base_env_config.get("action_delay", 0.5)  # seconds between actions
        
        self.logger.info(f"Discovery environment initialized with action feedback: {self.show_action_feedback}")
    
    def _discover_ui_elements(self) -> Dict[str, Any]:
        """
        Use the vision model to discover UI elements on the screen.
        
        Returns:
            Dictionary of discovered UI elements
        """
        vision_interface = self._get_vision_interface()
        if not vision_interface:
            self.logger.warning("No vision interface available, using fallback mode for UI discovery")
            # Return a minimal set of UI elements for fallback mode
            return {
                "ui_elements": [],
                "current_context": "fallback_mode",
                "is_tutorial_screen": False,
                "suggested_next_action": "use random actions"
            }
            
        try:
            # Capture the current screen
            screen = vision_interface.capture_screen()
            
            # Create a UI discovery prompt with special attention to tutorial screens
            ui_discovery_prompt = """
            You are a computer vision assistant that specializes in analyzing game interfaces.
            
            Looking at this screenshot from Cities: Skylines 2, identify all clickable UI elements.
            Pay special attention to tutorial screens, welcome dialogs, and buttons like "Let's build!" that may appear during startup.
            
            You MUST ONLY RETURN a valid JSON object with the following structure, and nothing else:
            {
                "ui_elements": [
                    {
                        "name": "element name",
                        "description": "what this element does",
                        "position": "approximate position",
                        "coordinates": [x, y],
                        "confidence": 0.9,
                        "is_clickable": true,
                        "action_type": "click"
                    }
                ],
                "current_context": "menu screen",
                "is_tutorial_screen": true,
                "tutorial_text": "any visible tutorial instructions",
                "suggested_next_action": "what to do next"
            }
            
            Your response must be ONLY THE JSON OBJECT, no other text. The coordinates should be pixel positions in the image.
            """
            
            # Query the vision model
            ui_info = vision_interface.query_vision_model(
                prompt=ui_discovery_prompt,
                image=screen,
                enforce_json=True
            )
            
            # Handle potentially different response formats
            if isinstance(ui_info, str):
                # Try to parse string as JSON if the model returned a string
                try:
                    import json
                    # Try to extract JSON if model added other text
                    json_start = ui_info.find('{')
                    json_end = ui_info.rfind('}') + 1
                    if json_start >= 0 and json_end > json_start:
                        json_str = ui_info[json_start:json_end]
                        ui_info = json.loads(json_str)
                    else:
                        ui_info = json.loads(ui_info)
                except Exception as e:
                    self.logger.warning(f"Couldn't parse vision model response as JSON: {str(e)}")
                    # Log the actual response for debugging
                    self.logger.debug(f"Raw response: {ui_info}")
                    ui_info = {}
            
            if not ui_info or not isinstance(ui_info, dict):
                self.logger.warning(f"Invalid response from vision model, using fallback. Response: {str(ui_info)[:200]}...")
                return {
                    "ui_elements": [],
                    "current_context": "fallback_mode",
                    "is_tutorial_screen": False,
                    "suggested_next_action": "use random actions"
                }
            
            # Ensure the response has the required keys
            if "ui_elements" not in ui_info:
                ui_info["ui_elements"] = []
            if "current_context" not in ui_info:
                ui_info["current_context"] = "unknown"
            if "is_tutorial_screen" not in ui_info:
                # Try to determine if it's a tutorial screen based on tutorial text
                ui_info["is_tutorial_screen"] = "tutorial_text" in ui_info and bool(ui_info.get("tutorial_text"))
                
                # Also check if the welcome screen is present by analyzing UI elements
                welcome_keywords = ["welcome", "let's build", "tutorial", "start the game"]
                for element in ui_info.get("ui_elements", []):
                    name = element.get("name", "").lower()
                    desc = element.get("description", "").lower()
                    if any(keyword in name or keyword in desc for keyword in welcome_keywords):
                        ui_info["is_tutorial_screen"] = True
                        break
            
            # Check if this is a tutorial or welcome screen
            is_tutorial_screen = ui_info.get("is_tutorial_screen", False)
            if is_tutorial_screen:
                self.logger.info("Tutorial screen detected!")
                
                # Look for specific welcome dialog buttons like "Let's build!"
                lets_build_button = None
                welcome_buttons = ["let's build!", "let's build", "start", "continue", "next", "begin"]
                
                for element in ui_info.get("ui_elements", []):
                    name = element.get("name", "").lower()
                    if any(btn in name for btn in welcome_buttons):
                        lets_build_button = element
                        break
                
                # If we found the button, click it
                if lets_build_button and "coordinates" in lets_build_button:
                    x, y = lets_build_button["coordinates"]
                    # Check if coordinates are a list or other format
                    if isinstance(x, (int, float)) and isinstance(y, (int, float)):
                        # Coordinates are already numeric
                        pass
                    elif isinstance(x, list) and len(x) > 0:
                        # If coordinates were parsed as a nested list
                        x, y = x[0], x[1] if len(x) > 1 else y
                    
                    if vision_interface and hasattr(vision_interface, "click_at_coordinates"):
                        try:
                            # Convert to integers to ensure valid coordinates
                            x_int, y_int = int(x), int(y)
                            success = vision_interface.click_at_coordinates(x_int, y_int)
                            if success:
                                self.logger.info(f"Clicked '{lets_build_button.get('name')}' button at ({x_int}, {y_int})")
                                # Track this as a tutorial step
                                if self.current_tutorial is None:
                                    self.current_tutorial = "Welcome Tutorial"
                                self.tutorial_progress += 1
                                
                                # Save the tutorial text for analysis
                                tutorial_text = ui_info.get("tutorial_text", "")
                                if tutorial_text:
                                    with open(os.path.join(self.debug_dir, "tutorial_text.txt"), "a") as f:
                                        f.write(f"--- Tutorial Screen {self.tutorial_progress} ---\n")
                                        f.write(tutorial_text)
                                        f.write("\n\n")
                        except (ValueError, TypeError) as e:
                            self.logger.error(f"Invalid coordinates: {x}, {y}. Error: {str(e)}")
            
            # Process and store discovered elements
            if "ui_elements" in ui_info:
                for element in ui_info.get("ui_elements", []):
                    element_id = element.get("name", "unknown")
                    
                    # Store with timestamp
                    if element_id not in self.discovered_ui_elements:
                        element["discovery_time"] = time.time()
                        element["times_seen"] = 1
                        element["times_clicked"] = 0
                        element["success_rate"] = 0.0
                        self.discovered_ui_elements[element_id] = element
                        self.stats["discoveries_made"] += 1
                    else:
                        self.discovered_ui_elements[element_id]["times_seen"] += 1
            
            return ui_info
        
        except Exception as e:
            self.logger.error(f"Error in UI discovery: {str(e)}")
            # Return a minimal set of UI elements for fallback mode
            return {
                "ui_elements": [],
                "current_context": "error_fallback",
                "is_tutorial_screen": False,
                "suggested_next_action": "use random actions"
            }
    
    def _discover_action_effect(self, action_idx: int) -> Dict[str, Any]:
        """
        Discover the effect of an action by taking it and observing the result.
        
        Args:
            action_idx: Index of action to take
            
        Returns:
            Dictionary with action effect information
        """
        try:
            # Capture before state
            vision_interface = self._get_vision_interface()
            if not vision_interface:
                return {"success": False, "reason": "No vision interface available"}
                
            before_screen = vision_interface.capture_screen()
            
            # Take the action
            action_name = self._get_action_name(action_idx)
            self.logger.info(f"Discovering effect of action: {action_name}")
            
            # Store in current sequence
            self.current_action_sequence.append({
                "action_idx": action_idx,
                "action_name": action_name,
                "timestamp": time.time()
            })
            
            # Take the action and get the result
            obs, reward, terminated, truncated, info = super().step(action_idx)
            
            # Wait a bit for any animations/effects
            time.sleep(0.5)
            
            # Capture after state
            after_screen = vision_interface.capture_screen()
            
            # Analyze the difference
            difference_prompt = """
            Compare these two screenshots from Cities: Skylines 2 (before and after an action was taken).
            
            Describe what changed between the two images in detail. Focus on:
            1. UI changes
            2. New elements that appeared
            3. Elements that disappeared
            4. Any visual feedback that indicates success or failure
            
            Provide your response in JSON format:
            {
                "changes_detected": true/false,
                "ui_changes": ["list of UI element changes"],
                "new_elements": ["list of new elements"],
                "removed_elements": ["list of removed elements"],
                "visual_feedback": "description of any success/failure indicators",
                "outcome": "success/failure/neutral",
                "effect_description": "detailed description of what the action did"
            }
            """
            
            # Query the vision model about the difference
            effect_info = vision_interface.query_vision_model(
                prompt=difference_prompt,
                image=[before_screen, after_screen],  # Send both images
                enforce_json=True
            )
            
            # Store the discovered effect
            if action_name not in self.discovered_actions:
                self.discovered_actions[action_name] = {
                    "times_used": 1,
                    "successful_uses": 0,
                    "average_reward": reward,
                    "effects": [effect_info],
                    "discovery_time": time.time()
                }
            else:
                self.discovered_actions[action_name]["times_used"] += 1
                self.discovered_actions[action_name]["average_reward"] = (
                    (self.discovered_actions[action_name]["average_reward"] * 
                     (self.discovered_actions[action_name]["times_used"] - 1) + reward) / 
                    self.discovered_actions[action_name]["times_used"]
                )
                self.discovered_actions[action_name]["effects"].append(effect_info)
            
            # Update success rate if we can determine success
            if effect_info.get("outcome") == "success":
                self.discovered_actions[action_name]["successful_uses"] += 1
            
            effect_info["reward"] = reward
            effect_info["action_name"] = action_name
            effect_info["action_idx"] = action_idx
            
            return effect_info
            
        except Exception as e:
            self.logger.error(f"Error in action effect discovery: {str(e)}")
            return {"success": False, "reason": str(e)}
    
    def _get_discovery_action(self) -> int:
        """
        Determine the next action based on discovery priorities.
        
        Returns:
            Action index to take
        """
        # Check if we should look for tutorials
        if random.random() < self.tutorial_frequency:
            return self._get_tutorial_guided_action()
        
        # Sometimes try a completely random action for exploration
        if random.random() < self.exploration_randomness:
            self.stats["total_steps"] += 1
            return self.action_space.sample()
        
        # Discover UI elements
        ui_info = self._discover_ui_elements()
        
        # If we found UI elements, try clicking one
        if ui_info and "ui_elements" in ui_info and ui_info["ui_elements"]:
            # Filter to clickable elements with reasonable confidence
            clickable_elements = [
                elem for elem in ui_info["ui_elements"] 
                if elem.get("is_clickable", True) and elem.get("confidence", 0) > 0.7
            ]
            
            if clickable_elements:
                # Choose an element, preferring ones we haven't tried before
                untried_elements = [
                    elem for elem in clickable_elements
                    if elem.get("name") in self.discovered_ui_elements and
                    self.discovered_ui_elements[elem.get("name")].get("times_clicked", 0) == 0
                ]
                
                if untried_elements and random.random() < 0.7:
                    element = random.choice(untried_elements)
                else:
                    element = random.choice(clickable_elements)
                
                # Try to click this element
                element_name = element.get("name", "unknown")
                if element_name in self.discovered_ui_elements:
                    self.discovered_ui_elements[element_name]["times_clicked"] += 1
                
                # Get coordinates
                coords = element.get("coordinates", None)
                if coords and len(coords) == 2:
                    x, y = coords
                    
                    # Use the vision interface to click at these coordinates
                    vision_interface = self._get_vision_interface()
                    if vision_interface and hasattr(vision_interface, "click_at_coordinates"):
                        success = vision_interface.click_at_coordinates(x, y)
                        if success:
                            self.logger.info(f"Clicked discovered UI element: {element_name} at ({x}, {y})")
                            # Return a dummy action since we've already performed the click
                            return 0
        
        # If we couldn't find elements to click, try an action that worked well before
        if self.discovered_actions and random.random() < 0.5:
            # Calculate success rates for actions
            success_rates = {}
            for action_name, data in self.discovered_actions.items():
                if data["times_used"] > 0:
                    success_rate = data["successful_uses"] / data["times_used"]
                    adjusted_rate = success_rate + (data["average_reward"] * 0.1)  # Boost for high rewards
                    success_rates[action_name] = adjusted_rate
            
            if success_rates:
                # Sometimes pick the best action, sometimes explore
                if random.random() < 0.7:
                    best_action = max(success_rates.items(), key=lambda x: x[1])[0]
                else:
                    best_action = random.choice(list(success_rates.keys()))
                
                # Get the action index
                action_idx = self._get_action_idx_by_name(best_action)
                if action_idx is not None:
                    self.stats["total_steps"] += 1
                    return action_idx
        
        # Default to a random action
        self.stats["total_steps"] += 1
        return self.action_space.sample()
    
    def _get_tutorial_guided_action(self) -> int:
        """
        Detect and follow tutorial guidance if available.
        
        Returns:
            Action index guided by tutorials
        """
        vision_interface = self._get_vision_interface()
        if not vision_interface:
            return self.action_space.sample()
            
        try:
            # Capture the screen
            screen = vision_interface.capture_screen()
            
            # Create a tutorial detection prompt
            tutorial_prompt = """
            Analyze this screenshot from Cities: Skylines 2 and identify if there are any tutorial messages,
            help prompts, or guided instructions visible.
            
            Provide your response in JSON format:
            {
                "tutorial_detected": true/false,
                "tutorial_name": "title or name of the tutorial",
                "instructions": "what the player is being instructed to do",
                "ui_elements_to_interact_with": ["list of UI elements to interact with"],
                "keys_to_press": ["list of keys to press"],
                "step_number": "current step number if shown",
                "total_steps": "total steps in tutorial if shown",
                "suggested_action": "specific action to take next"
            }
            """
            
            # Query the vision model
            tutorial_info = vision_interface.query_vision_model(
                prompt=tutorial_prompt,
                image=screen,
                enforce_json=True
            )
            
            # If a tutorial is detected
            if tutorial_info.get("tutorial_detected", False):
                tutorial_name = tutorial_info.get("tutorial_name", "Unknown Tutorial")
                
                # Update tutorial tracking
                if tutorial_name != self.current_tutorial:
                    self.current_tutorial = tutorial_name
                    self.tutorial_progress = 0
                    
                    if tutorial_name not in self.discovered_tutorials:
                        self.discovered_tutorials.add(tutorial_name)
                        self.logger.info(f"Discovered new tutorial: {tutorial_name}")
                
                # Try to follow the tutorial instructions
                suggested_action = tutorial_info.get("suggested_action", "")
                keys_to_press = tutorial_info.get("keys_to_press", [])
                ui_elements = tutorial_info.get("ui_elements_to_interact_with", [])
                
                # First try to press any suggested keys
                if keys_to_press:
                    key = keys_to_press[0].lower()
                    action_name = f"key_{key}"
                    action_idx = self._get_action_idx_by_name(action_name)
                    
                    if action_idx is not None:
                        self.logger.info(f"Following tutorial by pressing: {key}")
                        self.tutorial_progress += 1
                        return action_idx
                
                # Then try to interact with UI elements
                if ui_elements:
                    # For now, we'll just try to use mouse clicks to interact
                    # In a more advanced implementation, we could use the vision interface
                    # to locate and click on these elements precisely
                    return self._get_action_idx_by_name("mouse_click")
                
                # Try to match the suggested action to an available action
                if suggested_action:
                    # Simple keyword matching
                    for action_name in dir(self):
                        if callable(getattr(self, action_name)) and action_name.startswith("_"):
                            action_name = action_name[1:]  # Remove leading underscore
                            action_words = action_name.split("_")
                            
                            # Check if all words in the action name appear in the suggested action
                            if all(word.lower() in suggested_action.lower() for word in action_words):
                                action_idx = self._get_action_idx_by_name(action_name)
                                if action_idx is not None:
                                    self.logger.info(f"Following tutorial suggestion: {suggested_action}")
                                    self.tutorial_progress += 1
                                    return action_idx
            
            # If we couldn't find or follow a tutorial, fall back to discovery
            return self._get_discovery_action()
            
        except Exception as e:
            self.logger.error(f"Error in tutorial guidance: {str(e)}")
            return self.action_space.sample()
    
    def _get_action_name(self, action_idx: int) -> str:
        """Get the name of an action from its index."""
        if 0 <= action_idx < len(self.action_handlers):
            action = self.action_handlers[action_idx]
            return getattr(action, 'name', f"action_{action_idx}")
        return f"unknown_action_{action_idx}"
    
    def _get_action_idx_by_name(self, action_name: str) -> Optional[int]:
        """Get the index of an action from its name."""
        for i, action in enumerate(self.action_handlers):
            if hasattr(action, 'name') and action.name == action_name:
                return i
        return None
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """
        Take a step in the environment, using discovery and learning.
        
        Args:
            action: Action index to take
            
        Returns:
            Tuple of (observation, reward, terminated, truncated, info)
        """
        # Log action for debugging
        self.logger.info(f"Taking action: {action} - {self.action_handlers[action].name if action < len(self.action_handlers) else 'unknown'}")
        
        # Increment total steps counter
        self.total_steps += 1
        self.stats["total_steps"] += 1
        
        # Ensure we have focus on the game window
        self._ensure_game_focus()
        
        # Log the action to file
        self._log_action(action)
        
        # Add visual feedback for the action if enabled
        if self.show_action_feedback:
            self._show_action_feedback(action)
        
        # Handle discovery actions specially
        if action >= len(self.action_handlers) - 1 and hasattr(self, 'menu_explorer'):
            return self._handle_discovery_action()
        
        # Execute the action directly when possible
        success = self._execute_direct_action(action)
        
        # If direct execution failed, fall back to regular action execution
        if not success:
            try:
                self.action_handlers[action].execute()
            except Exception as e:
                self.logger.error(f"Error executing action {action}: {str(e)}")
                self.stats["failed_interactions"] += 1
        else:
            self.stats["successful_interactions"] += 1
        
        # Wait a bit to see the results of the action
        time.sleep(self.action_delay)
        
        # Get the new observation
        try:
            obs = self._get_observation()
        except Exception as e:
            self.logger.error(f"Failed to get observation after action: {str(e)}")
            obs = self._get_dummy_observation()
        
        # Calculate reward and check terminal conditions
        reward = self._calculate_reward()
        terminated = False
        truncated = False
        info = {"stats": self.stats}
        
        return obs, reward, terminated, truncated, info
    
    def _ensure_game_focus(self):
        """Make sure the game window has focus before taking action"""
        try:
            if self.window_manager and not self.window_manager.is_window_focused():
                self.logger.info("Game window not focused, attempting to focus")
                self.window_manager.focus_window()
                time.sleep(0.5)  # Wait for focus to take effect
        except Exception as e:
            self.logger.warning(f"Error when checking window focus: {str(e)}")
    
    def _execute_direct_action(self, action: int) -> bool:
        """
        Attempt to execute action directly using pyautogui
        
        Args:
            action: Action index
            
        Returns:
            True if action was executed successfully, False otherwise
        """
        try:
            if action >= len(self.action_handlers):
                return False
                
            action_handler = self.action_handlers[action]
            
            # Check if action_handler is a namedtuple Action
            if isinstance(action_handler, tuple) and hasattr(action_handler, '_fields') and 'action_fn' in action_handler._fields:
                # It's a namedtuple Action, call the action_fn
                action_handler.action_fn()
                return True
            
            # Otherwise, try to get the name and parameters
            action_name = action_handler.name if hasattr(action_handler, 'name') else "unknown"
            
            # Handle mouse actions
            if "mouse_click" in action_name:
                x, y = action_handler.params.get("x", None), action_handler.params.get("y", None) if hasattr(action_handler, 'params') else (None, None)
                if x is not None and y is not None:
                    pyautogui.click(x=x, y=y)
                    return True
            elif "mouse_right_click" in action_name:
                x, y = action_handler.params.get("x", None), action_handler.params.get("y", None) if hasattr(action_handler, 'params') else (None, None)
                if x is not None and y is not None:
                    pyautogui.rightClick(x=x, y=y)
                    return True
            elif "mouse_move" in action_name:
                x, y = action_handler.params.get("x", None), action_handler.params.get("y", None) if hasattr(action_handler, 'params') else (None, None)
                if x is not None and y is not None:
                    pyautogui.moveTo(x=x, y=y)
                    return True
                    
            # Handle keyboard actions
            elif action_name.startswith("key_"):
                key = action_name[4:]  # Remove "key_" prefix
                if len(key) == 1 or key.lower() in ["up", "down", "left", "right", "enter", "esc", "escape", "tab"]:
                    pyautogui.press(key)
                    return True
            
            # If we get here, try to call execute() if it exists
            if hasattr(action_handler, 'execute'):
                action_handler.execute()
                return True
                
            return False
        except Exception as e:
            self.logger.error(f"Error in direct action execution: {str(e)}")
            return False
    
    def _show_action_feedback(self, action: int):
        """
        Show visual feedback for the action being taken
        
        Args:
            action: Action index
        """
        try:
            if action >= len(self.action_handlers):
                return
                
            action_handler = self.action_handlers[action]
            if hasattr(action_handler, 'name') and "mouse" in action_handler.name:
                # For mouse actions, show a brief highlight
                x, y = action_handler.params.get("x", None), action_handler.params.get("y", None)
                if x is not None and y is not None:
                    # Move to position first
                    current_x, current_y = pyautogui.position()
                    pyautogui.moveTo(x, y, duration=0.2)
                    time.sleep(0.1)
                    # Return to original position if needed
                    if "move" not in action_handler.name:
                        pyautogui.moveTo(current_x, current_y, duration=0.1)
        except Exception as e:
            self.logger.warning(f"Error showing action feedback: {str(e)}")
    
    def _log_action(self, action: int):
        """
        Log action to file for tracking
        
        Args:
            action: Action index
        """
        try:
            action_name = "unknown"
            if action < len(self.action_handlers):
                action_handler = self.action_handlers[action]
                if hasattr(action_handler, 'name'):
                    action_name = action_handler.name
                    
            with open(self.action_log_path, "a") as f:
                timestamp = datetime.datetime.now().strftime("%H:%M:%S")
                f.write(f"[{timestamp}] Step {self.total_steps}: Action {action} - {action_name}\n")
        except Exception as e:
            self.logger.warning(f"Failed to log action: {str(e)}")
    
    def _handle_discovery_action(self) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """
        Handle special discovery actions like menu exploration
        
        Returns:
            observation, reward, terminated, truncated, info
        """
        try:
            # Run menu exploration
            self.logger.info("Exploring random menu")
            result = self.menu_explorer.explore_random_menu()
            
            # Update stats
            if result.get("success", False):
                self.stats["menus_explored"] += 1
                elements_found = len(result.get("elements", []))
                self.stats["ui_elements_discovered"] += elements_found
                self.logger.info(f"Menu exploration successful, found {elements_found} elements")
                reward = 0.5 + (0.1 * elements_found)  # Reward for exploration
            else:
                self.logger.info("Menu exploration failed")
                reward = -0.1  # Small penalty for failed exploration
                
            # Get observation
            obs = self._get_observation()
            terminated = False
            truncated = False
            info = {"stats": self.stats, "menu_exploration": result}
            
            return obs, reward, terminated, truncated, info
            
        except Exception as e:
            self.logger.error(f"Error in discovery action: {str(e)}")
            obs = self._get_dummy_observation()
            return obs, -0.2, False, False, {"stats": self.stats, "error": str(e)}
    
    def _get_dummy_observation(self) -> np.ndarray:
        """
        Create a dummy observation when regular observation fails
        
        Returns:
            A placeholder observation
        """
        # Create a black observation with correct shape
        shape = self.observation_space.shape
        return np.zeros(shape, dtype=np.uint8)
        
    def _get_observation(self) -> Dict[str, np.ndarray]:
        """
        Get the current observation from the environment
        
        Returns:
            Dictionary of observations
        """
        try:
            # Get the base observation from the parent class
            if hasattr(super(), 'get_observation'):
                return super().get_observation()
            
            # Fallback to using the interface directly
            if hasattr(self, 'interface') and self.interface is not None:
                # Get game metrics
                metrics = {}
                
                # Population
                population = self.interface.get_population()
                if population is not None:
                    metrics['population'] = np.array([population], dtype=np.float32)
                else:
                    metrics['population'] = np.array([0.0], dtype=np.float32)
                
                # Happiness
                happiness = self.interface.get_happiness()
                if happiness is not None:
                    metrics['happiness'] = np.array([happiness], dtype=np.float32)
                else:
                    metrics['happiness'] = np.array([0.0], dtype=np.float32)
                
                # Budget
                budget = self.interface.get_budget_balance()
                if budget is not None:
                    metrics['budget_balance'] = np.array([budget], dtype=np.float32)
                else:
                    metrics['budget_balance'] = np.array([0.0], dtype=np.float32)
                
                # Traffic
                traffic = self.interface.get_traffic()
                if traffic is not None:
                    metrics['traffic'] = np.array([traffic], dtype=np.float32)
                else:
                    metrics['traffic'] = np.array([0.0], dtype=np.float32)
                
                return metrics
            
            # If all else fails, return a dummy observation
            return {
                'population': np.array([0.0], dtype=np.float32),
                'happiness': np.array([0.0], dtype=np.float32),
                'budget_balance': np.array([0.0], dtype=np.float32),
                'traffic': np.array([0.0], dtype=np.float32)
            }
            
        except Exception as e:
            self.logger.error(f"Error getting observation: {str(e)}")
            return {
                'population': np.array([0.0], dtype=np.float32),
                'happiness': np.array([0.0], dtype=np.float32),
                'budget_balance': np.array([0.0], dtype=np.float32),
                'traffic': np.array([0.0], dtype=np.float32)
            }
    
    def _calculate_reward(self) -> float:
        """
        Calculate the reward based on the current state
        
        Returns:
            The calculated reward value
        """
        try:
            # Start with a small negative reward to encourage efficiency
            reward = -0.01
            
            # Get current metrics
            if hasattr(self, 'interface') and self.interface is not None:
                population = self.interface.get_population() or 0
                happiness = self.interface.get_happiness() or 0
                budget = self.interface.get_budget_balance() or 0
                
                # Reward for population growth
                if population > self.stats.get("last_population", 0):
                    population_increase = population - self.stats.get("last_population", 0)
                    reward += 0.1 * min(1.0, population_increase / 100.0)  # Cap at 1.0 for large increases
                
                # Reward for happiness improvement
                if happiness > self.stats.get("last_happiness", 0):
                    happiness_increase = happiness - self.stats.get("last_happiness", 0)
                    reward += 0.1 * min(1.0, happiness_increase / 10.0)  # Cap at 1.0 for large increases
                
                # Reward for budget improvement
                if budget > self.stats.get("last_budget", 0):
                    budget_increase = budget - self.stats.get("last_budget", 0)
                    reward += 0.05 * min(1.0, budget_increase / 1000.0)  # Cap at 1.0 for large increases
                
                # Update last metrics
                self.stats["last_population"] = population
                self.stats["last_happiness"] = happiness
                self.stats["last_budget"] = budget
            
            # Reward for successful interactions
            if self.stats.get("successful_interactions", 0) > self.stats.get("last_successful_interactions", 0):
                reward += 0.05
                self.stats["last_successful_interactions"] = self.stats.get("successful_interactions", 0)
            
            # Reward for discoveries
            if self.stats.get("ui_elements_discovered", 0) > self.stats.get("last_ui_elements_discovered", 0):
                elements_discovered = self.stats.get("ui_elements_discovered", 0) - self.stats.get("last_ui_elements_discovered", 0)
                reward += 0.1 * elements_discovered
                self.stats["last_ui_elements_discovered"] = self.stats.get("ui_elements_discovered", 0)
            
            # Update total rewards
            self.stats["total_rewards"] += reward
            
            return reward
            
        except Exception as e:
            self.logger.error(f"Error calculating reward: {str(e)}")
            return 0.0
    
    @property
    def action_delay(self) -> float:
        """Get the delay between actions"""
        return self._action_delay
        
    @action_delay.setter
    def action_delay(self, value: float):
        """Set the delay between actions"""
        self._action_delay = max(0.0, min(5.0, value))  # Clamp between 0 and 5 seconds
    
    def reset(self, **kwargs) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Reset the environment and discovery state.
        
        Returns:
            Tuple of (observation, info)
        """
        # Reset the parent environment
        obs, info = super().reset(**kwargs)
        
        # Reset the action sequence
        self.current_action_sequence = []
        
        # Reset tutorial tracking for this episode
        self.current_tutorial = None
        self.tutorial_progress = 0
        
        # Include discovery info
        info["discovery"] = {
            "discovered_ui_elements": len(self.discovered_ui_elements),
            "discovered_actions": len(self.discovered_actions),
            "completed_tutorials": len(self.discovered_tutorials),
            "discovery_phase": self.discovery_phase,
            "discovery_stats": self.stats
        }
        
        # Focus the game window
        if self.window_manager:
            self.window_manager.focus_window()
        time.sleep(1)  # Give time for the window to gain focus
        
        return obs, info
    
    def _save_successful_sequence(self, sequence: Dict[str, Any]) -> None:
        """Save a successful action sequence to file."""
        try:
            # Create a file for this sequence
            sequence_id = len(self.successful_action_sequences)
            filename = os.path.join(self.debug_dir, f"successful_sequence_{sequence_id}.json")
            
            with open(filename, "w") as f:
                json.dump(sequence, f, indent=2)
                
            self.logger.info(f"Saved successful action sequence to {filename}")
            
        except Exception as e:
            self.logger.error(f"Error saving successful sequence: {str(e)}")
    
    def close(self) -> None:
        """Close the environment and save discovery data."""
        # Save discovered UI elements
        try:
            with open(os.path.join(self.debug_dir, "discovered_ui_elements.json"), "w") as f:
                json.dump(self.discovered_ui_elements, f, indent=2)
                
            with open(os.path.join(self.debug_dir, "discovered_actions.json"), "w") as f:
                json.dump(self.discovered_actions, f, indent=2)
                
            with open(os.path.join(self.debug_dir, "discovery_stats.json"), "w") as f:
                json.dump(self.stats, f, indent=2)
                
            self.logger.info("Saved discovery data to files")
            
        except Exception as e:
            self.logger.error(f"Error saving discovery data: {str(e)}")
        
        # Close parent
        super().close() 