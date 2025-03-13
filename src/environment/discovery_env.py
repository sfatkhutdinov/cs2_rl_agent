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
import cv2
import winsound  # For audio feedback
from tkinter import Tk, Label, BOTH, TOP, X  # For overlay window

from src.environment.vision_guided_env import VisionGuidedCS2Environment
from src.interface.ollama_vision_interface import OllamaVisionInterface
from src.interface.window_manager import WindowManager
from src.interface.focus_helper import FocusHelper  # Import our new focus helper
from src.actions.action_handler import ActionHandler
from src.actions.menu_explorer import MenuExplorer
from src.environment.cs2_env import CS2Environment


class DiscoveryEnvironment(VisionGuidedCS2Environment):
    """
    An environment that focuses on discovering game mechanics through exploration,
    guided by vision intelligence, and learns from tutorials with minimal predefined structure.
    """
    
    def __init__(self, 
                 config: Dict[str, Any] = None,
                 discovery_frequency: float = 0.3,
                 tutorial_frequency: float = 0.3, 
                 random_action_frequency: float = 0.2,
                 exploration_randomness: float = 0.5,
                 logger: Optional[logging.Logger] = None):
        """
        Initialize the discovery-based environment.
        
        Args:
            config: Configuration dictionary
            discovery_frequency: How often to try discovering new UI elements
            tutorial_frequency: How often to look for tutorials
            random_action_frequency: How often to perform completely random actions
            exploration_randomness: How random exploration should be (0=focused, 1=random)
            logger: Logger instance
        """
        # Initialize logger first
        self.logger = logger or logging.getLogger("DiscoveryEnv")
        
        # Set up debug directory first to avoid attribute errors
        self.debug_dir = os.path.join("logs", "discovery_debug")
        os.makedirs(self.debug_dir, exist_ok=True)
        
        # Initialize tracking variables
        self.total_steps = 0  # Track total steps taken in the environment
        self.discovery_frequency = discovery_frequency
        self.tutorial_frequency = tutorial_frequency
        self.random_action_frequency = random_action_frequency
        self.exploration_randomness = exploration_randomness
        
        # Add vision guidance settings
        self.vision_guidance_enabled = True
        self.vision_guidance_frequency = 0.2
        self.consecutive_vision_attempts = 0
        self.max_consecutive_attempts = 3
        self.vision_cooldown = 0
        self.vision_cooldown_steps = 10
        self.last_vision_query = 0
        self.vision_cache_ttl = 300  # 5 minutes in seconds
        self.vision_cache = {}
        self.background_analysis = None
        self.vision_update_lock = threading.Lock()
        self.vision_guidance_cache = {}
        
        # Create overlay window for action feedback
        self.overlay_window = None
        self.overlay_label = None
        self.setup_overlay_window()
        
        # Action display duration in seconds
        self.action_display_duration = 2.0
        self.current_action_thread = None
        
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
        
        # Initialize focus helper for better window management
        self.focus_helper = FocusHelper()
        
        # Setup focus lost/restored callbacks
        def on_focus_lost():
            self.logger.warning("Game window lost focus - actions may fail")
            if self.overlay_window:
                self.display_action("FOCUS LOST", color="red")
        
        def on_focus_restored():
            self.logger.info("Game window focus restored")
            if self.overlay_window:
                self.display_action("FOCUS RESTORED", color="green")
        
        self.focus_helper.set_focus_callbacks(
            on_focus_lost=on_focus_lost,
            on_focus_restored=on_focus_restored
        )
        
        # Create a base environment to pass to the parent
        from src.environment.cs2_env import CS2Environment
        
        # Extract configs from the main config
        base_env_config = config.get("environment", {}) if config else {}
        observation_config = config.get("observation", {}) if config else {}
        vision_config = config.get("vision", {}) if config else {}
        
        # Create a proper config for CS2Environment
        cs2_config = {
            "environment": base_env_config,
            "observation": observation_config,
            "vision": vision_config,
            "interface": {
                "type": "vision",
                "vision": {
                    "screen_region": [0, 0, 1920, 1080],
                    "debug_mode": True,
                    "debug_dir": "debug/vision",
                    "use_ollama": True,
                    "ollama_model": "llama3.2-vision:latest",
                    "ollama_url": "http://localhost:11434/api/generate",
                    "ocr_confidence": 0.6,
                    "template_matching_threshold": 0.8,
                    "max_retries": 3,
                    "retry_delay": 1.0
                }
            }
        }
        
        # Create the base environment
        base_env = CS2Environment(cs2_config)
        
        # Call the parent constructor with the base environment
        from src.environment.autonomous_env import AutonomousEnvironment
        AutonomousEnvironment.__init__(
            self,
            base_env=base_env,
            exploration_frequency=discovery_frequency,
            random_action_frequency=random_action_frequency,
            menu_exploration_buffer_size=50,
            logger=self.logger
        )
        
        # Start focus monitoring
        self.focus_helper.start_focus_monitoring()
        
        # Verify window focus is set before proceeding
        self.focus_helper.ensure_focus(force_topmost=True)
        self.logger.info("Focus management initialized and active")
        
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
        
        # Set reward focus mode
        self.reward_focus = base_env_config.get("reward_focus", "balanced")  # Can be "goal", "explore", or "balanced"
        self.logger.info(f"Reward focus set to: {self.reward_focus}")
        
        self.logger.info(f"Discovery environment initialized with action feedback: {self.show_action_feedback}")
        
        # Initialize action handlers
        self.action_handlers = []
        self.env = None
    
    def setup_overlay_window(self):
        """Create a semi-transparent overlay window to display actions"""
        try:
            # Instead of using Tkinter, we'll use a simple flag and the existing OSD functionality
            self.current_action_text = "AGENT READY"
            self.last_action_time = time.time()
            self.action_display_duration = 2.0
            
            # Log successful creation
            if hasattr(self, 'logger') and self.logger:
                self.logger.info("Action feedback system initialized successfully")
                
        except Exception as e:
            # Handle the error safely
            error_msg = f"Error initializing action feedback: {str(e)}"
            if hasattr(self, 'logger') and self.logger:
                self.logger.error(error_msg)
            else:
                print(error_msg)
    
    def update_overlay(self):
        """This method is no longer needed with the CV2-based approach"""
        pass
    
    def display_action(self, action_name: str, color: str = "lime"):
        """Display an action and play a sound based on action type"""
        try:
            # Store the action text and time
            self.current_action_text = action_name
            self.last_action_time = time.time()
            
            # Play a sound based on action type
            self.play_action_sound(action_name)
            
            # Show the action on screen
            self._show_action_osd(action_name)
                
        except Exception as e:
            if hasattr(self, 'logger') and self.logger:
                self.logger.warning(f"Error displaying action: {str(e)}")
    
    def clear_action_display(self):
        """Clear the action display"""
        self.current_action_text = ""
    
    def play_action_sound(self, action_name: str):
        """Play a sound based on the action type"""
        try:
            # Determine which sound to play based on action type
            if "key" in action_name or "ctrl" in action_name or "shift" in action_name:
                # Keyboard action sound (higher pitch)
                winsound.Beep(1000, 100)  # 1000 Hz for 100ms
            elif "click" in action_name or "mouse" in action_name:
                # Mouse action sound (medium pitch)
                winsound.Beep(800, 100)  # 800 Hz for 100ms
            elif "zoom" in action_name or "pan" in action_name or "rotate" in action_name:
                # Camera action sound (lower pitch)
                winsound.Beep(600, 100)  # 600 Hz for 100ms
            elif "explore" in action_name:
                # Exploration action sound (double beep)
                winsound.Beep(900, 50)
                time.sleep(0.05)
                winsound.Beep(1200, 50)
            else:
                # Default sound for other actions
                winsound.Beep(700, 100)  # 700 Hz for 100ms
        except Exception as e:
            if hasattr(self, 'logger') and self.logger:
                self.logger.debug(f"Error playing action sound: {str(e)}")
    
    def _show_action_osd(self, action_name: str):
        """
        Display the action name as an on-screen display
        
        Args:
            action_name: Name of the action to display
        """
        try:
            # Capture current screen
            screen = pyautogui.screenshot()
            screen_np = np.array(screen)
            screen_np = cv2.cvtColor(screen_np, cv2.COLOR_RGB2BGR)
            
            # Add action name text
            font = cv2.FONT_HERSHEY_SIMPLEX
            text = f"ACTION: {action_name}"
            text_size = cv2.getTextSize(text, font, 1, 2)[0]
            
            # Position at top of screen
            x = (screen_np.shape[1] - text_size[0]) // 2  # Center horizontally
            y = 50
            
            # Draw background rectangle
            cv2.rectangle(
                screen_np,
                (x - 10, y - 40),
                (x + text_size[0] + 10, y + 10),
                (0, 0, 0),
                -1
            )
            
            # Draw text
            cv2.putText(
                screen_np,
                text,
                (x, y),
                font,
                1,
                (0, 255, 0),
                2,
                cv2.LINE_AA
            )
            
            # Save the annotated screen temporarily
            timestamp = int(time.time() * 1000)
            osd_path = os.path.join(self.debug_dir, f"action_osd_{timestamp}.png")
            cv2.imwrite(osd_path, screen_np)
            
            # Clean up older OSD images to prevent disk usage buildup
            self._cleanup_osd_images()
            
            # Create a small persistent window to display the action
            # This is more reliable than Tkinter in this context
            try:
                cv2.namedWindow("Agent Action", cv2.WINDOW_NORMAL)
                cv2.setWindowProperty("Agent Action", cv2.WND_PROP_TOPMOST, 1)
                cv2.resizeWindow("Agent Action", 400, 100)
                
                # Create a black image with text
                action_display = np.zeros((100, 400, 3), dtype=np.uint8)
                
                # Draw text
                cv2.putText(
                    action_display,
                    text,
                    (20, 60),
                    font,
                    0.8,
                    (0, 255, 0),
                    2,
                    cv2.LINE_AA
                )
                
                # Show the window
                cv2.imshow("Agent Action", action_display)
                cv2.waitKey(1)  # Update without blocking
                
                # Schedule window to close
                threading.Timer(2.0, lambda: cv2.destroyWindow("Agent Action")).start()
            except Exception as window_error:
                if hasattr(self, 'logger') and self.logger:
                    self.logger.debug(f"Could not create action window: {window_error}")
                # Not critical, can continue without the window
            
        except Exception as e:
            if hasattr(self, 'logger') and self.logger:
                self.logger.warning(f"Error showing on-screen display: {str(e)}")
            else:
                print(f"Error showing on-screen display: {str(e)}")
                
    def _cleanup_osd_images(self):
        """Clean up older OSD images to prevent disk usage buildup"""
        try:
            osd_files = [f for f in os.listdir(self.debug_dir) if f.startswith("action_osd_")]
            # Sort by creation time (oldest first)
            osd_files.sort()
            # Keep only the latest 20 images
            if len(osd_files) > 20:
                for old_file in osd_files[:-20]:
                    try:
                        os.remove(os.path.join(self.debug_dir, old_file))
                    except:
                        pass
        except Exception as e:
            if hasattr(self, 'logger') and self.logger:
                self.logger.warning(f"Error cleaning up OSD images: {str(e)}")
            else:
                print(f"Error cleaning up OSD images: {str(e)}")
    
    def _log_action(self, action: int = None, action_idx: int = None, action_name: str = None, is_vision_guided: bool = False, success: bool = False, reward: float = None, info: dict = None, **kwargs):
        """
        Log action to file for tracking
        
        Args:
            action: Action index
            action_idx: Alternative action index parameter (used by parent class)
            action_name: Name of the action (if provided directly)
            is_vision_guided: Whether the action was guided by vision
            success: Whether the action was successful
            reward: Reward received for the action
            info: Additional information about the action
            **kwargs: Additional arguments that might be passed
        """
        try:
            # Use action_idx if provided, otherwise use action
            actual_action = action_idx if action_idx is not None else action
            
            # If action_name is not provided, try to determine it from the action_handlers
            if action_name is None or action_name == "unknown":
                if actual_action is not None and actual_action < len(self.action_handlers):
                    action_handler = self.action_handlers[actual_action]
                    if hasattr(action_handler, 'name'):
                        action_name = action_handler.name
                    else:
                        action_name = "unknown"
                else:
                    action_name = "unknown"
                
            # Log to file
            with open(self.action_log_path, "a") as f:
                timestamp = datetime.datetime.now().strftime("%H:%M:%S")
                guided_str = " (vision guided)" if is_vision_guided else ""
                success_str = " (success)" if success else ""
                reward_str = f", reward: {reward:.4f}" if reward is not None else ""
                
                f.write(f"[{timestamp}] Step {self.total_steps}: Action {actual_action} - {action_name}{guided_str}{success_str}{reward_str}\n")
        except Exception as e:
            if hasattr(self, 'logger') and self.logger:
                self.logger.warning(f"Failed to log action: {str(e)}")
            else:
                print(f"Failed to log action: {str(e)}")
    
    def _handle_discovery_action(self) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """
        Handle special discovery actions like menu exploration
        
        Returns:
            observation, reward, terminated, truncated, info
        """
        try:
            # Run menu exploration
            if hasattr(self, 'logger') and self.logger:
                self.logger.info("Exploring random menu")
            
            # Check if menu_explorer exists
            if not hasattr(self, 'menu_explorer') or self.menu_explorer is None:
                if hasattr(self, 'logger') and self.logger:
                    self.logger.warning("Menu explorer not available")
                obs = self._get_observation()
                return obs, -0.1, False, False, {"stats": self.stats, "error": "Menu explorer not available"}
                
            # Try to explore a random menu
            result = self.menu_explorer.explore_random_menu()
            
            # Handle case where result is None
            if result is None:
                if hasattr(self, 'logger') and self.logger:
                    self.logger.warning("Menu exploration returned None")
                obs = self._get_observation()
                return obs, -0.1, False, False, {"stats": self.stats, "error": "Menu exploration failed"}
            
            # Update stats
            if result.get("success", False):
                self.stats["menus_explored"] += 1
                elements_found = len(result.get("elements", []))
                self.stats["ui_elements_discovered"] += elements_found
                if hasattr(self, 'logger') and self.logger:
                    self.logger.info(f"Menu exploration successful, found {elements_found} elements")
                reward = 0.5 + (0.1 * elements_found)  # Reward for exploration
            else:
                if hasattr(self, 'logger') and self.logger:
                    self.logger.info("Menu exploration failed")
                reward = -0.1  # Small penalty for failed exploration
                
            # Get observation
            obs = self._get_observation()
            terminated = False
            truncated = False
            info = {"stats": self.stats, "menu_exploration": result}
            
            return obs, reward, terminated, truncated, info
            
        except Exception as e:
            if hasattr(self, 'logger') and self.logger:
                self.logger.error(f"Error in discovery action: {str(e)}")
            else:
                print(f"Error in discovery action: {str(e)}")
            obs = self._get_dummy_observation()
            # Make sure obs is a dictionary with the expected keys
            if not isinstance(obs, dict):
                obs = {
                    'population': np.array([0.0], dtype=np.float32),
                    'happiness': np.array([0.0], dtype=np.float32),
                    'budget_balance': np.array([0.0], dtype=np.float32),
                    'traffic': np.array([0.0], dtype=np.float32)
                }
            return obs, -0.2, False, False, {"stats": self.stats, "error": str(e)}
    
    def _get_dummy_observation(self) -> Dict[str, np.ndarray]:
        """
        Create a dummy observation when regular observation fails
        
        Returns:
            A placeholder observation dictionary
        """
        # Create a dictionary with zero values for all expected metrics
        return {
            'population': np.array([0.0], dtype=np.float32),
            'happiness': np.array([0.0], dtype=np.float32),
            'budget_balance': np.array([0.0], dtype=np.float32),
            'traffic': np.array([0.0], dtype=np.float32)
        }
        
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
                
                # Collect metrics by explicitly hovering over UI elements
                collected_metrics = self._collect_game_metrics()
                
                # Population
                population = collected_metrics.get('population') or self.interface.get_population()
                if population is not None:
                    metrics['population'] = np.array([population], dtype=np.float32)
                else:
                    metrics['population'] = np.array([0.0], dtype=np.float32)
                
                # Happiness
                happiness = collected_metrics.get('happiness') or self.interface.get_happiness()
                if happiness is not None:
                    metrics['happiness'] = np.array([happiness], dtype=np.float32)
                else:
                    metrics['happiness'] = np.array([0.0], dtype=np.float32)
                
                # Budget
                budget = collected_metrics.get('budget') or self.interface.get_budget_balance()
                if budget is not None:
                    metrics['budget_balance'] = np.array([budget], dtype=np.float32)
                else:
                    metrics['budget_balance'] = np.array([0.0], dtype=np.float32)
                
                # Traffic
                traffic = collected_metrics.get('traffic') or self.interface.get_traffic()
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
    
    def _collect_game_metrics(self) -> Dict[str, float]:
        """
        Collect game metrics by hovering over UI elements and extracting values from tooltips
        
        Returns:
            Dictionary of metric values
        """
        # Initialize metrics dictionary
        metrics = {}
        
        # Check if it's time to refresh metrics
        current_time = time.time()
        if hasattr(self, 'last_metrics_time') and current_time - self.last_metrics_time < 30:
            # Use cached metrics if they're less than 30 seconds old
            if hasattr(self, 'cached_metrics'):
                return self.cached_metrics
        
        # Save current mouse position to restore later
        try:
            original_x, original_y = pyautogui.position()
            metrics_collected = False
            
            # Get the position of the UI elements using the interface
            if hasattr(self, 'interface') and self.interface is not None:
                # Get the screen coordinates of metrics UI elements
                ui_positions = self._get_metrics_ui_positions()
                
                if ui_positions:
                    # Try to collect population
                    if 'population' in ui_positions:
                        pop_x, pop_y = ui_positions['population']
                        population = self._hover_and_extract_value(pop_x, pop_y, metric_type='population')
                        if population is not None:
                            metrics['population'] = population
                    
                    # Try to collect happiness
                    if 'happiness' in ui_positions:
                        happy_x, happy_y = ui_positions['happiness']
                        happiness = self._hover_and_extract_value(happy_x, happy_y, metric_type='happiness')
                        if happiness is not None:
                            metrics['happiness'] = happiness
                    
                    # Try to collect budget
                    if 'budget' in ui_positions:
                        budget_x, budget_y = ui_positions['budget']
                        budget = self._hover_and_extract_value(budget_x, budget_y, metric_type='budget')
                        if budget is not None:
                            metrics['budget_balance'] = budget
                    
                    # Try to collect traffic
                    if 'traffic' in ui_positions:
                        traffic_x, traffic_y = ui_positions['traffic']
                        traffic = self._hover_and_extract_value(traffic_x, traffic_y, metric_type='traffic')
                        if traffic is not None:
                            metrics['traffic'] = traffic
                    
                    metrics_collected = True
            
            # If we collected any metrics, save them in the cache
            if metrics_collected:
                self.cached_metrics = metrics
                self.last_metrics_time = current_time
            
            # Return to original mouse position
            pyautogui.moveTo(original_x, original_y, duration=0.1)
            
        except Exception as e:
            if hasattr(self, 'logger') and self.logger:
                self.logger.warning(f"Error collecting game metrics: {str(e)}")
        
        return metrics
    
    def _get_metrics_ui_positions(self) -> Dict[str, Tuple[int, int]]:
        """
        Get the screen positions of key UI metric elements
        
        Returns:
            Dictionary mapping metric names to (x, y) screen coordinates
        """
        positions = {}
        
        try:
            # Use the vision interface to locate UI elements
            vision_interface = self._get_vision_interface()
            if not vision_interface:
                return positions
            
            # Find population icon position
            population_pos = vision_interface.find_template('population')
            if population_pos:
                positions['population'] = population_pos
            
            # Find happiness icon position
            happiness_pos = vision_interface.find_template('happiness')
            if happiness_pos:
                positions['happiness'] = happiness_pos
            
            # Find budget icon position
            budget_pos = vision_interface.find_template('budget')
            if budget_pos:
                positions['budget'] = budget_pos
            
            # Find traffic icon position
            traffic_pos = vision_interface.find_template('traffic')
            if traffic_pos:
                positions['traffic'] = traffic_pos
            
        except Exception as e:
            if hasattr(self, 'logger') and self.logger:
                self.logger.warning(f"Error finding UI metric positions: {str(e)}")
        
        return positions
    
    def _hover_and_extract_value(self, x: int, y: int, metric_type: str) -> Optional[float]:
        """
        Hover over a UI element and extract the numeric value from tooltip
        
        Args:
            x: X coordinate to hover over
            y: Y coordinate to hover over
            metric_type: Type of metric ('population', 'happiness', etc.)
            
        Returns:
            Numeric value if extraction successful, None otherwise
        """
        try:
            # Move to the position
            pyautogui.moveTo(x, y, duration=0.2)
            
            # Wait for tooltip to appear
            time.sleep(0.5)
            
            # Capture the screen with the tooltip
            screenshot = pyautogui.screenshot()
            screenshot_np = np.array(screenshot)
            
            # Convert to grayscale for OCR
            gray = cv2.cvtColor(screenshot_np, cv2.COLOR_RGB2GRAY)
            
            # Use OCR to extract text from the tooltip region
            # For now, we'll use the Ollama vision model if available
            if hasattr(self, 'interface') and hasattr(self.interface, 'query_vision_model'):
                # Create a prompt for the vision model to extract the number
                prompt = f"""
                Look at this screenshot from Cities: Skylines 2 showing a tooltip for the {metric_type}.
                Extract ONLY the numeric value for the {metric_type}.
                Return ONLY the number, nothing else. If you see "1.2K", convert it to 1200, etc.
                """
                
                # Query the vision model
                result = self.interface.query_vision_model(
                    prompt=prompt,
                    image=screenshot_np
                )
                
                # Try to parse the result as a number
                if result and isinstance(result, str):
                    # Remove any non-numeric characters except decimal points
                    numeric_chars = ''.join(c for c in result if c.isdigit() or c == '.')
                    
                    # Try to convert to float
                    try:
                        return float(numeric_chars)
                    except ValueError:
                        pass
            
            # If Ollama vision model failed, save the image for debugging
            debug_path = os.path.join(self.debug_dir, f"{metric_type}_tooltip_{int(time.time())}.png")
            cv2.imwrite(debug_path, cv2.cvtColor(screenshot_np, cv2.COLOR_RGB2BGR))
            
            # Return None if extraction failed
            return None
            
        except Exception as e:
            if hasattr(self, 'logger') and self.logger:
                self.logger.warning(f"Error extracting {metric_type} value: {str(e)}")
            return None
    
    def _calculate_reward(self) -> float:
        """
        Calculate the reward based on the current state
        
        Returns:
            The calculated reward value
        """
        try:
            # Set multipliers based on reward focus
            goal_multiplier = 1.0
            exploration_multiplier = 1.0
            
            if self.reward_focus == "goal":
                # Emphasize city-building goals
                goal_multiplier = 2.0
                exploration_multiplier = 0.5
                if hasattr(self, 'logger') and self.logger:
                    self.logger.debug("Using goal-focused reward multipliers")
            elif self.reward_focus == "explore":
                # Emphasize exploration
                goal_multiplier = 0.5
                exploration_multiplier = 2.0
                if hasattr(self, 'logger') and self.logger:
                    self.logger.debug("Using exploration-focused reward multipliers")
            else:
                # Balanced approach
                if hasattr(self, 'logger') and self.logger:
                    self.logger.debug("Using balanced reward multipliers")

            # Start with a small negative reward to encourage efficiency
            reward = -0.01
            
            # Get current metrics
            if hasattr(self, 'interface') and self.interface is not None:
                # Collect metrics by hovering over UI elements if needed
                if not hasattr(self, 'cached_metrics') or time.time() - self.last_metrics_time > 60:
                    self._collect_game_metrics()
                
                # Use cached metrics if available, otherwise use interface methods
                if hasattr(self, 'cached_metrics'):
                    population = self.cached_metrics.get('population')
                    happiness = self.cached_metrics.get('happiness')
                    budget = self.cached_metrics.get('budget_balance')
                else:
                    population = self.interface.get_population() or 0
                    happiness = self.interface.get_happiness() or 0
                    budget = self.interface.get_budget_balance() or 0

                # Initialize last metrics if not present
                if not hasattr(self, 'last_population'):
                    self.last_population = 0
                if not hasattr(self, 'last_happiness'):
                    self.last_happiness = 0
                if not hasattr(self, 'last_budget'):
                    self.last_budget = 0
                
                # POPULATION REWARDS - Higher weight for population growth
                if population > self.last_population:
                    population_increase = population - self.last_population
                    # Base reward for any increase (with goal multiplier)
                    population_reward = 0.2 * goal_multiplier * min(1.0, population_increase / 100.0)
                    
                    # Bonus rewards for significant milestones
                    if self.last_population < 1000 and population >= 1000:
                        population_reward += 1.0  # Bonus for reaching 1000 population
                    elif self.last_population < 5000 and population >= 5000:
                        population_reward += 2.0  # Bonus for reaching 5000 population
                    elif self.last_population < 10000 and population >= 10000:
                        population_reward += 3.0  # Bonus for reaching 10000 population
                    
                    reward += population_reward
                    
                    if hasattr(self, 'logger') and self.logger:
                        self.logger.info(f"Population increased from {self.last_population} to {population}, reward: +{population_reward:.2f}")
                elif population < self.last_population:
                    # Small penalty for population decrease
                    population_decrease = self.last_population - population
                    penalty = -0.1 * min(1.0, population_decrease / 100.0)
                    reward += penalty
                    
                    if hasattr(self, 'logger') and self.logger:
                        self.logger.info(f"Population decreased from {self.last_population} to {population}, penalty: {penalty:.2f}")
                
                # HAPPINESS REWARDS - Significant rewards for happiness improvement
                if happiness > self.last_happiness:
                    happiness_increase = happiness - self.last_happiness
                    # Higher coefficient for happiness (0.3 instead of 0.1)
                    happiness_reward = 0.3 * goal_multiplier * min(1.0, happiness_increase / 10.0)
                    
                    # Bonus for high happiness levels
                    if self.last_happiness < 80 and happiness >= 80:
                        happiness_reward += 0.5  # Bonus for high happiness
                    elif self.last_happiness < 90 and happiness >= 90:
                        happiness_reward += 1.0  # Bonus for very high happiness
                    
                    reward += happiness_reward
                    
                    if hasattr(self, 'logger') and self.logger:
                        self.logger.info(f"Happiness increased from {self.last_happiness} to {happiness}, reward: +{happiness_reward:.2f}")
                elif happiness < self.last_happiness:
                    # Larger penalty for happiness decrease
                    happiness_decrease = self.last_happiness - happiness
                    penalty = -0.2 * min(1.0, happiness_decrease / 10.0)
                    reward += penalty
                    
                    if hasattr(self, 'logger') and self.logger:
                        self.logger.info(f"Happiness decreased from {self.last_happiness} to {happiness}, penalty: {penalty:.2f}")
                
                # BUDGET REWARDS - Balanced rewards for budget
                if budget > self.last_budget:
                    budget_increase = budget - self.last_budget
                    budget_reward = 0.1 * goal_multiplier * min(1.0, budget_increase / 1000.0)
                    
                    # Bonus for getting out of debt
                    if self.last_budget < 0 and budget >= 0:
                        budget_reward += 1.0  # Significant bonus for getting out of debt
                    
                    reward += budget_reward
                    
                    if hasattr(self, 'logger') and self.logger:
                        self.logger.info(f"Budget increased from {self.last_budget} to {budget}, reward: +{budget_reward:.2f}")
                elif budget < self.last_budget:
                    # Penalty for budget decrease, especially into debt
                    budget_decrease = self.last_budget - budget
                    penalty_factor = 0.15 if budget < 0 else 0.1  # Higher penalty for going into debt
                    penalty = -penalty_factor * min(1.0, budget_decrease / 1000.0)
                    reward += penalty
                    
                    if hasattr(self, 'logger') and self.logger:
                        self.logger.info(f"Budget decreased from {self.last_budget} to {budget}, penalty: {penalty:.2f}")
                
                # SUSTAINED GROWTH BONUS - Reward consistent improvements over time
                if hasattr(self, 'consecutive_improvements'):
                    if population > self.last_population and happiness > self.last_happiness:
                        self.consecutive_improvements += 1
                        if self.consecutive_improvements >= 3:
                            # Bonus for maintaining growth for 3+ consecutive steps
                            sustained_bonus = 0.5
                            reward += sustained_bonus
                            if hasattr(self, 'logger') and self.logger:
                                self.logger.info(f"Sustained growth for {self.consecutive_improvements} steps, bonus: +{sustained_bonus}")
                    else:
                        self.consecutive_improvements = 0
                else:
                    self.consecutive_improvements = 0
                
                # Update last metrics
                self.last_population = population
                self.last_happiness = happiness
                self.last_budget = budget
            
            # EXPLORATION REWARDS (reduced compared to city metrics)
            # Reward for successful interactions (reduced from 0.05 to 0.02)
            if self.stats.get("successful_interactions", 0) > self.stats.get("last_successful_interactions", 0):
                exploration_reward = 0.02 * exploration_multiplier
                reward += exploration_reward
                self.stats["last_successful_interactions"] = self.stats.get("successful_interactions", 0)
                
                if hasattr(self, 'logger') and self.logger:
                    self.logger.info(f"Successful interaction reward: +{exploration_reward}")
            
            # Reward for discoveries (reduced from 0.1 to 0.05 per element)
            if self.stats.get("ui_elements_discovered", 0) > self.stats.get("last_ui_elements_discovered", 0):
                elements_discovered = self.stats.get("ui_elements_discovered", 0) - self.stats.get("last_ui_elements_discovered", 0)
                discovery_reward = 0.05 * exploration_multiplier * elements_discovered
                reward += discovery_reward
                self.stats["last_ui_elements_discovered"] = self.stats.get("ui_elements_discovered", 0)
                
                if hasattr(self, 'logger') and self.logger:
                    self.logger.info(f"Discovered {elements_discovered} UI elements, reward: +{discovery_reward}")
            
            # Update total rewards
            self.stats["total_rewards"] += reward
            
            # Log overall reward
            if hasattr(self, 'logger') and self.logger:
                self.logger.info(f"Total reward for this step: {reward:.4f}")
            
            return reward
            
        except Exception as e:
            if hasattr(self, 'logger') and self.logger:
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
    
    def reset(self, **kwargs):
        """
        Reset the environment.
        
        Returns:
            Initial observation
        """
        # Reset the base environment
        obs = self.base_env.reset(**kwargs)
        
        # Reset internal state
        self.current_action_sequence = []
        self.episode_steps = 0
        self.total_reward = 0.0
        self.last_action = None
        self.last_reward = 0.0
        
        # Reset focus
        self.focus_helper.ensure_focus()
        
        # Reset action feedback
        if self.overlay_window:
            self.display_action("RESET", color="blue")
        
        # Return the observation
        return self._process_observation(obs)
    
    def step(self, action: int) -> Tuple[Dict[str, np.ndarray], float, bool, bool, Dict[str, Any]]:
        """
        Take a step in the environment.
        
        Args:
            action: The action to take
            
        Returns:
            Tuple of (observation, reward, terminated, truncated, info)
        """
        # Ensure window focus before taking action
        if not self.focus_helper.is_window_focused():
            self.logger.warning("Window not focused before action, attempting to regain focus")
            self.focus_helper.ensure_focus()
            time.sleep(0.2)  # Small delay after focus attempt
        
        # Take the step in the base environment
        obs, reward, done, truncated, info = self.base_env.step(action)
        
        # Process the observation to ensure it's compatible with the model
        obs = self._process_observation(obs)
        
        # Update internal state
        self.episode_steps += 1
        self.total_steps += 1
        self.total_reward += reward
        self.last_action = action
        self.last_reward = reward
        
        # Update action statistics
        action_str = str(action)
        if action_str not in self.action_stats:
            self.action_stats[action_str] = {
                "count": 0,
                "total_reward": 0.0,
                "success_count": 0
            }
        self.action_stats[action_str]["count"] += 1
        self.action_stats[action_str]["total_reward"] += reward
        if reward > 0:
            self.action_stats[action_str]["success_count"] += 1
        
        # Display action feedback
        if self.overlay_window:
            color = "green" if reward > 0 else "red" if reward < 0 else "white"
            self.display_action(f"Action: {action}, Reward: {reward:.2f}", color=color)
        
        return obs, reward, done, truncated, info
    
    def _process_observation(self, obs):
        """
        Process the observation to ensure it's compatible with the model.
        
        Args:
            obs: Original observation
            
        Returns:
            Processed observation
        """
        # If the observation is a dictionary, ensure all values are numpy arrays
        if isinstance(obs, dict):
            processed_obs = {}
            for key, value in obs.items():
                if isinstance(value, np.ndarray):
                    processed_obs[key] = value
                elif isinstance(value, (int, float)):
                    processed_obs[key] = np.array([value], dtype=np.float32)
                else:
                    try:
                        processed_obs[key] = np.array(value, dtype=np.float32)
                    except:
                        self.logger.warning(f"Could not convert observation key {key} to numpy array. Using zeros.")
                        processed_obs[key] = np.zeros((1,), dtype=np.float32)
            return processed_obs
        return obs
    
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
        """
        Close the environment.
        """
        # Stop focus monitoring
        if hasattr(self, 'focus_helper'):
            self.focus_helper.stop_focus_monitoring()
            self.logger.info("Focus monitoring stopped")
        
        # Close overlay window
        if self.overlay_window:
            try:
                self.overlay_window.destroy()
                self.overlay_window = None
            except:
                pass
                
        # Call parent close method
        super().close()
        
        self.logger.info("Environment closed")
    
    def step(self, action: int) -> Tuple[Dict[str, np.ndarray], float, bool, bool, Dict[str, Any]]:
        """
        Take a step in the environment.
        
        Args:
            action: The action to take
            
        Returns:
            Tuple of (observation, reward, terminated, truncated, info)
        """
        # Ensure window focus before taking action
        if not self.focus_helper.is_window_focused():
            self.logger.warning("Window not focused before action, attempting to regain focus")
            self.focus_helper.ensure_focus()
            time.sleep(0.2)  # Small delay after focus attempt
        
        # Proceed with the original step method
        return super().step(action) 