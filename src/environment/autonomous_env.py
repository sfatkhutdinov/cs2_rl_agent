import logging
import time
import os
import gymnasium as gym
import numpy as np
from gymnasium import spaces
from src.interface.auto_vision_interface import AutoVisionInterface
from src.interface.menu_explorer import MenuExplorer
from src.environment.cs2_env import CS2Environment
from stable_baselines3.common.monitor import Monitor
import pyautogui
from enum import Enum, auto
from collections import namedtuple

# Define action types
class ActionType(Enum):
    CAMERA_CONTROL = auto()
    UI_INTERACTION = auto()
    GAME_ACTION = auto()
    KEYBOARD = auto()

# Define action structure
Action = namedtuple('Action', ['name', 'action_fn', 'action_type'])

class AutonomousCS2Environment(gym.Wrapper):
    """
    A wrapper around the base CS2Environment that adds autonomous exploration
    capabilities, allowing the agent to fully explore the game on its own.
    """
    
    def __init__(self, base_env, exploration_frequency=0.3, random_action_frequency=0.2, 
                 menu_exploration_buffer_size=50, logger=None):
        """
        Initialize the autonomous environment wrapper.
        
        Args:
            base_env: The base CS2Environment to wrap
            exploration_frequency: How often to trigger exploratory behavior (0-1)
            random_action_frequency: How often to take completely random actions (0-1)
            menu_exploration_buffer_size: Size of the buffer for storing discovered menu items
            logger: Optional logger instance
        """
        # Ensure base environment is properly initialized
        if not isinstance(base_env, gym.Env):
            raise ValueError("base_env must be a gymnasium.Env instance")
            
        super().__init__(base_env)
        
        self.logger = logger or logging.getLogger("AutonomousEnv")
        self.exploration_frequency = exploration_frequency
        self.random_action_frequency = random_action_frequency
        
        # Used for tracking exploration progress
        self.exploration_counter = 0
        self.total_steps = 0
        self.successful_actions = 0
        self.failed_actions = 0
        self.menu_exploration_counter = 0
        
        # Create menu explorer
        self.menu_explorer = MenuExplorer(logger)
        
        # Tracking game state for exploration
        self.last_observation = None
        self.last_reward = 0
        self.last_done = False
        self.last_info = {}
        
        # Buffer for storing information about discovered menu items
        self.menu_discovery_buffer = []
        self.menu_exploration_buffer_size = menu_exploration_buffer_size
        
        # Define a more comprehensive action space to enable autonomous exploration
        self._extend_action_space()
        
        self.logger.info("Autonomous environment initialized with extended action space")
    
    def _extend_action_space(self):
        """Extend the action space with additional control actions"""
        # Create wrappers for the interface methods
        actions = [
            # Define some basic exploration actions
            Action(
                name="explore_menu",
                action_fn=lambda: self._explore_menu(),
                action_type=ActionType.UI_INTERACTION
            ),
            Action(
                name="explore_ui",
                action_fn=lambda: self._explore_ui(),
                action_type=ActionType.UI_INTERACTION
            ),
            Action(
                name="random_action",
                action_fn=lambda: self._take_random_base_action(),
                action_type=ActionType.GAME_ACTION
            ),
            Action(
                name="repeat_last_action",
                action_fn=lambda: self._repeat_last_action(),
                action_type=ActionType.GAME_ACTION
            ),
            Action(
                name="zoom_in",
                action_fn=lambda: self.env.interface.zoom_with_wheel(5),
                action_type=ActionType.CAMERA_CONTROL
            ),
            Action(
                name="zoom_out",
                action_fn=lambda: self.env.interface.zoom_with_wheel(-5),
                action_type=ActionType.CAMERA_CONTROL
            ),
            Action(
                name="pan_left",
                action_fn=lambda: self.env.interface.pan_view("left"),
                action_type=ActionType.CAMERA_CONTROL
            ),
            Action(
                name="pan_right",
                action_fn=lambda: self.env.interface.pan_view("right"),
                action_type=ActionType.CAMERA_CONTROL
            ),
            Action(
                name="pan_up",
                action_fn=lambda: self.env.interface.pan_view("up"),
                action_type=ActionType.CAMERA_CONTROL
            ),
            Action(
                name="pan_down",
                action_fn=lambda: self.env.interface.pan_view("down"),
                action_type=ActionType.CAMERA_CONTROL
            ),
        ]
        
        # Add camera rotation actions
        actions.extend([
            Action(
                name="rotate_camera_left",
                action_fn=lambda: self.env.interface.rotate_camera_left(),
                action_type=ActionType.CAMERA_CONTROL
            ),
            Action(
                name="rotate_camera_right",
                action_fn=lambda: self.env.interface.rotate_camera_right(),
                action_type=ActionType.CAMERA_CONTROL
            ),
            Action(
                name="reset_camera_rotation",
                action_fn=lambda: self.env.interface.reset_camera_rotation(),
                action_type=ActionType.CAMERA_CONTROL
            ),
            Action(
                name="tilt_camera_up",
                action_fn=lambda: self.env.interface.tilt_camera_up(),
                action_type=ActionType.CAMERA_CONTROL
            ),
            Action(
                name="tilt_camera_down",
                action_fn=lambda: self.env.interface.tilt_camera_down(),
                action_type=ActionType.CAMERA_CONTROL
            ),
        ])
        
        # Add keyboard actions (all keys except restricted ones)
        alphabet = "abcdefghijklmnopqrstuvwxyz"
        numbers = "0123456789"
        special_keys = [
            'space', 'tab', 'enter', 'backspace', 'delete',
            'up', 'down', 'left', 'right', 'home', 'end',
            'pageup', 'pagedown', 'insert', '-', '=', '[', ']',
            '\\', ';', "'", ',', '.', '/'
        ]
        
        # Add all regular keys
        for key in alphabet + numbers + ''.join(special_keys):
            if key != 'esc':  # Skip ESC key
                actions.append(
                    Action(
                        name=f"press_key_{key}",
                        action_fn=lambda k=key: self.env.interface.press_key(k),
                        action_type=ActionType.KEYBOARD
                    )
                )
        
        # Add common modifier combinations (excluding ALT+F4)
        modifiers = ['shift', 'ctrl']
        for modifier in modifiers:
            for key in alphabet + numbers:
                actions.append(
                    Action(
                        name=f"{modifier}_{key}",
                        action_fn=lambda m=modifier, k=key: self.env.interface.press_hotkey(m, k),
                        action_type=ActionType.KEYBOARD
                    )
                )
        
        # Update action space
        original_size = self.env.action_space.n
        self.action_space = gym.spaces.Discrete(len(actions))
        self.action_handlers = actions
        self.logger.info(f"Extended action space from {original_size} to {len(actions)} actions")
    
    def reset(self, seed=None, options=None):
        """
        Reset the environment and optionally set the random seed.
        
        Args:
            seed: Optional random seed for reproducibility
            options: Optional dictionary with additional options
            
        Returns:
            observation: Initial observation
            info: Additional information
        """
        # Set seed if provided
        if seed is not None:
            np.random.seed(seed)
            
        # Reset exploration counters
        self.exploration_counter = 0
        self.menu_exploration_counter = 0
        
        # Reset base environment
        obs, info = self.env.reset()
        
        # Reset menu explorer
        self.menu_explorer.reset()
        
        return obs, info
    
    def step(self, action):
        """
        Take a step in the environment with autonomous exploration capabilities.
        
        Args:
            action: The action to take (index in the action space)
            
        Returns:
            observation, reward, terminated, truncated, info - Standard gymnasium step return values
        """
        self.total_steps += 1
        self.exploration_counter += 1
        
        # Safety check for action index
        if not isinstance(action, (int, np.integer)) or action < 0 or action >= self.action_space.n:
            self.logger.warning(f"Invalid action index: {action}")
            # Replace with a random valid action
            action = np.random.randint(0, self.action_space.n)
        
        # Sometimes override with exploratory actions if we're in exploration mode
        if np.random.random() < self.exploration_frequency:
            # Instead of purely random exploration, use guided exploration based on city state
            action = self._get_guided_exploratory_action()
        else:
            # Check if the action is valid given the current state
            action = self._apply_action_masking(action)
        
        # Process the action
        if 0 <= action < len(self.action_handlers):  # If it's an exploration action and in valid range
            observation, reward, terminated, truncated, info = self._handle_exploration_action(action)
        else:
            # Use base environment for regular actions
            try:
                observation, reward, terminated, truncated, info = self.env.step(action)
                self.successful_actions += 1
            except Exception as e:
                self.logger.warning(f"Error executing action {action}: {str(e)}")
                self.failed_actions += 1
                # Return previous observation with a small penalty
                observation = self.last_observation if self.last_observation is not None else self.reset()[0]
                reward = -0.1
                terminated = self.last_done
                truncated = False
                info = self.last_info.copy() if self.last_info is not None else {}
                info['error'] = str(e)
        
        # Store action history for learning patterns
        if not hasattr(self, 'action_history'):
            self.action_history = []
        self.action_history.append((action, reward))
        if len(self.action_history) > 100:  # Keep last 100 actions
            self.action_history = self.action_history[-100:]
        
        # Update success rate for actions to guide future exploration
        self._update_action_success_rates(action, reward)
        
        # Update last state
        self.last_observation = observation
        self.last_reward = reward
        self.last_done = terminated
        self.last_info = info
        
        # Log progress periodically
        if self.total_steps % 100 == 0:
            self.logger.info(f"Total steps: {self.total_steps}, "
                           f"Successful actions: {self.successful_actions}, "
                           f"Failed actions: {self.failed_actions}")
        
        return observation, reward, terminated, truncated, info
    
    def _apply_action_masking(self, action):
        """
        Apply action masking to prevent illogical or impossible actions.
        
        Args:
            action: The proposed action
            
        Returns:
            Potentially modified action
        """
        # If not a base environment action, no masking needed
        if action >= self.env.action_space.n:
            return action
            
        # Get current metrics to guide masking
        if hasattr(self.env, 'interface') and hasattr(self.env.interface, 'get_metrics'):
            try:
                metrics = self.env.interface.get_metrics()
                
                # Prevent budget actions when budget is healthy
                if action in range(11, 19) and metrics.get('budget_balance', 0) > 1000:
                    # Randomly select a more useful action instead
                    return np.random.randint(0, 11)  # Focus on zoning/infrastructure
                
                # Prevent building expensive infrastructure when budget is low
                if (action in range(5, 11) and metrics.get('budget_balance', 0) < 0 
                    and np.random.random() < 0.7):  # 70% chance to override
                    # Use budget-focused actions instead
                    budget_actions = list(range(11, 19))
                    return np.random.choice(budget_actions)
                    
                # If traffic is high, prioritize roads
                if metrics.get('traffic', 0) > 70 and action not in [5, 6]:  # Not a road action
                    if np.random.random() < 0.6:  # 60% chance to override
                        return 5  # Use road action instead
            except:
                pass  # If metrics can't be fetched, don't mask
        
        return action
    
    def _update_action_success_rates(self, action, reward):
        """
        Update success rates for actions based on rewards received.
        
        Args:
            action: The action taken
            reward: The reward received
        """
        if not hasattr(self, 'action_success_rates'):
            # Initialize success rates for all actions
            self.action_success_rates = {
                i: {'count': 0, 'success': 0, 'avg_reward': 0}
                for i in range(self.action_space.n)
            }
        
        # Update the count for this action
        self.action_success_rates[action]['count'] += 1
        
        # Update success (positive reward = success)
        if reward > 0:
            self.action_success_rates[action]['success'] += 1
            
        # Update average reward using running average
        count = self.action_success_rates[action]['count']
        old_avg = self.action_success_rates[action]['avg_reward']
        self.action_success_rates[action]['avg_reward'] = old_avg + (reward - old_avg) / count
    
    def _get_guided_exploratory_action(self):
        """Get a guided exploratory action that strategically explores the environment"""
        # 5% chance for completely random action to maintain some variety
        if np.random.random() < 0.05:
            return np.random.randint(0, self.action_space.n)
            
        # Use different exploration strategies based on probabilities
        exploration_strategy = np.random.choice([
            'camera_control',  # Move around the map
            'ui_exploration',  # Interact with UI elements
            'game_actions',    # Try game mechanics
            'keyboard_testing' # Try keyboard shortcuts
        ], p=[0.3, 0.3, 0.3, 0.1])
        
        # Get current metrics to guide actions
        metrics = self.env.get_observation().get('metrics', {})
        population = metrics.get('population', 0)
        
        if exploration_strategy == 'camera_control':
            # Prioritize uncovered map areas if we're tracking exploration
            if hasattr(self.env, 'exploration_grid') and hasattr(self.env, 'current_state') and 'camera_position' in self.env.current_state:
                # Get actions related to camera movement
                camera_actions = [i for i, a in enumerate(self.action_handlers) 
                                if a.action_type == ActionType.CAMERA_CONTROL]
                
                # Include rotation actions with higher probability when exploring
                rotation_actions = [i for i, a in enumerate(self.action_handlers)
                                  if a.name in ["rotate_camera_left", "rotate_camera_right", 
                                               "tilt_camera_up", "tilt_camera_down"]]
                
                # Combine regular camera movements with some rotation (70/30 split)
                if np.random.random() < 0.3 and rotation_actions:
                    return np.random.choice(rotation_actions)
                else:
                    return np.random.choice(camera_actions)
            else:
                # Fallback to any camera control action
                camera_actions = [i for i, a in enumerate(self.action_handlers) 
                                if a.action_type == ActionType.CAMERA_CONTROL]
                return np.random.choice(camera_actions) if camera_actions else np.random.randint(0, self.action_space.n)
        
        elif exploration_strategy == 'keyboard_testing':
            # Try keyboard actions
            keyboard_actions = [i for i, a in enumerate(self.action_handlers) 
                              if a.action_type == ActionType.KEYBOARD]
            
            # Prioritize common useful keys in city building games
            priority_keys = [i for i, a in enumerate(self.action_handlers) 
                            if a.action_type == ActionType.KEYBOARD and
                            any(key in a.name for key in ['press_key_b', 'press_key_r', 'press_key_p', 
                                                        'press_key_d', 'press_key_u', 'press_key_1', 
                                                        'press_key_2', 'press_key_3'])]
            
            # 70% chance to use priority keys if available
            if priority_keys and np.random.random() < 0.7:
                return np.random.choice(priority_keys)
            else:
                return np.random.choice(keyboard_actions) if keyboard_actions else np.random.randint(0, self.action_space.n)
                
        # ... existing code for other strategies ...

        # Default: return a random action if no strategy matched
        return np.random.randint(0, self.action_space.n)
    
    def _handle_exploration_action(self, action):
        """
        Handle exploration-specific actions.
        
        Args:
            action: The exploration action to take
            
        Returns:
            observation, reward, terminated, truncated, info - Standard gymnasium step return values
        """
        self.menu_exploration_counter += 1
        
        # Get current screenshot for exploration
        screenshot = None
        try:
            screenshot = self.env.interface.capture_screen()
        except Exception as e:
            self.logger.warning(f"Failed to capture screen: {str(e)}")
        
        # Default return values (will be overridden if action succeeds)
        observation = self.last_observation if self.last_observation is not None else self.reset()[0]
        reward = 0  # Neutral reward by default for exploration
        terminated = False
        truncated = False
        info = {}
        
        try:
            # Check if action index is valid
            if action < 0 or action >= len(self.action_handlers):
                self.logger.warning(f"Invalid exploration action index: {action}")
                return observation, -0.1, terminated, truncated, {"error": "Invalid action index"}
            
            # Get action handler details
            action_handler = self.action_handlers[action]
            action_name = action_handler.name
            action_type = action_handler.action_type
            
            # Add action details to info
            info["exploration_action"] = action_name
            
            # Execute the action using its action_fn
            try:
                success = action_handler.action_fn()
            except Exception as action_error:
                self.logger.warning(f"Error executing action {action_name}: {str(action_error)}")
                success = False
                info["error"] = str(action_error)
            
            # Update info based on action type
            if action_type == ActionType.CAMERA_CONTROL:
                info["camera_action"] = action_name
                reward = 0.01  # Small reward for camera control
            elif action_type == ActionType.UI_INTERACTION:
                info["ui_action"] = action_name
                # Menu exploration already gives rewards in the function
            elif action_type == ActionType.KEYBOARD:
                info["keyboard_action"] = action_name
            
            # Special actions that need additional handling
            if action_name == "explore_menu" and success:
                # Menu explorer might need additional processing beyond what's in action_fn
                if screenshot is not None and hasattr(self.menu_explorer, 'explore_screen'):
                    try:
                        exploration_result = self.menu_explorer.explore_screen(screenshot)
                        
                        if isinstance(exploration_result, dict) and "action" in exploration_result:
                            if exploration_result["action"] in ["click_menu", "click_submenu", "click_discovered"]:
                                # Add to discovery buffer if it's a new element
                                self._update_discovery_buffer(exploration_result)
                                
                                # Small positive reward for discovering new elements
                                reward = 0.05
                                
                                element_name = exploration_result.get("menu_name", 
                                                              exploration_result.get("element_name", "unknown"))
                                if element_name:
                                    info["exploration_element"] = element_name
                    except Exception as explore_error:
                        self.logger.warning(f"Menu exploration error: {str(explore_error)}")
                        info["menu_error"] = str(explore_error)
            
            # Wait a short time to let the game react
            time.sleep(0.05)
            
            # Get new observation after action
            try:
                new_observation = self.env.get_observation()
                if new_observation is not None:
                    observation = new_observation
            except Exception as obs_error:
                self.logger.warning(f"Failed to get observation: {str(obs_error)}")
            
            # Reward meaningful observation changes - with robust None checks
            if self.last_observation is not None and observation is not None:
                # Check if the observation changed meaningfully
                try:
                    if isinstance(observation, dict) and isinstance(self.last_observation, dict):
                        # For dictionary observations from multi-modal environments
                        if ("metrics" in observation and 
                            "metrics" in self.last_observation and
                            observation["metrics"] is not None and
                            self.last_observation["metrics"] is not None):
                            
                            # Make sure we can safely compare the metrics (they should be numpy arrays)
                            if (isinstance(observation["metrics"], np.ndarray) and 
                                isinstance(self.last_observation["metrics"], np.ndarray) and
                                observation["metrics"].shape == self.last_observation["metrics"].shape):
                                try:
                                    metrics_diff = np.sum(np.abs(observation["metrics"] - self.last_observation["metrics"]))
                                    if metrics_diff > 0.1:  # If metrics changed significantly
                                        reward += 0.03  # Small bonus for causing observable changes
                                except Exception as metrics_error:
                                    self.logger.debug(f"Error comparing metrics: {str(metrics_error)}")
                            
                        # Check for individual metrics as well
                        for metric in ["population", "happiness", "budget_balance", "traffic"]:
                            try:
                                if (metric in observation and 
                                    metric in self.last_observation and
                                    observation[metric] is not None and 
                                    self.last_observation[metric] is not None):
                                    
                                    # Ensure we're dealing with numpy arrays of compatible shape
                                    obs_metric = observation[metric]
                                    last_metric = self.last_observation[metric]
                                    
                                    # Convert to numpy arrays if they aren't already
                                    if not isinstance(obs_metric, np.ndarray):
                                        obs_metric = np.array(obs_metric)
                                    if not isinstance(last_metric, np.ndarray):
                                        last_metric = np.array(last_metric)
                                    
                                    # Make sure shapes match before comparison
                                    if obs_metric.shape == last_metric.shape:
                                        metric_diff = np.abs(obs_metric - last_metric).mean()
                                        if metric_diff > 0.1:
                                            reward += 0.01  # Smaller bonus for individual metric changes
                                            self.logger.debug(f"Metric {metric} changed: {metric_diff}")
                            except Exception as metric_error:
                                self.logger.debug(f"Error comparing metric {metric}: {str(metric_error)}")
                                continue  # Skip this metric but continue with others
                    elif isinstance(observation, np.ndarray) and isinstance(self.last_observation, np.ndarray):
                        # For image or vector observations
                        # Check dimensions and dtype match before comparison
                        try:
                            if (observation.shape == self.last_observation.shape and 
                                observation.dtype == self.last_observation.dtype):
                                obs_diff = np.mean(np.abs(observation - self.last_observation))
                                if obs_diff > 0.1:  # If observation changed significantly
                                    reward += 0.02
                        except Exception as array_error:
                            self.logger.debug(f"Error comparing array observations: {str(array_error)}")
                except Exception as compare_error:
                    self.logger.debug(f"Error comparing observations: {str(compare_error)}")
                    # Log additional details to help diagnose issues
                    if isinstance(observation, dict) and isinstance(self.last_observation, dict):
                        for key in set(observation.keys()).union(set(self.last_observation.keys())):
                            obs_val = observation.get(key)
                            last_val = self.last_observation.get(key)
                            self.logger.debug(f"Key: {key}, Current: {type(obs_val)}, Last: {type(last_val)}")
        
        except Exception as e:
            self.logger.error(f"Error in exploration action handling: {str(e)}")
            import traceback
            self.logger.debug(f"Traceback: {traceback.format_exc()}")
            info["error"] = str(e)
            reward = -0.1  # Penalty for errors
        
        return observation, reward, terminated, truncated, info
    
    def _update_discovery_buffer(self, exploration_result):
        """
        Update the buffer of discovered menu elements.
        
        Args:
            exploration_result: Result from menu exploration
        """
        # Create a discovery record
        discovery = {
            "name": exploration_result.get("menu_name", 
                                         exploration_result.get("element_name", "unknown")),
            "position": exploration_result["position"],
            "confidence": exploration_result["confidence"],
            "discovered_at": self.total_steps
        }
        
        # Check if this element is already in the buffer
        for i, item in enumerate(self.menu_discovery_buffer):
            if self.menu_explorer._positions_close(item["position"], discovery["position"]):
                # Update existing entry
                self.menu_discovery_buffer[i] = discovery
                return
        
        # Add new discovery to buffer
        self.menu_discovery_buffer.append(discovery)
        
        # Trim buffer if it exceeds the maximum size
        if len(self.menu_discovery_buffer) > self.menu_exploration_buffer_size:
            # Remove oldest entries
            self.menu_discovery_buffer = sorted(
                self.menu_discovery_buffer, 
                key=lambda x: x["discovered_at"], 
                reverse=True
            )[:self.menu_exploration_buffer_size]
    
    def _has_observation_changed(self, new_observation):
        """
        Check if the observation has changed significantly.
        
        Args:
            new_observation: The new observation to compare
            
        Returns:
            True if observation has changed significantly
        """
        if self.last_observation is None or new_observation is None:
            return True
        
        try:
            # For image observations, check if pixels have changed
            if isinstance(new_observation, np.ndarray) and new_observation.ndim >= 3:
                # Check if a significant number of pixels have changed
                if isinstance(self.last_observation, np.ndarray) and self.last_observation.shape == new_observation.shape:
                    diff = np.abs(new_observation - self.last_observation).mean()
                    return diff > 0.05  # Threshold for significant change
            
            # For dictionary observations, check components
            elif isinstance(new_observation, dict) and isinstance(self.last_observation, dict):
                # Check if visual component has changed
                if ('visual' in new_observation and 'visual' in self.last_observation and
                    new_observation['visual'] is not None and self.last_observation['visual'] is not None):
                    
                    visual_new = new_observation['visual']
                    visual_last = self.last_observation['visual']
                    
                    if (isinstance(visual_new, np.ndarray) and isinstance(visual_last, np.ndarray) and 
                        visual_new.shape == visual_last.shape):
                        visual_diff = np.abs(visual_new - visual_last).mean()
                        if visual_diff > 0.05:
                            return True
                
                # Check if metrics have changed
                for key in new_observation:
                    if key == 'visual':
                        continue
                    
                    if (key in self.last_observation and 
                        new_observation[key] is not None and 
                        self.last_observation[key] is not None):
                        
                        if isinstance(new_observation[key], (int, float)) and isinstance(self.last_observation[key], (int, float)):
                            if abs(new_observation[key] - self.last_observation[key]) > 0.01:
                                return True
                        elif isinstance(new_observation[key], np.ndarray) and isinstance(self.last_observation[key], np.ndarray):
                            if new_observation[key].shape == self.last_observation[key].shape:
                                diff = np.abs(new_observation[key] - self.last_observation[key]).mean()
                                if diff > 0.01:
                                    return True
        except Exception as e:
            self.logger.debug(f"Error in _has_observation_changed: {str(e)}")
            return True  # Assume change on error to be safe
        
        return False 

    def _explore_menu(self):
        """
        Explore the game menus by clicking on menu items.
        
        Returns:
            True if exploration was successful, False otherwise
        """
        # Use menu explorer to find and click on a menu item
        result = self.menu_explorer.explore_random_menu(self.env.interface)
        
        # Update discovery buffer with the result
        self._update_discovery_buffer(result)
        
        return result['success']
    
    def _explore_ui(self):
        """
        Explore the game UI by clicking on UI elements.
        
        Returns:
            True if exploration was successful, False otherwise
        """
        # Get screen dimensions
        screen_region = self.env.interface.screen_region
        width, height = screen_region[2], screen_region[3]
        
        # Generate random coordinates within the screen region
        x = np.random.randint(0, width)
        y = np.random.randint(0, height)
        
        # Click at the random coordinates
        success = self.env.interface.click_at_coordinates(x, y)
        
        return success
    
    def _take_random_base_action(self):
        """
        Take a random action from the base environment's action space.
        
        Returns:
            True if the action was successful, False otherwise
        """
        try:
            # Verify the base environment's action space exists
            if not hasattr(self.env, 'action_space') or self.env.action_space is None:
                self.logger.error("Base environment action space is not accessible")
                return False
                
            # Get the size of the action space safely
            action_space_size = getattr(self.env.action_space, 'n', 0)
            if action_space_size <= 0:
                self.logger.error("Invalid action space size")
                return False
            
            # Choose a random action from the base environment
            random_action = np.random.randint(0, action_space_size)
            
            # Perform the action using the base environment's step method
            try:
                observation, reward, terminated, truncated, info = self.env.step(random_action)
                
                # Track the observation and reward (with safety checks)
                if observation is not None:
                    self.last_observation = observation
                if isinstance(reward, (int, float)):
                    self.last_reward = reward
                self.last_done = bool(terminated or truncated)
                if isinstance(info, dict):
                    self.last_info = info
                
                return True
            except Exception as e:
                self.logger.error(f"Error taking random base action {random_action}: {e}")
                return False
        except Exception as outer_e:
            self.logger.error(f"Exception in random base action selection: {outer_e}")
            return False
    
    def _repeat_last_action(self):
        """
        Repeat the last action that was taken.
        
        Returns:
            True if the action was repeated successfully, False otherwise
        """
        if hasattr(self, 'last_action'):
            try:
                observation, reward, terminated, truncated, info = self.step(self.last_action)
                return True
            except Exception as e:
                self.logger.error(f"Error repeating last action: {e}")
                return False
        else:
            self.logger.warning("No last action to repeat")
            return False 