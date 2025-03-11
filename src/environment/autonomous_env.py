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
        """Extend the base environment's action space with exploration-specific actions."""
        # Get the original action space size
        base_action_space_n = self.env.action_space.n
        
        # Add exploration-specific actions
        exploration_actions = [
            "explore_menu",
            "explore_ui",
            "random_action",
            "repeat_last_action",
            "undo_last_action"
        ]
        
        # Create the extended action space
        self.action_space = spaces.Discrete(base_action_space_n + len(exploration_actions))
        
        # Store mapping of new actions
        self.exploration_actions = {
            base_action_space_n + i: action 
            for i, action in enumerate(exploration_actions)
        }
        
        # Store the size of the base action space
        self.base_action_space_n = base_action_space_n
        
        self.logger.info(f"Extended action space from {base_action_space_n} to {self.action_space.n} actions")
    
    def reset(self):
        """Reset the environment and initialize exploration state."""
        observation = self.env.reset()
        self.last_observation = observation
        self.exploration_counter = 0
        self.menu_discovery_buffer = []
        self.last_reward = 0
        self.last_done = False
        self.last_info = {}
        return observation
    
    def step(self, action):
        """
        Take a step in the environment with autonomous exploration capabilities.
        
        Args:
            action: The action to take (index in the action space)
            
        Returns:
            observation, reward, done, info - Standard gym step return values
        """
        self.total_steps += 1
        self.exploration_counter += 1
        
        # Sometimes override with exploratory actions if we're in exploration mode
        if np.random.random() < self.exploration_frequency:
            action = self._get_exploratory_action()
        
        # Process the action
        if action >= len(self.action_names) - 7:  # If it's an exploration action
            observation, reward, done, info = self._handle_exploration_action(action)
        else:
            # Use base environment for regular actions
            try:
                observation, reward, done, info = self.env.step(action)
                self.successful_actions += 1
            except Exception as e:
                self.logger.warning(f"Error executing action {action}: {str(e)}")
                self.failed_actions += 1
                # Return previous observation with a small penalty
                observation = self.last_observation
                reward = -0.1
                done = self.last_done
                info = self.last_info.copy()
                info['error'] = str(e)
        
        # Update last state
        self.last_observation = observation
        self.last_reward = reward
        self.last_done = done
        self.last_info = info
        
        # Log progress periodically
        if self.total_steps % 100 == 0:
            self.logger.info(f"Total steps: {self.total_steps}, "
                           f"Successful actions: {self.successful_actions}, "
                           f"Failed actions: {self.failed_actions}")
        
        return observation, reward, done, info
    
    def _get_exploratory_action(self):
        """
        Get an action that promotes exploration.
        
        Returns:
            An action index for exploration
        """
        # Different exploration strategies
        if np.random.random() < self.random_action_frequency:
            # Completely random action
            return np.random.randint(0, self.action_space.n)
        
        # Choose one of the exploration-specific actions
        exploration_options = list(self.exploration_actions.values())
        return np.random.choice(exploration_options)
    
    def _handle_exploration_action(self, action):
        """
        Handle exploration-specific actions.
        
        Args:
            action: The exploration action to take
            
        Returns:
            observation, reward, done, info - Standard gym step return values
        """
        self.menu_exploration_counter += 1
        
        # Get current screenshot for exploration
        screenshot = self.env.interface.capture_screen()
        
        # Default return values (will be overridden if action succeeds)
        observation = self.last_observation
        reward = 0  # Neutral reward by default for exploration
        done = False
        info = {"exploration_action": self.action_names.get(action, "unknown")}
        
        try:
            # Handle each exploration action type
            if action == self.exploration_actions["explore_menu"]:
                # Use menu explorer to find and click on menu elements
                exploration_result = self.menu_explorer.explore_screen(screenshot)
                
                if exploration_result["action"] in ["click_menu", "click_submenu", "click_discovered"]:
                    # Click on the discovered element
                    self.env.interface.click_at_coordinates(
                        exploration_result["position"][0], 
                        exploration_result["position"][1]
                    )
                    
                    # Add to discovery buffer if it's a new element
                    self._update_discovery_buffer(exploration_result)
                    
                    # Small positive reward for discovering new elements
                    reward = 0.05
                    
                    info["exploration_element"] = exploration_result.get("menu_name", 
                                                                       exploration_result.get("element_name", "unknown"))
                    time.sleep(0.5)  # Small delay to let UI update
                
                elif exploration_result["action"] == "random_click":
                    # Random exploration click
                    self.env.interface.click_at_coordinates(
                        exploration_result["position"][0], 
                        exploration_result["position"][1]
                    )
                    time.sleep(0.2)
            
            elif action == self.exploration_actions["click_discovered"]:
                # Click on a previously discovered element from the buffer
                if self.menu_discovery_buffer:
                    # Choose a random element from discovery buffer
                    element = np.random.choice(self.menu_discovery_buffer)
                    self.env.interface.click_at_coordinates(
                        element["position"][0], 
                        element["position"][1]
                    )
                    info["clicked_element"] = element.get("name", "unknown")
                    time.sleep(0.3)
            
            elif action == self.exploration_actions["random_click"]:
                # Make a completely random click
                screen_width, screen_height = self.env.interface.screen_size
                x = np.random.randint(0, screen_width)
                y = np.random.randint(0, screen_height)
                self.env.interface.click_at_coordinates(x, y)
                info["random_click"] = (x, y)
                time.sleep(0.2)
            
            elif action == self.exploration_actions["wait_and_observe"]:
                # Just wait and observe changes
                time.sleep(1.0)
                info["waited"] = True
            
            elif action == self.exploration_actions["esc_menu"]:
                # Press ESC to close menus
                self.env.interface.press_key("escape")
                time.sleep(0.3)
                info["pressed_esc"] = True
            
            elif action == self.exploration_actions["scroll_down"]:
                # Scroll down to see more content
                self.env.interface.scroll_down()
                time.sleep(0.2)
                info["scrolled"] = "down"
            
            elif action == self.exploration_actions["scroll_up"]:
                # Scroll up
                self.env.interface.scroll_up()
                time.sleep(0.2)
                info["scrolled"] = "up"
        
        except Exception as e:
            self.logger.warning(f"Error in exploration action {action}: {str(e)}")
            info["error"] = str(e)
            reward = -0.1  # Small penalty for errors
        
        # Capture updated observation
        observation = self.env.get_observation()
        
        # Check if any changes occurred from our exploration
        if self._has_observation_changed(observation):
            reward += 0.02  # Small bonus if observation changed
            info["observation_changed"] = True
        
        return observation, reward, done, info
    
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
        if self.last_observation is None:
            return True
        
        # For image observations, check if pixels have changed
        if isinstance(new_observation, np.ndarray) and new_observation.ndim >= 3:
            # Check if a significant number of pixels have changed
            if isinstance(self.last_observation, np.ndarray) and self.last_observation.shape == new_observation.shape:
                diff = np.abs(new_observation - self.last_observation).mean()
                return diff > 0.05  # Threshold for significant change
        
        # For dictionary observations, check components
        elif isinstance(new_observation, dict) and isinstance(self.last_observation, dict):
            # Check if visual component has changed
            if 'visual' in new_observation and 'visual' in self.last_observation:
                visual_diff = np.abs(new_observation['visual'] - self.last_observation['visual']).mean()
                if visual_diff > 0.05:
                    return True
            
            # Check if metrics have changed
            for key in new_observation:
                if key == 'visual':
                    continue
                if key in self.last_observation:
                    if isinstance(new_observation[key], (int, float)) and isinstance(self.last_observation[key], (int, float)):
                        if abs(new_observation[key] - self.last_observation[key]) > 0.01:
                            return True
        
        return False 