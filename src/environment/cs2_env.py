import time
import gymnasium as gym
from gymnasium import spaces
import numpy as np
from typing import Dict, Any, Tuple, List, Optional, Union
import logging

from ..interface.base_interface import BaseInterface
from ..interface.vision_interface import VisionInterface
from ..interface.api_interface import APIInterface


class CS2Environment(gym.Env):
    """
    Cities: Skylines 2 environment for reinforcement learning.
    
    This class wraps the game interface into a standard OpenAI Gym environment.
    """
    
    metadata = {'render.modes': ['human', 'rgb_array']}
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the environment.
        
        Args:
            config: Configuration dictionary
        """
        super().__init__()
        self.config = config
        self.logger = logging.getLogger("CS2Environment")
        
        # Create the game interface based on configuration
        interface_type = config["interface"]["type"]
        if interface_type == "api":
            try:
                self.interface = APIInterface(config)
                if not self.interface.connect():
                    self.logger.warning("Failed to connect to the game via API. Falling back to vision interface.")
                    self.interface = VisionInterface(config)
            except Exception as e:
                self.logger.warning(f"Error initializing API interface: {str(e)}. Falling back to vision interface.")
                self.interface = VisionInterface(config)
        else:
            self.interface = VisionInterface(config)
        
        # Connect to the game
        if not self.interface.connect():
            self.logger.error("Failed to connect to the game.")
            raise RuntimeError("Failed to connect to the game.")
        
        # Define observation space
        self._setup_observation_space()
        
        # Define action space
        self._setup_action_space()
        
        # State tracking
        self.episode_steps = 0
        self.max_episode_steps = config["environment"]["max_episode_steps"]
        self.current_state = None
        self.total_reward = 0.0
        self.last_metrics = {}
    
    def _setup_observation_space(self):
        """Set up the observation space."""
        obs_config = self.config["environment"]["observation_space"]
        spaces_dict = {}
        
        # Visual observation
        if obs_config["include_visual"]:
            height, width = obs_config["image_size"]
            if obs_config["grayscale"]:
                visual_space = spaces.Box(
                    low=0, high=255, 
                    shape=(height, width, 1),
                    dtype=np.uint8
                )
            else:
                visual_space = spaces.Box(
                    low=0, high=255, 
                    shape=(height, width, 3),
                    dtype=np.uint8
                )
            spaces_dict["visual"] = visual_space
        
        # Metrics observation
        if obs_config["include_metrics"]:
            # Define a space for each metric
            metrics = obs_config["metrics"]
            for metric in metrics:
                spaces_dict[metric] = spaces.Box(
                    low=-float('inf'), high=float('inf'),
                    shape=(1,), dtype=np.float32
                )
        
        # Use Dict space if we have multiple observation components
        if len(spaces_dict) > 1:
            self.observation_space = spaces.Dict(spaces_dict)
        else:
            # If only visual or only metrics, use that directly
            self.observation_space = list(spaces_dict.values())[0]
    
    def _setup_action_space(self):
        """Set up the action space."""
        action_config = self.config["environment"]["action_space"]
        
        # Collect all possible actions
        all_actions = []
        
        # Add zoning actions
        for zone_type in action_config["zone"]:
            all_actions.append({"type": "zone", "zone_type": zone_type})
        
        # Add infrastructure actions
        for infra_type in action_config["infrastructure"]:
            all_actions.append({"type": "infrastructure", "infra_type": infra_type})
        
        # Add budget actions
        for budget_action in action_config["budget"]:
            all_actions.append({"type": "budget", "budget_action": budget_action})
        
        # Define the action space as a Discrete space with one action per option
        self.action_space = spaces.Discrete(len(all_actions))
        
        # Store the action mapping for later use
        self.action_mapping = all_actions
    
    def step(self, action: int) -> Tuple[Dict[str, np.ndarray], float, bool, bool, Dict[str, Any]]:
        """
        Take a step in the environment.
        
        Args:
            action: Action to take (index in the action_mapping)
            
        Returns:
            Tuple of (observation, reward, terminated, truncated, info)
        """
        if not 0 <= action < len(self.action_mapping):
            self.logger.warning(f"Invalid action index: {action}")
            action = 0  # Default to the first action
        
        # Get the action details from the mapping
        action_details = self.action_mapping[action]
        
        # Perform the action
        success = self.interface.perform_action(action_details)
        if not success:
            self.logger.warning(f"Failed to perform action: {action_details}")
        
        # Wait for the game to process the action
        time.sleep(0.5)
        
        # Get the new state
        game_state = self.interface.get_game_state()
        
        # Extract the observation
        observation = self._extract_observation(game_state)
        
        # Calculate reward
        reward = self._calculate_reward(game_state)
        self.total_reward += reward
        
        # Check if the episode is done
        self.episode_steps += 1
        terminated = self.interface.is_game_over()
        truncated = self.episode_steps >= self.max_episode_steps
        
        # Store current state for rendering
        self.current_state = game_state
        
        # Additional info
        info = {
            "metrics": game_state.get("metrics", {}),
            "timestep": self.episode_steps,
            "total_reward": self.total_reward,
            "action": action_details
        }
        
        return observation, reward, terminated, truncated, info
    
    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
        """
        Reset the environment.
        
        Args:
            seed: Random seed
            options: Additional options
            
        Returns:
            Tuple of (observation, info)
        """
        super().reset(seed=seed)
        
        # Reset the game
        if not self.interface.reset_game():
            self.logger.warning("Failed to reset the game. Attempting to reconnect...")
            self.interface.disconnect()
            if not self.interface.connect():
                self.logger.error("Failed to reconnect to the game.")
                raise RuntimeError("Failed to reconnect to the game.")
        
        # Set the game speed
        game_speed = self.config["environment"]["time_scale"]
        self.interface.set_game_speed(game_speed)
        
        # Reset internal state
        self.episode_steps = 0
        self.total_reward = 0.0
        self.last_metrics = {}
        
        # Get initial state
        game_state = self.interface.get_game_state()
        observation = self._extract_observation(game_state)
        
        # Store current state for rendering
        self.current_state = game_state
        
        info = {
            "metrics": game_state.get("metrics", {}),
            "timestep": self.episode_steps,
            "total_reward": self.total_reward
        }
        
        return observation, info
    
    def render(self, mode='human'):
        """
        Render the environment.
        
        Args:
            mode: Rendering mode ('human' or 'rgb_array')
            
        Returns:
            If mode is 'rgb_array', returns the current visual observation
        """
        if self.current_state is None:
            return None
        
        if mode == 'rgb_array':
            return self.current_state.get("visual_observation", None)
        
        # 'human' mode - game is already being rendered in the window
        return None
    
    def close(self):
        """Close the environment."""
        if hasattr(self, 'interface') and self.interface is not None:
            self.interface.disconnect()
    
    def _extract_observation(self, game_state: Dict[str, Any]) -> Dict[str, np.ndarray]:
        """
        Extract observation from game state based on configuration.
        
        Args:
            game_state: Game state dictionary
            
        Returns:
            Observation dictionary or array
        """
        obs_config = self.config["environment"]["observation_space"]
        observation = {}
        
        # Visual observation
        if obs_config["include_visual"]:
            visual_obs = game_state.get("visual_observation", None)
            if visual_obs is None:
                # Create an empty image if no visual observation is available
                height, width = obs_config["image_size"]
                channels = 1 if obs_config["grayscale"] else 3
                visual_obs = np.zeros((height, width, channels), dtype=np.uint8)
            observation["visual"] = visual_obs
        
        # Metrics observation
        if obs_config["include_metrics"]:
            metrics = game_state.get("metrics", {})
            self.last_metrics = metrics.copy()  # Store for reward calculation
            
            for metric in obs_config["metrics"]:
                value = metrics.get(metric, 0.0)
                observation[metric] = np.array([value], dtype=np.float32)
        
        # If only one observation type, return it directly
        if len(observation) == 1 and "visual" in observation:
            return observation["visual"]
        
        return observation
    
    def _calculate_reward(self, game_state: Dict[str, Any]) -> float:
        """
        Calculate reward based on game state.
        
        Args:
            game_state: Game state dictionary
            
        Returns:
            Calculated reward
        """
        reward_config = self.config["environment"]["reward"]
        metrics = game_state.get("metrics", {})
        
        reward = 0.0
        
        # Calculate population growth reward
        if "population" in metrics and "population" in self.last_metrics:
            population_growth = metrics["population"] - self.last_metrics.get("population", 0)
            reward += population_growth * reward_config["population_growth"]
        
        # Calculate happiness reward
        if "happiness" in metrics:
            reward += metrics["happiness"] * reward_config["happiness"]
        
        # Calculate budget balance reward
        if "budget_balance" in metrics:
            reward += metrics["budget_balance"] * reward_config["budget_balance"]
        
        # Calculate traffic flow reward
        if "traffic_flow" in metrics:
            reward += metrics["traffic_flow"] * reward_config["traffic_flow"]
        
        # Apply bankruptcy penalty
        if "budget_balance" in metrics and metrics["budget_balance"] < 0:
            reward += reward_config["bankruptcy_penalty"]
        
        # Apply pollution penalty
        if "pollution" in metrics:
            reward += metrics["pollution"] * reward_config["pollution_penalty"]
        
        return reward 