import time
import gymnasium as gym
from gymnasium import spaces
import numpy as np
from typing import Dict, Any, Tuple, List, Optional, Union
import logging

from ..interface.base_interface import BaseInterface
from ..interface.vision_interface import VisionInterface
from ..interface.api_interface import APIInterface
from ..interface.auto_vision_interface import AutoVisionInterface


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
        elif interface_type == "auto_vision":
            try:
                self.logger.info("Using auto-detection vision interface.")
                self.interface = AutoVisionInterface(config)
            except Exception as e:
                self.logger.warning(f"Error initializing auto-detection interface: {str(e)}. Falling back to standard vision interface.")
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
        time.sleep(0.2)  # Reduced from 0.5 to speed up learning
        
        # Get the new state
        game_state = self.interface.get_game_state()
        
        # Extract the observation
        observation = self._extract_observation(game_state)
        
        # Calculate reward
        reward = self._calculate_reward(game_state)
        self.total_reward += reward
        
        # Update metrics tracking for early termination
        if not hasattr(self, 'recent_rewards'):
            self.recent_rewards = []
        
        self.recent_rewards.append(reward)
        if len(self.recent_rewards) > 20:  # Track last 20 rewards
            self.recent_rewards = self.recent_rewards[-20:]
        
        # Check if the episode is done
        self.episode_steps += 1
        terminated = self.interface.is_game_over()
        
        # Early termination conditions for efficiency:
        # 1. If there's a severe budget deficit that's been growing for several steps
        # 2. If the agent is stuck with negative rewards for too long
        if not terminated and self.episode_steps > 50:  # Don't terminate too early
            # Check for bankruptcy or financial crisis
            budget_balance = game_state.get("metrics", {}).get("budget_balance", 0)
            
            # Severe budget deficit that's likely unrecoverable
            if budget_balance < -5000:
                self.logger.info("Early termination due to severe budget deficit")
                terminated = True
            
            # Check if agent is stuck with negative rewards
            if len(self.recent_rewards) >= 20:
                avg_recent_reward = sum(self.recent_rewards) / len(self.recent_rewards)
                # If it's getting consistently negative rewards for a while, early terminate
                if avg_recent_reward < -0.1 and all(r <= 0 for r in self.recent_rewards[-10:]):
                    self.logger.info("Early termination due to consistent negative rewards")
                    terminated = True
        
        truncated = self.episode_steps >= self.max_episode_steps
        
        # Store current state for rendering
        self.current_state = game_state
        
        # Additional info
        info = {
            "metrics": game_state.get("metrics", {}),
            "timestep": self.episode_steps,
            "total_reward": self.total_reward,
            "action": action_details,
            "early_termination": terminated and not self.interface.is_game_over()
        }
        
        return observation, reward, terminated, truncated, info
    
    def reset(self, seed=None, options=None):
        """
        Reset the environment to initial state.
        
        Args:
            seed: Optional random seed for reproducibility
            options: Optional dictionary with additional options
            
        Returns:
            observation: Initial observation
            info: Additional information dictionary
        """
        # Set seed if provided
        if seed is not None:
            np.random.seed(seed)
            
        # Reset interface and get initial observation
        self.interface.reset()
        observation = self._get_observation()
        
        # Reset internal state
        self.current_step = 0
        self.last_metrics = None
        self.cumulative_reward = 0.0
        
        # Initialize info dictionary
        info = {
            "metrics": self._get_metrics(),
            "game_state": "running",
            "current_step": self.current_step
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
        reward_config = self.config["environment"].get("reward_function", {})
        metrics = game_state.get("metrics", {})
        
        reward = 0.0
        
        # Track growth trends using moving averages
        if not hasattr(self, 'metric_history'):
            self.metric_history = {key: [] for key in ['population', 'happiness', 'budget_balance', 'traffic']}
            self.history_max_size = 5  # Track last 5 values
            
        # Track explored areas for map exploration rewards
        if not hasattr(self, 'exploration_grid'):
            # Divide map into sections for tracking exploration
            self.exploration_grid = np.zeros((10, 10), dtype=bool)  # 10x10 grid representing the map
            self.exploration_reward_given = 0.0  # Track total exploration reward given
            self.max_exploration_reward = 10.0  # Cap on exploration rewards
        
        # Update history
        for key in self.metric_history:
            if key in metrics:
                self.metric_history[key].append(metrics[key])
                # Keep history limited to max size
                if len(self.metric_history[key]) > self.history_max_size:
                    self.metric_history[key] = self.metric_history[key][-self.history_max_size:]
        
        # SEVERE PENALTY FOR ZERO POPULATION after initial period
        if metrics.get("population", 0) == 0 and self.episode_steps > 50:
            reward -= 1.0  # Strong penalty for zero population after startup period
        
        # Calculate population growth reward (more sensitive to changes)
        if "population" in metrics and "population" in self.last_metrics:
            population_growth = metrics["population"] - self.last_metrics.get("population", 0)
            
            # Base reward for growth
            reward += population_growth * reward_config.get("population_growth", 0.1)
            
            # POPULATION DECLINE PENALTY - much harsher than before
            if population_growth < 0:
                decline_penalty = population_growth * 3.0  # Triple the penalty for decline
                reward += decline_penalty
            
            # Bonus reward for accelerating growth
            if len(self.metric_history['population']) >= 3:
                recent_growth_rate = population_growth
                previous_growth = self.metric_history['population'][-2] - self.metric_history['population'][-3]
                if recent_growth_rate > previous_growth and recent_growth_rate > 0:
                    reward += 0.2  # Bonus for accelerating positive growth
        
        # Calculate happiness reward (weighted by population)
        if "happiness" in metrics:
            happiness_score = metrics["happiness"]
            population = metrics.get("population", 1)
            
            # Scale happiness reward by population size (more valuable in larger cities)
            normalized_population = min(1.0, population / 10000)  # Cap at 10,000 population
            happiness_reward = happiness_score * reward_config.get("happiness", 0.05) * (0.5 + normalized_population)
            reward += happiness_reward
            
            # Extra reward for maintaining high happiness
            if happiness_score > 80:
                reward += 0.1  # Bonus for very happy citizens
        
        # Calculate budget balance reward - less emphasis on initial budget
        if "budget_balance" in metrics:
            budget_balance = metrics["budget_balance"]
            
            # Adjust for starting with 1M - care less about absolute value, more about trend
            starting_budget = 1000000  # 1 million starting budget
            budget_change = budget_balance - starting_budget
            
            # Only small reward for budget above starting
            if budget_balance > starting_budget:
                normalized_gain = min(1.0, (budget_balance - starting_budget) / 500000)  # Scale by 500K
                reward += normalized_gain * 0.05  # Very small reward for budget growth
            else:
                # More significant penalty for losing money from starting amount
                budget_penalty = max(-1.0, (budget_balance - starting_budget) / 500000)
                reward += budget_penalty * 0.1
        
        # Calculate traffic flow reward (lower is better)
        if "traffic" in metrics:
            traffic = metrics["traffic"]
            # Invert traffic score so lower traffic = better reward
            traffic_score = max(0, 100 - traffic) / 100
            reward += traffic_score * reward_config.get("traffic_flow", 0.1)
            
            # Bonus for very low traffic
            if traffic < 30:
                reward += 0.1
        
        # Land use balance reward
        if all(key in metrics for key in ["residential_ratio", "commercial_ratio", "industrial_ratio"]):
            # Calculate balance - closer to ideal ratios gets higher reward
            # Ideal: ~60% residential, ~25% commercial, ~15% industrial
            residential_diff = abs(metrics["residential_ratio"] - 0.6)
            commercial_diff = abs(metrics["commercial_ratio"] - 0.25)
            industrial_diff = abs(metrics["industrial_ratio"] - 0.15)
            
            balance_score = 1.0 - (residential_diff + commercial_diff + industrial_diff)
            reward += balance_score * 0.1  # Reward for good zone balance
            
        # MAP EXPLORATION REWARD - track and reward exploring new areas
        if hasattr(self, 'current_state') and 'camera_position' in self.current_state:
            camera_pos = self.current_state['camera_position']
            if isinstance(camera_pos, (list, tuple)) and len(camera_pos) >= 2:
                # Get position and convert to grid coordinates (0-9 range for 10x10 grid)
                x, y = camera_pos[0], camera_pos[1]
                
                # Convert world coordinates to grid indices (adjust these based on your map size)
                map_size = 16000  # Typical Cities:Skylines 2 map size in world units
                grid_x = min(9, max(0, int(x / map_size * 10)))
                grid_y = min(9, max(0, int(y / map_size * 10)))
                
                # If this is a newly explored cell, give reward
                if not self.exploration_grid[grid_x, grid_y] and self.exploration_reward_given < self.max_exploration_reward:
                    self.exploration_grid[grid_x, grid_y] = True
                    
                    # Calculate how much of the map has been explored
                    explored_percentage = np.sum(self.exploration_grid) / 100.0
                    
                    # Give a reward for discovering new areas, with diminishing returns
                    exploration_reward = 0.2
                    reward += exploration_reward
                    self.exploration_reward_given += exploration_reward
                    
                    # Log exploration progress
                    if explored_percentage % 0.2 < 0.01:  # Log at 20%, 40%, etc.
                        self.logger.info(f"Map exploration progress: {explored_percentage:.1%}")
        
        # Apply bankruptcy penalty 
        if "budget_balance" in metrics and metrics["budget_balance"] < 0:
            # Scale penalty by severity of deficit
            deficit_severity = abs(metrics["budget_balance"]) / max(1, metrics.get("population", 100))
            reward += min(0, -0.1 - deficit_severity) * reward_config.get("bankruptcy_penalty", 1.0)
        
        # Apply pollution penalty
        if "pollution" in metrics:
            pollution_penalty = -metrics["pollution"] * reward_config.get("pollution_penalty", 0.05)
            reward += pollution_penalty
        
        # Success bonuses for milestone achievements
        if "population" in metrics and "population" in self.last_metrics:
            # Population milestone rewards - more frequent early milestones
            milestones = [50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000, 50000, 100000]
            for milestone in milestones:
                if self.last_metrics.get("population", 0) < milestone <= metrics["population"]:
                    reward += 5.0  # Significant one-time reward for reaching milestone
                    self.logger.info(f"Milestone achieved: {milestone} population!")
        
        return reward
    
    def _get_observation(self):
        """
        Get the current observation from the game state.
        
        Returns:
            Observation dictionary or array based on game state
        """
        game_state = self.interface.get_game_state()
        return self._extract_observation(game_state)
    
    def get_observation(self):
        """
        Public method to get the current observation from the game state.
        
        Returns:
            Observation dictionary or array based on game state
        """
        return self._get_observation()
    
    def _get_metrics(self):
        """
        Get the current metrics from the game state.
        
        Returns:
            Dictionary of current game metrics
        """
        game_state = self.interface.get_game_state()
        return game_state.get("metrics", {}) 