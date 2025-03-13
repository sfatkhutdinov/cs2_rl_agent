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
from ..interface.ollama_vision_interface import OllamaVisionInterface


class CS2Environment(gym.Env):
    """
    Cities: Skylines 2 environment for reinforcement learning.
    
    This class wraps the game interface into a standard OpenAI Gym environment.
    """
    
    metadata = {'render.modes': ['human', 'rgb_array']}
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the CS2 environment.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.logger = logging.getLogger("CS2Environment")
        
        # Initialize fallback metrics
        self.fallback_metrics = {
            "population": 0,
            "happiness": 50.0,
            "budget_balance": 10000.0,
            "traffic": 50.0,
            "noise_pollution": 0.0,
            "air_pollution": 0.0
        }
        
        # Initialize fallback simulation parameters
        self.fallback_growth_rate = 0.05  # Population growth per action
        self.fallback_budget_rate = -100.0  # Budget drain per action
        self.fallback_happiness_decay = -0.1  # Happiness decay per action
        
        # Initialize interface based on config
        interface_type = config.get("interface", {}).get("type", "vision")
        
        if interface_type == "api":
            self.interface = APIInterface(config)
            self.logger.info("Using API interface.")
        elif interface_type == "vision":
            self.interface = VisionInterface(config)
            self.logger.info("Using vision interface.")
        elif interface_type == "ollama_vision":
            self.interface = OllamaVisionInterface(config)
            self.logger.info("Using Ollama vision interface.")
        else:
            self.logger.error(f"Unknown interface type: {interface_type}")
            raise ValueError(f"Unknown interface type: {interface_type}")
        
        # Set up observation and action spaces
        self._setup_observation_space()
        self._setup_action_space()
        
        # Initialize environment state
        self.connected = False
        self.in_fallback_mode = False
        self.total_steps = 0
        self.episode_steps = 0
        self.total_episodes = 0
        self.last_action = None
        self.last_reward = 0.0
        self.last_observation = self._get_fallback_observation()
        
        # Try to connect to the game
        try:
            self.connected = self.interface.connect()
            if not self.connected and config.get("environment", {}).get("use_fallback_mode", True):
                self._setup_fallback_mode()
        except Exception as e:
            self.logger.error(f"Error connecting to game: {str(e)}")
            if config.get("environment", {}).get("use_fallback_mode", True):
                self._setup_fallback_mode()
            else:
                raise e
        
        self.logger.info("Environment initialized successfully.")
        
        # Setup simulation parameters
        env_config = config.get("environment", {})
        self.max_episode_steps = env_config.get("max_episode_steps", 1000)
        self.metrics_update_freq = env_config.get("metrics_update_freq", 10)
        
        # Connect to the game - with fallback mode
        self.use_fallback = config.get("use_fallback_mode", False)
        
        # Initialize time step counter
        self.current_step = 0
        
        # State tracking
        self.current_state = None
        self.total_reward = 0.0
        self.last_metrics = {}
    
    def _setup_fallback_mode(self):
        """
        Setup fallback mode for the environment.
        """
        self.logger.warning("Setting up fallback mode.")
        self.in_fallback_mode = True
        self.connected = False
        
        # Fallback mode is already initialized in __init__
        # We can update values here if needed
        
        self.logger.info("Fallback mode setup complete.")
    
    def _setup_observation_space(self):
        """Set up the observation space."""
        # Get the observation space config, with some flexibility to find it
        obs_config = self.config.get("environment", {}).get("observation_space", {})
        
        # If not found, check for observation config at top level
        if not obs_config and "observation" in self.config:
            obs_config = self.config.get("observation", {})
            
        # Still nothing? Initialize with defaults
        if not obs_config:
            self.logger.warning("No observation configuration found. Using defaults.")
            obs_config = {
                "include_visual": False,
                "include_metrics": True,
                "include_minimap": False,
                "include_screenshot": True,
                "screenshot_width": 224,
                "screenshot_height": 224,
                "grayscale": False,
                "minimap_width": 84, 
                "minimap_height": 84
            }
        
        spaces_dict = {}
        
        # Add a vision key for compatibility with MultiInputPolicy
        # This is a placeholder that will be filled with actual visual data
        spaces_dict["vision"] = spaces.Box(
            low=0, 
            high=255, 
            shape=(84, 84, 3), 
            dtype=np.uint8
        )
        
        # Add decision_memory key for compatibility with the agent
        spaces_dict["decision_memory"] = spaces.Box(
            low=-float('inf'),
            high=float('inf'),
            shape=(10, 5),  # Assuming 10 past decisions with 5 features each
            dtype=np.float32
        )
        
        # Add metrics key for compatibility with the agent
        # Using shape (4,) to match the expected shape in the error message
        spaces_dict["metrics"] = spaces.Box(
            low=-float('inf'),
            high=float('inf'),
            shape=(4,),  # Using 4 metrics as mentioned in the error
            dtype=np.float32
        )
        
        # Add population metric
        spaces_dict["population"] = spaces.Box(
            low=0, 
            high=float('inf'), 
            shape=(1,), 
            dtype=np.float32
        )
        
        # Add happiness metric
        spaces_dict["happiness"] = spaces.Box(
            low=0, 
            high=100, 
            shape=(1,), 
            dtype=np.float32
        )
        
        # Add budget balance metric
        spaces_dict["budget_balance"] = spaces.Box(
            low=-float('inf'), 
            high=float('inf'), 
            shape=(1,), 
            dtype=np.float32
        )
        
        # Add traffic metric
        spaces_dict["traffic"] = spaces.Box(
            low=0, 
            high=100, 
            shape=(1,), 
            dtype=np.float32
        )
        
        # Create the observation space
        self.observation_space = spaces.Dict(spaces_dict)
    
    def _setup_action_space(self):
        """Set up the action space."""
        action_config = self.config.get("environment", {}).get("action_space", {})
        
        # Collect all possible actions
        all_actions = []
        
        # Add zoning actions
        for zone_type in action_config.get("zone", []):
            all_actions.append({"type": "zone", "zone_type": zone_type})
        
        # Add infrastructure actions
        for infra_type in action_config.get("infrastructure", []):
            all_actions.append({"type": "infrastructure", "infra_type": infra_type})
        
        # Add budget actions
        for budget_action in action_config.get("budget", []):
            all_actions.append({"type": "budget", "budget_action": budget_action})
        
        # If no actions defined, create some defaults
        if not all_actions:
            self.logger.warning("No actions defined in config, using defaults")
            all_actions = [
                {"type": "zone", "zone_type": "residential"},
                {"type": "zone", "zone_type": "commercial"},
                {"type": "zone", "zone_type": "industrial"},
                {"type": "infrastructure", "infra_type": "road"},
                {"type": "budget", "budget_action": "increase_tax"}
            ]
        
        # Define the action space as a Discrete space with one action per option
        self.action_space = spaces.Discrete(len(all_actions))
        
        # Store the action mapping for later use
        self.action_mapping = all_actions
    
    def step(self, action: int) -> Tuple[Dict[str, np.ndarray], float, bool, bool, Dict[str, Any]]:
        """
        Take a step in the environment.
        
        Args:
            action: Integer representing the action to take
            
        Returns:
            observation, reward, terminated, truncated, info
        """
        # Take the action in the environment
        self.current_step += 1
        self.episode_steps += 1
        
        if hasattr(self, 'in_fallback_mode') and self.in_fallback_mode:
            # In fallback mode, simulate the action
            success = True
            reward = self._simulate_fallback_action(action)
        else:
            # Normal mode, perform the actual action
            success = self.interface.perform_action(self._action_to_dict(action))
            if not success:
                # Action failed, apply small negative reward
                reward = -0.01
            else:
                # Get the current game state and calculate reward
                game_state = self.interface.get_game_state()
                reward = self._calculate_reward(game_state)
        
        # Check if episode is done
        terminated = self.interface.is_game_over() if not hasattr(self, 'in_fallback_mode') or not self.in_fallback_mode else False
        truncated = self.episode_steps >= self.max_episode_steps
        
        # Get observation
        observation = self._get_observation()
        
        # Assemble info dict
        info = {
            "action_success": success,
            "episode_steps": self.episode_steps,
            "metrics": self._get_metrics(),
            "action": action
        }
        
        # Remember the last observation
        self.last_observation = observation
        self.last_info = info
        
        return observation, reward, terminated, truncated, info
        
    def _simulate_fallback_action(self, action: int) -> float:
        """
        Simulate an action in fallback mode.
        
        Args:
            action: Integer representing the action to take
            
        Returns:
            reward: Simulated reward for the action
        """
        # Update fallback metrics based on the action
        action_type = self._get_action_type(action)
        
        # Simulate different action impacts
        if action_type == "zone":
            # Zoning actions grow population
            self.fallback_metrics["population"] += max(5, int(self.fallback_metrics["population"] * self.fallback_growth_rate))
            self.fallback_metrics["happiness"] -= self.fallback_happiness_decay
            self.fallback_metrics["budget_balance"] += self.fallback_budget_rate
            
            # More traffic with more population
            if self.fallback_metrics["population"] > 1000:
                self.fallback_metrics["traffic"] = min(100, self.fallback_metrics["traffic"] + 0.5)
                
            reward = 0.05
                
        elif action_type == "infrastructure":
            # Infrastructure improves happiness but costs money
            self.fallback_metrics["happiness"] = min(100, self.fallback_metrics["happiness"] + 1.0)
            self.fallback_metrics["budget_balance"] += self.fallback_budget_rate * 2
            
            # Reduces traffic if population is high
            if self.fallback_metrics["population"] > 1000:
                self.fallback_metrics["traffic"] = max(0, self.fallback_metrics["traffic"] - 1.0)
                
            reward = 0.02
                
        elif action_type == "budget":
            # Budget actions affect finances
            self.fallback_metrics["budget_balance"] += self.fallback_budget_rate / 2
            self.fallback_metrics["happiness"] -= self.fallback_happiness_decay / 2
            
            reward = 0.01
            
        else:
            # Other actions have minimal impact
            self.fallback_metrics["budget_balance"] += self.fallback_budget_rate / 4
            reward = 0.0
            
        # Apply natural growth if population exists
        if self.fallback_metrics["population"] > 0:
            # Apply small natural growth
            happiness_factor = max(0, self.fallback_metrics["happiness"] / 50)
            natural_growth = max(1, int(self.fallback_metrics["population"] * 0.01 * happiness_factor))
            self.fallback_metrics["population"] += natural_growth
            
            # Natural budget drain based on population
            budget_drain = -0.1 * self.fallback_metrics["population"]
            self.fallback_metrics["budget_balance"] += budget_drain
            
            # Add population growth component to reward
            reward += 0.01 * natural_growth
            
        # Bankruptcy check
        if self.fallback_metrics["budget_balance"] < 0:
            reward -= 0.1
            
        # Clamp metrics to reasonable ranges
        self.fallback_metrics["happiness"] = max(0, min(100, self.fallback_metrics["happiness"]))
        self.fallback_metrics["traffic"] = max(0, min(100, self.fallback_metrics["traffic"]))
        
        return reward
        
    def _get_action_type(self, action: int) -> str:
        """Determine the action type based on the action index."""
        if action >= len(self.action_mapping):
            return "unknown"
            
        return self.action_mapping[action].get("type", "unknown")
    
    def reset(self, seed=None, options=None):
        """
        Reset the environment to its initial state.
        
        Args:
            seed: Random seed
            options: Additional options
            
        Returns:
            observation, info
        """
        if seed is not None:
            np.random.seed(seed)
            
        self.episode_steps = 0
        self.total_reward = 0.0
        
        if hasattr(self, 'in_fallback_mode') and self.in_fallback_mode:
            # In fallback mode, reset to initial values
            self.fallback_metrics = {
                "population": 0,
                "happiness": 50.0,
                "budget_balance": 10000.0,
                "traffic": 50.0,
                "noise_pollution": 0.0,
                "air_pollution": 0.0
            }
        else:
            # Try to reset the actual game
            try:
                self.interface.reset()
            except Exception as e:
                self.logger.warning(f"Error resetting game: {str(e)}")
                if not hasattr(self, 'use_fallback') or not self.use_fallback:
                    raise e
                self._setup_fallback_mode()
        
        # Get the initial observation
        observation = self._get_observation()
        
        # Return info dict with metrics
        info = {
            "episode_steps": 0,
            "metrics": self._get_metrics(),
        }
        
        # Remember the last observation
        self.last_observation = observation
        self.last_info = info
        
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
        obs_config = self.config.get("environment", {}).get("observation_space", {})
        if not obs_config:
            obs_config = self.config.get("observation", {})
            
        observation = {}
        
        # Visual observation
        if obs_config.get("include_visual", False):
            visual_obs = game_state.get("visual_observation", None)
            if visual_obs is None:
                # Create an empty image if no visual observation is available
                height, width = obs_config.get("image_size", (224, 224))
                channels = 1 if obs_config.get("grayscale", False) else 3
                visual_obs = np.zeros((height, width, channels), dtype=np.uint8)
            else:
                # Ensure correct number of channels (convert RGBA to RGB if needed)
                if len(visual_obs.shape) == 3 and visual_obs.shape[2] == 4:
                    # Convert RGBA to RGB by discarding the alpha channel
                    self.logger.debug("Converting RGBA image to RGB")
                    visual_obs = visual_obs[:, :, :3]
                
                # Ensure correct image size
                target_height, target_width = obs_config.get("image_size", (224, 224))
                if visual_obs.shape[0] != target_height or visual_obs.shape[1] != target_width:
                    # Resize the image if needed
                    try:
                        import cv2
                        visual_obs = cv2.resize(visual_obs, (target_width, target_height))
                    except ImportError:
                        # Fallback to simple resizing if OpenCV is not available
                        from skimage.transform import resize
                        visual_obs = resize(visual_obs, (target_height, target_width), 
                                         anti_aliasing=True, preserve_range=True).astype(np.uint8)
                
                # Convert to grayscale if configured
                if obs_config.get("grayscale", False) and len(visual_obs.shape) == 3 and visual_obs.shape[2] > 1:
                    if visual_obs.shape[2] == 3:  # RGB
                        # Use standard RGB to grayscale conversion
                        visual_obs = np.dot(visual_obs[..., :3], [0.2989, 0.5870, 0.1140])
                    elif visual_obs.shape[2] == 4:  # RGBA
                        # Convert to grayscale ignoring alpha
                        visual_obs = np.dot(visual_obs[..., :3], [0.2989, 0.5870, 0.1140])
                    
                    # Reshape to have a single channel
                    visual_obs = visual_obs.reshape(visual_obs.shape[0], visual_obs.shape[1], 1)
            
            observation["visual"] = visual_obs
        
        # Metrics observation
        if obs_config.get("include_metrics", True):
            metrics = game_state.get("metrics", {})
            self.last_metrics = metrics.copy()  # Store for reward calculation
            
            metrics_list = obs_config.get("metrics", ["population", "happiness", "budget_balance", "traffic"])
            for metric in metrics_list:
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
        reward_config = self.config.get("environment", {}).get("reward_function", {})
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
        Get the current observation from the environment.
        
        Returns:
            Dictionary of observations
        """
        if self.in_fallback_mode:
            return self._get_fallback_observation()
        
        try:
            # Get game state from interface
            game_state = self.interface.get_game_state()
            
            # Extract observation from game state
            observation = {}
            
            # Add a placeholder vision observation
            observation["vision"] = np.zeros((84, 84, 3), dtype=np.uint8)
            
            # Add a placeholder decision_memory
            observation["decision_memory"] = np.zeros((10, 5), dtype=np.float32)
            
            # Add metrics
            metrics_dict = self._get_metrics()
            
            # Add individual metrics to observation
            for key, value in metrics_dict.items():
                if isinstance(value, (int, float)):
                    observation[key] = np.array([float(value)], dtype=np.float32)
            
            # Add combined metrics array with correct shape (4,)
            metrics_array = np.zeros(4, dtype=np.float32)
            # Only include the 4 main metrics
            if "population" in metrics_dict:
                metrics_array[0] = float(metrics_dict["population"])
            if "happiness" in metrics_dict:
                metrics_array[1] = float(metrics_dict["happiness"])
            if "budget_balance" in metrics_dict:
                metrics_array[2] = float(metrics_dict["budget_balance"])
            if "traffic" in metrics_dict:
                metrics_array[3] = float(metrics_dict["traffic"])
            observation["metrics"] = metrics_array
            
            return observation
            
        except Exception as e:
            self.logger.error(f"Error getting observation: {str(e)}")
            # Return a default observation
            return {
                "vision": np.zeros((84, 84, 3), dtype=np.uint8),
                "decision_memory": np.zeros((10, 5), dtype=np.float32),
                "metrics": np.zeros(4, dtype=np.float32),
                "population": np.array([0.0], dtype=np.float32),
                "happiness": np.array([0.0], dtype=np.float32),
                "budget_balance": np.array([0.0], dtype=np.float32),
                "traffic": np.array([0.0], dtype=np.float32)
            }
    
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
        if hasattr(self, 'in_fallback_mode') and self.in_fallback_mode:
            return self.fallback_metrics.copy()
            
        try:
            game_state = self.interface.get_game_state()
            return game_state.get("metrics", {})
        except Exception as e:
            self.logger.warning(f"Error getting metrics: {str(e)}")
            return {}
    
    def _action_to_dict(self, action: int) -> Dict[str, Any]:
        """
        Convert an action index to a dictionary representation for the interface.
        
        Args:
            action: Integer representing the action to take
            
        Returns:
            Dictionary representation of the action
        """
        # Check if action is valid
        if action < 0 or action >= len(self.action_mapping):
            self.logger.warning(f"Invalid action index: {action}")
            return {"type": "no_op"}
            
        # Return the mapped action
        return self.action_mapping[action]
    
    def _get_fallback_observation(self):
        """
        Get a simulated observation for fallback mode.
        
        Returns:
            Dictionary of simulated observations
        """
        # Create metrics array with correct shape (4,)
        metrics_array = np.zeros(4, dtype=np.float32)
        metrics_array[0] = float(self.fallback_metrics["population"])
        metrics_array[1] = float(self.fallback_metrics["happiness"])
        metrics_array[2] = float(self.fallback_metrics["budget_balance"])
        metrics_array[3] = float(self.fallback_metrics["traffic"])
        
        observation = {
            "vision": np.zeros((84, 84, 3), dtype=np.uint8),
            "decision_memory": np.zeros((10, 5), dtype=np.float32),
            "metrics": metrics_array,
            "population": np.array([float(self.fallback_metrics["population"])], dtype=np.float32),
            "happiness": np.array([float(self.fallback_metrics["happiness"])], dtype=np.float32),
            "budget_balance": np.array([float(self.fallback_metrics["budget_balance"])], dtype=np.float32),
            "traffic": np.array([float(self.fallback_metrics["traffic"])], dtype=np.float32)
        }
        return observation 