import gym
import numpy as np
from typing import Dict, Tuple, Any, Optional


class ObservationWrapper(gym.Wrapper):
    """
    A wrapper for creating vectorized observations from dictionary observations.
    
    This wrapper enables the environment to work with standard RL algorithms
    by converting dictionary observations to a format compatible with neural networks.
    """
    
    def __init__(self, env):
        """
        Initialize the observation wrapper.
        
        Args:
            env: The environment to wrap
        """
        super().__init__(env)
        
        # Check if the observation space is a dictionary
        assert isinstance(env.observation_space, gym.spaces.Dict), \
            "ObservationWrapper only works with Dict observation spaces"
        
        # Extract spaces from the dictionary
        self.observation_keys = list(env.observation_space.spaces.keys())
        
        # Build a new observation space
        self.observation_spaces = {}
        
        # Track flattened spaces for Box spaces
        flattened_spaces = {}
        
        # Process each space in the observation
        for key, space in env.observation_space.spaces.items():
            if isinstance(space, gym.spaces.Box):
                # For continuous spaces (e.g., metrics, screenshots)
                shape = space.shape
                if len(shape) == 1:
                    # For 1D arrays like metrics
                    flattened_spaces[key] = (0, shape[0])
                    self.observation_spaces[key] = shape[0]
                else:
                    # For images/screenshots
                    # Keep images as separate observations
                    self.observation_spaces[key] = space
            elif isinstance(space, gym.spaces.Discrete):
                # For discrete spaces
                flattened_spaces[key] = (0, 1)
                self.observation_spaces[key] = 1
            else:
                # For other spaces (e.g., MultiDiscrete)
                # This may need further customization
                self.observation_spaces[key] = space
        
        # Create the new observation space
        # For SB3's MultiInputPolicy, we need to define the new spaces as a dict
        self.observation_space = gym.spaces.Dict({
            k: v for k, v in env.observation_space.spaces.items()
        })
    
    def _process_obs(self, obs: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """
        Process the observation to convert it into the format expected by the model.
        
        Args:
            obs: Dictionary observation from the environment
            
        Returns:
            Processed observation dictionary
        """
        processed_obs = {}
        
        for key, value in obs.items():
            if isinstance(value, np.ndarray):
                # Ensure the array is in the right dtype
                if value.dtype == np.float64:
                    value = value.astype(np.float32)
                    
                # Make sure images have the right shape and range
                if len(value.shape) > 1:  # It's an image
                    # Check if it's grayscale and needs reshaping
                    if len(value.shape) == 2:
                        value = np.expand_dims(value, axis=-1)
                    
                    # Normalize image to [0, 1] if not already
                    if value.max() > 1.0:
                        value = value / 255.0
                        
                processed_obs[key] = value
            else:
                # Handle non-ndarray values
                processed_obs[key] = value
                
        return processed_obs
    
    def reset(self, **kwargs) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
        """
        Reset the environment and process the observation.
        
        Args:
            **kwargs: Keyword arguments to pass to the environment's reset method
            
        Returns:
            Tuple of (observation, info)
        """
        obs, info = self.env.reset(**kwargs)
        return self._process_obs(obs), info
    
    def step(self, action) -> Tuple[Dict[str, np.ndarray], float, bool, bool, Dict[str, Any]]:
        """
        Take a step in the environment and process the observation.
        
        Args:
            action: Action to take
            
        Returns:
            Tuple of (observation, reward, terminated, truncated, info)
        """
        obs, reward, terminated, truncated, info = self.env.step(action)
        return self._process_obs(obs), reward, terminated, truncated, info 