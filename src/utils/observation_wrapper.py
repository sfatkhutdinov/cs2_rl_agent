import gymnasium as gym
import numpy as np
from typing import Dict, Any, Tuple, List, Optional, Union

class FlattenObservationWrapper(gym.Wrapper):
    """
    A wrapper that flattens dictionary observations into a single array.
    This is useful for environments that use dictionary observations but need to work with
    algorithms that expect a single array.
    """
    
    def __init__(self, env):
        """
        Initialize the wrapper.
        
        Args:
            env: The environment to wrap
        """
        super().__init__(env)
        
        # Determine the shape of the flattened observation
        sample_obs = self.env.observation_space.sample()
        flattened_obs = self._flatten_observation(sample_obs)
        
        # Create a new observation space
        self.observation_space = gym.spaces.Box(
            low=-float('inf'),
            high=float('inf'),
            shape=flattened_obs.shape,
            dtype=np.float32
        )
    
    def _flatten_observation(self, obs: Dict[str, np.ndarray]) -> np.ndarray:
        """
        Flatten a dictionary observation into a single array.
        
        Args:
            obs: Dictionary observation
            
        Returns:
            Flattened observation
        """
        # Create a fixed-size array to ensure consistent shape
        flattened = np.zeros(58, dtype=np.float32)
        
        # Fill in the values we have
        index = 0
        
        # Process each key in a consistent order
        for key in sorted(obs.keys()):
            if key != 'vision':  # Skip vision data
                value = obs[key]
                if isinstance(value, np.ndarray):
                    flat_value = value.flatten()
                    # Ensure we don't exceed the array size
                    size = min(len(flat_value), 58 - index)
                    flattened[index:index+size] = flat_value[:size]
                    index += size
                elif isinstance(value, (int, float)):
                    if index < 58:
                        flattened[index] = float(value)
                        index += 1
        
        return flattened
    
    def reset(self, **kwargs):
        """
        Reset the environment.
        
        Returns:
            Flattened observation
        """
        obs, info = self.env.reset(**kwargs)
        return self._flatten_observation(obs), info
    
    def step(self, action) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """
        Take a step in the environment.
        
        Args:
            action: Action to take
            
        Returns:
            Tuple of (flattened_observation, reward, done, truncated, info)
        """
        obs, reward, done, truncated, info = self.env.step(action)
        return self._flatten_observation(obs), reward, done, truncated, info 