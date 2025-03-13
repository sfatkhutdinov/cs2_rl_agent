"""
Vision Environment - Environment with vision-based observations.
"""

import os
import logging
import numpy as np
from typing import Dict, Any, Optional, Tuple, List

import gymnasium as gym
from gymnasium import spaces

from src.environment.cs2_env import CS2Environment

class VisionEnvironment(CS2Environment):
    """
    Environment that processes and uses vision-based observations.
    """
    
    def __init__(self, config: Dict[str, Any], game_handle: Optional[int] = None):
        """
        Initialize the vision environment.
        
        Args:
            config: Configuration dictionary
            game_handle: Optional game window handle
        """
        super().__init__(config, game_handle)
        self.logger = logging.getLogger("VisionEnvironment")
        
        # Vision-specific settings
        self.vision_config = config.get("vision", {})
        self.use_frame_stacking = self.vision_config.get("frame_stacking", False)
        self.frame_stack_size = self.vision_config.get("frame_stack_size", 4)
        self.frame_stack = []
        
        # Set up observation space for vision
        self.setup_vision_observation_space()
        
        self.logger.info("Vision environment initialized")
    
    def setup_vision_observation_space(self):
        """Set up the observation space for vision-based learning."""
        vision_shape = self.vision_config.get("observation_shape", (84, 84, 3))
        
        if self.use_frame_stacking:
            # If using frame stacking, the observation is a stack of frames
            self.observation_space = spaces.Box(
                low=0, 
                high=255, 
                shape=(vision_shape[0], vision_shape[1], vision_shape[2] * self.frame_stack_size),
                dtype=np.uint8
            )
        else:
            # Single frame observation
            self.observation_space = spaces.Box(
                low=0, 
                high=255, 
                shape=vision_shape,
                dtype=np.uint8
            )
        
        self.logger.info(f"Vision observation space: {self.observation_space}")
    
    def preprocess_observation(self, observation):
        """
        Preprocess the raw observation for vision-based learning.
        
        Args:
            observation: Raw observation from the environment
        
        Returns:
            Processed observation
        """
        # Convert to grayscale if specified
        if self.vision_config.get("use_grayscale", False):
            import cv2
            if len(observation.shape) == 3 and observation.shape[2] == 3:
                observation = cv2.cvtColor(observation, cv2.COLOR_RGB2GRAY)
                observation = np.expand_dims(observation, -1)
        
        # Resize observation if specified
        if self.vision_config.get("resize_observation", True):
            import cv2
            target_shape = self.vision_config.get("observation_shape", (84, 84, 3))
            if observation.shape[:2] != target_shape[:2]:
                # Keep color channels if present
                channels = observation.shape[2] if len(observation.shape) == 3 else 1
                observation = cv2.resize(
                    observation, 
                    (target_shape[1], target_shape[0]),
                    interpolation=cv2.INTER_AREA
                )
                if len(observation.shape) == 2 and channels > 1:
                    observation = np.expand_dims(observation, -1)
        
        # Normalize pixel values if specified
        if self.vision_config.get("normalize_observation", False):
            observation = observation.astype(np.float32) / 255.0
        
        return observation
    
    def update_frame_stack(self, frame):
        """
        Update the frame stack with a new frame.
        
        Args:
            frame: New frame to add to the stack
        
        Returns:
            Updated frame stack as a single array
        """
        if not self.use_frame_stacking:
            return frame
        
        # Initialize frame stack if empty
        if not self.frame_stack:
            self.frame_stack = [frame] * self.frame_stack_size
        else:
            # Add new frame and remove oldest
            self.frame_stack.pop(0)
            self.frame_stack.append(frame)
        
        # Stack frames along the channel dimension
        stacked_frames = np.concatenate(self.frame_stack, axis=2)
        return stacked_frames
    
    def reset(self, **kwargs):
        """
        Reset the environment and return initial observation.
        
        Returns:
            Initial observation
        """
        observation, info = super().reset(**kwargs)
        
        # Process the observation for vision
        processed_observation = self.preprocess_observation(observation)
        
        # Update frame stack if using it
        if self.use_frame_stacking:
            self.frame_stack = []
            stacked_observation = self.update_frame_stack(processed_observation)
            return stacked_observation, info
        
        return processed_observation, info
    
    def step(self, action):
        """
        Take a step in the environment.
        
        Args:
            action: Action to take
        
        Returns:
            observation, reward, terminated, truncated, info
        """
        observation, reward, terminated, truncated, info = super().step(action)
        
        # Process the observation for vision
        processed_observation = self.preprocess_observation(observation)
        
        # Update frame stack if using it
        if self.use_frame_stacking:
            stacked_observation = self.update_frame_stack(processed_observation)
            return stacked_observation, reward, terminated, truncated, info
        
        return processed_observation, reward, terminated, truncated, info 