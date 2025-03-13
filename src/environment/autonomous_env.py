"""
Autonomous Environment - Environment with advanced autonomous operation capabilities.
"""

import os
import logging
import numpy as np
from typing import Dict, Any, Optional, Tuple, List

import gymnasium as gym
from gymnasium import spaces

from src.environment.vision_env import VisionEnvironment
from src.environment.cs2_env import CS2Environment

class AutonomousEnvironment:
    """
    Environment that supports autonomous operation with advanced decision capabilities.
    """
    
    def __init__(self, base_env, exploration_frequency: float = 0.3, 
                 random_action_frequency: float = 0.2,
                 menu_exploration_buffer_size: int = 50,
                 logger: Optional[logging.Logger] = None):
        """
        Initialize the autonomous environment.
        
        Args:
            base_env: The base environment to wrap
            exploration_frequency: How often to perform exploratory actions
            random_action_frequency: How often to perform completely random actions
            menu_exploration_buffer_size: Size of menu exploration buffer
            logger: Logger instance
        """
        self.base_env = base_env
        self.logger = logger or logging.getLogger("AutonomousEnvironment")
        
        # Autonomous-specific settings
        self.exploration_frequency = exploration_frequency
        self.random_action_frequency = random_action_frequency
        self.menu_exploration_buffer_size = menu_exploration_buffer_size
        
        # Get configuration from base environment
        self.config = getattr(base_env, 'config', {})
        self.autonomous_config = self.config.get("autonomous", {})
        
        self.use_decision_memory = self.autonomous_config.get("use_decision_memory", True)
        self.decision_memory_size = self.autonomous_config.get("decision_memory_size", 10)
        self.decision_memory = []
        
        # Set up action and observation spaces
        self.action_space = self.base_env.action_space
        self.observation_space = self.base_env.observation_space
        
        # Advanced metrics tracking
        self.performance_metrics = {
            "success_rate": 0.0,
            "confidence": 0.0,
            "efficiency": 0.0,
            "learning_progress": 0.0
        }
        
        # Extend observation space to include higher-level information
        self.extend_observation_space()
        
        self.logger.info("Autonomous environment initialized")
    
    def extend_observation_space(self):
        """Extend the observation space to include higher-level information."""
        # Preserve the original observation space
        self.vision_observation_space = self.observation_space
        
        # Get additional observation components
        additional_spaces = {}
        
        # Add metrics to observation if enabled
        if self.autonomous_config.get("observe_metrics", True):
            additional_spaces["metrics"] = spaces.Box(
                low=0.0, 
                high=1.0, 
                shape=(4,),  # success, confidence, efficiency, learning
                dtype=np.float32
            )
        
        # Add decision memory to observation if enabled
        if self.use_decision_memory:
            additional_spaces["decision_memory"] = spaces.Box(
                low=-np.inf, 
                high=np.inf, 
                shape=(self.decision_memory_size, self.action_space.n),
                dtype=np.float32
            )
        
        # Create a Dict observation space if we have additional components
        if additional_spaces:
            combined_spaces = {
                "vision": self.vision_observation_space,
                **additional_spaces
            }
            self.observation_space = spaces.Dict(combined_spaces)
            self.logger.info(f"Extended observation space: {self.observation_space}")
        else:
            self.logger.info("Using vision observation space without extensions")
    
    def update_decision_memory(self, action_probs):
        """
        Update the decision memory with new action probabilities.
        
        Args:
            action_probs: Action probability distribution
        """
        if not self.use_decision_memory:
            return
        
        # Initialize decision memory if empty
        if not self.decision_memory:
            self.decision_memory = [np.zeros(self.action_space.n) for _ in range(self.decision_memory_size)]
        
        # Add new action probs and remove oldest
        self.decision_memory.pop(0)
        self.decision_memory.append(action_probs)
    
    def update_performance_metrics(self, reward, done):
        """
        Update performance metrics based on recent experience.
        
        Args:
            reward: Last received reward
            done: Whether the episode is done
        """
        # Simple updates for the metrics, in a real system these would be more sophisticated
        alpha = 0.1  # Learning rate for metrics update
        
        # Update success rate based on reward
        if reward > 0:
            self.performance_metrics["success_rate"] += alpha * (1.0 - self.performance_metrics["success_rate"])
        else:
            self.performance_metrics["success_rate"] -= alpha * self.performance_metrics["success_rate"]
        
        # Bound success rate between 0 and 1
        self.performance_metrics["success_rate"] = max(0.0, min(1.0, self.performance_metrics["success_rate"]))
        
        # Update other metrics (simplified for demonstration)
        if done and reward > 0:
            self.performance_metrics["confidence"] += alpha
            self.performance_metrics["learning_progress"] += alpha * 0.5
        
        # Bound all metrics between 0 and 1
        for key in self.performance_metrics:
            self.performance_metrics[key] = max(0.0, min(1.0, self.performance_metrics[key]))
    
    def get_extended_observation(self, vision_observation):
        """
        Create extended observation with additional information.
        
        Args:
            vision_observation: Base vision observation
        
        Returns:
            Extended observation
        """
        if isinstance(self.observation_space, spaces.Dict):
            # Create the extended observation dictionary
            obs = {
                "vision": vision_observation
            }
            
            # Add metrics if needed
            if "metrics" in self.observation_space.spaces:
                obs["metrics"] = np.array([
                    self.performance_metrics["success_rate"],
                    self.performance_metrics["confidence"],
                    self.performance_metrics["efficiency"],
                    self.performance_metrics["learning_progress"]
                ], dtype=np.float32)
            
            # Add decision memory if needed
            if "decision_memory" in self.observation_space.spaces and self.use_decision_memory:
                obs["decision_memory"] = np.array(self.decision_memory, dtype=np.float32)
            
            return obs
        else:
            # If no extensions, return the original vision observation
            return vision_observation
    
    def reset(self, **kwargs):
        """
        Reset the environment and return initial observation.
        
        Returns:
            Initial observation
        """
        vision_observation, info = self.base_env.reset(**kwargs)
        
        # Reset decision memory and metrics
        self.decision_memory = []
        self.performance_metrics = {k: 0.0 for k in self.performance_metrics}
        
        # Return extended observation
        return self.get_extended_observation(vision_observation), info
    
    def step(self, action):
        """
        Take a step in the environment.
        
        Args:
            action: Action to take
        
        Returns:
            observation, reward, terminated, truncated, info
        """
        vision_observation, reward, terminated, truncated, info = self.base_env.step(action)
        
        # Update performance metrics
        self.update_performance_metrics(reward, terminated)
        
        # Update decision memory (in a real system, we'd get action_probs from the agent)
        action_probs = np.zeros(self.action_space.n)
        action_probs[action] = 1.0
        self.update_decision_memory(action_probs)
        
        # Add performance metrics to info
        info["performance_metrics"] = self.performance_metrics.copy()
        
        # Return extended observation
        return self.get_extended_observation(vision_observation), reward, terminated, truncated, info
        
    def close(self):
        """Close the environment."""
        return self.base_env.close()

class AutonomousCS2Environment(CS2Environment):
    """
    Cities: Skylines 2 environment with autonomous capabilities.
    This class combines CS2Environment with autonomous features.
    """
    
    def __init__(self, 
                 base_env_config: Dict[str, Any] = None,
                 observation_config: Dict[str, Any] = None,
                 vision_config: Dict[str, Any] = None,
                 use_fallback_mode: bool = True,
                 exploration_frequency: float = 0.3,
                 random_action_frequency: float = 0.2,
                 menu_exploration_buffer_size: int = 50, 
                 logger: Optional[logging.Logger] = None):
        """
        Initialize the autonomous CS2 environment.
        
        Args:
            base_env_config: Configuration for the base environment
            observation_config: Configuration for observations
            vision_config: Configuration for vision guidance
            use_fallback_mode: Whether to use fallback mode if game connection fails
            exploration_frequency: How often to perform exploratory actions
            random_action_frequency: How often to perform completely random actions
            menu_exploration_buffer_size: Size of menu exploration buffer
            logger: Logger instance
        """
        # Initialize the CS2Environment first
        super().__init__(config=base_env_config or {})
        
        # Set up autonomous features
        self.logger = logger or logging.getLogger("AutonomousCS2Environment")
        
        # Autonomous-specific settings
        self.exploration_frequency = exploration_frequency
        self.random_action_frequency = random_action_frequency
        self.menu_exploration_buffer_size = menu_exploration_buffer_size
        
        # Get configuration
        self.autonomous_config = self.config.get("autonomous", {})
        
        self.use_decision_memory = self.autonomous_config.get("use_decision_memory", True)
        self.decision_memory_size = self.autonomous_config.get("decision_memory_size", 10)
        self.decision_memory = []
        
        # Advanced metrics tracking
        self.performance_metrics = {
            "success_rate": 0.0,
            "confidence": 0.0,
            "efficiency": 0.0,
            "learning_progress": 0.0
        }
        
        # Extend observation space to include higher-level information
        self.extend_observation_space()
        
        self.logger.info("Autonomous environment initialized")
    
    def extend_observation_space(self):
        """Extend the observation space to include higher-level information."""
        # Preserve the original observation space
        self.vision_observation_space = self.observation_space
        
        # Get additional observation components
        additional_spaces = {}
        
        # Add metrics to observation if enabled
        if self.autonomous_config.get("observe_metrics", True):
            additional_spaces["metrics"] = spaces.Box(
                low=0.0, 
                high=1.0, 
                shape=(4,),  # success_rate, confidence, efficiency, learning_progress
                dtype=np.float32
            )
        
        # Add decision memory to observation if enabled
        if self.use_decision_memory:
            additional_spaces["decision_memory"] = spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(self.decision_memory_size, 5),  # action, reward, success, confidence, time
                dtype=np.float32
            )
        
        # Create the extended observation space
        self.observation_space = spaces.Dict({
            **self.observation_space.spaces,
            **additional_spaces
        })
    
    def update_decision_memory(self, action_probs):
        """
        Update the decision memory with new action probabilities.
        
        Args:
            action_probs: Action probability distribution
        """
        if not self.use_decision_memory:
            return
        
        # Initialize decision memory if empty
        if not self.decision_memory:
            self.decision_memory = [np.zeros(self.action_space.n) for _ in range(self.decision_memory_size)]
        
        # Add new action probs and remove oldest
        self.decision_memory.pop(0)
        self.decision_memory.append(action_probs)
    
    def update_performance_metrics(self, reward, done):
        """
        Update performance metrics based on recent experience.
        
        Args:
            reward: Last received reward
            done: Whether the episode is done
        """
        # Simple updates for the metrics, in a real system these would be more sophisticated
        alpha = 0.1  # Learning rate for metrics update
        
        # Update success rate based on reward
        if reward > 0:
            self.performance_metrics["success_rate"] += alpha * (1.0 - self.performance_metrics["success_rate"])
        else:
            self.performance_metrics["success_rate"] -= alpha * self.performance_metrics["success_rate"]
        
        # Bound success rate between 0 and 1
        self.performance_metrics["success_rate"] = max(0.0, min(1.0, self.performance_metrics["success_rate"]))
        
        # Update other metrics (simplified for demonstration)
        if done and reward > 0:
            self.performance_metrics["confidence"] += alpha
            self.performance_metrics["learning_progress"] += alpha * 0.5
        
        # Bound all metrics between 0 and 1
        for key in self.performance_metrics:
            self.performance_metrics[key] = max(0.0, min(1.0, self.performance_metrics[key]))
    
    def get_extended_observation(self, vision_observation):
        """
        Create extended observation with additional information.
        
        Args:
            vision_observation: Base vision observation
        
        Returns:
            Extended observation
        """
        if isinstance(self.observation_space, spaces.Dict):
            # Create the extended observation dictionary
            obs = {
                "vision": vision_observation
            }
            
            # Add metrics if needed
            if "metrics" in self.observation_space.spaces:
                obs["metrics"] = np.array([
                    self.performance_metrics["success_rate"],
                    self.performance_metrics["confidence"],
                    self.performance_metrics["efficiency"],
                    self.performance_metrics["learning_progress"]
                ], dtype=np.float32)
            
            # Add decision memory if needed
            if "decision_memory" in self.observation_space.spaces and self.use_decision_memory:
                obs["decision_memory"] = np.array(self.decision_memory, dtype=np.float32)
            
            return obs
        else:
            # If no extensions, return the original vision observation
            return vision_observation
    
    def reset(self, **kwargs):
        """
        Reset the environment and return initial observation.
        
        Returns:
            Initial observation
        """
        vision_observation, info = self.base_env.reset(**kwargs)
        
        # Reset decision memory and metrics
        self.decision_memory = []
        self.performance_metrics = {k: 0.0 for k in self.performance_metrics}
        
        # Return extended observation
        return self.get_extended_observation(vision_observation), info
    
    def step(self, action):
        """
        Take a step in the environment.
        
        Args:
            action: Action to take
        
        Returns:
            observation, reward, terminated, truncated, info
        """
        vision_observation, reward, terminated, truncated, info = self.base_env.step(action)
        
        # Update performance metrics
        self.update_performance_metrics(reward, terminated)
        
        # Update decision memory (in a real system, we'd get action_probs from the agent)
        action_probs = np.zeros(self.action_space.n)
        action_probs[action] = 1.0
        self.update_decision_memory(action_probs)
        
        # Add performance metrics to info
        info["performance_metrics"] = self.performance_metrics.copy()
        
        # Return extended observation
        return self.get_extended_observation(vision_observation), reward, terminated, truncated, info
        
    def close(self):
        """Close the environment."""
        return self.base_env.close()

# Create an alias for backward compatibility
AutonomousCS2Environment = AutonomousCS2Environment

# Define any missing types that might be imported from this module
class ActionType:
    """Type of action for the autonomous environment."""
    MENU = "menu"
    BUTTON = "button"
    KEYBOARD = "keyboard"
    MOUSE = "mouse"
    COMBINED = "combined"
    GAME_ACTION = "game_action"

class Action:
    """Action representation for the autonomous environment."""
    def __init__(self, action_type=None, value=None, name=None, description=None, action_fn=None):
        self.type = action_type
        self.value = value
        self.name = name or (f"{action_type}_{value}" if value is not None else action_type)
        self.description = description or (f"{action_type} action with value {value}" if value is not None else f"{action_type} action")
        self.action_fn = action_fn  # Add support for action_fn parameter
