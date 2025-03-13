"""
Tutorial Agent - Agent that follows guided tutorials to learn basic gameplay mechanics.
"""

import os
import logging
import numpy as np
from typing import Dict, Any, Optional, Tuple, List

import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback

class TutorialAgent:
    """
    An agent focused on completing guided tutorials to learn basic gameplay
    mechanics and understand the game interface.
    """
    
    def __init__(self, environment: gym.Env, config: Dict[str, Any]):
        """
        Initialize the tutorial agent.
        
        Args:
            environment: The tutorial environment
            config: Configuration dictionary
        """
        self.env = environment
        self.config = config
        self.logger = logging.getLogger("TutorialAgent")
        
        # Set up paths
        self.log_dir = config.get("paths", {}).get("log_dir", "logs/tutorial")
        self.model_dir = config.get("paths", {}).get("model_dir", "models/tutorial")
        
        # Set up model
        self.model = None
        self._setup_model()
        
        # Tracking metrics
        self.total_timesteps = 0
        self.total_episodes = 0
        self.completed_tutorials = 0
        
        self.logger.info("Tutorial agent initialized")
    
    def _setup_model(self):
        """Set up the PPO model for the agent."""
        # Create target directories
        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs(self.model_dir, exist_ok=True)
        
        # Get model parameters from config
        policy_kwargs = {}
        if self.config.get("model", {}).get("use_lstm", False):
            policy_kwargs["lstm_hidden_size"] = self.config.get("model", {}).get("lstm_hidden_size", 64)
            policy_kwargs["net_arch"] = [dict(pi=[64, 64], vf=[64, 64])]
        
        # Initialize the PPO model
        self.model = PPO(
            policy="MlpPolicy",
            env=self.env,
            learning_rate=self.config.get("model", {}).get("learning_rate", 3e-4),
            n_steps=self.config.get("model", {}).get("n_steps", 2048),
            batch_size=self.config.get("model", {}).get("batch_size", 64),
            n_epochs=self.config.get("model", {}).get("n_epochs", 10),
            gamma=self.config.get("model", {}).get("gamma", 0.99),
            gae_lambda=self.config.get("model", {}).get("gae_lambda", 0.95),
            clip_range=self.config.get("model", {}).get("clip_range", 0.2),
            clip_range_vf=self.config.get("model", {}).get("clip_range_vf", None),
            normalize_advantage=self.config.get("model", {}).get("normalize_advantage", True),
            ent_coef=self.config.get("model", {}).get("ent_coef", 0.01),
            vf_coef=self.config.get("model", {}).get("vf_coef", 0.5),
            max_grad_norm=self.config.get("model", {}).get("max_grad_norm", 0.5),
            verbose=1,
            policy_kwargs=policy_kwargs
        )
        
        self.logger.info("PPO model created")
    
    def _setup_callbacks(self, custom_callback=None):
        """Set up training callbacks."""
        callbacks = []
        
        # Add checkpoint callback
        checkpoint_freq = self.config.get("training", {}).get("checkpoint_freq", 10000)
        if checkpoint_freq > 0:
            checkpoint_callback = CheckpointCallback(
                save_freq=checkpoint_freq,
                save_path=self.model_dir,
                name_prefix="tutorial_model",
                verbose=1
            )
            callbacks.append(checkpoint_callback)
        
        # Add custom callback if provided
        if custom_callback is not None:
            callbacks.append(custom_callback)
        
        return callbacks
    
    def train(self, total_timesteps: int, callback=None):
        """
        Train the agent for a specified number of timesteps.
        
        Args:
            total_timesteps: Number of timesteps to train for
            callback: Optional callback for tracking progress
        
        Returns:
            Training metrics
        """
        self.logger.info(f"Starting training for {total_timesteps} timesteps")
        
        callbacks = self._setup_callbacks(callback)
        
        # Train the model
        self.model.learn(
            total_timesteps=total_timesteps,
            callback=callbacks,
            progress_bar=True
        )
        
        self.total_timesteps += total_timesteps
        
        # Save the final model
        final_model_path = os.path.join(self.model_dir, "final_model")
        self.model.save(final_model_path)
        self.logger.info(f"Saved final model to {final_model_path}")
        
        # Return training metrics
        return {
            "total_timesteps": self.total_timesteps,
            "total_episodes": self.total_episodes,
            "completed_tutorials": self.completed_tutorials
        }
    
    def predict(self, observation, deterministic=False):
        """
        Generate a prediction from the model for a given observation.
        
        Args:
            observation: The current observation
            deterministic: Whether to use deterministic predictions
        
        Returns:
            action, state
        """
        if self.model is None:
            self.logger.warning("Model not initialized, using random action")
            return np.random.randint(0, self.env.action_space.n), None
        
        action, state = self.model.predict(observation, deterministic=deterministic)
        return action, state
    
    def save(self, path: str):
        """
        Save the agent's model to a file.
        
        Args:
            path: Path to save the model to
        """
        if self.model is not None:
            self.model.save(path)
            self.logger.info(f"Model saved to {path}")
        else:
            self.logger.warning("Cannot save model: model not initialized")
    
    def load(self, path: str):
        """
        Load the agent's model from a file.
        
        Args:
            path: Path to load the model from
        """
        if os.path.exists(path):
            self.model = PPO.load(path, env=self.env)
            self.logger.info(f"Model loaded from {path}")
        else:
            self.logger.error(f"Cannot load model: file not found at {path}") 