#!/usr/bin/env python3
"""
Train a discovery-based agent for Cities: Skylines 2 using PPO.
"""

import os
import sys
import yaml
import time
import datetime
import logging
import argparse
from pathlib import Path
import json
import shutil
import traceback

import numpy as np

import stable_baselines3
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, CallbackList
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
from gymnasium import spaces
import gymnasium as gym

from src.environment.discovery_env import DiscoveryEnvironment
from src.models.features_extractor import CombinedExtractor


def main():
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(f"logs/training_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
        ]
    )
    logger = logging.getLogger("train_discovery")
    
    # Parse arguments
    parser = argparse.ArgumentParser(description="Train a discovery-based agent for Cities: Skylines 2")
    parser.add_argument("--config", type=str, default="config/discovery_config.yaml", help="Path to config file")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to checkpoint to continue from")
    args = parser.parse_args()
    
    # Load config
    try:
        with open(args.config, "r") as f:
            config = yaml.safe_load(f)
            logger.info(f"Loaded config from {args.config}")
    except Exception as e:
        logger.error(f"Error loading config: {e}")
        return 1
    
    # Create directories for models and logs
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    model_dir = os.path.join(config.get("training_config", {}).get("save_path", "models"), f"discovery_{timestamp}")
    log_dir = os.path.join(config.get("training_config", {}).get("log_path", "logs"), f"discovery_{timestamp}")
    
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    
    logger.info(f"Model directory: {model_dir}")
    logger.info(f"Log directory: {log_dir}")
    
    # Save the complete configuration to the model directory for reproducibility
    with open(os.path.join(model_dir, "config.yaml"), "w") as f:
        yaml.dump(config, f)
    
    # Create a file for tracking action history
    action_log_file = os.path.join(log_dir, "action_history.txt")
    with open(action_log_file, "w") as f:
        f.write(f"Action History - Started at {datetime.datetime.now()}\n")
        f.write("=" * 80 + "\n")
    
    # Print key configuration settings
    logger.info(f"Vision guidance frequency: {config.get('base_env_config', {}).get('vision_guidance_frequency', 0.2)}")
    logger.info(f"Action delay: {config.get('base_env_config', {}).get('action_delay', 0.5)} seconds")
    logger.info(f"Ollama model: {config.get('ollama_config', {}).get('model', 'granite3.2-vision:latest')}")
    
    try:
        # Create and wrap the environment
        env = DiscoveryEnvironment(config)
        env = Monitor(env, log_dir)
        env = ExplicitObservationWrapper(env)
        env = DummyVecEnv([lambda: env])
        
        logger.info("Environment created and wrapped")
        
        # Set up callbacks
        checkpoint_interval = config.get("training_config", {}).get("checkpoint_interval", 10000)
        checkpoint_callback = CheckpointCallback(
            save_freq=checkpoint_interval, 
            save_path=model_dir,
            name_prefix="discovery_model"
        )
        callbacks = CallbackList([checkpoint_callback])
        
        logger.info(f"Checkpoint callback configured with interval {checkpoint_interval}")
        
        # Set up the model
        try:
            if args.checkpoint:
                # Load model from checkpoint
                logger.info(f"Loading model from checkpoint: {args.checkpoint}")
                model = PPO.load(args.checkpoint, env=env)
                logger.info("Model loaded successfully")
            else:
                # Create a new model with the configured parameters
                logger.info("Creating new PPO model")
                
                # Get PPO configuration
                ppo_config = config.get("ppo_config", {})
                
                # Extract features extractor configuration if provided
                features_extractor_config = config.get("features_extractor_config", {})
                
                # Create policy kwargs with the features extractor
                policy_kwargs = {
                    "features_extractor_class": CombinedExtractor,
                    "features_extractor_kwargs": features_extractor_config
                }
                
                # Create the model
                model = PPO(
                    "MultiInputPolicy",
                    env, 
                    verbose=1,
                    tensorboard_log=log_dir,
                    policy_kwargs=policy_kwargs,
                    **ppo_config
                )
                
                logger.info("PPO model created successfully")
        except Exception as e:
            logger.error(f"Error initializing model: {e}")
            traceback.print_exc()
            return 1
        
        # Configure training settings
        training_config = config.get("training_config", {})
        total_timesteps = training_config.get("total_timesteps", 500000)
        log_interval = training_config.get("log_interval", 10)
        
        # Log training start
        logger.info(f"Starting training for {total_timesteps} timesteps")
        logger.info(f"Log interval: {log_interval}")
        
        # Train the model
        try:
            model.learn(
                total_timesteps=total_timesteps,
                callback=callbacks,
                log_interval=log_interval,
                reset_num_timesteps=training_config.get("reset_num_timesteps", False),
                progress_bar=training_config.get("progress_bar", True)
            )
            
            # Save the final model
            final_model_path = os.path.join(model_dir, "final_model")
            model.save(final_model_path)
            logger.info(f"Training completed, model saved to {final_model_path}")
            
        except KeyboardInterrupt:
            logger.info("Training interrupted by user")
            # Save the interrupted model
            interrupted_model_path = os.path.join(model_dir, "interrupted_model")
            model.save(interrupted_model_path)
            logger.info(f"Interrupted model saved to {interrupted_model_path}")
        except Exception as e:
            logger.error(f"Error during training: {e}")
            traceback.print_exc()
            # Try to save the model if possible
            try:
                error_model_path = os.path.join(model_dir, "error_model")
                model.save(error_model_path)
                logger.info(f"Model from error state saved to {error_model_path}")
            except Exception as save_e:
                logger.error(f"Failed to save model after error: {save_e}")
        
        # Close the environment
        env.close()
        
    except Exception as e:
        logger.error(f"Error in main function: {e}")
        traceback.print_exc()
        return 1
    
    return 0

class ExplicitObservationWrapper(gym.Wrapper):
    """
    A wrapper that explicitly sets the observation space if it is not already defined.
    """
    
    def __init__(self, env):
        """
        Initialize the wrapper.
        
        Args:
            env: The environment to wrap
        """
        super().__init__(env)
        
        # Set up logger for this class
        self.logger = logging.getLogger("ExplicitObservationWrapper")
        
        # Check if observation space is defined
        if self.observation_space is None:
            # Get a sample observation
            obs, _ = self.env.reset()
            
            # Create an explicit observation space based on the sample
            if isinstance(obs, dict):
                spaces_dict = {}
                for key, value in obs.items():
                    if isinstance(value, np.ndarray):
                        if len(value.shape) >= 3 and value.shape[-1] in [1, 3]:
                            # Image observation
                            spaces_dict[key] = spaces.Box(
                                low=0, high=255, shape=value.shape, dtype=np.uint8
                            )
                        else:
                            # Vector observation
                            spaces_dict[key] = spaces.Box(
                                low=-np.inf, high=np.inf, shape=value.shape, dtype=np.float32
                            )
                
                self.observation_space = spaces.Dict(spaces_dict)
            elif isinstance(obs, np.ndarray):
                # Simple array observation
                if len(obs.shape) >= 3 and obs.shape[-1] in [1, 3]:
                    # Image observation
                    self.observation_space = spaces.Box(
                        low=0, high=255, shape=obs.shape, dtype=np.uint8
                    )
                else:
                    # Vector observation
                    self.observation_space = spaces.Box(
                        low=-np.inf, high=np.inf, shape=obs.shape, dtype=np.float32
                    )
            else:
                # Fallback for unknown observation type
                self.observation_space = spaces.Box(
                    low=-np.inf, high=np.inf, shape=(1,), dtype=np.float32
                )
        
        self.logger.info(f"Observation space: {self.observation_space}")

if __name__ == "__main__":
    sys.exit(main()) 