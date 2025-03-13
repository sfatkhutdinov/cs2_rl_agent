#!/usr/bin/env python3
"""
Test script to check the observation space of the DiscoveryEnvironment.
"""

import os
import yaml
import logging
import gymnasium as gym
import numpy as np

from src.environment.discovery_env import DiscoveryEnvironment

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("test_observation_space")

# Load config
with open("config/discovery_config.yaml", "r") as f:
    config = yaml.safe_load(f)
    logger.info(f"Loaded config from config/discovery_config.yaml")

# Create environment
logger.info("Creating environment...")
env = DiscoveryEnvironment(config)

# Check observation space
logger.info(f"Observation space: {env.observation_space}")
if isinstance(env.observation_space, gym.spaces.Dict):
    # Print each subspace
    logger.info("Dictionary observation space with the following subspaces:")
    for key, subspace in env.observation_space.spaces.items():
        logger.info(f"  {key}: {subspace}")

# Reset environment and check observation
logger.info("Resetting environment...")
obs, info = env.reset()
logger.info(f"Observation type: {type(obs)}")
logger.info(f"Observation keys: {obs.keys() if isinstance(obs, dict) else 'Not a dict'}")

# Check each observation component
if isinstance(obs, dict):
    for key, value in obs.items():
        logger.info(f"  {key}: shape={value.shape if hasattr(value, 'shape') else 'No shape'}, type={type(value)}")

# Close environment
env.close()
logger.info("Test completed.") 