from typing import Dict, Any, Optional
import gymnasium as gym
import logging

import numpy as np
import torch as th
from stable_baselines3 import PPO, DQN, A2C
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv, VecTransposeImage
from stable_baselines3.common.preprocessing import maybe_transpose


def create_agent(env: gym.Env, config: Dict[str, Any], seed: Optional[int] = None) -> Any:
    """
    Create a reinforcement learning agent based on the configuration.
    
    Args:
        env: Gym environment
        config: Configuration dictionary
        seed: Random seed
        
    Returns:
        RL agent
    """
    logger = logging.getLogger("AgentFactory")
    
    # Get agent configuration
    agent_config = config["agent"]
    algorithm = agent_config["algorithm"]
    
    # Set random seed
    if seed is None:
        seed = config["training"]["random_seed"]
    
    # Create policy kwargs based on network architecture
    policy_kwargs = _create_policy_kwargs(config)
    
    # Create the agent based on the algorithm
    if algorithm == "PPO":
        logger.info("Creating PPO agent")
        agent = PPO(
            policy="CnnPolicy" if _has_image_observation(env) else "MlpPolicy",
            env=env,
            learning_rate=agent_config["ppo"]["learning_rate"],
            n_steps=agent_config["ppo"]["n_steps"],
            batch_size=agent_config["ppo"]["batch_size"],
            n_epochs=agent_config["ppo"]["n_epochs"],
            gamma=agent_config["ppo"]["gamma"],
            gae_lambda=agent_config["ppo"]["gae_lambda"],
            clip_range=agent_config["ppo"]["clip_range"],
            clip_range_vf=None,
            normalize_advantage=True,
            ent_coef=agent_config["ppo"]["ent_coef"],
            vf_coef=agent_config["ppo"]["vf_coef"],
            max_grad_norm=agent_config["ppo"]["max_grad_norm"],
            use_sde=False,
            sde_sample_freq=-1,
            target_kl=None,
            tensorboard_log=config["paths"]["tensorboard"],
            policy_kwargs=policy_kwargs,
            verbose=1,
            seed=seed,
            device="auto"
        )
    elif algorithm == "DQN":
        logger.info("Creating DQN agent")
        agent = DQN(
            policy="CnnPolicy" if _has_image_observation(env) else "MlpPolicy",
            env=env,
            learning_rate=1e-4,
            buffer_size=agent_config["dqn"]["buffer_size"],
            learning_starts=agent_config["dqn"]["learning_starts"],
            batch_size=agent_config["dqn"]["batch_size"],
            tau=agent_config["dqn"]["tau"],
            gamma=agent_config["dqn"]["gamma"],
            train_freq=agent_config["dqn"]["train_freq"],
            gradient_steps=1,
            replay_buffer_class=None,
            replay_buffer_kwargs=None,
            optimize_memory_usage=False,
            target_update_interval=agent_config["dqn"]["target_update_interval"],
            exploration_fraction=agent_config["dqn"]["exploration_fraction"],
            exploration_initial_eps=1.0,
            exploration_final_eps=agent_config["dqn"]["exploration_final_eps"],
            max_grad_norm=10,
            tensorboard_log=config["paths"]["tensorboard"],
            policy_kwargs=policy_kwargs,
            verbose=1,
            seed=seed,
            device="auto"
        )
    elif algorithm == "A2C":
        logger.info("Creating A2C agent")
        agent = A2C(
            policy="CnnPolicy" if _has_image_observation(env) else "MlpPolicy",
            env=env,
            learning_rate=0.0007,
            n_steps=5,
            gamma=0.99,
            gae_lambda=1.0,
            ent_coef=0.0,
            vf_coef=0.5,
            max_grad_norm=0.5,
            rms_prop_eps=1e-5,
            use_rms_prop=True,
            use_sde=False,
            sde_sample_freq=-1,
            normalize_advantage=False,
            tensorboard_log=config["paths"]["tensorboard"],
            policy_kwargs=policy_kwargs,
            verbose=1,
            seed=seed,
            device="auto"
        )
    else:
        logger.error(f"Unknown algorithm: {algorithm}")
        raise ValueError(f"Unknown algorithm: {algorithm}")
    
    return agent


def _create_policy_kwargs(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Create policy kwargs based on network architecture configuration.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Policy kwargs dictionary
    """
    network_config = config["agent"]["network"]
    policy_kwargs = {}
    
    # Feature extractor for CNN
    cnn_config = network_config["cnn"]
    
    # MLP configuration
    mlp_config = network_config["mlp"]
    policy_kwargs["net_arch"] = mlp_config["hidden_layers"] + [dict(pi=[64], vf=[64])]
    
    # LSTM configuration
    if network_config["use_lstm"]:
        lstm_units = network_config["lstm_units"]
        # Note: LSTM integration with SB3 requires custom policies
        # This is a placeholder and would need more implementation
        policy_kwargs["lstm_hidden_size"] = lstm_units
    
    return policy_kwargs


def _has_image_observation(env: gym.Env) -> bool:
    """
    Check if the environment has image observations.
    
    Args:
        env: Gym environment
        
    Returns:
        True if the environment has image observations, False otherwise
    """
    if isinstance(env.observation_space, gym.spaces.Dict):
        return "visual" in env.observation_space.spaces
    
    if isinstance(env.observation_space, gym.spaces.Box):
        return len(env.observation_space.shape) == 3  # (height, width, channels)
    
    return False


def wrap_env_if_needed(env: gym.Env, config: Dict[str, Any]) -> gym.Env:
    """
    Wrap the environment if needed (e.g., for image observations).
    
    Args:
        env: Gym environment
        config: Configuration dictionary
        
    Returns:
        Wrapped environment
    """
    # Vectorize the environment
    n_envs = config["training"]["n_envs"]
    if n_envs > 1:
        def make_env():
            return env
        env = DummyVecEnv([make_env for _ in range(n_envs)])
    
    # Transpose image observations if needed
    if _has_image_observation(env) and n_envs > 1:
        env = VecTransposeImage(env)
    
    return env 