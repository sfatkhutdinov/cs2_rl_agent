import os
import logging
import argparse
import time
import yaml
import numpy as np
import random
import torch
from datetime import datetime
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.evaluation import evaluate_policy

from src.environment.cs2_env import CS2Environment
from src.environment.autonomous_env import AutonomousCS2Environment
from src.interface.auto_vision_interface import AutoVisionInterface


# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("logs/autonomous_training.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("AutonomousTraining")


def make_env(rank, config, seed=0):
    """
    Helper function to create a vectorized environment.
    
    Args:
        rank: Subprocess rank
        config: Configuration dictionary
        seed: Random seed
        
    Returns:
        A function that creates and returns the environment
    """
    def _init():
        # Set environment-level random seed
        set_random_seed(seed + rank)
        
        # Create the base environment
        interface_type = config.get("interface", {}).get("type", "auto_vision")
        if interface_type == "auto_vision":
            # Use the automatic vision interface
            interface_config = {
                "interface": {
                    "type": "auto_vision",
                    "vision": {
                        "detection_method": config.get("interface", {}).get("detection_method", "ocr"),
                        "ocr_confidence": config.get("interface", {}).get("ocr_confidence", 0.6),
                        "template_threshold": config.get("interface", {}).get("template_threshold", 0.7),
                        "cache_detections": config.get("interface", {}).get("cache_detections", True),
                        "screen_region": config.get("interface", {}).get("vision", {}).get("screen_region", [0, 0, 1920, 1080])
                    }
                }
            }
            interface = AutoVisionInterface(config=interface_config)
            interface.logger = logger  # Set logger after initialization
        else:
            raise ValueError(f"Unknown interface type: {interface_type}")
            
        # Create base environment configuration
        env_config = {
            "interface": interface_config["interface"],  # Pass the same interface config
            "environment": {
                "type": config.get("environment", {}).get("type", "cs2"),
                "observation_space": {
                    "type": config.get("environment", {}).get("observation_space", {}).get("type", "combined"),
                    "visual": config.get("environment", {}).get("observation_space", {}).get("visual", {
                        "enabled": True,
                        "resolution": [84, 84],
                        "grayscale": True
                    }),
                    "metrics": config.get("environment", {}).get("observation_space", {}).get("metrics", {
                        "enabled": True,
                        "normalize": True
                    })
                },
                "action_space": {
                    "type": config.get("environment", {}).get("action_space", {}).get("type", "advanced"),
                    "continuous": config.get("environment", {}).get("action_space", {}).get("continuous", False)
                },
                "reward_function": {
                    "type": config.get("environment", {}).get("reward_function", {}).get("type", "balanced"),
                    "weights": config.get("environment", {}).get("reward_function", {}).get("weights", {
                        "population": 0.3,
                        "happiness": 0.2,
                        "budget": 0.2,
                        "traffic": 0.2,
                        "discovery": 0.1
                    })
                },
                "max_episode_steps": config.get("environment", {}).get("max_episode_steps", 2000),
                "metrics_update_freq": config.get("environment", {}).get("metrics_update_freq", 10),
                "pause_on_menu": config.get("environment", {}).get("pause_on_menu", False),
                "metrics": config.get("environment", {}).get("metrics", [
                    "population",
                    "happiness",
                    "budget_balance",
                    "traffic"
                ])
            }
        }
        
        # Create base environment
        base_env = CS2Environment(config=env_config)
        base_env.logger = logger  # Set logger after initialization
        
        # Wrap with autonomous environment
        env = AutonomousCS2Environment(
            base_env=base_env,
            exploration_frequency=config.get("exploration", {}).get("frequency", 0.3),
            random_action_frequency=config.get("exploration", {}).get("random_action_frequency", 0.2),
            menu_exploration_buffer_size=config.get("exploration", {}).get("menu_buffer_size", 50),
            logger=logger
        )
        
        # Wrap with Monitor for logging
        log_dir = config.get("paths", {}).get("log_dir", "logs")
        os.makedirs(log_dir, exist_ok=True)
        env = Monitor(env, os.path.join(log_dir, f"monitor_{rank}"))
        
        return env
    
    return _init


def train(config_path):
    """
    Train the agent using the provided configuration.
    
    Args:
        config_path: Path to the YAML configuration file
    """
    # Load configuration
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Set random seeds for reproducibility
    seed = config.get("seed", 42)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    # Set up logging and directories
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_name = config.get("experiment_name", "autonomous_agent")
    log_dir = os.path.join(config.get("paths", {}).get("log_dir", "logs"), f"{experiment_name}_{timestamp}")
    model_dir = os.path.join(config.get("paths", {}).get("model_dir", "models"), f"{experiment_name}_{timestamp}")
    tb_log_dir = os.path.join(config.get("paths", {}).get("tensorboard_dir", "tensorboard"), f"{experiment_name}_{timestamp}")
    
    # Create directories
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(tb_log_dir, exist_ok=True)
    
    # Save the configuration
    with open(os.path.join(log_dir, "config.yaml"), 'w') as f:
        yaml.dump(config, f)
    
    # Determine number of environments to run in parallel
    n_envs = config.get("training", {}).get("n_envs", 1)
    
    # Only use 1 environment for this type of game interaction
    n_envs = 1
    
    # Create vectorized environment
    logger.info(f"Creating {n_envs} environments")
    
    env_fns = [make_env(i, config, seed) for i in range(n_envs)]
    if n_envs == 1:
        env = DummyVecEnv(env_fns)
    else:
        env = SubprocVecEnv(env_fns)
    
    # Define callbacks
    checkpoint_callback = CheckpointCallback(
        save_freq=max(100, config.get("training", {}).get("checkpoint_freq", 10000) // n_envs),
        save_path=model_dir,
        name_prefix="autonomous_model"
    )
    
    # Create evaluation environment if needed
    if config.get("training", {}).get("evaluate_during_training", True):
        # Create a separate environment for evaluation
        eval_env = DummyVecEnv([make_env(0, config, seed + 1000)])  # Different seed
        
        eval_callback = EvalCallback(
            eval_env,
            best_model_save_path=os.path.join(model_dir, "best_model"),
            log_path=os.path.join(log_dir, "eval"),
            eval_freq=max(100, config.get("training", {}).get("eval_freq", 10000) // n_envs),
            deterministic=True,
            render=False
        )
        callbacks = [checkpoint_callback, eval_callback]
    else:
        callbacks = [checkpoint_callback]
    
    # Create policy kwargs
    policy_kwargs = config.get("agent", {}).get("policy_kwargs", {})
    
    # Convert string network architectures to actual objects if present
    if "net_arch" in policy_kwargs and isinstance(policy_kwargs["net_arch"], list):
        # Handle LSTM case
        if any(isinstance(item, dict) and "lstm" in item for item in policy_kwargs["net_arch"]):
            logger.info("Using LSTM network architecture")
            # Ensure we're using the correct feature extractor for LSTM
            if "features_extractor_class" not in policy_kwargs:
                pass  # Let SB3 handle this
    
    # Initialize the agent
    logger.info("Initializing PPO agent")
    model = PPO(
        policy=config.get("agent", {}).get("policy_type", "MlpPolicy"),
        env=env,
        learning_rate=config.get("agent", {}).get("learning_rate", 3e-4),
        n_steps=config.get("agent", {}).get("n_steps", 2048),
        batch_size=config.get("agent", {}).get("batch_size", 64),
        n_epochs=config.get("agent", {}).get("n_epochs", 10),
        gamma=config.get("agent", {}).get("gamma", 0.99),
        gae_lambda=config.get("agent", {}).get("gae_lambda", 0.95),
        clip_range=config.get("agent", {}).get("clip_range", 0.2),
        clip_range_vf=config.get("agent", {}).get("clip_range_vf", None),
        normalize_advantage=config.get("agent", {}).get("normalize_advantage", True),
        ent_coef=config.get("agent", {}).get("ent_coef", 0.01),
        vf_coef=config.get("agent", {}).get("vf_coef", 0.5),
        max_grad_norm=config.get("agent", {}).get("max_grad_norm", 0.5),
        use_sde=config.get("agent", {}).get("use_sde", False),
        sde_sample_freq=config.get("agent", {}).get("sde_sample_freq", -1),
        target_kl=config.get("agent", {}).get("target_kl", None),
        tensorboard_log=tb_log_dir,
        create_eval_env=False,
        policy_kwargs=policy_kwargs,
        verbose=1,
        seed=seed
    )
    
    # Train the agent
    total_timesteps = config.get("training", {}).get("total_timesteps", 1000000)
    logger.info(f"Training agent for {total_timesteps} timesteps")
    try:
        model.learn(
            total_timesteps=total_timesteps,
            callback=callbacks,
            tb_log_name=experiment_name
        )
        # Save the final model
        final_model_path = os.path.join(model_dir, "final_model")
        model.save(final_model_path)
        logger.info(f"Training completed. Final model saved to {final_model_path}")
    except Exception as e:
        logger.error(f"Training interrupted: {str(e)}")
        # Try to save the model
        try:
            interrupted_model_path = os.path.join(model_dir, "interrupted_model")
            model.save(interrupted_model_path)
            logger.info(f"Interrupted model saved to {interrupted_model_path}")
        except Exception as save_err:
            logger.error(f"Failed to save interrupted model: {str(save_err)}")
    
    # Close environment
    env.close()
    if config.get("training", {}).get("evaluate_during_training", True):
        eval_env.close()


def main():
    parser = argparse.ArgumentParser(description="Train an autonomous agent for Cities: Skylines 2")
    parser.add_argument("--config", type=str, required=True, help="Path to the configuration file")
    args = parser.parse_args()
    
    if not os.path.exists(args.config):
        logger.error(f"Config file not found: {args.config}")
        return
    
    logger.info(f"Starting autonomous agent training with config: {args.config}")
    train(args.config)


if __name__ == "__main__":
    main() 