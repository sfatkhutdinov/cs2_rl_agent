import os
import argparse
import yaml
import logging
import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
import datetime
import traceback
import sys

from src.environment.discovery_env import DiscoveryEnvironment
from src.utils.logging_utils import setup_logger
from src.utils.callbacks import TensorboardCallback
from src.utils.file_utils import ensure_dir
from src.utils.config_utils import load_config


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train a discovery-based RL agent for Cities: Skylines 2")
    parser.add_argument(
        "--config", 
        type=str, 
        default="config/discovery_config.yaml",
        help="Path to configuration file"
    )
    parser.add_argument(
        "--log-level", 
        type=str, 
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        default="INFO", 
        help="Set the logging level"
    )
    parser.add_argument(
        "--model-dir", 
        type=str, 
        default=None, 
        help="Directory to save models (default: models/discovery_YYYYMMDD_HHMMSS)"
    )
    parser.add_argument(
        "--log-dir", 
        type=str, 
        default=None, 
        help="Directory to save logs (default: logs/discovery_YYYYMMDD_HHMMSS)"
    )
    parser.add_argument(
        "--verify-config",
        action="store_true",
        help="Verify the configuration and exit without training"
    )
    parser.add_argument("--total_timesteps", type=int, default=None,
                       help="Total number of timesteps to train for")
    parser.add_argument("--save_freq", type=int, default=10000,
                       help="How often to save model checkpoints")
    return parser.parse_args()


def make_env(config, use_fallback_mode=True, rank=0):
    """
    Create a function that will create a Cities: Skylines 2 environment.
    
    Args:
        config: Configuration dictionary
        use_fallback_mode: Whether to use fallback mode if connection fails
        rank: Environment rank for parallel environments
        
    Returns:
        Function that creates an environment instance
    """
    def _init():
        try:
            # Set up logger for this environment
            env_logger = setup_logger(
                f"DiscoveryEnv_{rank}",
                log_level=logging.INFO,
                log_file=os.path.join(config.get("log_dir", "logs"), f"discovery_env_{rank}.log")
            )
            
            # Ensure base_env_config has observation_space with include_visual
            base_env_config = config.get("environment", {}).copy()
            if "observation_space" not in base_env_config:
                # Copy observation details to environment.observation_space
                obs_config = config.get("observation", {})
                
                base_env_config["observation_space"] = {
                    "type": "dict",
                    "spaces": {
                        "metrics": {
                            "type": "box",
                            "shape": [10],
                            "low": -1.0,
                            "high": 1.0
                        },
                        "minimap": {
                            "type": "box",
                            "shape": [84, 84, 3],
                            "low": 0,
                            "high": 255
                        },
                        "screenshot": {
                            "type": "box",
                            "shape": [224, 224, 3],
                            "low": 0,
                            "high": 255
                        }
                    },
                    "include_visual": True  # Explicitly add this key
                }
            else:
                # Ensure include_visual is in the observation_space
                base_env_config["observation_space"]["include_visual"] = True
                
            # Ensure observation config has include_visual
            observation_config = config.get("observation", {}).copy()
            observation_config["include_visual"] = True
            
            # Initialize environment with configurations
            env = DiscoveryEnvironment(
                base_env_config=base_env_config,
                observation_config=observation_config,
                vision_config=config.get("vision", {}),
                use_fallback_mode=use_fallback_mode,
                discovery_frequency=config.get("discovery_frequency", 0.3),
                tutorial_frequency=config.get("tutorial_frequency", 0.3),
                random_action_frequency=config.get("random_action_frequency", 0.2),
                exploration_randomness=config.get("exploration_randomness", 0.5),
                logger=env_logger
            )
            
            return env
        except Exception as e:
            # Print more detailed error information
            print(f"Error creating environment: {str(e)}")
            print("Traceback:")
            traceback.print_exc()
            
            # Re-raise to prevent training from continuing with a broken environment
            raise
    
    return _init


def verify_config(config):
    """
    Verify that the configuration has all required fields.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        bool: True if configuration is valid, False otherwise
    """
    required_sections = ["environment", "observation", "vision", "training"]
    required_env_fields = ["reward_type", "action_type", "interface_type"]
    
    # Check required sections
    for section in required_sections:
        if section not in config:
            print(f"ERROR: Missing required section '{section}' in configuration")
            return False
    
    # Check required environment fields
    for field in required_env_fields:
        if field not in config["environment"]:
            print(f"ERROR: Missing required field '{field}' in environment section")
            return False
    
    # Ensure observation_space is present or can be created
    if "observation_space" not in config["environment"]:
        if "observation" not in config or not isinstance(config["observation"], dict):
            print("ERROR: Either environment.observation_space or observation section must be defined")
            return False
    
    # Ensure vision configuration is valid
    if "ollama_model" not in config["vision"]:
        print("ERROR: Missing required field 'ollama_model' in vision section")
        return False
    
    # Ensure training configuration is valid
    if "policy" not in config["training"]:
        print("ERROR: Missing required field 'policy' in training section")
        return False
    
    print("Configuration verification passed!")
    return True


def train(config, args):
    """
    Train the discovery-based agent.
    
    Args:
        config: Configuration dictionary
        args: Command line arguments
    """
    # Create directories with timestamp
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    model_dir = args.model_dir if args.model_dir else os.path.join("models", f"discovery_{timestamp}")
    log_dir = args.log_dir if args.log_dir else os.path.join("logs", f"discovery_{timestamp}")
    
    ensure_dir(model_dir)
    ensure_dir(log_dir)
    
    # Set up logger
    logger = setup_logger(
        "DiscoveryTraining",
        log_level=getattr(logging, args.log_level),
        log_file=os.path.join(log_dir, "training.log")
    )
    
    logger.info(f"Starting discovery-based training with config: {args.config}")
    logger.info(f"Model directory: {model_dir}")
    logger.info(f"Log directory: {log_dir}")
    
    # Update config with directories
    config["model_dir"] = model_dir
    config["log_dir"] = log_dir
    
    # Save the complete configuration to the model directory for reproducibility
    config_save_path = os.path.join(model_dir, "used_config.yaml")
    with open(config_save_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False)
    logger.info(f"Saved configuration to {config_save_path}")
    
    # Training parameters
    n_envs = config["training"].get("n_envs", 1)
    total_timesteps = args.total_timesteps or config["training"].get("total_timesteps", 1000000)
    
    # Set up environment(s)
    logger.info(f"Creating {n_envs} environment{'s' if n_envs > 1 else ''}...")
    try:
        if n_envs == 1:
            env = DummyVecEnv([make_env(config, use_fallback_mode=True)])
            logger.info("Created single environment")
        else:
            env = SubprocVecEnv([make_env(config, use_fallback_mode=True, rank=i) for i in range(n_envs)])
            logger.info(f"Created {n_envs} parallel environments")
    except Exception as e:
        logger.error(f"Error creating environments: {str(e)}")
        logger.error("Traceback:", exc_info=True)
        print(f"Error creating environments: {str(e)}")
        return
    
    # Set up checkpoint callback
    checkpoint_dir = os.path.join(model_dir, "checkpoints")
    ensure_dir(checkpoint_dir)
    
    checkpoint_callback = CheckpointCallback(
        save_freq=args.save_freq,
        save_path=checkpoint_dir,
        name_prefix="discovery_model",
        save_replay_buffer=False,
        save_vecnormalize=True,
    )
    
    # Set up tensorboard callback
    tensorboard_callback = TensorboardCallback()
    
    # Initialize agent
    logger.info("Initializing PPO agent...")
    try:
        model = PPO(
            policy=config["training"].get("policy", "MultiInputPolicy"),
            env=env,
            learning_rate=config["training"].get("learning_rate", 3e-4),
            n_steps=config["training"].get("n_steps", 2048),
            batch_size=config["training"].get("batch_size", 64),
            n_epochs=config["training"].get("n_epochs", 10),
            gamma=config["training"].get("gamma", 0.99),
            gae_lambda=config["training"].get("gae_lambda", 0.95),
            clip_range=config["training"].get("clip_range", 0.2),
            clip_range_vf=config["training"].get("clip_range_vf", None),
            normalize_advantage=config["training"].get("normalize_advantage", True),
            ent_coef=config["training"].get("ent_coef", 0.01),
            vf_coef=config["training"].get("vf_coef", 0.5),
            max_grad_norm=config["training"].get("max_grad_norm", 0.5),
            use_sde=config["training"].get("use_sde", False),
            sde_sample_freq=config["training"].get("sde_sample_freq", -1),
            target_kl=config["training"].get("target_kl", None),
            tensorboard_log=log_dir,
            policy_kwargs=config.get("model", {}),
            verbose=1,
            device=config["training"].get("device", "auto"),
        )
        
        logger.info("Model initialized, starting training")
        
        # Train the agent
        model.learn(
            total_timesteps=total_timesteps,
            callback=[checkpoint_callback, tensorboard_callback],
            log_interval=10,
            progress_bar=True,
        )
        
        # Save the final model
        final_model_path = os.path.join(model_dir, "final_model")
        model.save(final_model_path)
        logger.info(f"Training complete. Final model saved to {final_model_path}")
        
    except Exception as e:
        logger.error(f"Error during training: {str(e)}")
        logger.error("Traceback:", exc_info=True)
        print(f"Error during training: {str(e)}")
    finally:
        # Close environment
        env.close()


def main():
    """Main function."""
    print("=== Discovery-Based CS2 Agent Training ===")
    
    args = parse_args()
    
    # Load configuration
    try:
        print(f"Loading config from: {args.config}")
        config = load_config(args.config)
        print("Config loaded successfully")
    except Exception as e:
        print(f"Error loading configuration: {str(e)}")
        return
    
    # Verify configuration
    if not verify_config(config):
        if args.verify_config:
            return
        print("Configuration has errors, but continuing anyway...")
    elif args.verify_config:
        print("Configuration is valid.")
        return
    
    # Train the agent
    train(config, args)
    
    print("Training complete!")


if __name__ == "__main__":
    main() 