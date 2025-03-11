import os
import argparse
import logging
from datetime import datetime
from typing import Dict, Any

import gymnasium as gym

from src.environment.cs2_env import CS2Environment
from src.environment.discovery_env import DiscoveryEnvironment
from src.agent.agent_factory import create_agent, wrap_env_if_needed
from src.utils.config_utils import load_config, override_config, get_full_path, save_config
from src.utils.logger import Logger


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train a reinforcement learning agent for Cities: Skylines 2")
    
    parser.add_argument("--config", type=str, default="src/config/default.yaml",
                       help="Path to the configuration file")
    parser.add_argument("--experiment_name", type=str, default=None,
                       help="Name of the experiment")
    parser.add_argument("--seed", type=int, default=None,
                       help="Random seed")
    parser.add_argument("--total_timesteps", type=int, default=None,
                       help="Total number of timesteps to train for")
    
    return parser.parse_args()


def setup_logging():
    """Set up logging."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )


def create_environment(config: Dict[str, Any]) -> gym.Env:
    """
    Create the Cities: Skylines 2 environment.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Gym environment
    """
    # Create the environment based on type
    env_type = config.get("environment", {}).get("type", "CS2Environment")
    
    print(f"Creating environment of type: {env_type}")
    
    if env_type == "DiscoveryEnvironment":
        print("Initializing Discovery Environment with config...")
        env = DiscoveryEnvironment(config)
    else:
        print("Initializing CS2Environment with config...")
        env = CS2Environment(config)
    
    # Wrap the environment if needed
    env = wrap_env_if_needed(env, config)
    
    return env


def train(config: Dict[str, Any], experiment_name: str = None, seed: int = None):
    """
    Train the reinforcement learning agent.
    
    Args:
        config: Configuration dictionary
        experiment_name: Name of the experiment
        seed: Random seed
    """
    logger = logging.getLogger("train")
    
    # Create experiment logger
    exp_logger = Logger(config, experiment_name)
    exp_logger.log_info("Starting training")
    
    # Create the environment
    env = create_environment(config)
    exp_logger.log_info(f"Environment created: {env}")
    
    # Create the agent
    agent = create_agent(env, config, seed)
    exp_logger.log_info(f"Agent created: {agent}")
    
    # Get training parameters
    total_timesteps = config["training"]["total_timesteps"]
    save_freq = config["training"]["save_freq"]
    eval_freq = config["training"]["eval_freq"]
    
    # Create model directory
    model_dir = os.path.normpath(os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        config["paths"]["models"]
    ))
    os.makedirs(model_dir, exist_ok=True)
    
    try:
        # Train the agent
        exp_logger.log_info(f"Starting training for {total_timesteps} timesteps")
        agent.learn(
            total_timesteps=total_timesteps,
            log_interval=config["training"]["log_interval"],
            tb_log_name=experiment_name or f"experiment_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )
        
        # Save the final model
        final_model_path = os.path.join(model_dir, f"{experiment_name or 'final_model'}.zip")
        agent.save(final_model_path)
        exp_logger.log_info(f"Final model saved to {final_model_path}")
    
    except KeyboardInterrupt:
        exp_logger.log_info("Training interrupted by user")
    
    except Exception as e:
        exp_logger.log_error(f"Training failed: {str(e)}")
        raise
    
    finally:
        # Close the environment
        env.close()
        
        # Close the logger
        exp_logger.close()


def main():
    """Main entry point."""
    # Parse command line arguments
    args = parse_args()
    
    # Set up logging
    setup_logging()
    logger = logging.getLogger("main")
    
    # Load configuration
    logger.info(f"Loading configuration from {args.config}")
    config = load_config(args.config)
    
    # Override configuration with command line arguments
    override_dict = {}
    if args.total_timesteps is not None:
        override_dict["training"] = {"total_timesteps": args.total_timesteps}
    
    config = override_config(config, override_dict)
    
    # Set experiment name
    experiment_name = args.experiment_name or f"experiment_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    # Start training
    logger.info(f"Starting experiment: {experiment_name}")
    train(config, experiment_name, args.seed)
    logger.info("Training completed")


if __name__ == "__main__":
    main() 