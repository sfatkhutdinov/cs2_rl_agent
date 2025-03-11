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
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv, VecTransposeImage
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
        interface_type = config.get("environment", {}).get("interface", {}).get("type", "auto_vision")
        
        # Get screen region configuration with fallback to default
        screen_region = config.get("environment", {}).get("interface", {}).get("screen_region", [0, 0, 1920, 1080])
        
        if interface_type == "auto_vision":
            # Use the automatic vision interface
            interface_config = {
                "interface": {
                    "type": "auto_vision",
                    "vision": {
                        "detection_method": "ocr",  # Default to OCR
                        "ocr_confidence": 0.6,
                        "template_threshold": 0.7,
                        "screen_region": screen_region
                    },
                    "templates_dir": config.get("environment", {}).get("interface", {}).get("templates_dir", "templates"),
                    "ocr_enabled": config.get("environment", {}).get("interface", {}).get("ocr_enabled", True)
                }
            }
        else:
            # Default to auto vision if type not recognized
            interface_config = {
                "interface": {
                    "type": "auto_vision",
                    "vision": {
                        "screen_region": screen_region
                    }
                }
            }
        
        # Get observation space configuration
        obs_space_config = config.get("environment", {}).get("observation_space", {})
        if not obs_space_config:
            # Default observation space configuration
            obs_space_config = {
                "type": "combined",
                "include_visual": True,
                "include_metrics": True,
                "image_size": [84, 84],
                "grayscale": True,
                "normalize_metrics": True,
                "metrics": ["population", "happiness", "budget_balance", "traffic"]
            }
        
        # Get environment parameters
        max_episode_steps = config.get("environment", {}).get("max_episode_steps", 2000)
        metrics_update_freq = config.get("environment", {}).get("metrics_update_freq", 10)
        pause_on_menu = config.get("environment", {}).get("pause_on_menu", False)
        
        # Create environment config with all default values
        env_config = {
            **interface_config,
            "environment": {
                "observation_space": obs_space_config,
                "action_space": {
                    "type": "advanced",
                    "zone": [
                        "residential",
                        "commercial",
                        "industrial",
                        "office",
                        "delete_zone"
                    ],
                    "infrastructure": [
                        "road",
                        "power_line",
                        "water_pipe",
                        "park",
                        "service_building", 
                        "delete_infrastructure"
                    ],
                    "budget": [
                        "increase_residential_tax",
                        "decrease_residential_tax", 
                        "increase_commercial_tax",
                        "decrease_commercial_tax",
                        "increase_industrial_tax",
                        "decrease_industrial_tax",
                        "increase_service_budget",
                        "decrease_service_budget"
                    ]
                },
                "reward_function": config.get("environment", {}).get("reward_function", {}),
                "max_episode_steps": max_episode_steps,
                "metrics_update_freq": metrics_update_freq,
                "pause_on_menu": pause_on_menu
            }
        }
        
        # Create base environment
        try:
            base_env = CS2Environment(config=env_config)
            base_env.logger = logger  # Set logger after initialization
            
            # Wrap with autonomous environment
            autonomous_config = config.get("autonomous", {})
            env = AutonomousCS2Environment(
                base_env=base_env,
                exploration_frequency=autonomous_config.get("exploration_frequency", 0.3),
                random_action_frequency=autonomous_config.get("random_action_frequency", 0.2),
                menu_exploration_buffer_size=autonomous_config.get("menu_exploration_buffer_size", 50),
                logger=logger
            )
            
            # Wrap with Monitor for logging
            env_name = f"cs2_autonomous_{rank}"
            env = Monitor(env, os.path.join("experiments", config.get("experiment_name", "autonomous"), "logs", env_name))
            
            return env
        except Exception as e:
            logger.error(f"Failed to create environment: {e}")
            # Log more detailed information about the configuration
            logger.error(f"Interface config: {interface_config}")
            logger.error(f"Full env config: {env_config}")
            raise
        
    return _init


def train(config_path):
    """
    Train an autonomous agent using the configuration file.
    
    Args:
        config_path: Path to the configuration file
    """
    # Load configuration
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Setup directories
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_name = config.get("experiment_name", f"autonomous_{timestamp}")
    
    base_dir = os.path.join("experiments", experiment_name)
    log_dir = os.path.join(base_dir, "logs")
    model_dir = os.path.join(base_dir, "models")
    tb_log_dir = os.path.join(base_dir, "tensorboard")
    
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(tb_log_dir, exist_ok=True)
    
    # Set seed for reproducibility
    seed = config.get("seed", 42)
    set_random_seed(seed)
    
    # Add memory monitoring
    memory_monitor_enabled = config.get("memory_monitor", {}).get("enabled", True)
    memory_usage_limit = config.get("memory_monitor", {}).get("memory_limit_gb", 12)  # Default 12GB limit
    disk_usage_limit = config.get("memory_monitor", {}).get("disk_limit_gb", 50)  # Default 50GB limit
    check_interval = config.get("memory_monitor", {}).get("check_interval", 10000)  # Every 10k steps
    
    if memory_monitor_enabled:
        logger.info(f"Memory monitoring enabled. Limits: {memory_usage_limit}GB RAM, {disk_usage_limit}GB disk")
    
    # Create environment from config
    n_envs = config.get("training", {}).get("n_envs", 1)
    logger.info(f"Creating {n_envs} environment(s)")
    
    # Create vectorized environment
    env = DummyVecEnv([make_env(i, config, seed + i) for i in range(n_envs)])
    
    # Only wrap with VecTransposeImage if using CnnPolicy (not for MultiInputPolicy)
    policy_type = config.get("agent", {}).get("policy_type", "").lower()
    if policy_type == "cnnpolicy":
        env = VecTransposeImage(env)
    
    # Set up checkpoint callback
    checkpoint_freq = config.get("training", {}).get("checkpoint_freq", 10000)
    checkpoint_callback = CheckpointCallback(
        save_freq=max(100, checkpoint_freq // n_envs),
        save_path=model_dir,
        name_prefix="autonomous",
        save_replay_buffer=False,
        save_vecnormalize=True
    )
    
    # Create evaluation environment if needed
    if config.get("training", {}).get("evaluate_during_training", True):
        # Create a separate environment for evaluation
        eval_env = DummyVecEnv([make_env(0, config, seed + 1000)])  # Different seed
        
        # Ensure the eval environment has the same wrappers as the training environment
        if policy_type == "cnnpolicy":
            eval_env = VecTransposeImage(eval_env)
        
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
    
    # Add memory monitoring callback if enabled
    if memory_monitor_enabled:
        from stable_baselines3.common.callbacks import BaseCallback
        
        class MemoryMonitorCallback(BaseCallback):
            """
            Callback for monitoring memory and disk usage during training.
            Inherits from BaseCallback for proper integration with Stable Baselines3.
            """
            def __init__(self, memory_limit_gb, disk_limit_gb, check_interval, model_dir, verbose=0):
                super().__init__(verbose)
                self.memory_limit_bytes = memory_limit_gb * 1024 * 1024 * 1024
                self.disk_limit_bytes = disk_limit_gb * 1024 * 1024 * 1024
                self.check_interval = check_interval
                self.last_check = 0
                self.model_dir = model_dir
            
            def _init_callback(self):
                # Called when the callback is initialized
                logger.info("Memory monitoring callback initialized")
            
            def _on_step(self):
                # Only check periodically to avoid performance impact
                if self.num_timesteps - self.last_check < self.check_interval:
                    return True
                
                self.last_check = self.num_timesteps
                
                # Check RAM usage
                try:
                    import psutil
                    process = psutil.Process(os.getpid())
                    memory_usage = process.memory_info().rss
                    memory_percent = memory_usage / self.memory_limit_bytes * 100
                    
                    # Check disk usage
                    disk_usage = self._get_dir_size(self.model_dir)
                    disk_percent = disk_usage / self.disk_limit_bytes * 100
                    
                    logger.info(f"Memory usage: {memory_usage / 1024 / 1024 / 1024:.2f}GB ({memory_percent:.1f}%), "
                                f"Disk usage: {disk_usage / 1024 / 1024 / 1024:.2f}GB ({disk_percent:.1f}%)")
                    
                    # Handle excessive memory usage
                    if memory_usage > self.memory_limit_bytes:
                        logger.warning(f"Memory limit exceeded ({memory_percent:.1f}%)! Saving model and stopping...")
                        # Save the model before stopping
                        model_path = os.path.join(self.model_dir, f"memory_limit_model_{self.num_timesteps}")
                        self.model.save(model_path)
                        return False  # Stop training
                        
                    # Handle excessive disk usage
                    if disk_usage > self.disk_limit_bytes:
                        logger.warning(f"Disk limit exceeded ({disk_percent:.1f}%)! Cleaning up old checkpoints...")
                        # Clean up old checkpoints except the best ones
                        self._cleanup_old_checkpoints()
                        
                except Exception as e:
                    logger.error(f"Error in memory monitoring: {e}")
                
                return True
                
            def _get_dir_size(self, path):
                """Get the size of a directory in bytes"""
                total_size = 0
                for dirpath, dirnames, filenames in os.walk(path):
                    for f in filenames:
                        fp = os.path.join(dirpath, f)
                        total_size += os.path.getsize(fp)
                return total_size
                
            def _cleanup_old_checkpoints(self):
                """Clean up old checkpoints, keeping only the best and latest ones"""
                try:
                    # Find all checkpoint files
                    checkpoint_files = []
                    for dirpath, dirnames, filenames in os.walk(self.model_dir):
                        for f in filenames:
                            if f.endswith('.zip') and not dirpath.endswith('best_model'):
                                fp = os.path.join(dirpath, f)
                                checkpoint_files.append((fp, os.path.getmtime(fp)))
                    
                    # Sort by modification time (oldest first)
                    checkpoint_files.sort(key=lambda x: x[1])
                    
                    # Keep the 5 most recent checkpoints, delete the rest
                    files_to_delete = checkpoint_files[:-5]
                    total_freed = 0
                    for file_path, _ in files_to_delete:
                        size = os.path.getsize(file_path)
                        os.remove(file_path)
                        total_freed += size
                        logger.info(f"Deleted old checkpoint: {file_path}")
                    
                    logger.info(f"Freed {total_freed / 1024 / 1024:.2f}MB of disk space")
                except Exception as e:
                    logger.error(f"Error cleaning up checkpoints: {e}")
        
        # Add memory monitor to callbacks
        memory_callback = MemoryMonitorCallback(
            memory_limit_gb=memory_usage_limit,
            disk_limit_gb=disk_usage_limit,
            check_interval=check_interval,
            model_dir=model_dir
        )
        callbacks.append(memory_callback)
    
    # Create optimized policy kwargs
    agent_config = config.get("agent", {})
    policy_kwargs = {}
    
    # Start with some sensible defaults
    policy_kwargs = {
        "net_arch": {
            "pi": [64, 32],  # Smaller policy network
            "vf": [64, 32]   # Smaller value network
        },
        "activation_fn": torch.nn.ReLU,
        "ortho_init": True,  # Use orthogonal initialization for stability
        "normalize_images": True,  # Normalize image inputs for faster convergence
        "optimizer_class": torch.optim.Adam,
        "optimizer_kwargs": {
            "eps": 1e-5,  # Prevent division by zero
        }
    }
    
    # Override with user config if provided
    if "policy_kwargs" in agent_config:
        user_policy_kwargs = agent_config["policy_kwargs"]
        # Carefully merge, prioritizing user settings where provided
        if "net_arch" in user_policy_kwargs:
            policy_kwargs["net_arch"] = user_policy_kwargs["net_arch"]
        
        # Handle activation function - convert from string to actual function if needed
        if "activation_fn" in user_policy_kwargs:
            activation_str = user_policy_kwargs["activation_fn"]
            if isinstance(activation_str, str):
                # Map common activation function strings to torch functions
                activation_map = {
                    "relu": torch.nn.ReLU,
                    "tanh": torch.nn.Tanh,
                    "sigmoid": torch.nn.Sigmoid,
                    "leaky_relu": torch.nn.LeakyReLU,
                    "elu": torch.nn.ELU
                }
                if activation_str.lower() in activation_map:
                    policy_kwargs["activation_fn"] = activation_map[activation_str.lower()]
                else:
                    logger.warning(f"Unknown activation function: {activation_str}, using default (ReLU)")
            else:
                policy_kwargs["activation_fn"] = user_policy_kwargs["activation_fn"]
        
        # Copy other kwargs
        for key in ["ortho_init", "normalize_images", "optimizer_kwargs"]:
            if key in user_policy_kwargs:
                policy_kwargs[key] = user_policy_kwargs[key]
    
    # Initialize the agent
    logger.info("Initializing PPO agent")
    
    # Get agent configuration
    use_sde = agent_config.get("use_sde", False)
    
    # Initialize PPO with appropriate parameters optimized for sample efficiency
    model = PPO(
        policy=agent_config.get("policy_type", "MultiInputPolicy"),
        env=env,
        learning_rate=agent_config.get("learning_rate", 3e-4),
        n_steps=agent_config.get("n_steps", 1024),  # Smaller buffer for faster updates
        batch_size=agent_config.get("batch_size", 128),  # Larger batch for better gradients
        n_epochs=agent_config.get("n_epochs", 8),  # Slightly fewer epochs
        gamma=agent_config.get("gamma", 0.99),
        gae_lambda=agent_config.get("gae_lambda", 0.95),
        clip_range=agent_config.get("clip_range", 0.2),
        clip_range_vf=agent_config.get("clip_range_vf", None),
        normalize_advantage=agent_config.get("normalize_advantage", True),
        ent_coef=agent_config.get("ent_coef", 0.005),  # Lower entropy for more exploitation
        vf_coef=agent_config.get("vf_coef", 0.75),  # Higher value loss weight
        max_grad_norm=agent_config.get("max_grad_norm", 0.5),
        use_sde=use_sde,
        sde_sample_freq=agent_config.get("sde_sample_freq", -1) if use_sde else None,
        target_kl=agent_config.get("target_kl", 0.015),  # Target KL for early stopping
        tensorboard_log=tb_log_dir,
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