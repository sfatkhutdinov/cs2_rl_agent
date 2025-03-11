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
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback, BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv, VecTransposeImage
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.evaluation import evaluate_policy
import gym
import gymnasium as gym

from src.environment.cs2_env import CS2Environment
from src.environment.autonomous_env import AutonomousCS2Environment
from src.environment.vision_guided_env import VisionGuidedCS2Environment
from src.interface.auto_vision_interface import AutoVisionInterface
from src.interface.ollama_vision_interface import OllamaVisionInterface


# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("logs/vision_guided_training.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("VisionGuidedTraining")


def make_env(rank, config, seed=0):
    """
    Create an environment creation function for stable-baselines3.
    
    Args:
        rank: Environment rank for parallel training
        config: Configuration dictionary
        seed: Random seed for reproducibility
        
    Returns:
        Function that creates the environment
    """
    def _init():
        # Extract interface type for logging
        interface_type = config["environment"]["interface"]["type"]
        
        # Build environment config
        env_config = {
            "interface": config["environment"]["interface"],
            "ollama": config["environment"].get("ollama", {}),
            "environment": {
                "observation_space": config["environment"]["observation_space"],
                "action_space": config["environment"]["action_space"],
                "reward_function": config["environment"]["reward_function"],
                "max_episode_steps": config["environment"]["max_episode_steps"],
                "metrics_update_freq": config["environment"]["metrics_update_freq"],
                "pause_on_menu": config["environment"].get("pause_on_menu", False),
                "action_repeat": config["environment"].get("action_repeat", 1),
                "vision_guidance": config["environment"].get("vision_guidance", {})
            },
            "use_fallback_mode": True  # Enable fallback mode
        }
        
        # Create base environment
        base_env = CS2Environment(config=env_config)
        base_env.logger = logging.getLogger(f"CS2Env_{rank}")
        
        # Wrap in VisionGuidedCS2Environment
        autonomous_config = config.get("autonomous", {})
        
        # Create the vision-guided environment
        env = VisionGuidedCS2Environment(
            base_env=base_env,
            exploration_frequency=autonomous_config.get("exploration_frequency", 0.3),
            random_action_frequency=autonomous_config.get("random_action_frequency", 0.2),
            menu_exploration_buffer_size=autonomous_config.get("menu_exploration_buffer_size", 50),
            logger=logging.getLogger(f"VisionGuidedEnv_{rank}")
        )
        
        # Create a Monitor to track episode statistics
        log_dir = f"logs/env_{rank}"
        os.makedirs(log_dir, exist_ok=True)
        env = Monitor(env, log_dir)
        
        return env
    
    return _init


def train(config_path):
    """
    Train a vision-guided agent using Proximal Policy Optimization.
    
    Args:
        config_path: Path to the YAML configuration file
    """
    # Load configuration
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
        
    # Set random seeds for reproducibility
    seed = config.get("seed", 0)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    # Create directories
    experiment_name = config.get("experiment_name", f"vision_guided_agent_{int(time.time())}")
    log_dir = f"logs/{experiment_name}"
    model_dir = f"models/{experiment_name}"
    tensorboard_dir = f"tensorboard/{experiment_name}"
    
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(tensorboard_dir, exist_ok=True)
    
    # Save config for reproducibility
    with open(f"{log_dir}/config.yaml", 'w') as f:
        yaml.dump(config, f)
    
    # Create vectorized environment
    n_envs = config["training"]["n_envs"]
    
    if n_envs > 1:
        logger.warning(f"Using {n_envs} environments, but this may cause issues with game control")
        
    env = DummyVecEnv([make_env(i, config, seed) for i in range(n_envs)])
    
    # If using images in observation, wrap with VecTransposeImage
    if config["environment"]["observation_space"].get("include_visual", True):
        env = VecTransposeImage(env)
    
    # Create the PPO agent
    logger.info("Creating PPO agent with vision-guided environment")
    
    policy_kwargs = {}
    if "policy_kwargs" in config["agent"]:
        policy_kwargs = config["agent"]["policy_kwargs"]
        
        # Convert string activation function to torch function
        if "activation_fn" in policy_kwargs and isinstance(policy_kwargs["activation_fn"], str):
            activation_map = {
                "relu": torch.nn.ReLU,
                "tanh": torch.nn.Tanh,
                "sigmoid": torch.nn.Sigmoid,
                "leaky_relu": torch.nn.LeakyReLU,
                "elu": torch.nn.ELU,
            }
            policy_kwargs["activation_fn"] = activation_map.get(
                policy_kwargs["activation_fn"].lower(), torch.nn.ReLU
            )
    
    # Define a callback for monitoring memory usage
    class MemoryMonitorCallback(BaseCallback):
        """
        Callback for monitoring memory usage and managing disk space.
        
        Attributes:
            memory_limit_gb: Maximum allowed memory usage in GB
            disk_limit_gb: Maximum allowed disk usage for model checkpoints in GB
            check_interval: How often to check memory usage (in timesteps)
            model_dir: Directory where model checkpoints are stored
        """
        
        def __init__(self, memory_limit_gb, disk_limit_gb, check_interval, model_dir, verbose=0):
            super().__init__(verbose)
            self.memory_limit_gb = memory_limit_gb
            self.disk_limit_gb = disk_limit_gb
            self.check_interval = check_interval
            self.model_dir = model_dir
            self.last_check_time = time.time()
        
        def _init_callback(self):
            # Called when the callback is initialized
            pass
        
        def _on_step(self):
            # Only check periodically to avoid performance impact
            if self.n_calls % self.check_interval == 0:
                try:
                    # Check memory usage (platform-specific)
                    if os.name == 'posix':  # Linux/Unix/MacOS
                        import psutil
                        process = psutil.Process(os.getpid())
                        mem_info = process.memory_info()
                        memory_gb = mem_info.rss / (1024 ** 3)  # Convert to GB
                    else:  # Windows
                        import psutil
                        process = psutil.Process(os.getpid())
                        memory_gb = process.memory_info().rss / (1024 ** 3)  # Convert to GB
                    
                    # Log memory usage
                    self.logger.record("system/memory_usage_gb", memory_gb)
                    
                    # Check if memory limit is exceeded
                    if memory_gb > self.memory_limit_gb:
                        self.logger.record("system/memory_limit_exceeded", 1.0)
                        logger.warning(f"Memory usage ({memory_gb:.2f} GB) exceeded limit ({self.memory_limit_gb} GB)")
                        
                        # Try to free some memory by performing garbage collection
                        import gc
                        gc.collect()
                        
                        # If it's still too high after GC, we could take more drastic actions here
                        # such as saving the model and terminating training
                    
                    # Check disk usage for model directory
                    dir_size_gb = self._get_dir_size(self.model_dir) / (1024 ** 3)
                    self.logger.record("system/model_disk_usage_gb", dir_size_gb)
                    
                    if dir_size_gb > self.disk_limit_gb:
                        logger.warning(f"Disk usage for models ({dir_size_gb:.2f} GB) exceeded limit ({self.disk_limit_gb} GB)")
                        self._cleanup_old_checkpoints()
                    
                    # Log some Vision-Guided specific metrics if available
                    try:
                        # These are available if using our VisionGuidedCS2Environment
                        for env_idx in range(len(self.training_env.envs)):
                            env = self.training_env.envs[env_idx].unwrapped
                            if hasattr(env, 'successful_vision_guided_actions') and hasattr(env, 'vision_guided_actions_taken'):
                                if env.vision_guided_actions_taken > 0:
                                    success_rate = env.successful_vision_guided_actions / max(1, env.vision_guided_actions_taken)
                                    self.logger.record(f"environment/vision_guided_success_rate_{env_idx}", success_rate)
                    except Exception as e:
                        logger.error(f"Error logging vision-guided metrics: {str(e)}")
                    
                    # Check how long this callback took
                    current_time = time.time()
                    callback_time = current_time - self.last_check_time
                    self.last_check_time = current_time
                    self.logger.record("time/memory_callback_seconds", callback_time)
                    
                except Exception as e:
                    logger.error(f"Error in memory monitoring callback: {str(e)}")
                
                # Always continue training
                return True
            
            return True
        
        def _get_dir_size(self, path):
            """Get the total size of files in a directory in bytes"""
            total_size = 0
            for dirpath, _, filenames in os.walk(path):
                for f in filenames:
                    fp = os.path.join(dirpath, f)
                    if not os.path.islink(fp):  # Skip if it's a symbolic link
                        total_size += os.path.getsize(fp)
            return total_size
        
        def _cleanup_old_checkpoints(self):
            """Remove older checkpoint files to free disk space, keeping the most recent ones"""
            try:
                checkpoint_files = []
                for f in os.listdir(self.model_dir):
                    if f.endswith('.zip') and 'rl_model_' in f:
                        full_path = os.path.join(self.model_dir, f)
                        checkpoint_files.append((full_path, os.path.getmtime(full_path)))
                
                # Sort by modification time (newest first)
                checkpoint_files.sort(key=lambda x: x[1], reverse=True)
                
                # Keep the 5 most recent checkpoints, remove the rest
                for path, _ in checkpoint_files[5:]:
                    logger.info(f"Removing old checkpoint: {path}")
                    os.remove(path)
            except Exception as e:
                logger.error(f"Error cleaning up old checkpoints: {str(e)}")
    
    # Create callbacks
    checkpoint_callback = CheckpointCallback(
        save_freq=config["training"]["checkpoint_freq"],
        save_path=model_dir,
        name_prefix="rl_model",
        save_replay_buffer=True,
        save_vecnormalize=True,
    )
    
    # Memory monitoring callback
    memory_config = config.get("memory_monitor", {})
    memory_monitor_callback = None
    if memory_config.get("enabled", False):
        memory_monitor_callback = MemoryMonitorCallback(
            memory_limit_gb=memory_config.get("memory_limit_gb", 12),
            disk_limit_gb=memory_config.get("disk_limit_gb", 50),
            check_interval=memory_config.get("check_interval", 10000),
            model_dir=model_dir,
        )
    
    # Put all callbacks together
    callbacks = [checkpoint_callback]
    if memory_monitor_callback:
        callbacks.append(memory_monitor_callback)
    
    # Optionally add evaluation callback
    if config["training"].get("evaluate_during_training", False):
        # Create a separate environment for evaluation
        eval_env = DummyVecEnv([make_env(0, config, seed+1000)])
        if config["environment"]["observation_space"].get("include_visual", True):
            eval_env = VecTransposeImage(eval_env)
        
        eval_callback = EvalCallback(
            eval_env=eval_env,
            best_model_save_path=f"{model_dir}/best_model",
            log_path=f"{log_dir}/eval",
            eval_freq=config["training"]["eval_freq"],
            deterministic=True,
            render=False,
            n_eval_episodes=config["training"]["eval_episodes"],
        )
        callbacks.append(eval_callback)
    
    # Create the agent
    model = PPO(
        policy=config["agent"]["policy_type"],
        env=env,
        learning_rate=config["agent"]["learning_rate"],
        n_steps=config["agent"]["n_steps"],
        batch_size=config["agent"]["batch_size"],
        n_epochs=config["agent"]["n_epochs"],
        gamma=config["agent"]["gamma"],
        gae_lambda=config["agent"]["gae_lambda"],
        ent_coef=config["agent"]["ent_coef"],
        vf_coef=config["agent"]["vf_coef"],
        max_grad_norm=config["agent"]["max_grad_norm"],
        use_sde=False,
        policy_kwargs=policy_kwargs,
        tensorboard_log=tensorboard_dir,
        verbose=1,
    )
    
    # Train the agent
    logger.info(f"Starting training for {config['training']['total_timesteps']} timesteps")
    try:
        model.learn(
            total_timesteps=config["training"]["total_timesteps"],
            callback=callbacks,
            tb_log_name=experiment_name,
        )
        
        # Save the final model
        final_model_path = f"{model_dir}/final_model.zip"
        model.save(final_model_path)
        logger.info(f"Training complete. Final model saved to {final_model_path}")
        
    except Exception as e:
        logger.error(f"Error during training: {str(e)}", exc_info=True)
        # Try to save model even if training failed
        try:
            emergency_model_path = f"{model_dir}/emergency_save_{int(time.time())}.zip"
            model.save(emergency_model_path)
            logger.info(f"Emergency model saved to {emergency_model_path}")
        except Exception as e2:
            logger.error(f"Failed to save emergency model: {str(e2)}")
    
    # Clean up
    env.close()
    if 'eval_env' in locals():
        eval_env.close()


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="Train a vision-guided agent for Cities: Skylines 2")
    parser.add_argument("--config", type=str, default="config/vision_guided_config.yaml",
                      help="Path to the configuration file")
    args = parser.parse_args()
    
    logger.info(f"Starting vision-guided training with config: {args.config}")
    train(args.config)


if __name__ == "__main__":
    main() 