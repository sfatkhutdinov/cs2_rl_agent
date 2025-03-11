import os
import yaml
import argparse
import numpy as np
import time
from datetime import datetime
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.callbacks import CheckpointCallback, CallbackList

from src.callbacks.metrics_callback import MetricsCallback
from src.environment.tutorial_guided_env import TutorialGuidedCS2Environment
from src.utils.observation_wrapper import ObservationWrapper
from src.utils.feature_extractor import CombinedExtractor


def make_env(config, seed=0, use_fallback_mode=True):
    """
    Create a TutorialGuidedCS2Environment with the given config.
    Wraps with ObservationWrapper if needed.
    """
    def _init():
        # Create environment with fallback mode enabled for robustness
        env = TutorialGuidedCS2Environment(
            base_env_config=config["environment"],
            observation_config=config["observation"],
            vision_config=config.get("vision", {}),
            tutorial_frequency=config.get("tutorial_frequency", 0.7),
            tutorial_timeout=config.get("tutorial_timeout", 300),
            tutorial_reward_multiplier=config.get("tutorial_reward_multiplier", 2.0),
            use_fallback_mode=use_fallback_mode
        )
        
        # Wrap environment with ObservationWrapper for vector observations
        env = ObservationWrapper(env)
        
        env.seed(seed)
        return env
        
    set_random_seed(seed)
    return _init


def parse_args():
    parser = argparse.ArgumentParser(description='Train a tutorial-guided agent for Cities: Skylines 2')
    parser.add_argument('--config', type=str, default='config/tutorial_guided_config.yaml',
                        help='Path to the YAML configuration file')
    parser.add_argument('--models-dir', type=str, default='models',
                        help='Directory to save models')
    parser.add_argument('--logs-dir', type=str, default='logs',
                        help='Directory to save logs')
    parser.add_argument('--n-envs', type=int, default=1,
                        help='Number of environments to run in parallel')
    parser.add_argument('--load-model', type=str, default=None,
                        help='Path to a model to load and continue training')
    parser.add_argument('--total-timesteps', type=int, default=1000000,
                        help='Total number of timesteps to train')
    return parser.parse_args()


def main():
    # Parse arguments
    args = parse_args()
    
    # Create timestamp for this run
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"tutorial_guided_{timestamp}"
    
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Create directories
    models_dir = os.path.join(args.models_dir, run_name)
    logs_dir = os.path.join(args.logs_dir, run_name)
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(logs_dir, exist_ok=True)
    
    # Save config file to the model directory for reproducibility
    with open(os.path.join(models_dir, 'config.yaml'), 'w') as f:
        yaml.dump(config, f)
    
    # Set up training parameters
    total_timesteps = args.total_timesteps or config.get("training", {}).get("total_timesteps", 1000000)
    n_envs = args.n_envs or config.get("training", {}).get("n_envs", 1)
    
    # Create vectorized environment
    env = SubprocVecEnv([make_env(config, i) for i in range(n_envs)])
    env = VecMonitor(env, os.path.join(logs_dir, "monitor.csv"))
    
    # Set up callbacks
    checkpoint_callback = CheckpointCallback(
        save_freq=max(10000 // n_envs, 1),
        save_path=models_dir,
        name_prefix="tutorial_model",
        save_replay_buffer=True,
        save_vecnormalize=True,
    )
    
    metrics_callback = MetricsCallback(
        log_dir=logs_dir,
        log_freq=100,
    )
    
    callback = CallbackList([checkpoint_callback, metrics_callback])
    
    # Create policy configuration
    policy_kwargs = {
        "features_extractor_class": CombinedExtractor,
        "features_extractor_kwargs": {
            "cnn_output_dim": config.get("model", {}).get("cnn_output_dim", 256),
            "mlp_extractor_hidden_sizes": config.get("model", {}).get("mlp_extractor_hidden_sizes", [256, 256])
        },
        "net_arch": config.get("model", {}).get("net_arch", [dict(pi=[256, 256], vf=[256, 256])])
    }
    
    # Initialize or load the model
    if args.load_model:
        print(f"Loading model from {args.load_model}")
        model = PPO.load(
            args.load_model,
            env=env,
            tensorboard_log=logs_dir,
            device=config.get("training", {}).get("device", "auto")
        )
    else:
        print("Initializing new model")
        model = PPO(
            "MultiInputPolicy",
            env,
            verbose=1,
            tensorboard_log=logs_dir,
            learning_rate=config.get("training", {}).get("learning_rate", 3e-4),
            n_steps=config.get("training", {}).get("n_steps", 2048),
            batch_size=config.get("training", {}).get("batch_size", 64),
            n_epochs=config.get("training", {}).get("n_epochs", 10),
            gamma=config.get("training", {}).get("gamma", 0.99),
            gae_lambda=config.get("training", {}).get("gae_lambda", 0.95),
            clip_range=config.get("training", {}).get("clip_range", 0.2),
            ent_coef=config.get("training", {}).get("ent_coef", 0.01),
            vf_coef=config.get("training", {}).get("vf_coef", 0.5),
            max_grad_norm=config.get("training", {}).get("max_grad_norm", 0.5),
            use_sde=config.get("training", {}).get("use_sde", False),
            sde_sample_freq=config.get("training", {}).get("sde_sample_freq", -1),
            policy_kwargs=policy_kwargs,
            device=config.get("training", {}).get("device", "auto")
        )
    
    print(f"Starting training for {total_timesteps} timesteps")
    try:
        model.learn(
            total_timesteps=total_timesteps,
            callback=callback,
            tb_log_name=run_name
        )
        
        # Save the final model
        final_model_path = os.path.join(models_dir, "final_model")
        model.save(final_model_path)
        print(f"Training completed. Final model saved to {final_model_path}")
        
    except KeyboardInterrupt:
        print("Training interrupted by user")
        # Save the model on interrupt
        interrupted_model_path = os.path.join(models_dir, "interrupted_model")
        model.save(interrupted_model_path)
        print(f"Interrupted model saved to {interrupted_model_path}")
    
    finally:
        # Close environments
        env.close()


if __name__ == "__main__":
    main() 