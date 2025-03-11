import os
import json
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional
import time
from datetime import datetime
from stable_baselines3.common.callbacks import BaseCallback


class MetricsCallback(BaseCallback):
    """
    Callback for saving metrics during training.
    
    This callback collects various metrics during training, including:
    - Episode rewards
    - Episode lengths
    - Learning rate
    - Explained variance
    - Policy entropy
    - Value loss
    - Policy loss
    
    It saves these metrics to CSV files in the specified log directory.
    """
    
    def __init__(self, log_dir: str, log_freq: int = 100, verbose: int = 0):
        """
        Initialize the callback.
        
        Args:
            log_dir: Directory to save metrics
            log_freq: How often to record metrics (in timesteps)
            verbose: Verbosity level
        """
        super().__init__(verbose)
        self.log_dir = log_dir
        self.log_freq = log_freq
        self.metrics_file = os.path.join(log_dir, "training_metrics.csv")
        self.ep_rewards_file = os.path.join(log_dir, "episode_rewards.csv")
        self.ep_lengths_file = os.path.join(log_dir, "episode_lengths.csv")
        
        # Metrics to track
        self.metrics = {
            "timesteps": [],
            "time_elapsed": [],
            "fps": [],
            "episodes": [],
            "mean_reward": [],
            "median_reward": [],
            "min_reward": [],
            "max_reward": [],
            "std_reward": [],
            "mean_episode_length": [],
            "learning_rate": [],
            "explained_variance": [],
            "entropy": [],
            "value_loss": [],
            "policy_loss": [],
        }
        
        # For FPS calculation
        self.start_time = time.time()
        self.last_timesteps = 0
        
    def _on_training_start(self) -> None:
        """
        Called at the start of training.
        """
        # Create log directory if it doesn't exist
        os.makedirs(self.log_dir, exist_ok=True)
        
        # Initialize metrics files with headers
        pd.DataFrame(columns=list(self.metrics.keys())).to_csv(
            self.metrics_file, index=False
        )
        
        pd.DataFrame(columns=["timestep", "episode", "reward"]).to_csv(
            self.ep_rewards_file, index=False
        )
        
        pd.DataFrame(columns=["timestep", "episode", "length"]).to_csv(
            self.ep_lengths_file, index=False
        )
        
        # Record initial time
        self.start_time = time.time()
        
    def _on_step(self) -> bool:
        """
        Called at each step of training.
        
        Returns:
            True to continue training, False to stop
        """
        # Only log on log_freq steps or at the end of training
        if self.n_calls % self.log_freq != 0:
            return True
            
        # Calculate metrics
        timesteps = self.num_timesteps
        time_elapsed = time.time() - self.start_time
        fps = (timesteps - self.last_timesteps) / (time_elapsed / (self.n_calls / self.log_freq))
        self.last_timesteps = timesteps
        
        # Get episode info
        episode_rewards = []
        episode_lengths = []
        episodes_done = 0
        
        for i in range(len(self.training_env.envs)):
            if self.training_env.episode_rewards[i]:
                for j in range(len(self.training_env.episode_rewards[i])):
                    episode_rewards.append(self.training_env.episode_rewards[i][j])
                    episode_lengths.append(self.training_env.episode_lengths[i][j])
                    episodes_done += 1
                    
                    # Save individual episode data
                    episode_data = pd.DataFrame({
                        "timestep": [timesteps],
                        "episode": [episodes_done],
                        "reward": [self.training_env.episode_rewards[i][j]]
                    })
                    episode_data.to_csv(self.ep_rewards_file, mode='a', header=False, index=False)
                    
                    episode_length_data = pd.DataFrame({
                        "timestep": [timesteps],
                        "episode": [episodes_done],
                        "length": [self.training_env.episode_lengths[i][j]]
                    })
                    episode_length_data.to_csv(self.ep_lengths_file, mode='a', header=False, index=False)
                
                # Clear the recorded episodes
                self.training_env.episode_rewards[i] = []
                self.training_env.episode_lengths[i] = []
        
        # Get other metrics from model
        explained_variance = float(np.mean(self.model.logger.name_to_value.get("train/explained_variance", [0])))
        entropy = float(np.mean(self.model.logger.name_to_value.get("train/entropy", [0])))
        value_loss = float(np.mean(self.model.logger.name_to_value.get("train/value_loss", [0])))
        policy_loss = float(np.mean(self.model.logger.name_to_value.get("train/policy_loss", [0])))
        learning_rate = self.model.learning_rate
        
        # Log metrics
        if episode_rewards:
            mean_reward = float(np.mean(episode_rewards))
            median_reward = float(np.median(episode_rewards))
            min_reward = float(np.min(episode_rewards))
            max_reward = float(np.max(episode_rewards))
            std_reward = float(np.std(episode_rewards))
            mean_episode_length = float(np.mean(episode_lengths))
        else:
            mean_reward = 0.0
            median_reward = 0.0
            min_reward = 0.0
            max_reward = 0.0
            std_reward = 0.0
            mean_episode_length = 0.0
        
        # Store metrics
        self.metrics["timesteps"].append(timesteps)
        self.metrics["time_elapsed"].append(time_elapsed)
        self.metrics["fps"].append(fps)
        self.metrics["episodes"].append(len(episode_rewards))
        self.metrics["mean_reward"].append(mean_reward)
        self.metrics["median_reward"].append(median_reward)
        self.metrics["min_reward"].append(min_reward)
        self.metrics["max_reward"].append(max_reward)
        self.metrics["std_reward"].append(std_reward)
        self.metrics["mean_episode_length"].append(mean_episode_length)
        self.metrics["learning_rate"].append(learning_rate)
        self.metrics["explained_variance"].append(explained_variance)
        self.metrics["entropy"].append(entropy)
        self.metrics["value_loss"].append(value_loss)
        self.metrics["policy_loss"].append(policy_loss)
        
        # Save metrics to CSV
        metrics_df = pd.DataFrame({k: [v[-1]] for k, v in self.metrics.items()})
        metrics_df.to_csv(self.metrics_file, mode='a', header=False, index=False)
        
        # Print metrics to console if verbose
        if self.verbose > 0:
            print(f"Steps: {timesteps}, FPS: {fps:.1f}, Episodes: {episodes_done}")
            if episode_rewards:
                print(f"Mean reward: {mean_reward:.2f}, Max reward: {max_reward:.2f}")
                print(f"Mean episode length: {mean_episode_length:.1f}")
            
        return True
        
    def _on_training_end(self) -> None:
        """
        Called at the end of training.
        """
        # One final update
        self._on_step()
        
        # Save all metrics as a single CSV
        pd.DataFrame(self.metrics).to_csv(
            os.path.join(self.log_dir, "all_metrics.csv"), index=False
        )
        
        # Save summary statistics
        summary = {
            "total_timesteps": self.num_timesteps,
            "total_time_elapsed": time.time() - self.start_time,
            "mean_fps": self.num_timesteps / (time.time() - self.start_time),
            "final_learning_rate": self.model.learning_rate,
        }
        
        with open(os.path.join(self.log_dir, "training_summary.json"), "w") as f:
            json.dump(summary, f, indent=4) 