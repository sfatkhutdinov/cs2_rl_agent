from typing import Dict, Any, List
import numpy as np
from stable_baselines3.common.callbacks import BaseCallback
import os
import json

class TensorboardCallback(BaseCallback):
    """
    Custom callback for logging additional metrics to tensorboard.
    """
    
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.episode_rewards = []
        self.episode_lengths = []
        self.discovery_stats = {}
    
    def _on_step(self) -> bool:
        """
        Log metrics on each step.
        
        Returns:
            Whether to continue training
        """
        # Check if there are discovery stats in the infos
        if len(self.model.ep_info_buffer) > 0 and self.model.ep_info_buffer[-1].get("discovery") is not None:
            discovery_info = self.model.ep_info_buffer[-1]["discovery"]
            
            # Log discovery statistics to tensorboard
            if "discovered_ui_elements" in discovery_info:
                self.logger.record("discovery/ui_elements_discovered", discovery_info["discovered_ui_elements"])
            
            if "discovered_actions" in discovery_info:
                self.logger.record("discovery/actions_discovered", discovery_info["discovered_actions"])
            
            if "completed_tutorials" in discovery_info:
                self.logger.record("discovery/tutorials_completed", discovery_info["completed_tutorials"])
            
            # Log discovery stats if available
            if "discovery_stats" in discovery_info:
                stats = discovery_info["discovery_stats"]
                for key, value in stats.items():
                    self.logger.record(f"discovery/{key}", value)
                
                # Store for later analysis
                self.discovery_stats = stats
        
        # Continue training
        return True
    
    def _on_rollout_end(self) -> None:
        """
        Log episode statistics at the end of the rollout.
        """
        # Calculate average reward and episode length
        if len(self.model.ep_info_buffer) > 0:
            ep_reward_mean = np.mean([ep_info["r"] for ep_info in self.model.ep_info_buffer])
            ep_len_mean = np.mean([ep_info["l"] for ep_info in self.model.ep_info_buffer])
            
            # Log to tensorboard
            self.logger.record("rollout/ep_rew_mean", ep_reward_mean)
            self.logger.record("rollout/ep_len_mean", ep_len_mean)
            
            # Store for later analysis
            self.episode_rewards.append(ep_reward_mean)
            self.episode_lengths.append(ep_len_mean)
        
        # Make sure we flush everything to disk
        self.logger.dump(self.num_timesteps)
    
    def save_stats(self, save_path: str) -> None:
        """
        Save episode statistics to file.
        
        Args:
            save_path: Directory to save statistics
        """
        os.makedirs(save_path, exist_ok=True)
        
        # Save episode rewards and lengths
        with open(os.path.join(save_path, "episode_rewards.json"), "w") as f:
            json.dump(self.episode_rewards, f)
        
        with open(os.path.join(save_path, "episode_lengths.json"), "w") as f:
            json.dump(self.episode_lengths, f)
        
        # Save discovery stats
        with open(os.path.join(save_path, "discovery_stats.json"), "w") as f:
            json.dump(self.discovery_stats, f) 