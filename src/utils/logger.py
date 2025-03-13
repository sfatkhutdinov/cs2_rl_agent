import os
import logging
import time
from typing import Dict, Any, Optional
import json
from datetime import datetime
# Remove TensorFlow/TensorBoard imports
# import tensorboard
# from tensorboard.plugins.hparams import api as hp
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd


class Logger:
    """
    Logger class for tracking experiment metrics and visualizing results.
    """
    
    def __init__(self, config: Dict[str, Any], experiment_name: Optional[str] = None):
        """
        Initialize the logger.
        
        Args:
            config: Configuration dictionary
            experiment_name: Name of the experiment (if None, will generate a timestamp-based name)
        """
        self.config = config
        
        # Set up experiment name and paths
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.experiment_name = experiment_name or f"experiment_{timestamp}"
        
        log_dir = os.path.normpath(os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
            config["paths"]["logs"]
        ))
        self.experiment_dir = os.path.join(log_dir, self.experiment_name)
        # Keep reference but don't use TensorBoard
        self.tensorboard_dir = os.path.join(self.experiment_dir, "tensorboard")
        
        # Create directories
        os.makedirs(self.experiment_dir, exist_ok=True)
        os.makedirs(self.tensorboard_dir, exist_ok=True)
        
        # Set up Python logger
        self.logger = logging.getLogger(self.experiment_name)
        self.logger.setLevel(logging.INFO)
        
        # File handler
        log_file = os.path.join(self.experiment_dir, "experiment.log")
        file_handler = logging.FileHandler(log_file)
        file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(file_formatter)
        self.logger.addHandler(file_handler)
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_formatter = logging.Formatter('%(levelname)s - %(message)s')
        console_handler.setFormatter(console_formatter)
        self.logger.addHandler(console_handler)
        
        # Set up metrics tracking
        self.metrics = {
            "timesteps": [],
            "episodes": [],
            "rewards": [],
            "population": [],
            "happiness": [],
            "budget_balance": [],
            "traffic_flow": []
        }
        
        # Save configuration
        self.save_config()
        
        self.logger.info(f"Logger initialized. Experiment: {self.experiment_name}")
        self.logger.info(f"Logs will be saved to: {self.experiment_dir}")
    
    def log_info(self, message: str) -> None:
        """Log informational message."""
        self.logger.info(message)
    
    def log_warning(self, message: str) -> None:
        """Log warning message."""
        self.logger.warning(message)
    
    def log_error(self, message: str) -> None:
        """Log error message."""
        self.logger.error(message)
    
    def log_metrics(self, 
                   timestep: int, 
                   episode: int, 
                   reward: float, 
                   metrics: Dict[str, float]) -> None:
        """
        Log training metrics.
        
        Args:
            timestep: Current timestep
            episode: Current episode
            reward: Episode reward
            metrics: Dictionary of metrics to log
        """
        self.metrics["timesteps"].append(timestep)
        self.metrics["episodes"].append(episode)
        self.metrics["rewards"].append(reward)
        
        # Log specific game metrics if available
        if "population" in metrics:
            self.metrics["population"].append(metrics["population"])
        if "happiness" in metrics:
            self.metrics["happiness"].append(metrics["happiness"])
        if "budget_balance" in metrics:
            self.metrics["budget_balance"].append(metrics["budget_balance"])
        if "traffic_flow" in metrics:
            self.metrics["traffic_flow"].append(metrics["traffic_flow"])
        
        # Log to console
        self.logger.info(f"Episode {episode} | Timestep {timestep} | Reward {reward:.2f}")
        
        # Save metrics periodically
        if len(self.metrics["timesteps"]) % 10 == 0:
            self.save_metrics()
    
    def save_metrics(self) -> None:
        """Save metrics to CSV and create visualizations."""
        metrics_file = os.path.join(self.experiment_dir, "metrics.csv")
        pd.DataFrame(self.metrics).to_csv(metrics_file, index=False)
        
        # Create visualizations
        self._plot_rewards()
        self._plot_game_metrics()
    
    def _plot_rewards(self) -> None:
        """Plot episode rewards."""
        plt.figure(figsize=(10, 6))
        plt.plot(self.metrics["episodes"], self.metrics["rewards"])
        plt.title("Episode Rewards")
        plt.xlabel("Episode")
        plt.ylabel("Reward")
        plt.grid(True)
        plt.savefig(os.path.join(self.experiment_dir, "rewards.png"))
        plt.close()
    
    def _plot_game_metrics(self) -> None:
        """Plot game-specific metrics."""
        if len(self.metrics["population"]) == 0:
            return
        
        plt.figure(figsize=(12, 8))
        
        plt.subplot(2, 2, 1)
        plt.plot(self.metrics["episodes"], self.metrics["population"])
        plt.title("Population Growth")
        plt.xlabel("Episode")
        plt.ylabel("Population")
        plt.grid(True)
        
        if len(self.metrics["happiness"]) > 0:
            plt.subplot(2, 2, 2)
            plt.plot(self.metrics["episodes"], self.metrics["happiness"])
            plt.title("Citizen Happiness")
            plt.xlabel("Episode")
            plt.ylabel("Happiness")
            plt.grid(True)
        
        if len(self.metrics["budget_balance"]) > 0:
            plt.subplot(2, 2, 3)
            plt.plot(self.metrics["episodes"], self.metrics["budget_balance"])
            plt.title("Budget Balance")
            plt.xlabel("Episode")
            plt.ylabel("Balance")
            plt.grid(True)
        
        if len(self.metrics["traffic_flow"]) > 0:
            plt.subplot(2, 2, 4)
            plt.plot(self.metrics["episodes"], self.metrics["traffic_flow"])
            plt.title("Traffic Flow")
            plt.xlabel("Episode")
            plt.ylabel("Flow Efficiency")
            plt.grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.experiment_dir, "game_metrics.png"))
        plt.close()
    
    def save_config(self) -> None:
        """Save experiment configuration."""
        config_file = os.path.join(self.experiment_dir, "config.yaml")
        with open(config_file, 'w') as f:
            json.dump(self.config, f, indent=4)
    
    def close(self) -> None:
        """Close the logger and save final metrics."""
        self.save_metrics()
        self.logger.info(f"Experiment {self.experiment_name} completed.") 