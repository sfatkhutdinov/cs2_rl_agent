#!/usr/bin/env python
"""
Adaptive Agent Training Script for CS2 RL Agent.

This script trains an agent that can dynamically switch between different 
training modes (discovery, tutorial, vision, autonomous) based on performance
and game feedback.
"""

import os
import sys
import time
import json
import yaml
import argparse
import logging
from pathlib import Path
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Any, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(f"logs/adaptive_training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    ]
)
logger = logging.getLogger("AdaptiveTraining")

# Import the adaptive agent class
from src.agent.adaptive_agent import AdaptiveAgent, TrainingMode

def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load configuration from a YAML file.
    
    Args:
        config_path: Path to the config file
        
    Returns:
        Configuration dictionary
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def setup_directories() -> None:
    """
    Set up necessary directories for training.
    """
    os.makedirs("models", exist_ok=True)
    os.makedirs("logs", exist_ok=True)
    os.makedirs("logs/tensorboard", exist_ok=True)
    os.makedirs("logs/plots", exist_ok=True)

def plot_mode_history(adaptive_agent, save_path: str) -> None:
    """
    Plot the agent's mode history over time.
    
    Args:
        adaptive_agent: The trained adaptive agent
        save_path: Path to save the plot
    """
    if not adaptive_agent.mode_history:
        logger.warning("No mode history to plot")
        return
    
    plt.figure(figsize=(12, 6))
    
    # Extract timestamps and modes
    timestamps = [ts for ts, _, _ in adaptive_agent.mode_history]
    start_time = timestamps[0]
    rel_times = [(ts - start_time) / 60 for ts in timestamps]  # Minutes
    
    # Get unique modes in the order they first appear
    mode_values = []
    seen_modes = set()
    for _, old_mode, _ in adaptive_agent.mode_history:
        if old_mode.value not in seen_modes:
            seen_modes.add(old_mode.value)
            mode_values.append(old_mode.value)
    
    # Add the final mode if not already included
    final_mode = adaptive_agent.current_mode.value
    if final_mode not in seen_modes:
        mode_values.append(final_mode)
    
    # Create mapping from mode names to numbers for plotting
    mode_to_num = {mode: i for i, mode in enumerate(mode_values)}
    mode_sequence = [mode_to_num[m.value] for _, m, _ in adaptive_agent.mode_history]
    
    # Add the current mode at the end
    rel_times.append((time.time() - start_time) / 60)
    mode_sequence.append(mode_to_num[adaptive_agent.current_mode.value])
    
    # Plot the mode changes
    plt.step(rel_times, mode_sequence, where='post', linewidth=2)
    
    # Add markers for mode switches
    plt.scatter(rel_times[:-1], mode_sequence[:-1], marker='o', s=100, c='red')
    
    # Add annotations for switch reasons
    for i, (ts, reason) in enumerate(adaptive_agent.mode_switch_reasons):
        rel_time = (ts - start_time) / 60
        mode_num = mode_sequence[i]
        plt.annotate(
            reason,
            xy=(rel_time, mode_num),
            xytext=(10, 10),
            textcoords="offset points",
            fontsize=8,
            arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=.2")
        )
    
    # Set up the plot
    plt.yticks(range(len(mode_values)), mode_values)
    plt.xlabel('Time (minutes)')
    plt.ylabel('Training Mode')
    plt.title('Agent Mode Switching Over Time')
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Save the plot
    plt.tight_layout()
    plt.savefig(save_path)
    logger.info(f"Mode history plot saved to {save_path}")

def plot_metrics(adaptive_agent, save_dir: str) -> None:
    """
    Plot performance metrics for each mode.
    
    Args:
        adaptive_agent: The trained adaptive agent
        save_dir: Directory to save the plots
    """
    for mode in TrainingMode:
        history = adaptive_agent.metrics_history.get(mode, [])
        if not history:
            continue
            
        plt.figure(figsize=(12, 8))
        
        # Extract metrics
        metrics = {}
        for metric in history[0].keys():
            metrics[metric] = [h[metric] for h in history]
        
        # Plot each metric
        for metric, values in metrics.items():
            if metric == "stuck_episodes":
                continue  # Skip this as it's not as informative
                
            plt.plot(values, label=metric)
        
        plt.xlabel('Episodes')
        plt.ylabel('Value')
        plt.title(f'Training Metrics for {mode.value} Mode')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        
        # Save the plot
        plt.tight_layout()
        save_path = os.path.join(save_dir, f"metrics_{mode.value}.png")
        plt.savefig(save_path)
        logger.info(f"Metrics plot for {mode.value} saved to {save_path}")

def progress_callback(progress: Dict[str, Any]) -> None:
    """
    Callback function for training progress updates.
    
    Args:
        progress: Dictionary containing progress information
    """
    timesteps = progress["timesteps_used"]
    total = progress["total_timesteps"]
    percent = (timesteps / total) * 100
    mode = progress["current_mode"]
    episodes = progress["episode_count"]
    
    # Get metrics for current mode
    metrics = progress["mode_metrics"][TrainingMode(mode)]
    confidence = metrics["confidence"]
    reward = metrics["reward_avg"]
    
    logger.info(f"Progress: {timesteps}/{total} steps ({percent:.1f}%) - "
                f"Mode: {mode}, Episodes: {episodes}, "
                f"Confidence: {confidence:.2f}, Reward: {reward:.2f}")

def main():
    """Main function for adaptive agent training."""
    parser = argparse.ArgumentParser(description='Train the adaptive agent for CS2')
    parser.add_argument('--config', type=str, default='config/adaptive_config.yaml',
                        help='Path to configuration file')
    parser.add_argument('--timesteps', type=int, default=None,
                        help='Total timesteps to train (overrides config)')
    parser.add_argument('--focus', action='store_true',
                        help='Auto-focus the game window')
    parser.add_argument('--starting-mode', type=str, default=None,
                        choices=['discovery', 'tutorial', 'vision', 'autonomous'],
                        help='Starting training mode (overrides config)')
    parser.add_argument('--load', type=str, default=None,
                        help='Path to load a saved adaptive agent from')
    parser.add_argument('--save-dir', type=str, default='models/adaptive',
                        help='Directory to save the trained agent')
    
    args = parser.parse_args()
    
    # Set up directories
    setup_directories()
    
    # Load configuration
    config = load_config(args.config)
    
    # Override config with command line arguments
    if args.timesteps:
        config['training']['total_timesteps'] = args.timesteps
    if args.focus is not None:
        config['training']['auto_focus'] = args.focus
    if args.starting_mode:
        config['training']['starting_mode'] = args.starting_mode
    
    # Create save directory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    save_dir = os.path.join(args.save_dir, f"run_{timestamp}")
    os.makedirs(save_dir, exist_ok=True)
    
    # Save the configuration
    with open(os.path.join(save_dir, 'config.yaml'), 'w') as f:
        yaml.dump(config, f)
    
    # Initialize the adaptive agent
    logger.info("Initializing adaptive agent")
    adaptive_agent = AdaptiveAgent(
        config,
        discovery_config_path=config['mode_configs']['discovery'],
        vision_config_path=config['mode_configs']['vision'],
        autonomous_config_path=config['mode_configs']['autonomous'],
        tutorial_config_path=config['mode_configs']['tutorial']
    )
    
    # Set the starting mode if specified
    if config['training']['starting_mode'] != "discovery":
        starting_mode = TrainingMode(config['training']['starting_mode'])
        adaptive_agent.current_mode = starting_mode
        logger.info(f"Starting with {starting_mode.value} mode")
    
    # Load a saved agent if specified
    if args.load:
        logger.info(f"Loading adaptive agent from {args.load}")
        if adaptive_agent.load(args.load):
            logger.info("Agent loaded successfully")
        else:
            logger.error("Failed to load agent")
            return
    
    # Train the agent
    logger.info(f"Starting adaptive training for {config['training']['total_timesteps']} timesteps")
    start_time = time.time()
    
    training_results = adaptive_agent.train(
        config['training']['total_timesteps'],
        progress_callback=progress_callback
    )
    
    training_time = time.time() - start_time
    logger.info(f"Training completed in {training_time:.2f} seconds")
    
    # Save the trained agent
    logger.info(f"Saving adaptive agent to {save_dir}")
    adaptive_agent.save(save_dir)
    
    # Save training results
    with open(os.path.join(save_dir, 'training_results.json'), 'w') as f:
        # Convert non-serializable objects to strings
        serializable_results = {
            k: (str(v) if not isinstance(v, (int, float, str, list, dict, bool, type(None))) else v)
            for k, v in training_results.items()
        }
        json.dump(serializable_results, f, indent=2)
    
    # Generate plots
    plots_dir = os.path.join(save_dir, 'plots')
    os.makedirs(plots_dir, exist_ok=True)
    
    plot_mode_history(adaptive_agent, os.path.join(plots_dir, 'mode_history.png'))
    plot_metrics(adaptive_agent, plots_dir)
    
    logger.info("Training script completed successfully")
    print(f"\nTraining completed! Results saved to {save_dir}")
    print(f"Final mode: {training_results['final_mode']}")
    print(f"Mode switches: {len(adaptive_agent.mode_history)}")
    print(f"Total episodes: {training_results['episode_count']}")

if __name__ == "__main__":
    main() 