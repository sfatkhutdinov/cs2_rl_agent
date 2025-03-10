import os
import argparse
import logging
from typing import Dict, Any
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime

import gymnasium as gym
from stable_baselines3 import PPO, DQN, A2C

from environment.cs2_env import CS2Environment
from agent.agent_factory import wrap_env_if_needed
from utils.config_utils import load_config, get_full_path
from utils.logger import Logger


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Evaluate a trained reinforcement learning agent for Cities: Skylines 2")
    
    parser.add_argument("--config", type=str, default="src/config/default.yaml",
                       help="Path to the configuration file")
    parser.add_argument("--model", type=str, required=True,
                       help="Path to the trained model")
    parser.add_argument("--episodes", type=int, default=5,
                       help="Number of episodes to evaluate")
    parser.add_argument("--render", action="store_true",
                       help="Render the environment")
    parser.add_argument("--seed", type=int, default=None,
                       help="Random seed")
    
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
    # Create the environment
    env = CS2Environment(config)
    
    # Wrap the environment if needed
    env = wrap_env_if_needed(env, config)
    
    return env


def load_agent(model_path: str, env: gym.Env, config: Dict[str, Any]):
    """
    Load a trained agent.
    
    Args:
        model_path: Path to the trained model
        env: Gym environment
        config: Configuration dictionary
        
    Returns:
        Trained agent
    """
    logger = logging.getLogger("evaluate")
    
    # Determine the algorithm from the config
    algorithm = config["agent"]["algorithm"]
    
    # Load the agent
    if algorithm == "PPO":
        logger.info(f"Loading PPO agent from {model_path}")
        agent = PPO.load(model_path, env=env)
    elif algorithm == "DQN":
        logger.info(f"Loading DQN agent from {model_path}")
        agent = DQN.load(model_path, env=env)
    elif algorithm == "A2C":
        logger.info(f"Loading A2C agent from {model_path}")
        agent = A2C.load(model_path, env=env)
    else:
        logger.error(f"Unknown algorithm: {algorithm}")
        raise ValueError(f"Unknown algorithm: {algorithm}")
    
    return agent


def evaluate(agent, env: gym.Env, num_episodes: int, render: bool = False):
    """
    Evaluate a trained agent.
    
    Args:
        agent: Trained agent
        env: Gym environment
        num_episodes: Number of episodes to evaluate
        render: Whether to render the environment
        
    Returns:
        Dictionary of evaluation metrics
    """
    logger = logging.getLogger("evaluate")
    
    # Metrics to track
    episode_rewards = []
    episode_lengths = []
    episode_metrics = {
        "population": [],
        "happiness": [],
        "budget_balance": [],
        "traffic_flow": []
    }
    
    # Run evaluation episodes
    for episode in range(num_episodes):
        logger.info(f"Starting evaluation episode {episode+1}/{num_episodes}")
        
        # Reset the environment
        obs, info = env.reset()
        done = False
        truncated = False
        episode_reward = 0
        episode_step = 0
        
        # Store metrics for this episode
        episode_population = []
        episode_happiness = []
        episode_budget = []
        episode_traffic = []
        
        # Run the episode
        while not (done or truncated):
            # Select action
            action, _ = agent.predict(obs, deterministic=True)
            
            # Take step
            obs, reward, done, truncated, info = env.step(action)
            episode_reward += reward
            episode_step += 1
            
            # Store metrics
            metrics = info.get("metrics", {})
            if "population" in metrics:
                episode_population.append(metrics["population"])
            if "happiness" in metrics:
                episode_happiness.append(metrics["happiness"])
            if "budget" in metrics:
                episode_budget.append(metrics["budget"])
            if "traffic" in metrics:
                episode_traffic.append(metrics["traffic"])
            
            # Render if requested
            if render:
                env.render()
        
        # Store episode results
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_step)
        
        # Store average metrics for this episode
        if episode_population:
            episode_metrics["population"].append(np.mean(episode_population))
        if episode_happiness:
            episode_metrics["happiness"].append(np.mean(episode_happiness))
        if episode_budget:
            episode_metrics["budget_balance"].append(np.mean(episode_budget))
        if episode_traffic:
            episode_metrics["traffic_flow"].append(np.mean(episode_traffic))
        
        logger.info(f"Episode {episode+1} finished: reward={episode_reward:.2f}, length={episode_step}")
    
    # Calculate overall metrics
    eval_metrics = {
        "mean_reward": np.mean(episode_rewards),
        "std_reward": np.std(episode_rewards),
        "mean_length": np.mean(episode_lengths),
        "std_length": np.std(episode_lengths)
    }
    
    # Add game-specific metrics
    for metric, values in episode_metrics.items():
        if values:
            eval_metrics[f"mean_{metric}"] = np.mean(values)
            eval_metrics[f"std_{metric}"] = np.std(values)
    
    return eval_metrics, episode_rewards, episode_metrics


def plot_results(eval_metrics: Dict[str, float], episode_rewards: list, episode_metrics: Dict[str, list], output_dir: str):
    """
    Plot evaluation results.
    
    Args:
        eval_metrics: Dictionary of evaluation metrics
        episode_rewards: List of episode rewards
        episode_metrics: Dictionary of episode metrics
        output_dir: Directory to save plots
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Plot episode rewards
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(episode_rewards) + 1), episode_rewards, marker='o')
    plt.title(f"Episode Rewards (Mean: {eval_metrics['mean_reward']:.2f})")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, "episode_rewards.png"))
    plt.close()
    
    # Plot game metrics
    if any(episode_metrics.values()):
        plt.figure(figsize=(12, 8))
        
        plot_idx = 1
        for metric, values in episode_metrics.items():
            if values:
                plt.subplot(2, 2, plot_idx)
                plt.plot(range(1, len(values) + 1), values, marker='o')
                plt.title(f"{metric.capitalize()} (Mean: {eval_metrics.get(f'mean_{metric}', 0):.2f})")
                plt.xlabel("Episode")
                plt.ylabel(metric.capitalize())
                plt.grid(True)
                plot_idx += 1
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "game_metrics.png"))
        plt.close()
    
    # Save metrics to CSV
    metrics_df = pd.DataFrame({
        "metric": list(eval_metrics.keys()),
        "value": list(eval_metrics.values())
    })
    metrics_df.to_csv(os.path.join(output_dir, "eval_metrics.csv"), index=False)


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
    
    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.normpath(os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        config["paths"]["logs"],
        f"eval_{timestamp}"
    ))
    os.makedirs(output_dir, exist_ok=True)
    
    # Create environment
    env = create_environment(config)
    
    # Load agent
    agent = load_agent(args.model, env, config)
    
    try:
        # Evaluate agent
        logger.info(f"Evaluating agent for {args.episodes} episodes")
        eval_metrics, episode_rewards, episode_metrics = evaluate(
            agent, env, args.episodes, args.render
        )
        
        # Log results
        logger.info("Evaluation results:")
        for metric, value in eval_metrics.items():
            logger.info(f"  {metric}: {value:.4f}")
        
        # Plot results
        logger.info(f"Plotting results to {output_dir}")
        plot_results(eval_metrics, episode_rewards, episode_metrics, output_dir)
        
    except KeyboardInterrupt:
        logger.info("Evaluation interrupted by user")
    
    except Exception as e:
        logger.error(f"Evaluation failed: {str(e)}")
        raise
    
    finally:
        # Close the environment
        env.close()
    
    logger.info("Evaluation completed")


if __name__ == "__main__":
    main() 