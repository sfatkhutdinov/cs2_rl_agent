#!/usr/bin/env python3
"""
Training script for the strategic agent that builds on the adaptive agent.
This script trains an agent that autonomously discovers and optimizes game strategies.
"""

import os
import sys
import time
import yaml
import logging
import argparse
from pathlib import Path

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('logs/strategic_training.log', mode='w')
    ]
)
logger = logging.getLogger("StrategicTraining")

# Ensure necessary directories exist
os.makedirs("logs", exist_ok=True)
os.makedirs("models", exist_ok=True)

# Import dependencies after ensuring directories exist
from src.agent.adaptive_agent import AdaptiveAgent, TrainingMode
from src.agent.strategic_agent import StrategicAgent
from src.environment.strategic_env import StrategicEnvironment

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train a strategic agent for Cities: Skylines 2")
    
    parser.add_argument('--config', type=str, default='config/strategic_config.yaml',
                       help='Path to the configuration file')
    
    parser.add_argument('--timesteps', type=int, default=10000000,
                       help='Total training timesteps')
    
    parser.add_argument('--checkpoint-freq', type=int, default=50000,
                       help='Save checkpoints every N timesteps')
    
    parser.add_argument('--load-checkpoint', type=str, default=None,
                       help='Path to load a checkpoint from')
    
    parser.add_argument('--use-adaptive', action='store_true',
                       help='Use the adaptive agent as a wrapper instead of direct strategic training')
    
    parser.add_argument('--eval-freq', type=int, default=100000,
                       help='Run evaluation every N timesteps')
    
    parser.add_argument('--knowledge-bootstrap', action='store_true',
                       help='Use knowledge bootstrapping to accelerate learning')
    
    return parser.parse_args()

def progress_callback(progress):
    """Callback function to track training progress."""
    if progress and isinstance(progress, dict):
        # Log progress information
        step = progress.get('step', 0)
        total = progress.get('total', 1)
        percentage = (step / total) * 100
        
        # Only log occasionally to avoid spamming
        if step % 1000 == 0 or step == total:
            logger.info(f"Training progress: {step}/{total} steps ({percentage:.1f}%)")
        
        # Log metrics if available
        if 'metrics' in progress:
            metrics = progress['metrics']
            if step % 5000 == 0:  # Log metrics less frequently
                metrics_str = ", ".join([f"{k}: {v:.3f}" for k, v in metrics.items()])
                logger.info(f"Metrics: {metrics_str}")

def main():
    """Main training function."""
    args = parse_arguments()
    
    # Load configuration
    try:
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
            logger.info(f"Loaded configuration from {args.config}")
    except Exception as e:
        logger.error(f"Error loading configuration: {e}")
        return
    
    # Apply command-line overrides
    config['training']['total_timesteps'] = args.timesteps
    config['training']['checkpoint_freq'] = args.checkpoint_freq
    config['training']['eval_freq'] = args.eval_freq
    
    # Apply knowledge bootstrapping if specified
    if args.knowledge_bootstrap and 'knowledge_bootstrapping' in config:
        bootstrap_knowledge = config['knowledge_bootstrapping']
        
        # Initialize knowledge base if needed
        if 'knowledge_base' not in config:
            config['knowledge_base'] = {}
        
        # Add important metrics to prioritize
        if 'important_metrics' in bootstrap_knowledge:
            config['knowledge_base']['prioritized_metrics'] = bootstrap_knowledge['important_metrics']
        
        # Add metric types (positive vs negative)
        if 'metric_types' in bootstrap_knowledge:
            config['knowledge_base']['metric_types'] = bootstrap_knowledge['metric_types']
        
        # Add causal hints
        if 'causal_hints' in bootstrap_knowledge:
            if 'causal_links' not in config['knowledge_base']:
                config['knowledge_base']['causal_links'] = {}
            
            # Convert hints to causal links
            for hint in bootstrap_knowledge['causal_hints']:
                link_id = f"{hint['action']}→{hint['affects']}"
                config['knowledge_base']['causal_links'][link_id] = {
                    'action': hint['action'],
                    'metric': hint['affects'],
                    'direction': hint['direction'],
                    'strength': 0.7,  # Default moderate confidence
                    'average_change': 1.0 if hint['direction'] == 'positive' else -1.0
                }
        
        logger.info("Applied knowledge bootstrapping")
    
    # Determine training approach
    if args.use_adaptive:
        # Use adaptive agent with strategic mode
        logger.info("Using adaptive agent as a wrapper for strategic training")
        
        # Modify config paths for the adaptive agent
        adaptive_config = config.copy()
        adaptive_config['discovery_config_path'] = 'config/discovery_config.yaml'
        adaptive_config['vision_config_path'] = 'config/vision_guided_config.yaml'
        adaptive_config['autonomous_config_path'] = 'config/autonomous_config.yaml'
        adaptive_config['tutorial_config_path'] = 'config/tutorial_guided_config.yaml'
        adaptive_config['strategic_config_path'] = args.config
        
        # Create adaptive agent
        agent = AdaptiveAgent(adaptive_config)
        
        # Force the mode to strategic
        agent.current_mode = TrainingMode.STRATEGIC
        agent.initialize_current_mode()
        
        # Train with the adaptive agent
        logger.info("Starting strategic training via adaptive agent")
        agent.train(args.timesteps, progress_callback=progress_callback)
        
    else:
        # Directly train the strategic agent
        logger.info("Creating strategic environment")
        strategic_env = StrategicEnvironment(config)
        
        logger.info("Creating strategic agent")
        agent = StrategicAgent(strategic_env, config)
        
        # Load from checkpoint if specified
        if args.load_checkpoint:
            try:
                logger.info(f"Loading checkpoint from {args.load_checkpoint}")
                agent.load(args.load_checkpoint)
            except Exception as e:
                logger.error(f"Error loading checkpoint: {e}")
        
        # Train the agent
        logger.info(f"Starting strategic training for {args.timesteps} timesteps")
        start_time = time.time()
        agent.train(args.timesteps, callback=progress_callback)
        
        # Log training completion
        total_time = time.time() - start_time
        logger.info(f"Training completed in {total_time:.2f} seconds")
        
        # Final evaluation
        logger.info("Running final evaluation")
        strategic_score = agent.evaluate_strategic_understanding()
        logger.info(f"Final strategic understanding score: {strategic_score:.4f}")
        
        # Save the final model
        final_save_path = os.path.join(config['paths']['model_dir'], 'final_model')
        os.makedirs(Path(final_save_path).parent, exist_ok=True)
        agent.save(final_save_path)
        logger.info(f"Final model saved to {final_save_path}")
    
    logger.info("Strategic training completed successfully")

if __name__ == "__main__":
    main() 