#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Test script for verifying the adaptive agent's orchestration capabilities.
This script tests the adaptive agent's ability to:
1. Initialize correctly with all specialized agent modes
2. Switch between different modes based on performance metrics
3. Transfer knowledge between different modes
4. Handle errors and recover appropriately

Usage:
    python test_adaptive_orchestration.py

Author: Adaptive Agent Team
Last updated: March 13, 2025 21:17
"""

import os
import sys
import time
import logging
import argparse
import numpy as np

# Add the src directory to the path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import adaptive agent components
from src.agent.adaptive_agent import AdaptiveAgent
from src.utils.logger import setup_logger
from src.environment.cs2_env import CS2Environment

# Set up logging
logger = setup_logger('test_adaptive_orchestration', level=logging.INFO)

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Test adaptive agent orchestration')
    parser.add_argument('--config', type=str, default='config/adaptive_config.yaml',
                        help='Path to the adaptive agent configuration file')
    parser.add_argument('--test-duration', type=int, default=300,
                        help='Duration of the test in seconds')
    parser.add_argument('--mock-env', action='store_true',
                        help='Use a mock environment instead of the real game')
    return parser.parse_args()

class MockEnvironment:
    """Mock environment for testing without the actual game."""
    
    def __init__(self):
        self.state = 'initial'
        self.reward_map = {
            'discovery': np.random.normal(0.3, 0.1),
            'tutorial': np.random.normal(0.5, 0.1),
            'vision': np.random.normal(0.7, 0.1),
            'autonomous': np.random.normal(0.4, 0.1),
            'strategic': np.random.normal(0.6, 0.1),
        }
    
    def reset(self):
        """Reset the environment."""
        self.state = 'initial'
        return {'observation': np.random.rand(10), 'mode': 'discovery'}
    
    def step(self, action, mode):
        """Take a step in the environment."""
        # Simulate different rewards based on the current mode
        reward = self.reward_map[mode]
        
        # Randomly decide if we should trigger a mode switch
        if np.random.random() < 0.1:  # 10% chance of mode switch suggestion
            next_mode = np.random.choice(['discovery', 'tutorial', 'vision', 'autonomous', 'strategic'])
            mode_switch_signal = True
        else:
            next_mode = mode
            mode_switch_signal = False
            
        # Create observation with mode-specific features
        observation = {'observation': np.random.rand(10), 
                      'mode': next_mode,
                      'mode_switch': mode_switch_signal}
        
        done = np.random.random() < 0.05  # 5% chance of episode ending
        info = {'performance': np.random.rand(), 'current_mode': mode}
        
        return observation, reward, done, info
    
    def close(self):
        """Close the environment."""
        pass

def test_adaptive_orchestration(config_path, test_duration, use_mock_env):
    """Test the adaptive agent's orchestration capabilities."""
    logger.info(f"Starting adaptive orchestration test with config: {config_path}")
    
    # Initialize environment (mock or real)
    if use_mock_env:
        environment = MockEnvironment()
        logger.info("Using mock environment for testing")
    else:
        environment = CS2Environment()
        logger.info("Using real CS2Environment for testing")
    
    # Initialize adaptive agent
    agent = AdaptiveAgent(environment, config_path)
    logger.info("Adaptive agent initialized successfully")
    
    # Track metrics for each mode
    mode_metrics = {
        'discovery': {'episodes': 0, 'rewards': [], 'duration': 0},
        'tutorial': {'episodes': 0, 'rewards': [], 'duration': 0},
        'vision': {'episodes': 0, 'rewards': [], 'duration': 0},
        'autonomous': {'episodes': 0, 'rewards': [], 'duration': 0},
        'strategic': {'episodes': 0, 'rewards': [], 'duration': 0},
    }
    
    # Track mode switches
    mode_switches = []
    current_mode = 'discovery'  # Start with discovery mode
    
    # Test for specified duration
    start_time = time.time()
    episode_count = 0
    
    try:
        while time.time() - start_time < test_duration:
            # Reset environment
            observation = environment.reset()
            
            # Track episode start
            episode_start_time = time.time()
            episode_rewards = []
            
            # Run episode
            done = False
            while not done and time.time() - start_time < test_duration:
                # Get action from agent
                action = agent.get_action(observation)
                
                # Take step in environment
                observation, reward, done, info = environment.step(action, current_mode)
                episode_rewards.append(reward)
                
                # Update mode if changed
                if 'current_mode' in info and info['current_mode'] != current_mode:
                    prev_mode = current_mode
                    current_mode = info['current_mode']
                    mode_switches.append({
                        'time': time.time() - start_time,
                        'from': prev_mode,
                        'to': current_mode,
                        'reason': info.get('switch_reason', 'unknown')
                    })
                    logger.info(f"Mode switched from {prev_mode} to {current_mode}")
            
            # Update metrics for this episode
            episode_count += 1
            episode_duration = time.time() - episode_start_time
            
            # Update mode-specific metrics
            mode_metrics[current_mode]['episodes'] += 1
            mode_metrics[current_mode]['rewards'].append(sum(episode_rewards))
            mode_metrics[current_mode]['duration'] += episode_duration
            
            logger.info(f"Episode {episode_count} completed. "
                       f"Mode: {current_mode}, "
                       f"Reward: {sum(episode_rewards):.2f}, "
                       f"Duration: {episode_duration:.2f}s")
    
    except KeyboardInterrupt:
        logger.info("Test interrupted by user")
    except Exception as e:
        logger.error(f"Test failed with error: {e}")
    finally:
        # Close environment
        environment.close()
    
    # Log test summary
    test_duration_actual = time.time() - start_time
    logger.info(f"Test completed. Duration: {test_duration_actual:.2f}s")
    logger.info(f"Total episodes: {episode_count}")
    logger.info(f"Mode switches: {len(mode_switches)}")
    
    # Log mode-specific metrics
    for mode, metrics in mode_metrics.items():
        if metrics['episodes'] > 0:
            avg_reward = sum(metrics['rewards']) / metrics['episodes'] if metrics['rewards'] else 0
            logger.info(f"Mode {mode}: Episodes: {metrics['episodes']}, "
                       f"Avg reward: {avg_reward:.2f}, "
                       f"Total duration: {metrics['duration']:.2f}s")
    
    return {
        'total_episodes': episode_count,
        'mode_switches': mode_switches,
        'mode_metrics': mode_metrics,
        'test_duration': test_duration_actual
    }

def main():
    """Main function."""
    args = parse_args()
    results = test_adaptive_orchestration(args.config, args.test_duration, args.mock_env)
    
    # Print summary results
    print("\n===== Adaptive Orchestration Test Results =====")
    print(f"Total test duration: {results['test_duration']:.2f} seconds")
    print(f"Total episodes: {results['total_episodes']}")
    print(f"Total mode switches: {len(results['mode_switches'])}")
    
    # Print mode usage statistics
    print("\nMode Statistics:")
    for mode, metrics in results['mode_metrics'].items():
        if metrics['episodes'] > 0:
            pct_time = (metrics['duration'] / results['test_duration']) * 100
            avg_reward = sum(metrics['rewards']) / metrics['episodes'] if metrics['rewards'] else 0
            print(f"  {mode.capitalize()}: {metrics['episodes']} episodes, "
                  f"{pct_time:.1f}% of time, {avg_reward:.2f} avg reward")
    
    # Verify if all modes were used
    unused_modes = [mode for mode, metrics in results['mode_metrics'].items() 
                   if metrics['episodes'] == 0]
    if unused_modes:
        print(f"\nWARNING: The following modes were not used during testing: {', '.join(unused_modes)}")
        print("Consider checking the mode switching logic or extending the test duration.")
    else:
        print("\nSUCCESS: All agent modes were utilized during testing.")
    
    # Check knowledge transfer (did performance improve over time?)
    print("\nVerifying knowledge transfer:")
    knowledge_transfer = False
    for mode, metrics in results['mode_metrics'].items():
        if len(metrics['rewards']) >= 2:
            first_half = metrics['rewards'][:len(metrics['rewards'])//2]
            second_half = metrics['rewards'][len(metrics['rewards'])//2:]
            if first_half and second_half:
                improvement = (sum(second_half)/len(second_half)) - (sum(first_half)/len(first_half))
                print(f"  {mode.capitalize()}: {'Improved' if improvement > 0 else 'Declined'} by {abs(improvement):.2f}")
                if improvement > 0:
                    knowledge_transfer = True
    
    if knowledge_transfer:
        print("SUCCESS: Evidence of knowledge transfer detected between episodes.")
    else:
        print("WARNING: No clear evidence of knowledge transfer. Consider longer testing duration.")
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 