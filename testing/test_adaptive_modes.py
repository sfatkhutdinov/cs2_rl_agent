#!/usr/bin/env python3
"""
Test script to verify that the adaptive agent can initialize and
access all training modes, including the strategic mode.
"""

import logging
import yaml
import sys
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("TestAdaptiveModes")

# Import the adaptive agent
from src.agent.adaptive_agent import AdaptiveAgent, TrainingMode

def main():
    """
    Test all training modes in the adaptive agent to verify they are accessible.
    """
    logger.info("Testing adaptive agent's access to all training modes...")
    
    # Load the adaptive agent config
    with open('config/adaptive_config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Ensure required configurations are available
    config_files = [
        "config/discovery_config.yaml",
        "config/tutorial_guided_config.yaml",
        "config/vision_guided_config.yaml",
        "config/autonomous_config.yaml",
        "config/strategic_config.yaml"
    ]
    
    missing_files = [f for f in config_files if not Path(f).exists()]
    if missing_files:
        logger.error(f"Missing configuration files: {missing_files}")
        return False
    
    logger.info("All configuration files found.")
    
    # Create the adaptive agent instance
    try:
        agent = AdaptiveAgent(config)
        logger.info("Adaptive agent initialized successfully.")
    except Exception as e:
        logger.error(f"Failed to initialize adaptive agent: {e}")
        return False
    
    # Test all modes
    all_modes = list(TrainingMode)
    results = {}
    
    for mode in all_modes:
        logger.info(f"Testing mode: {mode.value}")
        
        # Set the current mode
        agent.current_mode = mode
        
        try:
            # Initialize the current mode
            agent.initialize_current_mode()
            results[mode.value] = "Success"
            logger.info(f"✅ Successfully initialized {mode.value} mode")
        except Exception as e:
            results[mode.value] = f"Failed: {str(e)}"
            logger.error(f"❌ Failed to initialize {mode.value} mode: {e}")
    
    # Print summary
    logger.info("\n--- Mode Initialization Test Results ---")
    success_count = sum(1 for result in results.values() if result == "Success")
    logger.info(f"Success: {success_count}/{len(all_modes)} modes")
    
    for mode, result in results.items():
        status = "✅" if result == "Success" else "❌"
        logger.info(f"{status} {mode}: {result}")
    
    # Test mode switching logic
    logger.info("\nTesting mode switching logic...")
    agent.current_mode = TrainingMode.AUTONOMOUS
    
    # Set up metrics to trigger a switch to strategic mode
    agent.mode_metrics[TrainingMode.AUTONOMOUS]["confidence"] = 0.9
    agent.mode_metrics[TrainingMode.AUTONOMOUS]["game_cycles_completed"] = 15
    
    # Check if the agent decides to switch to strategic mode
    should_switch, new_mode, reason = agent.should_switch_mode()
    
    if should_switch and new_mode == TrainingMode.STRATEGIC:
        logger.info("✅ Mode switching logic correctly transitions to strategic mode")
        logger.info(f"Reason: {reason}")
    else:
        logger.error("❌ Mode switching logic failed to transition to strategic mode")
        logger.error(f"Decision: switch={should_switch}, new_mode={new_mode}, reason={reason}")
    
    logger.info("\nTest completed.")
    return success_count == len(all_modes)

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 