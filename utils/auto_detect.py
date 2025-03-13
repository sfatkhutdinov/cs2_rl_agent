#!/usr/bin/env python
"""
Auto-detection script for Cities: Skylines 2 RL Agent.

This script uses computer vision to automatically detect UI elements
without requiring manual coordinate input.
"""

import os
import sys
import time
import argparse
import logging
from typing import Dict, Any

# Add the project root to path
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

from src.interface.auto_vision_interface import AutoVisionInterface
from src.utils.config_utils import load_config, override_config


def setup_logging():
    """Set up logging."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    return logging.getLogger("auto_detect")


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Auto-detection for Cities: Skylines 2 RL Agent")
    
    parser.add_argument("--config", type=str, default=os.path.join("src", "config", "default.yaml"),
                       help="Path to the configuration file")
    parser.add_argument("--capture-templates", action="store_true",
                       help="Capture templates for UI elements")
    parser.add_argument("--method", type=str, choices=["ocr", "template"], default="ocr",
                       help="Detection method: 'ocr' or 'template'")
    
    return parser.parse_args()


def run_template_capture(interface, logger):
    """Run template capture mode."""
    logger.info("Starting template capture mode")
    logger.info("Make sure the game is running and UI elements are visible")
    
    try:
        # Capture templates
        interface.capture_templates()
        
        logger.info("Template capture completed!")
        logger.info("You can now use these templates for automatic UI detection")
        
        return True
    except KeyboardInterrupt:
        logger.info("Template capture aborted by user")
        return False
    except Exception as e:
        logger.error(f"Template capture failed: {str(e)}")
        return False


def run_detection_test(interface, logger):
    """Run detection test."""
    logger.info("Testing automatic UI detection")
    
    # Connect to the game
    logger.info("Connecting to Cities: Skylines 2...")
    if interface.connect():
        logger.info("Connection successful!")
        
        # Test UI element detection
        logger.info("Detecting UI elements...")
        if interface.detect_ui_elements():
            logger.info("UI elements detected successfully!")
            
            # Print detected elements
            logger.info("Detected UI elements:")
            for element, data in interface.ui_element_cache.items():
                region = data["region"]
                confidence = data["confidence"]
                logger.info(f"  {element}: region={region}, confidence={confidence:.2f}")
            
            # Get metrics
            logger.info("Extracting metrics...")
            metrics = interface.get_metrics()
            logger.info(f"Metrics: {metrics}")
            
            # Test game speed control
            logger.info("Testing game speed control...")
            success = interface.set_game_speed(3)
            logger.info(f"Set game speed to 3: {'Success' if success else 'Failed'}")
            
            time.sleep(2)
            
            logger.info("Setting game speed back to normal...")
            interface.set_game_speed(1)
            
            logger.info("Test completed successfully!")
        else:
            logger.error("Failed to detect UI elements")
        
        # Disconnect
        interface.disconnect()
        logger.info("Disconnected from the game")
        
        return True
    else:
        logger.error("Failed to connect to the game")
        return False


def main():
    """Main entry point."""
    # Set up logging
    logger = setup_logging()
    
    # Parse arguments
    args = parse_args()
    
    # Ensure the config path is correct
    config_path = os.path.normpath(os.path.join(current_dir, args.config))
    logger.info(f"Loading configuration from: {config_path}")
    
    # Load configuration
    try:
        config = load_config(config_path)
    except FileNotFoundError:
        logger.error(f"Configuration file not found: {config_path}")
        return
    
    # Override configuration
    config_override = {
        "interface": {
            "type": "auto_vision",
            "vision": {
                "detection_method": args.method
            }
        }
    }
    config = override_config(config, config_override)
    
    # Create auto vision interface
    interface = AutoVisionInterface(config)
    
    # Choose mode based on arguments
    if args.capture_templates:
        run_template_capture(interface, logger)
    else:
        run_detection_test(interface, logger)


if __name__ == "__main__":
    main() 