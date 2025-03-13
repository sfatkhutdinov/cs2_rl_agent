#!/usr/bin/env python
"""
Test script for the Cities: Skylines 2 vision interface, optimized for Windows.
"""

import os
import sys
import time
import argparse
from typing import Dict, Any
import logging

# Add the project root to path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.append(project_root)

from src.interface.vision_interface import VisionInterface
from src.utils.config_utils import load_config, override_config


def setup_logging():
    """Set up logging."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    return logging.getLogger("test_vision")


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Test the Cities: Skylines 2 vision interface")
    parser.add_argument("--config", type=str, default=os.path.join("src", "config", "default.yaml"),
                       help="Path to the configuration file")
    parser.add_argument("--calibrate", action="store_true",
                       help="Run in calibration mode to find UI elements")
    return parser.parse_args()


def run_standard_test(config, logger):
    """Run standard connection and action tests."""
    # Create vision interface
    interface = VisionInterface(config)
    
    # Connect to the game
    logger.info("Attempting to connect to Cities: Skylines 2 via vision interface...")
    if interface.connect():
        logger.info("Connection successful!")
        
        # Get game state
        logger.info("Getting game state...")
        game_state = interface.get_game_state()
        logger.info(f"Game state: {game_state}")
        
        # Get metrics
        logger.info("Getting metrics...")
        metrics = interface.get_metrics()
        logger.info(f"Metrics: {metrics}")
        
        # Test game speed control
        logger.info("Testing game speed control...")
        success = interface.set_game_speed(3)
        logger.info(f"Set game speed to 3: {'Success' if success else 'Failed'}")
        
        time.sleep(2)
        
        logger.info("Setting game speed back to normal...")
        interface.set_game_speed(1)
        
        # Disconnect
        interface.disconnect()
        logger.info("Disconnected from the game.")
        return True
    else:
        logger.error("Failed to connect to the game.")
        return False


def run_calibration_mode(config, logger):
    """Run in calibration mode to help find UI elements."""
    import pyautogui
    import mss
    import numpy as np
    import cv2
    
    logger.info("Starting calibration mode...")
    logger.info("This will capture screenshots to help identify UI element positions")
    logger.info("Press Ctrl+C to exit calibration mode")
    
    # Initialize screen capture
    sct = mss.mss()
    
    try:
        # Prompt user to prepare the game
        input("Position Cities: Skylines 2 so UI elements are visible, then press Enter...")
        
        # Capture full screen
        screen_region = config["interface"]["vision"]["screen_region"]
        monitor = {"top": screen_region[0], 
                  "left": screen_region[1], 
                  "width": screen_region[2], 
                  "height": screen_region[3]}
        
        # Capture and save screenshot
        logger.info("Capturing screenshot...")
        screenshot = np.array(sct.grab(monitor))
        
        # Save the screenshot
        screenshot_path = os.path.join(project_root, "calibration_screenshot.png")
        cv2.imwrite(screenshot_path, screenshot)
        logger.info(f"Screenshot saved to: {screenshot_path}")
        
        # Show mouse position tracker
        logger.info("Now showing mouse position. Move your mouse over UI elements and note the coordinates.")
        logger.info("Press Ctrl+C to exit.")
        
        try:
            while True:
                x, y = pyautogui.position()
                position_str = f"Mouse Position: X: {x}, Y: {y}"
                print(position_str, end='\r')
                time.sleep(0.1)
        except KeyboardInterrupt:
            print("\nMouse position tracking stopped.")
        
        # Guide for updating the configuration
        logger.info("\nTo update UI element positions:")
        logger.info("1. Edit src/interface/vision_interface.py")
        logger.info("2. Find the self.ui_elements dictionary in the __init__ method")
        logger.info("3. Update the coordinate values based on your observations")
        logger.info("Example: \"population\": {\"region\": (x1, y1, x2, y2)}")
        
        return True
        
    except KeyboardInterrupt:
        logger.info("Calibration mode exited.")
        return True
    finally:
        sct.close()


def main():
    """Main entry point."""
    # Set up logging
    logger = setup_logging()
    
    # Parse arguments
    args = parse_args()
    
    # Ensure the config path is correct for Windows
    config_path = os.path.normpath(os.path.join(project_root, args.config))
    logger.info(f"Loading configuration from: {config_path}")
    
    # Load configuration
    try:
        config = load_config(config_path)
    except FileNotFoundError:
        logger.error(f"Configuration file not found: {config_path}")
        return
    
    # Ensure vision interface is selected
    config_override = {
        "interface": {
            "type": "vision"
        }
    }
    config = override_config(config, config_override)
    
    # Choose mode based on arguments
    if args.calibrate:
        run_calibration_mode(config, logger)
    else:
        run_standard_test(config, logger)


if __name__ == "__main__":
    main() 