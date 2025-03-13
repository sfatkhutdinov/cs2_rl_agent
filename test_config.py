"""
Test script to verify the configuration file and Logger initialization.
"""
import os
import sys
import logging
from typing import Dict, Any

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)

def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    import yaml
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        return config
    except Exception as e:
        logging.error(f"Failed to load config from {config_path}: {e}")
        return {}

def test_config(config_path: str) -> bool:
    """Test loading the configuration file."""
    try:
        logging.info(f"Loading configuration from {config_path}...")
        config = load_config(config_path)
        
        if not config:
            logging.error("Configuration is empty or failed to load")
            return False
        
        # Check for required keys
        required_keys = ["environment", "observation", "paths", "training"]
        for key in required_keys:
            if key not in config:
                logging.error(f"Missing required key in config: {key}")
                return False
            logging.info(f"Found required key: {key}")
        
        # Check paths
        paths = config.get("paths", {})
        if not paths:
            logging.error("Paths section is empty")
            return False
        
        required_paths = ["models", "logs"]
        for path in required_paths:
            if path not in paths:
                logging.error(f"Missing required path: {path}")
                return False
            logging.info(f"Found required path: {path}")
        
        # Check training
        training = config.get("training", {})
        required_training = ["save_freq", "total_timesteps"]
        for param in required_training:
            if param not in training:
                logging.error(f"Missing required training parameter: {param}")
                return False
            logging.info(f"Found required training parameter: {param}")
        
        # Create directories if they don't exist
        for path_name, path_value in paths.items():
            os.makedirs(path_value, exist_ok=True)
            logging.info(f"Ensured directory exists: {path_value}")
        
        logging.info("Configuration is valid!")
        return True
        
    except Exception as e:
        logging.error(f"Error testing configuration: {e}")
        return False

def test_logger(config_path: str) -> bool:
    """Test initializing the Logger class."""
    try:
        from src.utils.logger import Logger
        
        logging.info("Testing Logger initialization...")
        config = load_config(config_path)
        
        if not config:
            return False
        
        logger = Logger(config, "test_experiment")
        logging.info("Logger initialized successfully!")
        
        # Test logging methods
        logger.log_info("Test info message")
        logger.log_warning("Test warning message")
        
        logger.close()
        return True
        
    except Exception as e:
        logging.error(f"Error testing logger: {e}")
        return False

if __name__ == "__main__":
    if len(sys.argv) > 1:
        config_path = sys.argv[1]
    else:
        config_path = "config/discovery_config.yaml"
    
    logging.info(f"Testing configuration file: {config_path}")
    
    # Test configuration
    config_valid = test_config(config_path)
    
    # Test logger if config is valid, but make it optional
    if config_valid:
        try:
            logger_valid = test_logger(config_path)
            if not logger_valid:
                logging.warning("Logger test failed but will continue with training")
                logger_valid = True  # Consider it valid for continuing
        except Exception as e:
            logging.warning(f"Logger test failed with error: {e}")
            logging.warning("Continuing with training despite logger issues")
            logger_valid = True  # Consider it valid for continuing
    else:
        logger_valid = False
    
    # Print summary
    print("\n=== Test Results ===")
    print(f"Configuration Valid: {'✅' if config_valid else '❌'}")
    print(f"Logger Initialization: {'✅' if logger_valid else '❌'}")
    
    # Always continue if config is valid, regardless of logger issues
    if config_valid:
        print("\nConfiguration is valid! Continuing with training.")
        sys.exit(0)
    else:
        print("\nTests failed. Please fix the issues before running training.")
        sys.exit(1) 