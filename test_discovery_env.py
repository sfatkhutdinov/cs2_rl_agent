"""
Simple test script to verify the DiscoveryEnvironment class can be imported properly.
"""
import sys
import os
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)

def test_import():
    """Test importing the DiscoveryEnvironment class."""
    try:
        logging.info("Attempting to import DiscoveryEnvironment...")
        from src.environment.discovery_env import DiscoveryEnvironment
        logging.info("Successfully imported DiscoveryEnvironment!")
        return True
    except ImportError as e:
        logging.error(f"Failed to import DiscoveryEnvironment: {e}")
        return False

def test_create():
    """Test creating an instance of DiscoveryEnvironment."""
    try:
        logging.info("Attempting to create DiscoveryEnvironment instance...")
        from src.environment.discovery_env import DiscoveryEnvironment
        from src.environment.cs2_env import CS2Environment
        
        # Create a minimal base environment config
        base_env_config = {
            "environment": {
                "type": "CS2Environment",
                "action_space": {
                    "zone": ["residential", "commercial", "industrial"],
                    "infrastructure": ["road", "water", "electricity"]
                }
            },
            "interface": {
                "type": "ollama_vision"
            },
            "ollama": {
                "url": "http://localhost:11434/api/generate",
                "model": "llama3.2-vision:latest"
            },
            "use_fallback_mode": True
        }
        
        # Create observation config
        observation_config = {
            "include_visual": True
        }
        
        # Create a CS2Environment instance first
        base_env = CS2Environment(base_env_config)
        
        # Try to create the environment with the base environment
        env = DiscoveryEnvironment(
            base_env_config=base_env_config,
            observation_config=observation_config
        )
        
        logging.info("Successfully created DiscoveryEnvironment instance!")
        return True
    except Exception as e:
        logging.error(f"Failed to create DiscoveryEnvironment instance: {e}")
        return False

if __name__ == "__main__":
    logging.info("=== Testing DiscoveryEnvironment ===")
    
    # Run import test
    import_success = test_import()
    
    # Run creation test if import succeeded
    if import_success:
        create_success = test_create()
    else:
        create_success = False
    
    # Print summary
    if import_success and create_success:
        logging.info("All tests passed! DiscoveryEnvironment is working properly.")
        sys.exit(0)
    else:
        logging.error("Tests failed! Please check the logs above for details.")
        sys.exit(1) 