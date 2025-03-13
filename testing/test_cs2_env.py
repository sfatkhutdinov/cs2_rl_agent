"""
Test script that verifies the CS2Environment can be properly imported.
"""
import sys
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)

def test_import():
    """Test importing the CS2Environment class."""
    try:
        logging.info("Attempting to import CS2Environment...")
        from src.environment.cs2_env import CS2Environment
        logging.info("Successfully imported CS2Environment!")
        return True
    except Exception as e:
        logging.error(f"Failed to import CS2Environment: {e}")
        return False

def test_create():
    """Test creating a CS2Environment instance with a minimal configuration."""
    try:
        logging.info("Attempting to create a CS2Environment instance...")
        from src.environment.cs2_env import CS2Environment
        
        # Create a minimal configuration
        config = {
            "environment": {
                "type": "CS2Environment",
                "observation_space": {
                    "include_visual": True,
                    "include_metrics": True,
                    "metrics": ["population", "happiness", "budget_balance"]
                },
                "action_space": {
                    "zone": ["residential", "commercial", "industrial"],
                    "infrastructure": ["road", "water", "electricity"],
                    "budget": ["increase_tax", "decrease_tax"]
                }
            },
            "interface": {
                "type": "ollama_vision",
                "vision": {
                    "ocr_confidence": 0.7,
                    "detection_method": "ocr"
                }
            },
            "ollama": {
                "url": "http://localhost:11434/api/generate",
                "model": "llama3.2-vision:latest",
                "max_tokens": 1000,
                "temperature": 0.7
            },
            "game_window_handle": 67258,  # Handle for Cities Skylines II window
            "use_fallback_mode": True
        }
        
        # Create the environment
        env = CS2Environment(config)
        logging.info("Successfully created CS2Environment instance!")
        return True
    except Exception as e:
        logging.error(f"Failed to create CS2Environment instance: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    logging.info("=== Testing CS2Environment ===")
    
    # Test importing
    import_success = test_import()
    
    # Test creating instance
    if import_success:
        create_success = test_create()
    else:
        create_success = False
    
    # Print results
    print("\n=== Test Results ===")
    print(f"Import Test: {'✅' if import_success else '❌'}")
    print(f"Creation Test: {'✅' if create_success else '❌'}")
    
    if import_success and create_success:
        print("\nCS2Environment is working correctly!")
        sys.exit(0)
    else:
        print("\nThere were issues with the CS2Environment. Check the logs for details.")
        sys.exit(1) 