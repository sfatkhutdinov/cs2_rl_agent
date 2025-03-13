import unittest
import os
import sys
import tempfile
import yaml
from typing import Dict, Any

# Add the parent directory to the path so we can import the modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.config_validator import validate_config, validate_file, CONFIG_SCHEMA
from src.utils.config_utils import load_config, save_config


class TestConfigValidator(unittest.TestCase):
    """Test the configuration validator."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create a valid test configuration
        self.valid_config = {
            "agent": {
                "agent_type": "adaptive",
                "learning_rate": 0.001,
                "gamma": 0.99,
                "n_steps": 2048,
                "batch_size": 64
            },
            "environment": {
                "action_delay": 0.5,
                "max_episode_steps": 1000,
                "reward_scales": {
                    "population": 1.0,
                    "happiness": 0.8
                }
            },
            "interface": {
                "type": "api",
                "api": {
                    "host": "localhost",
                    "port": 5000,
                    "timeout": 10.0
                }
            },
            "logging": {
                "level": "INFO",
                "log_mode_switches": True,
                "log_metrics": True
            }
        }
        
        # Create a temporary directory for test files
        self.temp_dir = tempfile.TemporaryDirectory()
    
    def tearDown(self):
        """Clean up test fixtures."""
        self.temp_dir.cleanup()
    
    def test_validate_valid_config(self):
        """Test that a valid configuration passes validation."""
        errors = validate_config(self.valid_config)
        self.assertEqual(len(errors), 0, "Valid configuration should have no validation errors")
    
    def test_validate_invalid_agent_type(self):
        """Test that an invalid agent type is caught."""
        invalid_config = self.valid_config.copy()
        invalid_config["agent"]["agent_type"] = "invalid_type"
        
        errors = validate_config(invalid_config)
        self.assertGreater(len(errors), 0, "Invalid agent type should cause validation error")
        self.assertTrue(any("agent_type" in error for error in errors), "Error should mention agent_type")
    
    def test_validate_invalid_learning_rate(self):
        """Test that an invalid learning rate is caught."""
        invalid_config = self.valid_config.copy()
        invalid_config["agent"]["learning_rate"] = 2.0  # Out of range
        
        errors = validate_config(invalid_config)
        self.assertGreater(len(errors), 0, "Invalid learning rate should cause validation error")
        self.assertTrue(any("learning_rate" in error for error in errors), "Error should mention learning_rate")
    
    def test_validate_missing_required_field(self):
        """Test that a missing required field is caught."""
        invalid_config = self.valid_config.copy()
        del invalid_config["agent"]["agent_type"]
        
        errors = validate_config(invalid_config)
        self.assertGreater(len(errors), 0, "Missing required field should cause validation error")
        self.assertTrue(any("agent_type" in error for error in errors), "Error should mention agent_type")
    
    def test_validate_file(self):
        """Test validating a configuration file."""
        # Save the valid configuration to a file
        config_path = os.path.join(self.temp_dir.name, "test_config.yaml")
        with open(config_path, "w") as f:
            yaml.dump(self.valid_config, f)
        
        # Validate the file
        errors = validate_file(config_path)
        self.assertEqual(len(errors), 0, "Valid configuration file should have no validation errors")
    
    def test_validate_nonexistent_file(self):
        """Test validating a nonexistent file."""
        config_path = os.path.join(self.temp_dir.name, "nonexistent.yaml")
        
        errors = validate_file(config_path)
        self.assertGreater(len(errors), 0, "Nonexistent file should cause validation error")
        self.assertTrue(any("not found" in error for error in errors), "Error should mention file not found")
    
    def test_integration_with_config_utils(self):
        """Test integration with config_utils."""
        # Save the valid configuration to a file
        config_path = os.path.join(self.temp_dir.name, "test_config.yaml")
        save_config(self.valid_config, config_path)
        
        # Load the configuration with validation
        try:
            loaded_config = load_config(config_path, validate=True)
            self.assertEqual(loaded_config["agent"]["agent_type"], "adaptive", "Loaded config should match saved config")
        except ValueError:
            self.fail("Valid configuration should not raise ValueError")
        
        # Create an invalid configuration
        invalid_config = self.valid_config.copy()
        invalid_config["agent"]["learning_rate"] = 2.0  # Out of range
        
        # Save the invalid configuration
        invalid_path = os.path.join(self.temp_dir.name, "invalid_config.yaml")
        save_config(invalid_config, invalid_path, validate=False)  # Skip validation on save
        
        # Try to load with validation
        with self.assertRaises(ValueError):
            load_config(invalid_path, validate=True)


if __name__ == "__main__":
    unittest.main() 