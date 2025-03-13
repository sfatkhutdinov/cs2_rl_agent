import os
import json
import logging
from typing import Dict, Any, List, Optional, Union
import jsonschema
from jsonschema import validate, ValidationError

logger = logging.getLogger("ConfigValidator")

# Base schemas for different configuration types
AGENT_SCHEMA = {
    "type": "object",
    "required": ["agent_type"],
    "properties": {
        "agent_type": {"type": "string", "enum": ["adaptive", "strategic", "discovery", "autonomous"]},
        "learning_rate": {"type": "number", "minimum": 0.0, "maximum": 1.0},
        "gamma": {"type": "number", "minimum": 0.0, "maximum": 1.0},
        "n_steps": {"type": "integer", "minimum": 1},
        "batch_size": {"type": "integer", "minimum": 1},
        "buffer_size": {"type": "integer", "minimum": 1}
    }
}

ENVIRONMENT_SCHEMA = {
    "type": "object",
    "properties": {
        "action_delay": {"type": "number", "minimum": 0.0},
        "max_episode_steps": {"type": "integer", "minimum": 1},
        "reward_scales": {
            "type": "object",
            "additionalProperties": {"type": "number"}
        },
        "use_vision_model": {"type": "boolean"}
    }
}

INTERFACE_SCHEMA = {
    "type": "object",
    "required": ["type"],
    "properties": {
        "type": {"type": "string", "enum": ["api", "vision", "auto_vision", "ollama_vision"]},
        "api": {
            "type": "object",
            "properties": {
                "host": {"type": "string"},
                "port": {"type": "integer", "minimum": 1, "maximum": 65535},
                "timeout": {"type": "number", "minimum": 0.1},
                "max_retries": {"type": "integer", "minimum": 0},
                "retry_delay": {"type": "number", "minimum": 0.0}
            },
            "required": ["host", "port"]
        },
        "vision": {
            "type": "object",
            "properties": {
                "tesseract_path": {"type": "string"},
                "confidence_threshold": {"type": "number", "minimum": 0.0, "maximum": 1.0}
            }
        }
    }
}

# Combined schema for the full configuration
CONFIG_SCHEMA = {
    "type": "object",
    "properties": {
        "agent": AGENT_SCHEMA,
        "environment": ENVIRONMENT_SCHEMA,
        "interface": INTERFACE_SCHEMA,
        "logging": {
            "type": "object",
            "properties": {
                "level": {"type": "string", "enum": ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]},
                "log_mode_switches": {"type": "boolean"},
                "log_metrics": {"type": "boolean"},
                "use_tensorboard": {"type": "boolean"},
                "checkpoint_frequency": {"type": "integer", "minimum": 1}
            }
        },
        "training": {
            "type": "object",
            "properties": {
                "total_timesteps": {"type": "integer", "minimum": 1},
                "learning_rate": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                "batch_size": {"type": "integer", "minimum": 1},
                "gamma": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                "n_steps": {"type": "integer", "minimum": 1},
                "starting_mode": {"type": "string"}
            }
        }
    }
}


def validate_config(config: Dict[str, Any], schema: Dict[str, Any] = None) -> List[str]:
    """
    Validate configuration against a JSON schema.
    
    Args:
        config: Configuration dictionary to validate
        schema: JSON schema to validate against (uses CONFIG_SCHEMA if None)
        
    Returns:
        List of validation error messages (empty if validation passed)
    """
    if schema is None:
        schema = CONFIG_SCHEMA
    
    errors = []
    
    try:
        validate(instance=config, schema=schema)
    except ValidationError as e:
        errors.append(f"Validation error: {e.message}")
        # Log path to the error
        error_path = '/'.join(str(p) for p in e.path)
        errors.append(f"Error path: {error_path}")
    
    return errors


def validate_file(config_path: str) -> List[str]:
    """
    Validate a configuration file against the schema.
    
    Args:
        config_path: Path to the configuration file (YAML)
        
    Returns:
        List of validation error messages (empty if validation passed)
    """
    from src.utils.config_utils import load_config
    
    try:
        config = load_config(config_path)
        return validate_config(config)
    except Exception as e:
        return [f"Error loading config file: {str(e)}"]


def print_validation_report(errors: List[str], config_name: str) -> None:
    """
    Print a validation report.
    
    Args:
        errors: List of validation error messages
        config_name: Name of the configuration being validated
    """
    if not errors:
        logger.info(f"✅ Configuration '{config_name}' is valid")
    else:
        logger.error(f"❌ Configuration '{config_name}' has validation errors:")
        for error in errors:
            logger.error(f"  - {error}")


if __name__ == "__main__":
    # Simple CLI for validating configuration files
    import argparse
    import sys
    
    logging.basicConfig(level=logging.INFO)
    
    parser = argparse.ArgumentParser(description="Validate configuration files")
    parser.add_argument("config_file", help="Path to the configuration file to validate")
    args = parser.parse_args()
    
    errors = validate_file(args.config_file)
    print_validation_report(errors, args.config_file)
    
    sys.exit(1 if errors else 0) 