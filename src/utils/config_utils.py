import os
import yaml
import logging
from typing import Dict, Any, Optional, List, Tuple

# Import the validator
from src.utils.config_validator import validate_config, print_validation_report

logger = logging.getLogger("ConfigUtils")

def load_config(config_path: str, validate: bool = True) -> Dict[str, Any]:
    """
    Load configuration from a YAML file.
    
    Args:
        config_path: Path to the YAML configuration file
        validate: Whether to validate the configuration against the schema
        
    Returns:
        Dictionary containing configuration parameters
        
    Raises:
        FileNotFoundError: If the configuration file doesn't exist
        ValueError: If the configuration is invalid and validation is enabled
    """
    # Normalize path for Windows compatibility
    config_path = os.path.normpath(config_path)
    
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    
    # Validate configuration if requested
    if validate:
        errors = validate_config(config)
        if errors:
            print_validation_report(errors, config_path)
            raise ValueError(f"Configuration validation failed for {config_path}")
    
    return config


def override_config(base_config: Dict[str, Any], override_dict: Optional[Dict[str, Any]] = None, validate: bool = True) -> Dict[str, Any]:
    """
    Override base configuration with values from override_dict.
    
    Args:
        base_config: Base configuration dictionary
        override_dict: Dictionary containing override values
        validate: Whether to validate the resulting configuration
        
    Returns:
        Updated configuration dictionary
        
    Raises:
        ValueError: If the resulting configuration is invalid and validation is enabled
    """
    if override_dict is None:
        return base_config
    
    result = base_config.copy()
    
    def _update_dict_recursive(d, u):
        for k, v in u.items():
            if isinstance(v, dict) and k in d and isinstance(d[k], dict):
                d[k] = _update_dict_recursive(d[k], v)
            else:
                d[k] = v
        return d
    
    result = _update_dict_recursive(result, override_dict)
    
    # Validate the resulting configuration if requested
    if validate:
        errors = validate_config(result)
        if errors:
            logger.warning("Configuration validation failed after override:")
            for error in errors:
                logger.warning(f"  - {error}")
            raise ValueError("Configuration validation failed after override")
    
    return result


def get_full_path(relative_path: str, config: Dict[str, Any]) -> str:
    """
    Convert a relative path to a full path based on the configuration.
    
    Args:
        relative_path: Relative path
        config: Configuration dictionary containing path information
        
    Returns:
        Full path
    """
    # Get the directory where the script is located
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    # Normalize path for Windows compatibility
    return os.path.normpath(os.path.join(base_dir, relative_path))


def save_config(config: Dict[str, Any], save_path: str, validate: bool = True) -> None:
    """
    Save configuration to a YAML file.
    
    Args:
        config: Configuration dictionary
        save_path: Path where to save the configuration
        validate: Whether to validate the configuration before saving
        
    Raises:
        ValueError: If the configuration is invalid and validation is enabled
    """
    # Validate the configuration before saving if requested
    if validate:
        errors = validate_config(config)
        if errors:
            print_validation_report(errors, save_path)
            raise ValueError(f"Configuration validation failed for {save_path}")
    
    # Normalize path for Windows compatibility
    save_path = os.path.normpath(save_path)
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    with open(save_path, 'w') as file:
        yaml.dump(config, file, default_flow_style=False)


def merge_configs(configs: List[Dict[str, Any]], validate: bool = True) -> Dict[str, Any]:
    """
    Merge multiple configurations into one.
    
    Args:
        configs: List of configuration dictionaries to merge
        validate: Whether to validate the resulting configuration
        
    Returns:
        Merged configuration dictionary
        
    Raises:
        ValueError: If the resulting configuration is invalid and validation is enabled
    """
    if not configs:
        return {}
    
    result = configs[0].copy()
    
    for config in configs[1:]:
        result = override_config(result, config, validate=False)
    
    # Validate the resulting configuration if requested
    if validate:
        errors = validate_config(result)
        if errors:
            logger.warning("Configuration validation failed after merge:")
            for error in errors:
                logger.warning(f"  - {error}")
            raise ValueError("Configuration validation failed after merge")
    
    return result 