import os
import yaml
from typing import Dict, Any, Optional


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load configuration from a YAML file.
    
    Args:
        config_path: Path to the YAML configuration file
        
    Returns:
        Dictionary containing configuration parameters
        
    Raises:
        FileNotFoundError: If the configuration file doesn't exist
    """
    # Normalize path for Windows compatibility
    config_path = os.path.normpath(config_path)
    
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    
    return config


def override_config(base_config: Dict[str, Any], override_dict: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Override base configuration with values from override_dict.
    
    Args:
        base_config: Base configuration dictionary
        override_dict: Dictionary containing override values
        
    Returns:
        Updated configuration dictionary
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
    
    return _update_dict_recursive(result, override_dict)


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


def save_config(config: Dict[str, Any], save_path: str) -> None:
    """
    Save configuration to a YAML file.
    
    Args:
        config: Configuration dictionary
        save_path: Path where to save the configuration
    """
    # Normalize path for Windows compatibility
    save_path = os.path.normpath(save_path)
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    with open(save_path, 'w') as file:
        yaml.dump(config, file, default_flow_style=False) 