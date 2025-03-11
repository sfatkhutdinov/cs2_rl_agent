import logging
import os
from typing import Optional

def setup_logger(
    name: str, 
    log_level: int = logging.INFO, 
    log_file: Optional[str] = None, 
    console_output: bool = True
) -> logging.Logger:
    """
    Set up a logger with file and/or console output.
    
    Args:
        name: Name of the logger
        log_level: Logging level (e.g., logging.INFO)
        log_file: Path to log file (None for no file output)
        console_output: Whether to also log to console
        
    Returns:
        Configured logger instance
    """
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(log_level)
    logger.propagate = False  # Don't propagate to parent loggers
    
    # Remove existing handlers if any
    if logger.handlers:
        logger.handlers.clear()
    
    # Create formatters
    detailed_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    simple_formatter = logging.Formatter('%(levelname)s - %(message)s')
    
    # Add file handler if log_file is specified
    if log_file:
        # Make sure directory exists
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(log_level)
        file_handler.setFormatter(detailed_formatter)
        logger.addHandler(file_handler)
    
    # Add console handler if requested
    if console_output:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(log_level)
        console_handler.setFormatter(simple_formatter)
        logger.addHandler(console_handler)
    
    return logger 