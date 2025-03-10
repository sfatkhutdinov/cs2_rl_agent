from abc import ABC, abstractmethod
from typing import Dict, Any, Tuple, List, Optional
import numpy as np


class BaseInterface(ABC):
    """
    Abstract base class for game interfaces.
    
    This class defines the interface for interacting with Cities: Skylines 2.
    Concrete implementations will handle the actual communication with the game,
    either through a mod API or through computer vision.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the interface.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.connected = False
    
    @abstractmethod
    def connect(self) -> bool:
        """
        Connect to the game.
        
        Returns:
            True if connection was successful, False otherwise
        """
        pass
    
    @abstractmethod
    def disconnect(self) -> None:
        """Disconnect from the game."""
        pass
    
    @abstractmethod
    def get_game_state(self) -> Dict[str, Any]:
        """
        Get the current game state.
        
        Returns:
            Dictionary containing the current game state
        """
        pass
    
    @abstractmethod
    def get_visual_observation(self) -> np.ndarray:
        """
        Get visual observation (screenshot) from the game.
        
        Returns:
            NumPy array containing the screenshot
        """
        pass
    
    @abstractmethod
    def get_metrics(self) -> Dict[str, float]:
        """
        Get current game metrics (population, happiness, etc.)
        
        Returns:
            Dictionary containing game metrics
        """
        pass
    
    @abstractmethod
    def perform_action(self, action: Dict[str, Any]) -> bool:
        """
        Perform an action in the game.
        
        Args:
            action: Dictionary describing the action to perform
            
        Returns:
            True if the action was performed successfully, False otherwise
        """
        pass
    
    @abstractmethod
    def set_game_speed(self, speed: int) -> bool:
        """
        Set the game speed.
        
        Args:
            speed: Game speed (1-3)
            
        Returns:
            True if the speed was set successfully, False otherwise
        """
        pass
    
    @abstractmethod
    def is_game_over(self) -> bool:
        """
        Check if the game is over (e.g., city went bankrupt).
        
        Returns:
            True if the game is over, False otherwise
        """
        pass
    
    @abstractmethod
    def reset_game(self) -> bool:
        """
        Reset the game to its initial state.
        
        Returns:
            True if the game was reset successfully, False otherwise
        """
        pass 