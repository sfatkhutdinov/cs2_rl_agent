import time
import json
import logging
import requests
from typing import Dict, Any, Optional
import numpy as np
from .base_interface import BaseInterface


class APIInterface(BaseInterface):
    """
    API-based interface for interacting with Cities: Skylines 2.
    
    This interface communicates with the game through a bridge mod that
    exposes a REST API for game interaction.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the API interface.
        
        Args:
            config: Configuration dictionary
        """
        super().__init__(config)
        self.logger = logging.getLogger("APIInterface")
        
        # API configuration
        self.host = config["interface"]["api"]["host"]
        self.port = config["interface"]["api"]["port"]
        self.timeout = config["interface"]["api"]["timeout"]
        self.base_url = f"http://{self.host}:{self.port}"
        
        # State tracking
        self.last_state = {}
        self.last_metrics = {}
        self.last_screenshot = None
        self.game_speed = 1
    
    def connect(self) -> bool:
        """
        Connect to the game via the bridge mod API.
        
        Returns:
            True if connection was successful, False otherwise
        """
        try:
            # Try to get the game state to verify connection
            response = requests.get(f"{self.base_url}/state", timeout=self.timeout)
            
            if response.status_code == 200:
                self.connected = True
                self.logger.info("Successfully connected to Cities: Skylines 2 via bridge mod API.")
                
                # Store initial state
                self.last_state = response.json()
                if "metrics" in self.last_state:
                    self.last_metrics = self.last_state["metrics"]
                
                return True
            else:
                self.logger.error(f"Failed to connect: API returned status code {response.status_code}")
                return False
                
        except requests.exceptions.RequestException as e:
            self.logger.error(f"Failed to connect: {str(e)}")
            return False
    
    def disconnect(self) -> None:
        """Disconnect from the game."""
        self.connected = False
        self.logger.info("Disconnected from the game.")
    
    def get_game_state(self) -> Dict[str, Any]:
        """
        Get the current game state.
        
        Returns:
            Dictionary containing the current game state
        """
        if not self.connected:
            self.logger.warning("Not connected to the game.")
            return {}
        
        try:
            response = requests.get(f"{self.base_url}/state", timeout=self.timeout)
            
            if response.status_code == 200:
                self.last_state = response.json()
                if "metrics" in self.last_state:
                    self.last_metrics = self.last_state["metrics"]
                
                # Add visual observation if needed
                if self.config["environment"]["observation_space"]["include_visual"]:
                    # Note: The bridge mod doesn't provide visual data yet
                    # This is a placeholder for future implementation
                    self.last_state["visual_observation"] = self.get_visual_observation()
                
                return self.last_state
            else:
                self.logger.warning(f"Failed to get game state: API returned status code {response.status_code}")
                return self.last_state
                
        except requests.exceptions.RequestException as e:
            self.logger.error(f"Failed to get game state: {str(e)}")
            return self.last_state
    
    def get_visual_observation(self) -> np.ndarray:
        """
        Get visual observation (screenshot) from the game.
        
        Returns:
            NumPy array containing the screenshot
        """
        if not self.connected:
            self.logger.warning("Not connected to the game.")
            return np.zeros((84, 84, 3), dtype=np.uint8)
        
        # Note: The bridge mod doesn't provide visual data yet
        # This is a placeholder for future implementation
        # For now, return a blank image of the configured size
        image_size = self.config["environment"]["observation_space"]["image_size"]
        if self.config["environment"]["observation_space"]["grayscale"]:
            return np.zeros((image_size[0], image_size[1], 1), dtype=np.uint8)
        else:
            return np.zeros((image_size[0], image_size[1], 3), dtype=np.uint8)
    
    def get_metrics(self) -> Dict[str, float]:
        """
        Get current game metrics (population, happiness, etc.)
        
        Returns:
            Dictionary containing game metrics
        """
        if not self.connected:
            self.logger.warning("Not connected to the game.")
            return {}
        
        # The metrics are already fetched in get_game_state()
        return self.last_metrics
    
    def perform_action(self, action: Dict[str, Any]) -> bool:
        """
        Perform an action in the game.
        
        Args:
            action: Dictionary describing the action to perform
            
        Returns:
            True if the action was performed successfully, False otherwise
        """
        if not self.connected:
            self.logger.warning("Not connected to the game.")
            return False
        
        try:
            response = requests.post(
                f"{self.base_url}/action",
                json=action,
                timeout=self.timeout
            )
            
            if response.status_code == 200:
                result = response.json()
                return result.get("success", False)
            else:
                self.logger.warning(f"Failed to perform action: API returned status code {response.status_code}")
                return False
                
        except requests.exceptions.RequestException as e:
            self.logger.error(f"Failed to perform action: {str(e)}")
            return False
    
    def set_game_speed(self, speed: int) -> bool:
        """
        Set the game speed.
        
        Args:
            speed: Game speed (1-3)
            
        Returns:
            True if the speed was set successfully, False otherwise
        """
        if not self.connected:
            self.logger.warning("Not connected to the game.")
            return False
        
        if speed < 1 or speed > 3:
            self.logger.warning(f"Invalid game speed: {speed}. Must be between 1 and 3.")
            return False
        
        try:
            # Send a speed change action
            action = {
                "type": "game_control",
                "control_type": "speed",
                "value": speed
            }
            
            success = self.perform_action(action)
            if success:
                self.game_speed = speed
            
            return success
            
        except Exception as e:
            self.logger.error(f"Failed to set game speed: {str(e)}")
            return False
    
    def is_game_over(self) -> bool:
        """
        Check if the game is over (e.g., city went bankrupt).
        
        Returns:
            True if the game is over, False otherwise
        """
        if not self.connected:
            self.logger.warning("Not connected to the game.")
            return True
        
        # Check for bankruptcy - this is a simple implementation that assumes
        # bankruptcy is indicated by a negative budget value
        metrics = self.get_metrics()
        if metrics.get("budget_balance", 0) < 0:
            return True
        
        return False
    
    def reset_game(self) -> bool:
        """
        Reset the game to its initial state.
        
        Returns:
            True if the game was reset successfully, False otherwise
        """
        if not self.connected:
            self.logger.warning("Not connected to the game.")
            return False
        
        try:
            # Send a reset action
            action = {
                "type": "game_control",
                "control_type": "reset"
            }
            
            return self.perform_action(action)
            
        except Exception as e:
            self.logger.error(f"Failed to reset game: {str(e)}")
            return False 