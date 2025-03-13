import time
import json
import logging
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import base64
from io import BytesIO
from PIL import Image
from typing import Dict, Any, Optional, List, Tuple, Union
import numpy as np
from concurrent.futures import ThreadPoolExecutor
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
        self.timeout = config["interface"]["api"].get("timeout", 10.0)
        self.max_retries = config["interface"]["api"].get("max_retries", 3)
        self.retry_delay = config["interface"]["api"].get("retry_delay", 0.5)
        self.base_url = f"http://{self.host}:{self.port}"
        
        # Performance optimization settings
        self.use_connection_pooling = config["interface"]["api"].get("use_connection_pooling", True)
        self.use_binary_serialization = config["interface"]["api"].get("use_binary_serialization", True)
        self.max_workers = config["interface"]["api"].get("max_workers", 4)
        self.batch_size = config["interface"]["api"].get("batch_size", 10)
        
        # State tracking
        self.last_state = {}
        self.last_metrics = {}
        self.last_screenshot = None
        self.game_speed = 1
        
        # Pending actions for batching
        self.pending_actions = []
        self.action_executor = ThreadPoolExecutor(max_workers=self.max_workers)
        
        # Set up connection pooling with retry logic
        if self.use_connection_pooling:
            self.session = self._create_session_with_retries()
        else:
            self.session = requests.Session()
    
    def _create_session_with_retries(self) -> requests.Session:
        """
        Create a requests session with connection pooling and retry logic.
        
        Returns:
            Configured requests session
        """
        session = requests.Session()
        
        # Configure retry strategy
        retry_strategy = Retry(
            total=self.max_retries,
            backoff_factor=self.retry_delay,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["GET", "POST"]
        )
        
        # Configure connection pooling
        adapter = HTTPAdapter(
            max_retries=retry_strategy,
            pool_connections=self.max_workers,
            pool_maxsize=self.max_workers * 2
        )
        
        # Mount the adapter for both HTTP and HTTPS
        session.mount("http://", adapter)
        session.mount("https://", adapter)
        
        return session
    
    def connect(self) -> bool:
        """
        Connect to the game via the bridge mod API.
        
        Returns:
            True if connection was successful, False otherwise
        """
        try:
            # Try to get the game state to verify connection
            response = self.session.get(f"{self.base_url}/state", timeout=self.timeout)
            
            if response.status_code == 200:
                self.last_state = response.json()
                self.connected = True
                self.logger.info(f"Connected to bridge mod at {self.base_url}")
                return True
            else:
                self.logger.error(f"Failed to connect: API returned status code {response.status_code}")
                return False
                
        except requests.exceptions.RequestException as e:
            self.logger.error(f"Failed to connect: {str(e)}")
            return False
    
    def disconnect(self) -> None:
        """Disconnect from the game."""
        # Execute any pending batched actions before disconnecting
        self._execute_pending_actions()
        
        self.connected = False
        self.session.close()
        self.logger.info("Disconnected from bridge mod")
    
    def get_game_state(self) -> Dict[str, Any]:
        """
        Get the current game state.
        
        Returns:
            Dictionary containing the current game state
        """
        if not self.connected:
            self.logger.warning("Not connected to the game.")
            return self.last_state
        
        try:
            response = self.session.get(f"{self.base_url}/state", timeout=self.timeout)
            
            if response.status_code == 200:
                self.last_state = response.json()
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
            return self.last_screenshot if self.last_screenshot is not None else np.zeros((720, 1280, 3), dtype=np.uint8)
        
        try:
            # Use binary serialization if enabled
            if self.use_binary_serialization:
                headers = {"Accept": "application/octet-stream"}
                response = self.session.get(f"{self.base_url}/screenshot", headers=headers, timeout=self.timeout)
                
                if response.status_code == 200:
                    # Convert binary data directly to numpy array
                    img = Image.open(BytesIO(response.content))
                    self.last_screenshot = np.array(img)
                    return self.last_screenshot
            else:
                # Use JSON serialization (base64)
                response = self.session.get(f"{self.base_url}/screenshot", timeout=self.timeout)
                
                if response.status_code == 200:
                    # Decode base64 image data
                    img_data = response.json().get("screenshot", "")
                    if img_data:
                        img_bytes = base64.b64decode(img_data)
                        img = Image.open(BytesIO(img_bytes))
                        self.last_screenshot = np.array(img)
                        return self.last_screenshot
            
            self.logger.warning(f"Failed to get screenshot: API returned status code {response.status_code}")
            return self.last_screenshot if self.last_screenshot is not None else np.zeros((720, 1280, 3), dtype=np.uint8)
                
        except requests.exceptions.RequestException as e:
            self.logger.error(f"Failed to get screenshot: {str(e)}")
            return self.last_screenshot if self.last_screenshot is not None else np.zeros((720, 1280, 3), dtype=np.uint8)
    
    def get_metrics(self) -> Dict[str, float]:
        """
        Get current game metrics (population, happiness, etc.)
        
        Returns:
            Dictionary containing game metrics
        """
        if not self.connected:
            self.logger.warning("Not connected to the game.")
            return self.last_metrics
        
        try:
            response = self.session.get(f"{self.base_url}/metrics", timeout=self.timeout)
            
            if response.status_code == 200:
                self.last_metrics = response.json()
                return self.last_metrics
            else:
                self.logger.warning(f"Failed to get metrics: API returned status code {response.status_code}")
                return self.last_metrics
                
        except requests.exceptions.RequestException as e:
            self.logger.error(f"Failed to get metrics: {str(e)}")
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
        
        # Add to pending actions if batching is enabled and it's not a critical action
        if self.batch_size > 1 and action.get("type") != "game_control":
            self.pending_actions.append(action)
            
            # Execute batch if we've reached the batch size
            if len(self.pending_actions) >= self.batch_size:
                return self._execute_pending_actions()
            
            # Assume success for now, actual result will be determined when batch is executed
            return True
        
        # Execute immediately for non-batched actions
        return self._execute_action(action)
    
    def _execute_action(self, action: Dict[str, Any]) -> bool:
        """
        Execute a single action.
        
        Args:
            action: Dictionary describing the action to perform
            
        Returns:
            True if the action was performed successfully, False otherwise
        """
        try:
            response = self.session.post(
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
    
    def _execute_pending_actions(self) -> bool:
        """
        Execute all pending actions in a batch.
        
        Returns:
            True if all actions were performed successfully, False otherwise
        """
        if not self.pending_actions:
            return True
        
        try:
            # Create batch request
            batch_request = {
                "actions": self.pending_actions
            }
            
            response = self.session.post(
                f"{self.base_url}/batch_actions",
                json=batch_request,
                timeout=self.timeout * 2  # Longer timeout for batch requests
            )
            
            # Clear pending actions regardless of result
            pending_count = len(self.pending_actions)
            self.pending_actions = []
            
            if response.status_code == 200:
                results = response.json().get("results", [])
                success_count = sum(1 for r in results if r.get("success", False))
                self.logger.debug(f"Batch execution: {success_count}/{pending_count} actions succeeded")
                return success_count == pending_count
            else:
                self.logger.warning(f"Failed to execute batch: API returned status code {response.status_code}")
                return False
                
        except requests.exceptions.RequestException as e:
            self.logger.error(f"Failed to execute batch: {str(e)}")
            self.pending_actions = []  # Clear pending actions on error
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
        
        # Execute any pending batched actions before changing game speed
        self._execute_pending_actions()
        
        try:
            # Send a speed change action
            action = {
                "type": "game_control",
                "control_type": "speed",
                "value": speed
            }
            
            success = self._execute_action(action)
            
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
            return False
        
        # Execute any pending batched actions before checking game state
        self._execute_pending_actions()
        
        game_state = self.get_game_state()
        return game_state.get("game_over", False)
    
    def reset_game(self) -> bool:
        """
        Reset the game to its initial state.
        
        Returns:
            True if the game was reset successfully, False otherwise
        """
        if not self.connected:
            self.logger.warning("Not connected to the game.")
            return False
        
        # Clear any pending batched actions
        self.pending_actions = []
        
        try:
            action = {
                "type": "game_control",
                "control_type": "reset"
            }
            
            return self._execute_action(action)
            
        except Exception as e:
            self.logger.error(f"Failed to reset game: {str(e)}")
            return False 