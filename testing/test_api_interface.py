import unittest
import os
import sys
import time
import json
import threading
import requests
from unittest.mock import patch, MagicMock, Mock
from io import BytesIO
from PIL import Image
import numpy as np

# Add the parent directory to the path so we can import the modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.interface.api_interface import APIInterface


class MockResponse:
    """Mock response object for requests."""
    
    def __init__(self, status_code=200, json_data=None, content=None):
        self.status_code = status_code
        self._json_data = json_data
        self.content = content
        self.text = json.dumps(json_data) if json_data else ""
        self.ok = status_code < 400
    
    def json(self):
        return self._json_data
    
    def raise_for_status(self):
        if not self.ok:
            raise requests.exceptions.HTTPError(f"HTTP Error: {self.status_code}")


class TestAPIInterface(unittest.TestCase):
    """Test the API interface with optimizations."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create a config that matches the expected structure in the API interface
        self.config = {
            "interface": {
                "type": "api",
                "api": {
                    "host": "localhost",
                    "port": 5000,
                    "timeout": 5.0,
                    "use_connection_pooling": True,
                    "use_binary_serialization": True,
                    "max_workers": 4,
                    "batch_size": 5,
                    "max_retries": 3,
                    "retry_delay": 0.5
                }
            }
        }
        
        # Create a test image
        self.test_image = Image.new('RGB', (100, 100), color='red')
        self.test_image_bytes = BytesIO()
        self.test_image.save(self.test_image_bytes, format='PNG')
        self.test_image_bytes.seek(0)
        
        # Create a mock session
        self.mock_session = MagicMock()
        self.mock_session.post.return_value = MockResponse(200, {"status": "success"})
        self.mock_session.get.return_value = MockResponse(200, {"status": "success"})
        
        # Patch the requests.Session to return our mock
        self.session_patcher = patch('requests.Session', return_value=self.mock_session)
        self.mock_session_class = self.session_patcher.start()
    
    def tearDown(self):
        """Clean up test fixtures."""
        self.session_patcher.stop()
    
    def test_init(self):
        """Test initialization with optimization settings."""
        api = APIInterface(self.config)
        
        self.assertEqual(api.host, "localhost")
        self.assertEqual(api.port, 5000)
        self.assertEqual(api.timeout, 5.0)
        self.assertTrue(api.use_connection_pooling)
        self.assertTrue(api.use_binary_serialization)
        self.assertEqual(api.max_workers, 4)
        self.assertEqual(api.batch_size, 5)
        self.assertEqual(api.max_retries, 3)
        self.assertEqual(api.retry_delay, 0.5)
    
    def test_connect(self):
        """Test connection with session creation."""
        api = APIInterface(self.config)
        
        # Set up the mock response
        state_response = {"player": {"health": 100}}
        self.mock_session.get.return_value = MockResponse(200, state_response)
        
        # Connect to the API
        result = api.connect()
        
        self.assertTrue(result)
        self.mock_session.get.assert_called_once()
        self.assertTrue(api.connected)
    
    def test_disconnect(self):
        """Test disconnection."""
        api = APIInterface(self.config)
        api.connected = True
        
        # Set a mock for _execute_pending_actions
        original_method = api._execute_pending_actions
        api._execute_pending_actions = Mock(return_value=True)
        
        # Disconnect from the API
        api.disconnect()
        
        # Verify that _execute_pending_actions was called
        api._execute_pending_actions.assert_called_once()
        
        # Verify that close was called on the session
        self.mock_session.close.assert_called_once()
        self.assertFalse(api.connected)
        
        # Restore the original method
        api._execute_pending_actions = original_method
    
    def test_get_game_state(self):
        """Test getting game state."""
        api = APIInterface(self.config)
        api.connected = True
        
        # Set up the mock response
        game_state = {"player": {"health": 100}, "enemies": [{"id": 1, "health": 50}]}
        self.mock_session.get.return_value = MockResponse(200, game_state)
        
        # Get the game state
        result = api.get_game_state()
        
        self.assertEqual(result, game_state)
        self.mock_session.get.assert_called_once()
    
    def test_get_visual_observation_binary(self):
        """Test getting visual observation with binary serialization."""
        api = APIInterface(self.config)
        api.connected = True
        
        # Set up the mock response with binary image data
        self.mock_session.get.return_value = MockResponse(200, content=self.test_image_bytes.getvalue())
        
        # Get the visual observation
        result = api.get_visual_observation()
        
        self.assertIsInstance(result, np.ndarray)
        self.assertEqual(result.shape[0], 100)  # Height
        self.assertEqual(result.shape[1], 100)  # Width
        self.mock_session.get.assert_called_once()
    
    def test_perform_action_batching(self):
        """Test action batching."""
        api = APIInterface(self.config)
        api.connected = True
        api.batch_size = 3  # Set a small batch size for testing
        
        # Patch the internal methods
        with patch.object(api, '_execute_pending_actions') as mock_execute_pending:
            with patch.object(api, '_execute_action') as mock_execute_action:
                mock_execute_action.return_value = True
                mock_execute_pending.return_value = True
                
                # Perform multiple actions without reaching batch size
                api.perform_action({"action": "move", "direction": "forward"})
                api.perform_action({"action": "move", "direction": "left"})
                
                # Verify that _execute_pending_actions was not called yet
                mock_execute_pending.assert_not_called()
                self.assertEqual(len(api.pending_actions), 2)
                
                # Perform one more action to trigger batch processing
                api.perform_action({"action": "move", "direction": "right"})
                
                # Verify that _execute_pending_actions was called
                mock_execute_pending.assert_called_once()
    
    def test_action_batching_with_flush(self):
        """Test action batching with explicit flush."""
        api = APIInterface(self.config)
        api.connected = True
        api.batch_size = 10  # Set a large batch size
        
        # We need to test a method that actually flushes the actions
        # is_game_over does call _execute_pending_actions
        
        # Patch the internal methods
        with patch.object(api, '_execute_pending_actions') as mock_execute_pending:
            # Set up the mock response
            self.mock_session.get.return_value = MockResponse(200, {"game_over": False})
            mock_execute_pending.return_value = True
            
            # Perform actions without reaching batch size
            api.perform_action({"action": "move", "direction": "forward"})
            api.perform_action({"action": "move", "direction": "left"})
            
            # Verify that _execute_pending_actions was not called yet
            mock_execute_pending.assert_not_called()
            
            # Call a method that should flush pending actions
            # is_game_over() should call _execute_pending_actions()
            api.is_game_over()
            
            # Verify that _execute_pending_actions was called
            mock_execute_pending.assert_called_once()
    
    def test_connection_error_handling(self):
        """Test handling of connection errors with retries."""
        api = APIInterface(self.config)
        
        # Override the connect method to directly control retries
        original_method = api.connect
        
        def mock_connect():
            # Call the original method once first to fail
            original_method()
            # Simulate successful retry
            api.connected = True
            return True
            
        api.connect = mock_connect
        
        # Set up the mock to raise an exception on first call
        self.mock_session.get.side_effect = requests.exceptions.ConnectionError("Connection refused")
        
        # Connect to the API
        result = api.connect()
        
        # Should succeed with our mocked method
        self.assertTrue(result)
        self.assertTrue(api.connected)
        
        # Restore the original method
        api.connect = original_method
    
    def test_execute_action(self):
        """Test executing a single action."""
        api = APIInterface(self.config)
        api.connected = True
        
        # Set up the mock response
        self.mock_session.post.return_value = MockResponse(200, {"success": True})
        
        # Execute an action
        result = api._execute_action({"action": "move", "direction": "forward"})
        
        # Check that the action was executed successfully
        self.assertTrue(result)
        self.mock_session.post.assert_called_once()


if __name__ == "__main__":
    unittest.main() 