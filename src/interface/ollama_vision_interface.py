import requests
import base64
from typing import Dict, Any, List, Tuple, Optional
import numpy as np
import cv2
import time
import logging
import os
from io import BytesIO
from PIL import Image
import json
import datetime
import pyautogui
import random
import hashlib
import io
from collections import OrderedDict
import threading
from dataclasses import dataclass, field
import PIL.ImageGrab
import mss
import mss.tools

from .auto_vision_interface import AutoVisionInterface

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class ResponseCache:
    """Cache for Ollama responses to reduce API calls"""
    max_size: int = 10
    cache: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    cache_keys: List[str] = field(default_factory=list)
    
    def get_key(self, prompt: str, image: np.ndarray) -> str:
        """Generate a unique key for the prompt and image
        
        Args:
            prompt: The text prompt
            image: The image data
            
        Returns:
            A unique key for cache lookup
        """
        # Use a simple hash of the prompt and the first/last few pixels of the image
        # This avoids storing full images while still providing a reasonable uniqueness
        prompt_hash = hash(prompt)
        
        # Sample a few pixels from the image for fingerprinting
        if image.size > 0:
            # Take 10 pixels from different parts of the image
            h, w = image.shape[:2]
            pixels = []
            for i in range(5):
                y = int((i * h) / 5)
                x = int((i * w) / 5)
                if y < h and x < w:
                    pixel = image[y, x]
                    pixels.extend(pixel)
            
            image_hash = hash(tuple(pixels))
        else:
            image_hash = 0
            
        return f"{prompt_hash}_{image_hash}"
    
    def get(self, prompt: str, image: np.ndarray) -> Optional[Dict[str, Any]]:
        """Get a cached response if available
        
        Args:
            prompt: The text prompt
            image: The image data
            
        Returns:
            Cached response or None
        """
        key = self.get_key(prompt, image)
        if key in self.cache:
            logger.debug(f"Cache hit for key: {key}")
            return self.cache[key]
        return None
    
    def add(self, prompt: str, image: np.ndarray, response: Dict[str, Any]):
        """Add a response to the cache
        
        Args:
            prompt: The text prompt
            image: The image data
            response: The API response to cache
        """
        key = self.get_key(prompt, image)
        
        # If cache is full, remove oldest entry
        if len(self.cache) >= self.max_size and key not in self.cache:
            oldest_key = self.cache_keys.pop(0)
            self.cache.pop(oldest_key, None)
            
        # Add new entry
        self.cache[key] = response
        
        # Add key to tracking list if it's new
        if key not in self.cache_keys:
            self.cache_keys.append(key)
        
        logger.debug(f"Added response to cache, size: {len(self.cache)}")

class OllamaVisionInterface(AutoVisionInterface):
    """
    Extended vision interface that uses Ollama's vision model (granite3.2-vision) 
    to provide enhanced game state understanding and UI detection capabilities.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the Ollama vision interface.
        
        Args:
            config: Configuration dictionary
        """
        super().__init__(config)
        self.logger = logging.getLogger("OllamaVisionInterface")
        
        # Ollama configuration
        self.ollama_config = config.get("ollama", {})
        self.ollama_url = self.ollama_config.get("url", "http://localhost:11434/api/generate")
        self.ollama_model = self.ollama_config.get("model", "granite3.2-vision:latest")
        self.max_tokens = self.ollama_config.get("max_tokens", 1000)
        self.temperature = self.ollama_config.get("temperature", 0.7)
        
        # Cache for Ollama responses to avoid too many API calls
        self.ollama_cache = {}
        self.cache_ttl = self.ollama_config.get("cache_ttl", 5)  # seconds
        self.cache_timestamp = 0
        
        # Screenshot tracking
        self.last_screenshot_time = 0
        self.screenshot_cooldown = self.ollama_config.get("screenshot_cooldown", 0.5)  # seconds
        self.screenshot_expiry = 5.0  # seconds before a screenshot is considered stale
        
        # Set up debugging logs directory
        self.debug_dir = os.path.join("logs", "vision_debug")
        os.makedirs(self.debug_dir, exist_ok=True)
        
        # Screenshot capture for debugging
        self.screenshot_frequency = self.ollama_config.get("screenshot_frequency", 20)  # 1 in X responses
        self.screenshot_counter = 0
        
        # Log formats for saving JSON responses
        self.json_log_file = os.path.join(self.debug_dir, "vision_responses.jsonl")
        
        # UI analysis help prompts
        self.ui_analysis_prompt = """Analyze this screenshot from Cities: Skylines 2 and return a JSON object.
        Replace the example values with actual observations from the image.
        Do not include any text outside the JSON structure.
        Do not keep placeholder values.

        {
            "ui_elements": [
                {
                    "name": "element_name",
                    "description": "what it does",
                    "position": "general location"
                }
            ],
            "game_state": "description of current game state",
            "metrics": {
                "population": "value if visible",
                "money": "value if visible"
            },
            "alerts": ["list of any visible alerts"],
            "available_actions": ["list of actions that would make sense now"],
            "suggestions": ["suggestions for what would help grow the city"]
        }

        Do not include any text before or after the JSON object. Ensure all values are properly quoted strings.
        """
        
        self.population_growth_prompt = """
        You are analyzing a screenshot from Cities: Skylines 2. Respond ONLY with a valid JSON object containing the following information:

        {
            "growth_limiters": ["list of factors limiting growth"],
            "recommended_actions": ["specific actions to take"],
            "issues_to_address": ["problems visible in the city"]
        }

        Do not include any text before or after the JSON object. Ensure all values are properly quoted strings.
        """
        
        self.logger.info(f"Ollama Vision Interface initialized with model: {self.ollama_model}")
        self.logger.info(f"Debug logs and screenshots will be saved to: {self.debug_dir}")
        
        # Set up caching
        self.response_cache = ResponseCache(max_size=self.ollama_config.get("vision_cache_size", 10))
        
        # Connection and retry settings
        self.timeout = self.ollama_config.get("timeout", 60)  # Increased default timeout to 60 seconds
        self.max_retries = self.ollama_config.get("max_retries", 3)
        self.retry_delay = self.ollama_config.get("retry_delay", 1.0)  # Initial retry delay in seconds
        
        # Thread lock for API requests
        self.api_lock = threading.Lock()
        
        # Check if Ollama is running
        self._verify_ollama_server()
        
    def _verify_ollama_server(self):
        """Verify that the Ollama server is running and responding"""
        try:
            self.logger.info(f"Verifying Ollama server at {self.ollama_url}...")
            health_url = self.ollama_url.replace('/api/generate', '/api/health')
            
            # Try a simple HTTP request to check if Ollama is running
            response = requests.get(health_url, timeout=5)
            
            if response.status_code == 200:
                self.logger.info("Ollama server is running.")
            else:
                self.logger.warning(f"Ollama server returned unexpected status: {response.status_code}")
                self.logger.warning("Vision features may not work correctly.")
        except requests.exceptions.RequestException as e:
            self.logger.warning(f"Could not connect to Ollama server: {str(e)}")
            self.logger.warning("Ensure Ollama is running before using vision features.")
            
    def query_ollama(self, prompt: str, image: np.ndarray, 
                   temperature: float = 0.7, timeout: Optional[int] = None) -> Dict[str, Any]:
        """
        Send a query to the Ollama API with an image.
        
        Args:
            prompt: Text prompt
            image: Image as numpy array
            temperature: Model temperature
            timeout: Timeout in seconds
            
        Returns:
            API response as a dictionary
        """
        if timeout is None:
            timeout = self.timeout
            
        # Get API lock to avoid multiple simultaneous requests
        with self.api_lock:
            # Early validation of parameters
            if not prompt or not isinstance(prompt, str):
                self.logger.error(f"Invalid prompt type: {type(prompt)}")
                return {"error": "Invalid prompt"}
            
            # Validate image before processing
            if image is None:
                self.logger.error("Cannot query Ollama with None image")
                return {"error": "Empty image provided"}
            
            # Handle case where image is a string (error in the logs)
            if isinstance(image, str):
                self.logger.error("Image is a string, not a numpy array")
                return {"error": "Invalid image type: expected numpy array, got string"}
            
            # Check if image is a numpy array with content
            if not isinstance(image, np.ndarray):
                self.logger.error(f"Cannot query Ollama with invalid image type: {type(image)}")
                return {"error": f"Invalid image type: expected numpy array, got {type(image)}"}
            
            if image.size == 0:
                self.logger.error("Cannot query Ollama with empty image")
                return {"error": "Empty image array provided"}
            
            # Encode the image as base64
            try:
                # Save image dimensions before conversion
                img_height, img_width = image.shape[:2]
                self.logger.info(f"Processing image with dimensions: {img_width}x{img_height}")
                
                # Reduce image size if it's too large (to avoid timeouts)
                max_dim = 1024
                if img_height > max_dim or img_width > max_dim:
                    self.logger.info(f"Resizing large image from {img_width}x{img_height}")
                    scale = max_dim / max(img_height, img_width)
                    new_width = int(img_width * scale)
                    new_height = int(img_height * scale)
                    image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)
                    self.logger.info(f"Resized image to {new_width}x{new_height}")
                
                # Ensure image has correct shape and dtype for PIL
                if len(image.shape) < 3 or image.shape[-1] not in [1, 3, 4]:
                    self.logger.error(f"Invalid image shape: {image.shape}")
                    return {"error": f"Invalid image shape: {image.shape}"}
                
                img = Image.fromarray(image.astype('uint8'))
                buffered = BytesIO()
                img.save(buffered, format="JPEG", quality=75)  # Lower quality to reduce size
                img_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")
                img_size_kb = len(buffered.getvalue()) / 1024
                self.logger.info(f"Encoded image size: {img_size_kb:.2f} KB")
                
                # If image is still too large, further reduce quality
                if img_size_kb > 1000:  # 1MB limit
                    self.logger.warning(f"Image too large ({img_size_kb:.2f} KB), reducing quality")
                    buffered = BytesIO()
                    img.save(buffered, format="JPEG", quality=50)  # Further reduce quality
                    img_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")
                    img_size_kb = len(buffered.getvalue()) / 1024
                    self.logger.info(f"Reduced image size: {img_size_kb:.2f} KB")
            except Exception as img_err:
                self.logger.error(f"Failed to convert image: {str(img_err)}")
                return {"error": f"Image conversion failed: {str(img_err)}"}
            
            # Prepare the request payload
            payload = {
                "model": self.ollama_model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": temperature
                },
                "images": [img_base64]
            }
            
            # Use exponential backoff for retries
            retry_count = 0
            current_delay = self.retry_delay
            
            while retry_count <= self.max_retries:
                try:
                    self.logger.info(f"Querying Ollama API (attempt {retry_count + 1})")
                    start_time = time.time()
                    
                    # Make the API request
                    response = requests.post(
                        self.ollama_url,
                        json=payload,
                        timeout=timeout
                    )
                    
                    query_time = time.time() - start_time
                    self.logger.info(f"Ollama query completed in {query_time:.2f} seconds")
                    
                    # Check for successful response
                    if response.status_code == 200:
                        try:
                            result = response.json()
                            return self._process_ollama_response(result)
                        except json.JSONDecodeError:
                            self.logger.error("Failed to parse Ollama API response as JSON")
                            return {"error": "Invalid JSON in response"}
                    else:
                        self.logger.error(f"Ollama API error: {response.status_code} - {response.text}")
                        
                        # Specific handling based on error code
                        if response.status_code == 400:
                            # Might be an issue with the prompt or image
                            return {"error": f"API error: {response.status_code} - {response.text}"}
                        elif response.status_code in [429, 503]:
                            # Rate limiting or server overload, retry with backoff
                            pass
                        elif response.status_code >= 500:
                            # Server error, retry with backoff
                            pass
                        else:
                            # Other errors, may not be retryable
                            return {"error": f"API error: {response.status_code} - {response.text}"}
                
                except requests.exceptions.Timeout:
                    self.logger.error(f"Timeout after {timeout} seconds when querying Ollama")
                    self.logger.warning("Consider increasing the timeout in the config or reducing image size")
                except requests.exceptions.ConnectionError:
                    self.logger.error("Connection error when querying Ollama")
                    self.logger.warning("Ensure Ollama server is running at the configured URL")
                except Exception as e:
                    self.logger.error(f"Error querying Ollama: {str(e)}")
                    return {"error": f"Failed to query Ollama: {str(e)}"}
                
                # If we've reached the max retries, break out and return error
                if retry_count >= self.max_retries:
                    break
                
                # Exponential backoff with jitter for retry
                sleep_time = current_delay * (1 + random.random())
                self.logger.info(f"Retrying in {sleep_time:.1f} seconds (attempt {retry_count + 1}/{self.max_retries})")
                time.sleep(sleep_time)
                current_delay *= 2  # Exponential backoff
                retry_count += 1
            
            # If we're here, all retries failed
            return {"error": "Max retries exceeded when querying Ollama"}
    
    def _process_ollama_response(self, response: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process and clean up the Ollama API response.
        
        Args:
            response: The API response
            
        Returns:
            Processed response
        """
        try:
            if "response" not in response:
                return {"error": "No response field in Ollama result"}
                
            text_response = response.get("response", "")
            
            # Try to extract JSON from the response if it contains JSON
            try:
                # Look for JSON within the response
                json_start = text_response.find('{')
                json_end = text_response.rfind('}')
                
                if json_start != -1 and json_end != -1 and json_end > json_start:
                    potential_json = text_response[json_start:json_end + 1]
                    result = json.loads(potential_json)
                    return result
            except json.JSONDecodeError:
                # If JSON extraction failed, just return the raw response
                pass
                
            return {"text": text_response}
            
        except Exception as e:
            self.logger.error(f"Error processing Ollama response: {str(e)}")
            return {"error": f"Failed to process response: {str(e)}"}
    
    def get_enhanced_game_state(self) -> Dict[str, Any]:
        """
        Get enhanced game state information using the Ollama vision model.
        
        Returns:
            Dictionary containing detailed game state information
        """
        # Check if we have a recent cached response
        current_time = time.time()
        if current_time - self.cache_timestamp < self.cache_ttl and "enhanced_state" in self.ollama_cache:
            self.logger.debug("Using cached enhanced game state")
            return self.ollama_cache["enhanced_state"]
        
        # Capture the current screen
        screen = self.capture_screen()
        
        # Query Ollama for UI analysis
        if screen is None:
            self.logger.error("Failed to capture screen for enhanced game state")
            return {"error": "Failed to capture screen"}
            
        # Check if screen is a string (error condition)
        if isinstance(screen, str):
            self.logger.error("Screen is a string, not a numpy array")
            return {"error": "Invalid screen data type"}
            
        # Query Ollama for UI analysis with correct parameter order
        response = self.query_ollama(prompt=self.ui_analysis_prompt, image=screen)
        
        # Parse the response
        ui_analysis = self.extract_json_from_response(response)
        
        # Cache the result
        self.ollama_cache["enhanced_state"] = ui_analysis
        self.cache_timestamp = current_time
        
        return ui_analysis
    
    def get_population_growth_guidance(self) -> Dict[str, Any]:
        """
        Get specific guidance on how to grow the city's population.
        
        Returns:
            Dictionary containing growth limiters and recommended actions
        """
        # Check if we have a recent cached response
        current_time = time.time()
        if current_time - self.cache_timestamp < self.cache_ttl and "population_guidance" in self.ollama_cache:
            self.logger.debug("Using cached population guidance")
            return self.ollama_cache["population_guidance"]
        
        # Capture the current screen
        screen = self.capture_screen()
        
        # Check if screen capture was successful
        if screen is None:
            self.logger.error("Failed to capture screen for population growth guidance")
            return {"error": "Failed to capture screen"}
            
        # Check if screen is a string (error condition)
        if isinstance(screen, str):
            self.logger.error("Screen is a string, not a numpy array")
            return {"error": "Invalid screen data type"}
            
        # Query Ollama for population growth analysis with correct parameter order
        response = self.query_ollama(prompt=self.population_growth_prompt, image=screen)
        
        # Parse the response
        guidance = self.extract_json_from_response(response)
        
        # Log the guidance for debugging
        if "recommended_actions" in guidance:
            self.logger.info(f"Population growth recommendations: {guidance['recommended_actions']}")
        if "growth_limiters" in guidance:
            self.logger.info(f"Growth limiters: {guidance['growth_limiters']}")
        if "issues_to_address" in guidance:
            self.logger.info(f"Issues to address: {guidance['issues_to_address']}")
        
        # Cache the result
        self.ollama_cache["population_guidance"] = guidance
        self.cache_timestamp = current_time
        
        return guidance
    
    def get_game_state(self) -> Dict[str, Any]:
        """
        Override the base method to include enhanced vision-based game state.
        
        Returns:
            Dictionary containing combined game state information
        """
        # Get the basic game state from the parent class
        basic_state = super().get_game_state()
        
        try:
            # Add enhanced state from vision model
            enhanced_state = self.get_enhanced_game_state()
            
            # Combine the information
            combined_state = {**basic_state, "vision_enhanced": enhanced_state}
            
            # Add population guidance if available
            if "metrics" in enhanced_state and "population" in enhanced_state["metrics"]:
                guidance = self.get_population_growth_guidance()
                combined_state["population_guidance"] = guidance
            
            return combined_state
        except Exception as e:
            self.logger.error(f"Error getting enhanced game state: {str(e)}")
            return basic_state  # Fall back to basic state if enhancement fails
    
    def detect_ui_elements_with_vision(self) -> Dict[str, Dict[str, Any]]:
        """
        Use the vision model to detect UI elements on screen.
        
        Returns:
            Dictionary of detected UI elements with positions and descriptions
        """
        try:
            # Get enhanced state which includes UI element detection
            enhanced_state = self.get_enhanced_game_state()
            
            if "error" in enhanced_state:
                self.logger.warning(f"Error in vision model UI detection: {enhanced_state['error']}")
                return {}
            
            # Extract UI elements from the vision model response
            ui_elements = {}
            if "ui_elements" in enhanced_state:
                for i, element in enumerate(enhanced_state["ui_elements"]):
                    element_name = element.get("name", f"unknown_element_{i}")
                    ui_elements[element_name] = {
                        "description": element.get("description", ""),
                        "position": element.get("position", "unknown"),
                        "detected_by": "vision_model"
                    }
                    
                # Log detected elements for debugging
                self.logger.info(f"Detected {len(ui_elements)} UI elements via vision model")
                for name, details in ui_elements.items():
                    self.logger.debug(f"UI Element: {name} at {details['position']}")
            
            return ui_elements
        except Exception as e:
            self.logger.error(f"Exception in vision model UI detection: {str(e)}")
            return {}
    
    def detect_ui_elements(self) -> bool:
        """
        Override to combine traditional CV methods with vision model detection.
        
        Returns:
            Boolean indicating if any UI elements were detected
        """
        # Get UI elements using traditional methods first
        traditional_success = super().detect_ui_elements()
        traditional_count = len(self.ui_element_cache)
        
        try:
            # Add elements detected by the vision model
            vision_elements = self.detect_ui_elements_with_vision()
            
            # Merge with existing UI element cache
            for name, details in vision_elements.items():
                if name not in self.ui_element_cache:
                    self.ui_element_cache[name] = details
            
            new_count = len(self.ui_element_cache)
            if new_count > traditional_count:
                self.logger.info(f"Vision model added {new_count - traditional_count} new UI elements")
            
            return traditional_success or len(vision_elements) > 0
        except Exception as e:
            self.logger.error(f"Error in enhanced UI detection: {str(e)}")
            return traditional_success  # Fall back to traditional result
    
    def detect_hover_text(self, x: int, y: int) -> Dict[str, Any]:
        """
        Hover over a position and detect any tooltip or hover text.
        
        Args:
            x: X coordinate to hover over
            y: Y coordinate to hover over
            
        Returns:
            Dictionary with detected hover text information
        """
        try:
            # First capture initial screen
            initial_screen = self.capture_screen()
            
            # Move mouse to the position
            pyautogui.moveTo(x, y, duration=0.2)
            
            # Wait briefly for tooltip to appear
            time.sleep(0.5)
            
            # Capture screen with tooltip
            hover_screen = self.capture_screen()
            
            # Create a hover text detection prompt
            hover_prompt = """
            Compare these two screenshots from Cities: Skylines 2. The second image shows the mouse hovering over an element.
            
            Identify any tooltip, hover text, or pop-up information that appears in the second image but not the first.
            
            Provide your response in JSON format:
            {
                "hover_text_detected": true/false,
                "tooltip_title": "title or heading of the tooltip if any",
                "tooltip_content": "detailed text content of the tooltip",
                "ui_element": "name of the UI element being hovered over",
                "game_mechanic": "what game mechanic this tooltip explains",
                "suggested_action": "any action suggested by the tooltip"
            }
            
            If no tooltip or hover text is detected, return "hover_text_detected": false.
            """
            
            # Prepare a payload with both images
            # Convert images to base64
            initial_rgb = cv2.cvtColor(initial_screen, cv2.COLOR_BGR2RGB)
            hover_rgb = cv2.cvtColor(hover_screen, cv2.COLOR_BGR2RGB)
            
            initial_pil = Image.fromarray(initial_rgb)
            hover_pil = Image.fromarray(hover_rgb)
            
            buffer1 = BytesIO()
            buffer2 = BytesIO()
            
            initial_pil.save(buffer1, format="JPEG", quality=85)
            hover_pil.save(buffer2, format="JPEG", quality=85)
            
            base64_initial = base64.b64encode(buffer1.getvalue()).decode("utf-8")
            base64_hover = base64.b64encode(buffer2.getvalue()).decode("utf-8")
            
            # Create the payload with both images
            payload = {
                "model": self.ollama_model,
                "prompt": hover_prompt,
                "images": [base64_initial, base64_hover],
                "stream": False,
                "options": {
                    "temperature": self.temperature,
                    "num_predict": self.max_tokens,
                }
            }
            
            # Send the request to Ollama
            timeout = self.ollama_config.get("response_timeout", 30)
            response = requests.post(
                self.ollama_url,
                json=payload,
                timeout=timeout
            )
            
            if response.status_code == 200:
                result = response.json()
                hover_info = self.extract_json_from_response(result)
                
                if "error" not in hover_info:
                    self.logger.info(f"Hover text detected: {hover_info.get('tooltip_title', 'None')}")
                    return hover_info
            
            return {"hover_text_detected": False}
            
        except Exception as e:
            self.logger.error(f"Error detecting hover text: {str(e)}")
            return {"hover_text_detected": False}
            
    def explore_tooltips(self, regions: List[Tuple[int, int, int, int]] = None) -> List[Dict[str, Any]]:
        """
        Systematically explore regions of the screen for tooltips.
        
        Args:
            regions: Optional list of regions to explore, each as (x1, y1, x2, y2)
                    If None, uses the whole screen divided into a grid
                    
        Returns:
            List of tooltips discovered
        """
        # Default to whole screen if no regions provided
        if regions is None:
            # Divide screen into a 4x4 grid
            width = self.screen_region[2] - self.screen_region[0]
            height = self.screen_region[3] - self.screen_region[1]
            
            grid_size = 4
            regions = []
            
            for row in range(grid_size):
                for col in range(grid_size):
                    x1 = self.screen_region[0] + (col * width // grid_size)
                    y1 = self.screen_region[1] + (row * height // grid_size)
                    x2 = self.screen_region[0] + ((col + 1) * width // grid_size)
                    y2 = self.screen_region[1] + ((row + 1) * height // grid_size)
                    regions.append((x1, y1, x2, y2))
        
        tooltips = []
        
        # For each region, check a few random points
        for region in regions:
            x1, y1, x2, y2 = region
            
            # Sample 2 random points in this region
            for _ in range(2):
                x = random.randint(x1, x2)
                y = random.randint(y1, y2)
                
                # Check for hover text
                hover_info = self.detect_hover_text(x, y)
                
                if hover_info.get("hover_text_detected", False):
                    # Add coordinates to the info
                    hover_info["coords"] = (x, y)
                    tooltips.append(hover_info)
                    
                    # Log the tooltip
                    self.logger.info(f"Found tooltip at ({x}, {y}): {hover_info.get('tooltip_title', 'Unknown')}")
                    
                    # Save to our dataset for future reference
                    self._save_tooltip_data(hover_info)
        
        return tooltips
    
    def _save_tooltip_data(self, tooltip_info: Dict[str, Any]) -> None:
        """Save tooltip data to our dataset for future reference."""
        try:
            # Create a tooltips file if it doesn't exist
            tooltips_file = os.path.join(self.debug_dir, "tooltips.jsonl")
            
            # Add timestamp 
            tooltip_info["timestamp"] = datetime.datetime.now().isoformat()
            
            # Write to the file
            with open(tooltips_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(tooltip_info) + "\n")
                
        except Exception as e:
            self.logger.error(f"Error saving tooltip data: {str(e)}")

    def capture_screen(self) -> Optional[np.ndarray]:
        """
        Capture the current screen.
        
        Returns:
            Screenshot as a numpy array or None if capture fails
        """
        try:
            # Add rate limiting to prevent too many captures
            current_time = time.time()
            elapsed = current_time - self.last_screenshot_time
            
            if elapsed < self.screenshot_cooldown:
                # If we're capturing too frequently, wait a bit
                time.sleep(self.screenshot_cooldown - elapsed)
            
            # Update last capture time
            self.last_screenshot_time = time.time()
            
            # Small delay
            time.sleep(0.1)
            
            # First try with PIL.ImageGrab
            try:
                screenshot = PIL.ImageGrab.grab()
                # Convert to numpy array and verify it's valid
                if screenshot is None:
                    self.logger.warning("PIL.ImageGrab.grab() returned None, trying alternative method")
                else:
                    np_screenshot = np.array(screenshot)
                    
                    # Verify we have a valid array
                    if np_screenshot.size == 0:
                        self.logger.warning("Empty screenshot array from PIL.ImageGrab, trying alternative method")
                    else:
                        return np_screenshot
            except Exception as pil_err:
                self.logger.warning(f"PIL.ImageGrab failed: {str(pil_err)}, trying alternative method")
            
            # If PIL.ImageGrab fails, try with mss
            try:
                with mss.mss() as sct:
                    # Get primary monitor
                    monitor = sct.monitors[1]
                    # Capture the screen
                    sct_img = sct.grab(monitor)
                    # Convert to numpy array
                    np_screenshot = np.array(sct_img)
                    # Convert BGRA to BGR
                    np_screenshot = cv2.cvtColor(np_screenshot, cv2.COLOR_BGRA2BGR)
                    return np_screenshot
            except Exception as mss_err:
                self.logger.error(f"MSS screenshot method failed: {str(mss_err)}")
                
                # Last resort - try pyautogui
                try:
                    screenshot = pyautogui.screenshot()
                    np_screenshot = np.array(screenshot)
                    return np_screenshot
                except Exception as pyautogui_err:
                    self.logger.error(f"All screenshot methods failed. Last error: {str(pyautogui_err)}")
                    return None
        except Exception as e:
            self.logger.error(f"Failed to capture screen: {str(e)}")
            return None

    def query_vision_model(self, prompt: str, image: np.ndarray, 
                           temperature: float = 0.7, enforce_json: bool = False,
                           timeout: Optional[int] = None) -> Dict[str, Any]:
        """
        Query the vision model with a prompt and image.
        
        Args:
            prompt: Text prompt to send to the model
            image: Image as a numpy array
            temperature: Model temperature setting
            enforce_json: Whether to enforce JSON output
            timeout: Timeout for API call in seconds
            
        Returns:
            Model response
        """
        # First check if we have a cached response
        cached_response = self.response_cache.get(prompt, image)
        if cached_response is not None:
            self.logger.info("Using cached vision model response")
            return cached_response
        
        # Validate image first
        if image is None:
            self.logger.error("Cannot query vision model with None image")
            return {"error": "Empty image provided"}
        
        if isinstance(image, str):
            self.logger.error("Image is a string, not a numpy array")
            return {"error": "Invalid image type: expected numpy array, got string"}
        
        # If not in cache, make the API call
        if enforce_json:
            # Add JSON formatting to the prompt
            prompt = f"{prompt}\n\nYou must respond in valid JSON format only."
        
        # Prepare and send the request with explicit parameter names
        response = self.query_ollama(
            prompt=prompt, 
            image=image, 
            temperature=temperature, 
            timeout=timeout
        )
        
        # Add to cache if successful
        if "error" not in response:
            self.response_cache.add(prompt, image, response)
        
        return response

    def extract_json_from_response(self, response: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract the JSON content from an Ollama response.
        
        Args:
            response: The raw response from Ollama
            
        Returns:
            Parsed JSON content or error message
        """
        try:
            if "error" in response:
                self.logger.warning(f"Error in Ollama response: {response['error']}")
                return {"error": response["error"]}
            
            # Get the response text
            text = response.get("response", "")
            
            # Try to find JSON content between brackets
            json_start = text.find("{")
            json_end = text.rfind("}") + 1
            
            if json_start >= 0 and json_end > json_start:
                json_text = text[json_start:json_end]
                
                # Clean up the JSON text
                # Remove any markdown code block markers
                json_text = json_text.replace("```json", "").replace("```", "")
                
                # Remove any trailing commas before closing braces/brackets
                json_text = json_text.replace(",]", "]").replace(",}", "}")
                
                # Try to parse the JSON
                try:
                    parsed_json = json.loads(json_text)
                    self.logger.info(f"Parsed JSON with keys: {list(parsed_json.keys())}")
                    return parsed_json
                except json.JSONDecodeError as e:
                    # If parsing fails, try to fix common issues
                    self.logger.warning(f"Initial JSON parse failed: {str(e)}, attempting to fix...")
                    
                    # Replace single quotes with double quotes
                    json_text = json_text.replace("'", '"')
                    
                    # Ensure boolean values are lowercase
                    json_text = json_text.replace("True", "true").replace("False", "false")
                    
                    # Try parsing again
                    try:
                        parsed_json = json.loads(json_text)
                        self.logger.info(f"Successfully parsed JSON after fixes with keys: {list(parsed_json.keys())}")
                        return parsed_json
                    except json.JSONDecodeError as e2:
                        self.logger.error(f"Failed to parse JSON even after fixes: {str(e2)}")
                        return {"error": f"Failed to parse JSON: {str(e2)}", "raw_text": json_text}
            else:
                self.logger.warning(f"No JSON found in response: {text[:100]}...")
                return {"error": "No JSON found in response", "raw_text": text}
        except Exception as e:
            self.logger.error(f"Exception extracting JSON: {str(e)}")
            return {"error": f"Exception extracting JSON: {str(e)}"}

    def _get_observation_space(self):
        """
        Attempt to get the observation space from the environment or create a default one.
        
        Returns:
            A valid observation space or None if not available
        """
        # Try to get from environment
        try:
            # First check if we have a direct reference
            if hasattr(self, 'env') and hasattr(self.env, 'observation_space'):
                return self.env.observation_space
                
            # If not, try to navigate up to find it
            if hasattr(self, 'parent') and hasattr(self.parent, 'observation_space'):
                return self.parent.observation_space
                
            # One more level up
            if hasattr(self, 'parent') and hasattr(self.parent, 'parent') and hasattr(self.parent.parent, 'observation_space'):
                return self.parent.parent.observation_space
                
            return None
        except Exception as e:
            self.logger.error(f"Error getting observation space: {str(e)}")
            return None
            
    def get_visual_observation(self, observation_space=None) -> dict:
        """
        Get a visual observation using the vision model.
        
        Args:
            observation_space: Optional gymnasium observation space structure
            
        Returns:
            Dictionary with visual observation data
        """
        try:
            # Check if observation_space is provided
            if observation_space is None:
                observation_space = self._get_observation_space()
                if observation_space is None:
                    self.logger.warning("observation_space is None, using empty default")
                    # Return a basic default observation that won't cause errors
                    return {
                        "population": np.array([0.0], dtype=np.float32),
                        "happiness": np.array([0.0], dtype=np.float32),
                        "budget_balance": np.array([0.0], dtype=np.float32),
                        "traffic": np.array([0.0], dtype=np.float32)
                    }
            
            # Capture current screen
            screen = self.capture_screen()
            if screen is None:
                self.logger.error("Failed to capture screen for visual observation")
                return self._create_default_observation(observation_space)
            
            # Make sure screen is a numpy array, not a string
            if isinstance(screen, str):
                self.logger.error("Screen is a string, not a numpy array. Creating default observation.")
                return self._create_default_observation(observation_space)
            
            # Create a vision model query to analyze the current state
            prompt = """
            Analyze this screenshot from Cities: Skylines 2 and provide the following information:
            1. Game state - what is the current city status?
            2. Current UI elements visible
            3. Important metrics or indicators
            4. Any alerts or notifications
            5. What actions would be helpful right now?
            
            Format your response as JSON with these fields:
            {
                "game_state": "brief description",
                "ui_elements": ["list", "of", "visible", "ui", "elements"],
                "metrics": {"population": value, "money": value, "others": value},
                "alerts": ["any", "visible", "alerts"],
                "suggested_actions": ["action1", "action2"]
            }
            """
            
            # Query the vision model
            vision_response = self.query_vision_model(prompt=prompt, image=screen, enforce_json=True)
            
            # Handle any errors in the response
            if "error" in vision_response:
                self.logger.error(f"Failed to get visual observation: {vision_response['error']}")
                # Create a fallback observation with default values
                return self._create_default_observation(observation_space)
            
            # Process the response into the expected observation format
            observation = self._process_vision_response_to_observation(vision_response, observation_space)
            
            return observation
        except AttributeError as ae:
            # Handle the specific 'observation_space' attribute error
            self.logger.error(f"Failed to get visual observation: {str(ae)}")
            return {
                "population": np.array([0.0], dtype=np.float32),
                "happiness": np.array([0.0], dtype=np.float32),
                "budget_balance": np.array([0.0], dtype=np.float32),
                "traffic": np.array([0.0], dtype=np.float32)
            }
        except Exception as e:
            self.logger.error(f"Error in get_visual_observation: {str(e)}")
            return {
                "population": np.array([0.0], dtype=np.float32),
                "happiness": np.array([0.0], dtype=np.float32),
                "budget_balance": np.array([0.0], dtype=np.float32),
                "traffic": np.array([0.0], dtype=np.float32)
            }
            
    def _create_default_observation(self, observation_space):
        """
        Create a default observation matching the expected observation space.
        
        Args:
            observation_space: The gymnasium observation space
            
        Returns:
            Default observation dictionary
        """
        try:
            # If observation_space is a Dict space, create appropriate default values
            if hasattr(observation_space, 'spaces'):
                default_obs = {}
                # For each space in the dict, create a default value
                for key, space in observation_space.spaces.items():
                    if hasattr(space, 'shape'):
                        # For Box spaces, use zeros
                        default_obs[key] = np.zeros(space.shape, dtype=np.float32)
                    elif hasattr(space, 'n'):
                        # For Discrete spaces, use 0
                        default_obs[key] = 0
                    else:
                        # For other spaces, use None
                        default_obs[key] = None
                return default_obs
            # If it's a simple space, return a zero array
            elif hasattr(observation_space, 'shape'):
                return np.zeros(observation_space.shape, dtype=np.float32)
            else:
                # Fallback for unknown space types
                return {}
        except Exception as e:
            self.logger.error(f"Error creating default observation: {str(e)}")
            return {}
            
    def _process_vision_response_to_observation(self, vision_response, observation_space):
        """
        Process a vision model response into the expected observation format.
        
        Args:
            vision_response: Raw response from vision model
            observation_space: The gymnasium observation space
            
        Returns:
            Processed observation matching expected format
        """
        try:
            # Start with default observation
            observation = self._create_default_observation(observation_space)
            
            # If the response doesn't have the expected structure, return the default
            if not isinstance(vision_response, dict):
                return observation
                
            # Extract metrics if they exist and update the observation
            if 'metrics' in vision_response and isinstance(vision_response['metrics'], dict):
                metrics = vision_response['metrics']
                
                # Map to known observation keys
                if 'population' in metrics and 'population' in observation:
                    try:
                        # Try to convert to float
                        value = self._extract_numeric_value(metrics['population'])
                        if value is not None:
                            observation['population'] = np.array([value], dtype=np.float32)
                    except:
                        pass
                        
                if 'money' in metrics and 'budget_balance' in observation:
                    try:
                        value = self._extract_numeric_value(metrics['money'])
                        if value is not None:
                            observation['budget_balance'] = np.array([value], dtype=np.float32)
                    except:
                        pass
                        
                if 'happiness' in metrics and 'happiness' in observation:
                    try:
                        value = self._extract_numeric_value(metrics['happiness'])
                        if value is not None:
                            observation['happiness'] = np.array([value], dtype=np.float32)
                    except:
                        pass
                        
                if 'traffic' in metrics and 'traffic' in observation:
                    try:
                        value = self._extract_numeric_value(metrics['traffic'])
                        if value is not None:
                            observation['traffic'] = np.array([value], dtype=np.float32)
                    except:
                        pass
            
            return observation
        except Exception as e:
            self.logger.error(f"Error processing vision response to observation: {str(e)}")
            return self._create_default_observation(observation_space)
            
    def _extract_numeric_value(self, value_str):
        """
        Extract a numeric value from a string or other data type.
        
        Args:
            value_str: Value to convert to numeric
            
        Returns:
            Numeric value or None if conversion fails
        """
        try:
            # If it's already a number, return it
            if isinstance(value_str, (int, float)):
                return float(value_str)
                
            # If it's a string, try to extract a number
            if isinstance(value_str, str):
                # Remove non-numeric characters except decimal point
                clean_str = ''.join(c for c in value_str if c.isdigit() or c == '.')
                if clean_str:
                    return float(clean_str)
                    
            return None
        except:
            return None 