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

from .auto_vision_interface import AutoVisionInterface

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
        
    def query_ollama(self, image: np.ndarray, prompt: str) -> Dict[str, Any]:
        """
        Send an image to Ollama vision model and get a response.
        
        Args:
            image: The image to analyze as numpy array (BGR format from OpenCV)
            prompt: The prompt to send to the model
            
        Returns:
            Dictionary containing the model's response
        """
        # Convert from BGR to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Save screenshot for debugging (periodically)
        self.screenshot_counter += 1
        if self.screenshot_counter % self.screenshot_frequency == 0:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            screenshot_path = os.path.join(self.debug_dir, f"screen_{timestamp}.jpg")
            cv2.imwrite(screenshot_path, image)
            self.logger.info(f"Saved debug screenshot to {screenshot_path}")
        
        # Convert to PIL Image and then to base64
        pil_image = Image.fromarray(image_rgb)
        buffer = BytesIO()
        pil_image.save(buffer, format="JPEG", quality=85)  # Reduced quality for smaller payload
        base64_image = base64.b64encode(buffer.getvalue()).decode("utf-8")
        
        # Create the payload
        payload = {
            "model": self.ollama_model,
            "prompt": prompt,
            "images": [base64_image],
            "stream": False,
            "options": {
                "temperature": self.temperature,
                "num_predict": self.max_tokens,
                "stop": ["}"]  # Stop after closing brace to prevent extra text
            }
        }
        
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        prompt_type = "ui_analysis" if self.ui_analysis_prompt in prompt else "population_guidance"
        
        # Get retry configuration
        max_retries = self.ollama_config.get("retries", 3)
        retry_delay = self.ollama_config.get("retry_delay", 1.0)
        timeout = self.ollama_config.get("response_timeout", 30)
        
        for attempt in range(max_retries):
            try:
                self.logger.info(f"Querying Ollama with {prompt_type} prompt (attempt {attempt + 1}/{max_retries})")
                start_time = time.time()
                
                # Make the API call with timeout
                response = requests.post(
                    self.ollama_url,
                    json=payload,
                    timeout=timeout
                )
                
                query_time = time.time() - start_time
                self.logger.info(f"Ollama query took {query_time:.2f} seconds")
                
                if response.status_code == 200:
                    result = response.json()
                    
                    # Log the response for debugging
                    with open(self.json_log_file, "a", encoding="utf-8") as f:
                        log_entry = {
                            "timestamp": timestamp,
                            "prompt_type": prompt_type,
                            "response": result,
                            "query_time_seconds": query_time,
                            "attempt": attempt + 1
                        }
                        f.write(json.dumps(log_entry) + "\n")
                    
                    self.logger.info(f"Logged {prompt_type} response to {self.json_log_file}")
                    
                    # Validate JSON if enabled
                    if self.ollama_config.get("json_validation", True):
                        parsed = self.extract_json_from_response(result)
                        if "error" not in parsed:
                            return result
                        elif attempt < max_retries - 1:
                            self.logger.warning(f"Invalid JSON on attempt {attempt + 1}, retrying...")
                            time.sleep(retry_delay)
                            continue
                    
                    return result
                else:
                    error_msg = f"Error querying Ollama API: {response.status_code} - {response.text}"
                    self.logger.error(error_msg)
                    
                    if attempt < max_retries - 1:
                        self.logger.warning(f"Request failed on attempt {attempt + 1}, retrying...")
                        time.sleep(retry_delay)
                        continue
                    
                    # Log the final error
                    with open(self.json_log_file, "a", encoding="utf-8") as f:
                        log_entry = {
                            "timestamp": timestamp,
                            "prompt_type": prompt_type,
                            "error": error_msg,
                            "status_code": response.status_code,
                            "attempt": attempt + 1
                        }
                        f.write(json.dumps(log_entry) + "\n")
                    
                    return {"error": f"Status code: {response.status_code}"}
                    
            except requests.Timeout:
                error_msg = f"Timeout after {timeout} seconds when querying Ollama"
                self.logger.error(error_msg)
                
                if attempt < max_retries - 1:
                    self.logger.warning(f"Request timed out on attempt {attempt + 1}, retrying...")
                    time.sleep(retry_delay)
                    continue
                
                # Log the final timeout
                with open(self.json_log_file, "a", encoding="utf-8") as f:
                    log_entry = {
                        "timestamp": timestamp,
                        "prompt_type": prompt_type,
                        "error": error_msg,
                        "type": "timeout",
                        "attempt": attempt + 1
                    }
                    f.write(json.dumps(log_entry) + "\n")
                
                return {"error": error_msg}
                
            except Exception as e:
                error_msg = f"Exception when querying Ollama: {str(e)}"
                self.logger.error(error_msg)
                
                if attempt < max_retries - 1:
                    self.logger.warning(f"Request failed on attempt {attempt + 1}, retrying...")
                    time.sleep(retry_delay)
                    continue
                
                # Log the final error
                with open(self.json_log_file, "a", encoding="utf-8") as f:
                    log_entry = {
                        "timestamp": timestamp,
                        "prompt_type": prompt_type,
                        "error": error_msg,
                        "attempt": attempt + 1
                    }
                    f.write(json.dumps(log_entry) + "\n")
                
                return {"error": str(e)}
    
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
        response = self.query_ollama(screen, self.ui_analysis_prompt)
        
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
        
        # Query Ollama for population growth analysis
        response = self.query_ollama(screen, self.population_growth_prompt)
        
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