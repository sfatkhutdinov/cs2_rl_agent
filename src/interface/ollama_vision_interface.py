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
        
        # UI analysis help prompts
        self.ui_analysis_prompt = """
        You are analyzing a screenshot from the city-building game Cities: Skylines 2.
        
        Identify and describe:
        1. Key UI elements visible in the image
        2. Current game state (building mode, economy panel, etc.)
        3. Important metrics visible (population, happiness, money)
        4. Any alerts or notifications
        5. Available tools or actions
        
        Format your response as JSON with these keys:
        {
            "ui_elements": [{"name": "element_name", "description": "what it does", "position": "general location"}],
            "game_state": "description of current game state",
            "metrics": {"population": "value if visible", "money": "value if visible"},
            "alerts": ["list of any visible alerts"],
            "available_actions": ["list of actions that would make sense now"],
            "suggestions": ["suggestions for what would help grow the city"]
        }
        """
        
        self.population_growth_prompt = """
        You are analyzing a screenshot from the city-building game Cities: Skylines 2.
        
        Based on what you see, provide guidance on:
        1. What is currently limiting population growth
        2. What specific actions would help grow the population
        3. Any issues that need to be addressed (like traffic, pollution, etc.)
        
        Format your response as JSON with these keys:
        {
            "growth_limiters": ["list of factors limiting growth"],
            "recommended_actions": ["specific actions to take"],
            "issues_to_address": ["problems visible in the city"]
        }
        """
        
        self.logger.info(f"Ollama Vision Interface initialized with model: {self.ollama_model}")
        
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
        
        # Convert to PIL Image and then to base64
        pil_image = Image.fromarray(image_rgb)
        buffer = BytesIO()
        pil_image.save(buffer, format="JPEG")
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
            }
        }
        
        try:
            response = requests.post(self.ollama_url, json=payload)
            if response.status_code == 200:
                result = response.json()
                return result
            else:
                self.logger.error(f"Error querying Ollama API: {response.status_code} - {response.text}")
                return {"error": f"Status code: {response.status_code}"}
        except Exception as e:
            self.logger.error(f"Exception when querying Ollama: {str(e)}")
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
                return {"error": response["error"]}
            
            # Get the response text
            text = response.get("response", "")
            
            # Try to find JSON content between brackets
            json_start = text.find("{")
            json_end = text.rfind("}") + 1
            
            if json_start >= 0 and json_end > json_start:
                json_text = text[json_start:json_end]
                return json.loads(json_text)
            else:
                return {"error": "No JSON found in response", "raw_text": text}
        except json.JSONDecodeError:
            return {"error": "Failed to parse JSON", "raw_text": text}
        except Exception as e:
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
            return self.ollama_cache["population_guidance"]
        
        # Capture the current screen
        screen = self.capture_screen()
        
        # Query Ollama for population growth analysis
        response = self.query_ollama(screen, self.population_growth_prompt)
        
        # Parse the response
        guidance = self.extract_json_from_response(response)
        
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
        
        try:
            # Add elements detected by the vision model
            vision_elements = self.detect_ui_elements_with_vision()
            
            # Merge with existing UI element cache
            for name, details in vision_elements.items():
                if name not in self.ui_element_cache:
                    self.ui_element_cache[name] = details
            
            return traditional_success or len(vision_elements) > 0
        except Exception as e:
            self.logger.error(f"Error in enhanced UI detection: {str(e)}")
            return traditional_success  # Fall back to traditional result 