import logging
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import random
import datetime
import json
import os
import time
import threading
from typing import Dict, Any, Tuple, List, Optional, Union

from src.environment.vision_guided_env import VisionGuidedCS2Environment
from src.interface.ollama_vision_interface import OllamaVisionInterface


class TutorialGuidedCS2Environment(VisionGuidedCS2Environment):
    """
    An extension of the vision-guided environment that follows game tutorials
    to learn game mechanics in a structured way.
    """
    
    def __init__(self, base_env, exploration_frequency=0.2, random_action_frequency=0.1, 
                 menu_exploration_buffer_size=50, logger=None):
        """Initialize the tutorial-guided environment wrapper."""
        super().__init__(
            base_env, 
            exploration_frequency=exploration_frequency,
            random_action_frequency=random_action_frequency,
            menu_exploration_buffer_size=menu_exploration_buffer_size,
            logger=logger
        )
        
        self.logger = logger or logging.getLogger("TutorialGuidedEnv")
        
        # Tutorial tracking
        self.current_tutorial = None
        self.completed_tutorials = set()
        self.tutorial_steps = []
        self.in_tutorial_mode = False
        self.tutorial_progress = 0
        
        # Debugging
        self.debug_dir = os.path.join("logs", "tutorial_debug")
        os.makedirs(self.debug_dir, exist_ok=True)
        
        # Stats
        self.tutorial_stats = {
            "tutorials_started": 0,
            "tutorials_completed": 0,
            "tutorial_steps_completed": 0
        }
        
        self.logger.info("Tutorial-guided environment initialized")
    
    def _get_vision_interface(self) -> Optional[OllamaVisionInterface]:
        """Helper to get the vision interface if it's the right type"""
        if isinstance(self.env.interface, OllamaVisionInterface):
            return self.env.interface
        return None
    
    def _detect_tutorial_text(self) -> Dict[str, Any]:
        """
        Capture the screen and analyze it for tutorial information.
        
        Returns:
            Dict with tutorial information if detected
        """
        # Check if we have access to the Ollama vision interface
        if not hasattr(self.env, "interface") or not isinstance(self.env.interface, OllamaVisionInterface):
            self.logger.warning("No Ollama vision interface available for tutorial detection")
            return {"detected": False}
            
        try:
            # Capture the current screen
            screen = self.env.interface.capture_screen()
            
            # Create a tutorial detection prompt
            tutorial_prompt = """
            Analyze this screenshot from Cities: Skylines 2 game and identify if there are any tutorial messages, 
            help prompts, or guided instructions visible.
            
            Provide your response in JSON format:
            {
                "detected": true/false,
                "tutorial_title": "title or name of the tutorial",
                "instructions": "what the player is being instructed to do",
                "ui_elements_mentioned": ["list of UI elements mentioned"],
                "actions_suggested": ["list of actions the tutorial is suggesting"],
                "current_step": "description of the current step",
                "next_step": "description of what to do next",
                "progress_indicator": "any progress indicator shown (e.g., '2/5')"
            }
            
            If no tutorial or help text is visible, return "detected": false.
            """
            
            # Send the request to Ollama via the vision interface
            tutorial_info = self.env.interface.query_vision_model(
                prompt=tutorial_prompt, 
                image=screen,
                enforce_json=True
            )
            
            if tutorial_info.get("detected", False):
                self.logger.info(f"Tutorial detected: {tutorial_info.get('tutorial_title', 'Unknown')}")
            
            return tutorial_info
            
        except Exception as e:
            self.logger.error(f"Error detecting tutorial text: {str(e)}")
            return {"detected": False}
    
    def _follow_tutorial_instruction(self, tutorial_info: Dict[str, Any]) -> int:
        """
        Follow the current tutorial instruction based on detected text.
        
        Args:
            tutorial_info: Dictionary with tutorial information
            
        Returns:
            Action index to take
        """
        # Extract the tutorial title and instructions
        tutorial_title = tutorial_info.get("tutorial_title", "").lower()
        instructions = tutorial_info.get("instructions", "").lower()
        actions_suggested = tutorial_info.get("actions_suggested", [])
        ui_elements = tutorial_info.get("ui_elements_mentioned", [])
        
        # Track this tutorial if it's new
        if tutorial_title and tutorial_title != self.current_tutorial:
            self.current_tutorial = tutorial_title
            self.tutorial_progress = 0
            self.tutorial_stats["tutorials_started"] += 1
            self.logger.info(f"Starting new tutorial: {tutorial_title}")
        
        # Try to map instructions to specific actions
        action_mapping = {
            "move the camera": ["pan_camera", "rotate_camera", "zoom_camera"],
            "zoom in": ["zoom_in"],
            "zoom out": ["zoom_out"],
            "rotate": ["rotate_camera_left", "rotate_camera_right"],
            "click": ["left_click"],
            "select": ["left_click"],
            "open": ["left_click"],
            "build": ["left_click", "b"],
            "road": ["r"],
            "bulldoze": ["b"],
            "pause": ["space"],
            "play": ["space"],
            "speed up": ["1", "2", "3"],
            "slow down": ["1"],
            "zone": ["z"],
            "residential": ["z", "q"],
            "commercial": ["z", "w"],
            "industrial": ["z", "e"],
            "water pipe": ["w", "q"],
            "electricity": ["w", "e"],
            "services": ["u"],
        }
        
        # Determine which actions might apply to the current instructions
        potential_actions = []
        
        # Check if any of our keywords match the instruction text
        for keyword, actions in action_mapping.items():
            if keyword in instructions:
                potential_actions.extend(actions)
        
        # If we found potential actions, select one
        if potential_actions:
            # Convert action name to action index
            action_names = [a.__name__ if callable(a) else str(a) for a in self.action_handlers]
            matching_indices = []
            
            for action in potential_actions:
                for i, name in enumerate(action_names):
                    if action.lower() in name.lower():
                        matching_indices.append(i)
            
            if matching_indices:
                chosen_action = random.choice(matching_indices)
                self.logger.info(f"Following tutorial instruction: {instructions} with action {action_names[chosen_action]}")
                self.tutorial_progress += 1
                self.tutorial_stats["tutorial_steps_completed"] += 1
                return chosen_action
        
        # If we couldn't determine a specific action, take a random action
        self.logger.info(f"Couldn't determine specific action for instruction: {instructions}. Taking random action.")
        return self.action_space.sample()
    
    def _should_follow_tutorial(self) -> bool:
        """
        Determine if the agent should be following tutorials during this step.
        
        Returns:
            True if the agent should follow tutorials, False otherwise
        """
        # Always try to follow tutorials with a certain probability
        return random.random() < 0.8  # 80% chance to look for and follow tutorials
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """
        Take a step in the environment, potentially following tutorials.
        
        Args:
            action: Action index to take
            
        Returns:
            Tuple of (observation, reward, terminated, truncated, info)
        """
        is_tutorial_guided = False
        original_action = action
        
        # Decide whether to follow tutorials
        if self._should_follow_tutorial():
            tutorial_info = self._detect_tutorial_text()
            
            if tutorial_info.get("detected", False):
                action = self._follow_tutorial_instruction(tutorial_info)
                is_tutorial_guided = True
        
        # Get the action name for logging
        action_name = "unknown"
        if 0 <= action < len(self.action_handlers):
            action_name = self.action_handlers[action].name
        
        # Take the step using the parent class
        observation, reward, terminated, truncated, info = super().step(action)
        
        # Add tutorial info to the info dictionary
        if is_tutorial_guided:
            info["tutorial_guided"] = True
            info["current_tutorial"] = self.current_tutorial
            info["tutorial_progress"] = self.tutorial_progress
        
        # If we're following a tutorial, apply the reward multiplier
        if self.current_tutorial:
            reward *= 2.0
            info["tutorial"] = self.current_tutorial
            info["tutorial_progress"] = self.tutorial_progress
            
            # Check if tutorial has completed (detect "completed" message)
            if "completed" in str(info.get("vision_info", {})).lower() or "success" in str(info.get("vision_info", {})).lower():
                self.logger.info(f"Tutorial completed: {self.current_tutorial}")
                self.completed_tutorials.add(self.current_tutorial)
                self.current_tutorial = None
                self.tutorial_progress = 0
                
                # Give a large completion reward
                reward += 10.0
                info["tutorial_completed"] = True
        
        return observation, reward, terminated, truncated, info
    
    def reset(self, **kwargs):
        """Reset the environment and tutorial state."""
        self.current_tutorial = None
        self.tutorial_progress = 0
        return super().reset(**kwargs) 