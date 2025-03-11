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

from src.environment.autonomous_env import AutonomousCS2Environment, ActionType, Action
from src.interface.ollama_vision_interface import OllamaVisionInterface


class VisionGuidedCS2Environment(AutonomousCS2Environment):
    """
    An extension of the autonomous environment that uses the Ollama vision model
    to provide strategic guidance, but relies primarily on direct learning.
    """
    
    def __init__(self, base_env, exploration_frequency=0.3, random_action_frequency=0.2, 
                 menu_exploration_buffer_size=50, logger=None):
        """Initialize the vision-guided environment wrapper."""
        super().__init__(
            base_env, 
            exploration_frequency=exploration_frequency,
            random_action_frequency=random_action_frequency,
            menu_exploration_buffer_size=menu_exploration_buffer_size,
            logger=logger
        )
        
        self.logger = logger or logging.getLogger("VisionGuidedEnv")
        
        # Vision guidance configuration
        vision_config = self.env.config.get("environment", {}).get("vision_guidance", {})
        self.vision_guidance_enabled = vision_config.get("enabled", True)
        self.vision_guidance_frequency = vision_config.get("frequency", 0.1)
        self.vision_cache_ttl = vision_config.get("cache_ttl", 30)
        self.min_confidence = vision_config.get("min_confidence", 0.7)
        self.max_consecutive_attempts = vision_config.get("max_consecutive_attempts", 3)
        self.background_analysis = vision_config.get("background_analysis", True)
        
        # Vision guidance state
        self.vision_guidance_cache = {}
        self.last_vision_query = 0
        self.consecutive_vision_attempts = 0
        self.vision_update_thread = None
        self.vision_update_lock = threading.Lock()
        
        # Setup debugging directory
        self.debug_dir = os.path.join("logs", "actions_debug")
        os.makedirs(self.debug_dir, exist_ok=True)
        
        # Action tracking log
        self.action_log_file = os.path.join(self.debug_dir, "action_history.jsonl")
        
        # Action statistics
        self.action_stats = {
            "total_actions": 0,
            "vision_guided_actions": 0,
            "successful_actions": 0,
            "action_counts": {},
            "vision_action_success_rate": {}
        }
        
        # Add vision-guided actions
        self._add_vision_guided_actions()
        
        self.logger.info("Vision-guided environment initialized with extended action space")
        self.logger.info(f"Action debug logs will be saved to: {self.debug_dir}")
    
    def _add_vision_guided_actions(self):
        """Add vision-specific guided actions to the action space."""
        vision_actions = [
            Action(
                name="follow_population_advice",
                action_fn=lambda: self._follow_population_growth_advice(),
                action_type=ActionType.GAME_ACTION
            ),
            Action(
                name="address_visible_issue",
                action_fn=lambda: self._address_visible_issue(),
                action_type=ActionType.GAME_ACTION
            )
        ]
        
        # Add the new actions to our action list
        for action in vision_actions:
            self.action_handlers.append(action)
        
        # Update the action space
        self.action_space = spaces.Discrete(len(self.action_handlers))
        
        # Initialize action statistics
        for action in self.action_handlers:
            self.action_stats["action_counts"][action.name] = 0
            self.action_stats["vision_action_success_rate"][action.name] = {
                "attempts": 0,
                "successes": 0
            }
    
    def _update_vision_guidance_async(self):
        """Update vision guidance in a background thread."""
        if not self.vision_guidance_enabled:
            return
            
        def update_thread():
            try:
                with self.vision_update_lock:
                    if not isinstance(self.env.interface, OllamaVisionInterface):
                        return
                    
                    try:
                        screen = self.env.interface.capture_screen()
                    except Exception as e:
                        self.logger.warning(f"Failed to capture screen: {str(e)}")
                        return
                        
                    try:
                        response = self.env.interface.query_ollama(screen, self.env.interface.population_growth_prompt)
                        
                        if isinstance(response, dict) and "response" in response:
                            guidance = self.env.interface.extract_json_from_response(response)
                            if "error" not in guidance:
                                self.vision_guidance_cache = {
                                    "timestamp": time.time(),
                                    "guidance": guidance,
                                    "confidence": self.min_confidence
                                }
                                self.logger.debug("Updated vision guidance cache")
                    except Exception as e:
                        self.logger.warning(f"Failed to process vision guidance: {str(e)}")
                        
            except Exception as e:
                self.logger.warning(f"Vision guidance update failed: {str(e)}")
        
        # Start the update in a background thread
        if self.background_analysis:
            if self.vision_update_thread is None or not self.vision_update_thread.is_alive():
                self.vision_update_thread = threading.Thread(target=update_thread)
                self.vision_update_thread.daemon = True
                self.vision_update_thread.start()
        else:
            update_thread()
    
    def _should_use_vision_guidance(self) -> bool:
        """Determine if vision guidance should be used for the current step."""
        if not self.vision_guidance_enabled:
            return False
            
        # Check if we've exceeded consecutive attempts
        if self.consecutive_vision_attempts >= self.max_consecutive_attempts:
            self.consecutive_vision_attempts = 0
            return False
            
        # Check if guidance is stale
        current_time = time.time()
        if current_time - self.last_vision_query > self.vision_cache_ttl:
            self._update_vision_guidance_async()
            
        # Use vision guidance with configured probability
        return (
            random.random() < self.vision_guidance_frequency and
            "guidance" in self.vision_guidance_cache and
            current_time - self.vision_guidance_cache["timestamp"] < self.vision_cache_ttl
        )
    
    def _get_vision_guided_action(self) -> int:
        """Get an action recommendation based on cached vision guidance."""
        try:
            guidance = self.vision_guidance_cache.get("guidance", {})
            
            if "recommended_actions" in guidance and guidance["recommended_actions"]:
                # Map recommended actions to available actions
                action_names = []
                for recommendation in guidance["recommended_actions"]:
                    matching_actions = self._find_matching_actions(recommendation)
                    action_names.extend(matching_actions)
                
                if action_names:
                    # Select an action with preference for population-focused actions
                    population_actions = [name for name in action_names if "population" in name]
                    if population_actions and random.random() < 0.7:  # 70% chance to use population action
                        selected_action = random.choice(population_actions)
                    else:
                        selected_action = random.choice(action_names)
                        
                    action_idx = self._get_action_idx_by_name(selected_action)
                    if action_idx is not None:
                        self.consecutive_vision_attempts += 1
                        return action_idx
            
            self.consecutive_vision_attempts = 0
            return self._get_guided_exploratory_action()
            
        except Exception as e:
            self.logger.error(f"Error getting vision-guided action: {str(e)}")
            self.consecutive_vision_attempts = 0
            return self._get_guided_exploratory_action()
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Take a step in the environment with optional vision guidance."""
        is_vision_guided = False
        original_action = action
        
        # Decide whether to use vision guidance
        if self._should_use_vision_guidance():
            guided_action = self._get_vision_guided_action()
            self.logger.debug(f"Using vision guidance to select action {guided_action} instead of {action}")
            action = guided_action
            is_vision_guided = True
        else:
            self.consecutive_vision_attempts = 0
        
        # Get the action name for logging
        action_name = "unknown"
        if 0 <= action < len(self.action_handlers):
            action_name = self.action_handlers[action].name
        
        # Take the step using the parent class
        observation, reward, terminated, truncated, info = super().step(action)
        
        # Log the action
        self._log_action(
            action_idx=action,
            action_name=action_name,
            is_vision_guided=is_vision_guided,
            success=reward > 0,
            reward=reward,
            info=info
        )
        
        return observation, reward, terminated, truncated, info
    
    def reset(self, **kwargs):
        """Reset the environment and vision guidance state."""
        self.vision_guidance_cache = {}
        self.last_vision_query = 0
        self.consecutive_vision_attempts = 0
        return super().reset(**kwargs)
    
    def close(self):
        """Clean up resources."""
        if self.vision_update_thread and self.vision_update_thread.is_alive():
            self.vision_update_thread.join(timeout=1.0)
        super().close()
    
    def _log_action(self, action_idx: int, action_name: str, is_vision_guided: bool, success: bool, reward: float, info: Dict[str, Any]) -> None:
        """Log an action and its outcome"""
        try:
            # Convert numpy types to Python types for JSON serialization
            if isinstance(action_idx, (np.int64, np.int32)):
                action_idx = int(action_idx)
            if isinstance(reward, (np.float64, np.float32)):
                reward = float(reward)
            
            # Create log entry
            log_entry = {
                "timestamp": datetime.datetime.now().isoformat(),
                "action_idx": action_idx,
                "action_name": action_name,
                "is_vision_guided": is_vision_guided,
                "success": success,
                "reward": reward,
                "info": {k: str(v) if isinstance(v, (np.int64, np.int32, np.float64, np.float32)) else v 
                        for k, v in info.items()}
            }
            
            # Write to log file
            with open(self.action_log_file, "a", encoding="utf-8") as f:
                json.dump(log_entry, f)
                f.write("\n")
            
            # Update statistics
            self.action_stats["total_actions"] += 1
            if is_vision_guided:
                self.action_stats["vision_guided_actions"] += 1
            if success:
                self.action_stats["successful_actions"] += 1
            
            # Update action-specific stats
            if action_name not in self.action_stats["action_counts"]:
                self.action_stats["action_counts"][action_name] = 0
            self.action_stats["action_counts"][action_name] += 1
            
            if is_vision_guided:
                if action_name not in self.action_stats["vision_action_success_rate"]:
                    self.action_stats["vision_action_success_rate"][action_name] = {
                        "attempts": 0,
                        "successes": 0
                    }
                self.action_stats["vision_action_success_rate"][action_name]["attempts"] += 1
                if success:
                    self.action_stats["vision_action_success_rate"][action_name]["successes"] += 1
                    
        except Exception as e:
            self.logger.error(f"Error logging action: {str(e)}")
    
    def _find_matching_actions(self, text_description: str) -> List[str]:
        """Find actions in our action space that match a text description"""
        text_description = text_description.lower()
        matching_actions = []
        
        # Keywords to action mappings
        keyword_mappings = {
            "zone": ["place_residential_zone", "place_commercial_zone", "place_industrial_zone"],
            "residential": ["place_residential_zone"],
            "commercial": ["place_commercial_zone"],
            "industrial": ["place_industrial_zone"],
            "road": ["build_road"],
            "power": ["build_power_lines", "build_power_plant"],
            "water": ["build_water_pipe", "build_water_tower", "build_water_treatment"],
            "service": ["build_services"],
            "building": ["build_services", "build_education", "build_healthcare"],
            "budget": ["increase_budget", "open_budget_panel"],
            "tax": ["decrease_taxes", "increase_taxes", "open_budget_panel"],
            "public transport": ["build_public_transportation"],
            "bus": ["build_public_transportation"],
            "train": ["build_public_transportation"],
            "metro": ["build_public_transportation"],
            "education": ["build_education"],
            "school": ["build_education"],
            "university": ["build_education"],
            "healthcare": ["build_healthcare"],
            "hospital": ["build_healthcare"],
            "police": ["build_services"],
            "fire": ["build_services"],
            "park": ["build_parks_and_recreation"],
            "recreation": ["build_parks_and_recreation"],
            "pollution": ["address_visible_issue"],
            "traffic": ["address_visible_issue", "upgrade_roads"],
            "upgrade": ["upgrade_roads", "upgrade_services"],
            "speed": ["set_game_speed"],
            "explore": ["explore_menu", "explore_ui"]
        }
        
        # Log all keyword matches for debugging
        matched_keywords = []
        
        # Check for keyword matches
        for keyword, actions in keyword_mappings.items():
            if keyword in text_description:
                matching_actions.extend(actions)
                matched_keywords.append(keyword)
        
        if matched_keywords:
            self.logger.debug(f"Text '{text_description}' matched keywords: {matched_keywords}")
        
        # If we have our vision-guided actions, always include them
        vision_actions = ["follow_population_advice", "address_visible_issue"]
        
        # Add our vision-guided actions with lower priority
        matching_actions.extend(vision_actions)
        
        # Remove duplicates while preserving order
        seen = set()
        unique_actions = []
        for action in matching_actions:
            if action not in seen:
                seen.add(action)
                
                # Only add if the action exists in our action space
                if self._get_action_idx_by_name(action) is not None:
                    unique_actions.append(action)
        
        return unique_actions
    
    def _get_action_idx_by_name(self, action_name: str) -> Optional[int]:
        """Get the index of an action by its name"""
        for i, action in enumerate(self.action_handlers):
            if action.name == action_name:
                return i
        return None
    
    def _follow_population_growth_advice(self) -> bool:
        """Follow the advice from the vision model to grow population"""
        vision_interface = self._get_vision_interface()
        if not vision_interface:
            self.logger.warning("Vision interface not available for population growth advice")
            return False
        
        try:
            # Get population growth guidance
            guidance = vision_interface.get_population_growth_guidance()
            
            if "error" in guidance:
                self.logger.warning(f"Error in population guidance: {guidance['error']}")
                return False
            
            # Extract recommended actions
            if "recommended_actions" in guidance and guidance["recommended_actions"]:
                # Take the first recommended action
                recommendation = guidance["recommended_actions"][0]
                
                # Log the recommendation
                self.logger.info(f"Following vision guidance: {recommendation}")
                
                # Try to map this to a concrete action
                matching_actions = self._find_matching_actions(recommendation)
                
                if matching_actions:
                    # Take the first matching action
                    action_name = matching_actions[0]
                    action_idx = self._get_action_idx_by_name(action_name)
                    
                    if action_idx is not None:
                        # Get the action and execute it
                        action = self.action_handlers[action_idx]
                        success = action.action_fn()
                        
                        # Update statistics
                        self.vision_guided_actions_taken += 1
                        if success:
                            self.successful_vision_guided_actions += 1
                            self.logger.info(f"Successfully followed population advice with action: {action_name}")
                        else:
                            self.logger.warning(f"Failed to follow population advice with action: {action_name}")
                        
                        # Log this action
                        self._log_action(
                            action_idx=action_idx,
                            action_name=action_name,
                            is_vision_guided=True,
                            success=success,
                            reward=0,
                            info={}
                        )
                        
                        return success
            
            self.logger.warning("No actionable population growth advice found")
            return False
        except Exception as e:
            self.logger.error(f"Error following population growth advice: {str(e)}")
            return False
    
    def _address_visible_issue(self) -> bool:
        """Address issues detected by the vision model"""
        vision_interface = self._get_vision_interface()
        if not vision_interface:
            self.logger.warning("Vision interface not available for addressing issues")
            return False
        
        try:
            # Get population growth guidance which contains issues
            guidance = vision_interface.get_population_growth_guidance()
            
            if "error" in guidance:
                self.logger.warning(f"Error getting issues: {guidance['error']}")
                return False
            
            # Extract issues to address
            if "issues_to_address" in guidance and guidance["issues_to_address"]:
                # Take the first issue
                issue = guidance["issues_to_address"][0]
                
                # Log the issue
                self.logger.info(f"Addressing vision-detected issue: {issue}")
                
                # Try to map this to a concrete action
                matching_actions = self._find_matching_actions(issue)
                
                if matching_actions:
                    # Take the first matching action
                    action_name = matching_actions[0]
                    action_idx = self._get_action_idx_by_name(action_name)
                    
                    if action_idx is not None:
                        # Get the action and execute it
                        action = self.action_handlers[action_idx]
                        success = action.action_fn()
                        
                        # Update statistics
                        self.vision_guided_actions_taken += 1
                        if success:
                            self.successful_vision_guided_actions += 1
                            self.logger.info(f"Successfully addressed issue with action: {action_name}")
                        else:
                            self.logger.warning(f"Failed to address issue with action: {action_name}")
                        
                        # Log this action
                        self._log_action(
                            action_idx=action_idx,
                            action_name=action_name,
                            is_vision_guided=True,
                            success=success,
                            reward=0,
                            info={}
                        )
                        
                        return success
            
            self.logger.warning("No actionable issues found")
            return False
        except Exception as e:
            self.logger.error(f"Error addressing visible issue: {str(e)}")
            return False
    
    def _get_vision_interface(self) -> Optional[OllamaVisionInterface]:
        """Helper to get the vision interface if it's the right type"""
        if isinstance(self.env.interface, OllamaVisionInterface):
            return self.env.interface
        return None
    
    def _parse_position_text(self, position_text: str) -> Optional[Tuple[int, int]]:
        """Try to parse a position description into screen coordinates"""
        position_text = position_text.lower()
        
        # Get screen dimensions
        if isinstance(self.env.interface, OllamaVisionInterface):
            width = self.env.interface.screen_region[2]
            height = self.env.interface.screen_region[3]
        else:
            # Fallback to standard resolution
            width, height = 1920, 1080
        
        self.logger.debug(f"Parsing position text: '{position_text}' for screen {width}x{height}")
        
        # Map text positions to coordinates
        if "top left" in position_text:
            return (width // 4, height // 4)
        elif "top right" in position_text:
            return (width * 3 // 4, height // 4)
        elif "bottom left" in position_text:
            return (width // 4, height * 3 // 4)
        elif "bottom right" in position_text:
            return (width * 3 // 4, height * 3 // 4)
        elif "top" in position_text:
            return (width // 2, height // 4)
        elif "bottom" in position_text:
            return (width // 2, height * 3 // 4)
        elif "left" in position_text:
            return (width // 4, height // 2)
        elif "right" in position_text:
            return (width * 3 // 4, height // 2)
        elif "center" in position_text or "middle" in position_text:
            return (width // 2, height // 2)
        
        # No match found
        self.logger.warning(f"Could not parse position text: '{position_text}'")
        return None 