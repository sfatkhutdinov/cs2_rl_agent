import logging
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import random
from typing import Dict, Any, Tuple, List, Optional, Union

from src.environment.autonomous_env import AutonomousCS2Environment, ActionType, Action
from src.interface.ollama_vision_interface import OllamaVisionInterface


class VisionGuidedCS2Environment(AutonomousCS2Environment):
    """
    An extension of the autonomous environment that uses the Ollama vision model
    to provide more intelligent guidance to the agent, especially for population growth.
    """
    
    def __init__(self, base_env, exploration_frequency=0.3, random_action_frequency=0.2, 
                 menu_exploration_buffer_size=50, logger=None, vision_guidance_frequency=0.5):
        """
        Initialize the vision-guided environment wrapper.
        
        Args:
            base_env: The base CS2Environment to wrap
            exploration_frequency: How often to trigger exploratory behavior (0-1)
            random_action_frequency: How often to take completely random actions (0-1)
            menu_exploration_buffer_size: Size of the buffer for storing discovered menu items
            logger: Optional logger instance
            vision_guidance_frequency: How often to use vision guidance for actions (0-1)
        """
        super().__init__(
            base_env, 
            exploration_frequency=exploration_frequency,
            random_action_frequency=random_action_frequency,
            menu_exploration_buffer_size=menu_exploration_buffer_size,
            logger=logger
        )
        
        self.logger = logger or logging.getLogger("VisionGuidedEnv")
        
        # Check if our interface is OllamaVisionInterface
        if not isinstance(self.game_interface, OllamaVisionInterface):
            self.logger.warning("The provided interface is not an OllamaVisionInterface. Some features may not work.")
        
        # Parameters for vision guidance
        self.vision_guidance_frequency = vision_guidance_frequency
        
        # Tracking for vision guided actions
        self.vision_guided_actions_taken = 0
        self.successful_vision_guided_actions = 0
        
        # Add vision-guided actions to our action space
        self._add_vision_guided_actions()
        
        self.logger.info("Vision-guided environment initialized with extended action space")
    
    def _add_vision_guided_actions(self):
        """Add vision-specific guided actions to the action space"""
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
            ),
            Action(
                name="click_suggested_ui_element",
                action_fn=lambda: self._click_suggested_ui_element(),
                action_type=ActionType.UI_INTERACTION
            ),
            Action(
                name="take_suggested_action",
                action_fn=lambda: self._take_suggested_action(),
                action_type=ActionType.GAME_ACTION
            )
        ]
        
        # Add the new actions to our action list
        for action in vision_actions:
            self.actions.append(action)
        
        # Update the action space
        self.action_space = spaces.Discrete(len(self.actions))
        
        self.logger.info(f"Added {len(vision_actions)} vision-guided actions to action space")
    
    def _get_vision_interface(self) -> Optional[OllamaVisionInterface]:
        """Helper to get the vision interface if it's the right type"""
        if isinstance(self.game_interface, OllamaVisionInterface):
            return self.game_interface
        return None
    
    def _get_vision_guided_action(self) -> int:
        """Get an action recommendation based on vision model analysis"""
        vision_interface = self._get_vision_interface()
        if not vision_interface:
            return self._get_guided_exploratory_action()
        
        try:
            # Get game state with vision enhancements
            game_state = vision_interface.get_game_state()
            
            # If we have vision-enhanced data and population guidance
            if "vision_enhanced" in game_state and "population_guidance" in game_state:
                guidance = game_state["population_guidance"]
                
                if "error" in guidance:
                    self.logger.warning(f"Error in population guidance: {guidance['error']}")
                    return self._get_guided_exploratory_action()
                
                # If we have recommended actions, pick from them
                if "recommended_actions" in guidance and guidance["recommended_actions"]:
                    action_names = []
                    
                    # Try to map the recommended text actions to our action space
                    for recommendation in guidance["recommended_actions"]:
                        # Find relevant actions in our action space that might match
                        matching_actions = self._find_matching_actions(recommendation)
                        action_names.extend(matching_actions)
                    
                    # If we found any matching actions, randomly select one
                    if action_names:
                        action_idx = self._get_action_idx_by_name(random.choice(action_names))
                        if action_idx is not None:
                            return action_idx
            
            # If we have vision_enhanced but can't get specific guidance, try suggestions
            if "vision_enhanced" in game_state and "suggestions" in game_state["vision_enhanced"]:
                suggestions = game_state["vision_enhanced"]["suggestions"]
                if suggestions:
                    # Find matching actions for any suggestion
                    action_names = []
                    for suggestion in suggestions:
                        matching_actions = self._find_matching_actions(suggestion)
                        action_names.extend(matching_actions)
                    
                    if action_names:
                        action_idx = self._get_action_idx_by_name(random.choice(action_names))
                        if action_idx is not None:
                            return action_idx
            
            # If we can't get specific guidance, use exploration actions
            # Try population-specific actions first
            population_actions = ["follow_population_advice", "address_visible_issue", 
                                 "click_suggested_ui_element", "take_suggested_action"]
            
            # 50% chance to try one of our population-focused actions
            if random.random() < 0.5:
                action_name = random.choice(population_actions)
                action_idx = self._get_action_idx_by_name(action_name)
                if action_idx is not None:
                    return action_idx
            
            # Fall back to regular exploration
            return self._get_guided_exploratory_action()
        
        except Exception as e:
            self.logger.error(f"Error getting vision-guided action: {str(e)}")
            return self._get_guided_exploratory_action()
    
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
        
        # Check for keyword matches
        for keyword, actions in keyword_mappings.items():
            if keyword in text_description:
                matching_actions.extend(actions)
        
        # If we have our vision-guided actions, always include them
        vision_actions = ["follow_population_advice", "address_visible_issue", 
                          "click_suggested_ui_element", "take_suggested_action"]
        
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
        for i, action in enumerate(self.actions):
            if action.name == action_name:
                return i
        return None
    
    def _follow_population_growth_advice(self) -> bool:
        """Follow the advice from the vision model to grow population"""
        vision_interface = self._get_vision_interface()
        if not vision_interface:
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
                        action = self.actions[action_idx]
                        return action.action_fn()
            
            return False
        except Exception as e:
            self.logger.error(f"Error following population growth advice: {str(e)}")
            return False
    
    def _address_visible_issue(self) -> bool:
        """Address issues detected by the vision model"""
        vision_interface = self._get_vision_interface()
        if not vision_interface:
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
                        action = self.actions[action_idx]
                        return action.action_fn()
            
            return False
        except Exception as e:
            self.logger.error(f"Error addressing visible issue: {str(e)}")
            return False
    
    def _click_suggested_ui_element(self) -> bool:
        """Click on a UI element suggested by the vision model"""
        vision_interface = self._get_vision_interface()
        if not vision_interface:
            return False
        
        try:
            # Get enhanced game state
            enhanced_state = vision_interface.get_enhanced_game_state()
            
            if "error" in enhanced_state:
                self.logger.warning(f"Error getting UI elements: {enhanced_state['error']}")
                return False
            
            # Extract UI elements
            if "ui_elements" in enhanced_state and enhanced_state["ui_elements"]:
                # Randomly select a UI element to click
                ui_element = random.choice(enhanced_state["ui_elements"])
                
                element_name = ui_element.get("name", "unknown")
                position = ui_element.get("position", "")
                
                self.logger.info(f"Clicking suggested UI element: {element_name} at {position}")
                
                # Try to map position text to screen coordinates
                coords = self._parse_position_text(position)
                
                if coords:
                    # Click at the coordinates
                    x, y = coords
                    return vision_interface.click_at_coordinates(x, y)
                else:
                    # If we can't get coordinates, see if the element is in our cached UI elements
                    ui_elements = vision_interface.ui_element_cache
                    
                    for name, details in ui_elements.items():
                        if element_name.lower() in name.lower():
                            # Try to click on this element using the interface
                            return vision_interface.perform_action({"type": "click", "target": name})
            
            return False
        except Exception as e:
            self.logger.error(f"Error clicking suggested UI element: {str(e)}")
            return False
    
    def _take_suggested_action(self) -> bool:
        """Take an action suggested by the vision model"""
        vision_interface = self._get_vision_interface()
        if not vision_interface:
            return False
        
        try:
            # Get enhanced game state
            enhanced_state = vision_interface.get_enhanced_game_state()
            
            if "error" in enhanced_state:
                self.logger.warning(f"Error getting suggested actions: {enhanced_state['error']}")
                return False
            
            # Extract available actions
            if "available_actions" in enhanced_state and enhanced_state["available_actions"]:
                # Take the first suggested action
                suggestion = enhanced_state["available_actions"][0]
                
                self.logger.info(f"Taking suggested action: {suggestion}")
                
                # Try to map this to a concrete action
                matching_actions = self._find_matching_actions(suggestion)
                
                if matching_actions:
                    # Take the first matching action
                    action_name = matching_actions[0]
                    action_idx = self._get_action_idx_by_name(action_name)
                    
                    if action_idx is not None:
                        # Get the action and execute it
                        action = self.actions[action_idx]
                        return action.action_fn()
            
            # If no available actions, check suggestions
            if "suggestions" in enhanced_state and enhanced_state["suggestions"]:
                # Take the first suggestion
                suggestion = enhanced_state["suggestions"][0]
                
                self.logger.info(f"Taking suggested improvement: {suggestion}")
                
                # Try to map this to a concrete action
                matching_actions = self._find_matching_actions(suggestion)
                
                if matching_actions:
                    # Take the first matching action
                    action_name = matching_actions[0]
                    action_idx = self._get_action_idx_by_name(action_name)
                    
                    if action_idx is not None:
                        # Get the action and execute it
                        action = self.actions[action_idx]
                        return action.action_fn()
            
            return False
        except Exception as e:
            self.logger.error(f"Error taking suggested action: {str(e)}")
            return False
    
    def _parse_position_text(self, position_text: str) -> Optional[Tuple[int, int]]:
        """Try to parse a position description into screen coordinates"""
        position_text = position_text.lower()
        
        # Get screen dimensions
        if isinstance(self.game_interface, OllamaVisionInterface):
            width = self.game_interface.screen_region[2]
            height = self.game_interface.screen_region[3]
        else:
            # Fallback to standard resolution
            width, height = 1920, 1080
        
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
        
        return None
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """
        Override the step method to incorporate vision guidance.
        
        Args:
            action: The action to take
            
        Returns:
            observation, reward, terminated, truncated, info
        """
        # Decide whether to use vision guidance
        if random.random() < self.vision_guidance_frequency:
            guided_action = self._get_vision_guided_action()
            self.logger.info(f"Using vision guidance to select action {guided_action} instead of {action}")
            action = guided_action
        
        # Take the step using the parent class
        return super().step(action) 