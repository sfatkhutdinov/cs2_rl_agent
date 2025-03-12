"""
Action handlers for the CS2 reinforcement learning agent.
"""

import logging
from enum import Enum
from typing import Any, Callable, Dict, Optional


class ActionType(Enum):
    """Types of actions that can be performed."""
    MOUSE = "mouse"
    KEYBOARD = "keyboard"
    CAMERA = "camera"
    GAME_ACTION = "game_action"
    MENU_ACTION = "menu_action"
    COMBINATION = "combination"


class Action:
    """
    Represents an action that can be performed in the game.
    """
    
    def __init__(self, 
                 name: str, 
                 action_fn: Callable[[], bool], 
                 action_type: ActionType = ActionType.GAME_ACTION,
                 params: Optional[Dict[str, Any]] = None):
        """
        Initialize an action.
        
        Args:
            name: The name of the action
            action_fn: The function to call to perform the action
            action_type: The type of action
            params: Additional parameters for the action
        """
        self.name = name
        self.action_fn = action_fn
        self.action_type = action_type
        self.params = params or {}
        self.logger = logging.getLogger(f"Action.{name}")
    
    def execute(self) -> bool:
        """
        Execute the action.
        
        Returns:
            True if the action was successful, False otherwise
        """
        try:
            result = self.action_fn()
            if isinstance(result, bool):
                return result
            return True  # Assume success if no boolean result
        except Exception as e:
            self.logger.error(f"Error executing action {self.name}: {str(e)}")
            return False


class ActionHandler:
    """
    Base class for handling game actions.
    """
    
    def __init__(self, name: str = "ActionHandler"):
        """
        Initialize the action handler.
        
        Args:
            name: The name of this handler
        """
        self.name = name
        self.logger = logging.getLogger(f"ActionHandler.{name}")
        self.actions = {}
    
    def register_action(self, action: Action):
        """
        Register an action with this handler.
        
        Args:
            action: The action to register
        """
        self.actions[action.name] = action
        self.logger.debug(f"Registered action: {action.name}")
    
    def execute_action(self, action_name: str) -> bool:
        """
        Execute an action by name.
        
        Args:
            action_name: The name of the action to execute
            
        Returns:
            True if the action was successful, False otherwise
        """
        if action_name in self.actions:
            self.logger.debug(f"Executing action: {action_name}")
            return self.actions[action_name].execute()
        else:
            self.logger.warning(f"Action not found: {action_name}")
            return False
    
    def get_action_names(self):
        """
        Get the names of all registered actions.
        
        Returns:
            List of action names
        """
        return list(self.actions.keys()) 