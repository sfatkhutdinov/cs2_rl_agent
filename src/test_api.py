#!/usr/bin/env python
"""
Test script for the Cities: Skylines 2 RL Agent Bridge API.

This script tests the connection to the bridge mod and performs some basic actions.
"""

import time
import json
import argparse
import requests
from typing import Dict, Any


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Test the Cities: Skylines 2 RL Agent Bridge API")
    
    parser.add_argument("--host", type=str, default="localhost",
                        help="API host (default: localhost)")
    parser.add_argument("--port", type=int, default=5000,
                        help="API port (default: 5000)")
    parser.add_argument("--timeout", type=int, default=10,
                        help="API request timeout in seconds (default: 10)")
    
    return parser.parse_args()


def test_connection(base_url: str, timeout: int) -> bool:
    """
    Test connection to the API.
    
    Args:
        base_url: Base URL for the API
        timeout: Request timeout in seconds
        
    Returns:
        True if connection was successful, False otherwise
    """
    try:
        print(f"Testing connection to {base_url}...")
        response = requests.get(f"{base_url}/state", timeout=timeout)
        
        if response.status_code == 200:
            print("Connection successful!")
            print(f"Game state: {json.dumps(response.json(), indent=2)}")
            return True
        else:
            print(f"Connection failed: API returned status code {response.status_code}")
            return False
    
    except requests.exceptions.RequestException as e:
        print(f"Connection failed: {str(e)}")
        return False


def perform_action(base_url: str, action: Dict[str, Any], timeout: int) -> bool:
    """
    Perform an action via the API.
    
    Args:
        base_url: Base URL for the API
        action: Action to perform
        timeout: Request timeout in seconds
        
    Returns:
        True if the action was performed successfully, False otherwise
    """
    try:
        print(f"Performing action: {json.dumps(action)}")
        response = requests.post(
            f"{base_url}/action",
            json=action,
            timeout=timeout
        )
        
        if response.status_code == 200:
            result = response.json()
            success = result.get("success", False)
            print(f"Action {'successful' if success else 'failed'}")
            return success
        else:
            print(f"Action failed: API returned status code {response.status_code}")
            return False
    
    except requests.exceptions.RequestException as e:
        print(f"Action failed: {str(e)}")
        return False


def test_basic_actions(base_url: str, timeout: int) -> None:
    """
    Test basic game actions.
    
    Args:
        base_url: Base URL for the API
        timeout: Request timeout in seconds
    """
    # Test game speed control
    action = {
        "type": "game_control",
        "control_type": "speed",
        "value": 3
    }
    
    if perform_action(base_url, action, timeout):
        print("Game speed set to 3 (fast)")
    
    time.sleep(2)
    
    # Get updated state
    response = requests.get(f"{base_url}/state", timeout=timeout)
    if response.status_code == 200:
        state = response.json()
        speed = state.get("simulationSpeed", 0)
        print(f"Current game speed: {speed}")
    
    # Set game speed back to normal
    action = {
        "type": "game_control",
        "control_type": "speed",
        "value": 1
    }
    
    perform_action(base_url, action, timeout)
    print("Game speed set back to 1 (normal)")


def test_zone_creation(base_url: str, timeout: int) -> None:
    """
    Test zone creation.
    
    Args:
        base_url: Base URL for the API
        timeout: Request timeout in seconds
    """
    # Create a residential zone
    action = {
        "type": "zone",
        "zone_type": "residential",
        "position": {
            "x": 0,
            "y": 0,
            "z": 0
        }
    }
    
    if perform_action(base_url, action, timeout):
        print("Created residential zone at (0, 0, 0)")
    
    # Create a commercial zone
    action = {
        "type": "zone",
        "zone_type": "commercial",
        "position": {
            "x": 100,
            "y": 0,
            "z": 0
        }
    }
    
    if perform_action(base_url, action, timeout):
        print("Created commercial zone at (100, 0, 0)")
    
    # Create an industrial zone
    action = {
        "type": "zone",
        "zone_type": "industrial",
        "position": {
            "x": 200,
            "y": 0,
            "z": 0
        }
    }
    
    if perform_action(base_url, action, timeout):
        print("Created industrial zone at (200, 0, 0)")


def test_infrastructure_creation(base_url: str, timeout: int) -> None:
    """
    Test infrastructure creation.
    
    Args:
        base_url: Base URL for the API
        timeout: Request timeout in seconds
    """
    # Create a road
    action = {
        "type": "infrastructure",
        "infra_type": "road_straight",
        "position": {
            "x": 0,
            "y": 0,
            "z": 0
        },
        "end_position": {
            "x": 100,
            "y": 0,
            "z": 0
        }
    }
    
    if perform_action(base_url, action, timeout):
        print("Created road from (0, 0, 0) to (100, 0, 0)")
    
    # Create a power line
    action = {
        "type": "infrastructure",
        "infra_type": "power",
        "position": {
            "x": 0,
            "y": 0,
            "z": 50
        },
        "end_position": {
            "x": 100,
            "y": 0,
            "z": 50
        }
    }
    
    if perform_action(base_url, action, timeout):
        print("Created power line from (0, 0, 50) to (100, 0, 50)")
    
    # Create a water pipe
    action = {
        "type": "infrastructure",
        "infra_type": "water",
        "position": {
            "x": 0,
            "y": 0,
            "z": 100
        },
        "end_position": {
            "x": 100,
            "y": 0,
            "z": 100
        }
    }
    
    if perform_action(base_url, action, timeout):
        print("Created water pipe from (0, 0, 100) to (100, 0, 100)")


def test_budget_adjustment(base_url: str, timeout: int) -> None:
    """
    Test budget adjustment.
    
    Args:
        base_url: Base URL for the API
        timeout: Request timeout in seconds
    """
    # Increase residential budget
    action = {
        "type": "budget",
        "budget_action": "increase_residential_budget"
    }
    
    if perform_action(base_url, action, timeout):
        print("Increased residential budget")
    
    # Decrease commercial budget
    action = {
        "type": "budget",
        "budget_action": "decrease_commercial_budget"
    }
    
    if perform_action(base_url, action, timeout):
        print("Decreased commercial budget")


def main():
    """Main entry point."""
    args = parse_args()
    
    base_url = f"http://{args.host}:{args.port}"
    
    if not test_connection(base_url, args.timeout):
        print("Connection test failed. Make sure the game is running and the RL Agent Bridge mod is enabled.")
        return
    
    print("\n--- Testing Basic Actions ---")
    test_basic_actions(base_url, args.timeout)
    
    print("\n--- Testing Zone Creation ---")
    test_zone_creation(base_url, args.timeout)
    
    print("\n--- Testing Infrastructure Creation ---")
    test_infrastructure_creation(base_url, args.timeout)
    
    print("\n--- Testing Budget Adjustment ---")
    test_budget_adjustment(base_url, args.timeout)
    
    print("\nAll tests completed!")


if __name__ == "__main__":
    main() 