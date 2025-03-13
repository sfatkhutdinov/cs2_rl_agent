# Integration Testing

The CS2 reinforcement learning agent implements several integration testing approaches to validate the interactions between different components of the system. These tests focus on ensuring that the various subsystems work together correctly, particularly at critical integration points.

## Integration Test Categories

The integration tests in the codebase can be classified into several categories:

### 1. Agent-Environment Integration

Tests that verify the proper interaction between agent implementations and environment interfaces:

- `test_adaptive_modes.py`: Tests the adaptive agent's ability to switch between different operating modes based on environmental conditions and performance metrics.

```python
def main():
    """Test the AdaptiveAgent class."""
    logger.info("Loading configuration...")
    
    # Create a configuration for testing
    config = {
        "agent": {
            "adaptive": {
                "base_dir": "models/adaptive",
                "mode_switch_thresholds": {
                    "autonomous_to_strategic": {
                        "confidence": 0.8,
                        "game_cycles": 10
                    },
                    "strategic_to_autonomous": {
                        "confidence": 0.3,
                        "timeout": 60
                    }
                }
            }
        }
    }
    
    # Create the agent
    logger.info("Creating AdaptiveAgent...")
    agent = AdaptiveAgent(config, test_mode=True)
    
    # Test all modes
    all_modes = [
        TrainingMode.AUTONOMOUS,
        TrainingMode.STRATEGIC,
        TrainingMode.HYBRID
    ]
    
    results = {}
    
    for mode in all_modes:
        logger.info(f"Testing mode: {mode.value}")
        
        # Set the current mode
        agent.current_mode = mode
        
        try:
            # Initialize the current mode
            agent.initialize_current_mode()
            results[mode.value] = "Success"
            logger.info(f"✅ Successfully initialized {mode.value} mode")
        except Exception as e:
            results[mode.value] = f"Failed: {str(e)}"
            logger.error(f"❌ Failed to initialize {mode.value} mode: {e}")
```

This test verifies:
- Proper initialization of the adaptive agent
- Mode switching logic
- Configuration of different agent modes
- Error handling during mode transitions

### 2. Vision-Interface Integration

Tests that verify the interaction between vision processing components and game interfaces:

- `test_vision_windows.py`: Tests the vision window management system's integration with the game interface.
- `auto_detect.py`: Tests the automatic UI detection system's integration with the game interface.

```python
def run_detection_test(interface, logger):
    """Run detection test."""
    logger.info("Testing automatic UI detection")
    
    # Connect to the game
    logger.info("Connecting to Cities: Skylines 2...")
    if interface.connect():
        logger.info("Connection successful!")
        
        # Test UI element detection
        logger.info("Detecting UI elements...")
        if interface.detect_ui_elements():
            logger.info("UI elements detected successfully!")
            
            # Print detected elements
            logger.info("Detected UI elements:")
            for element, data in interface.ui_element_cache.items():
                region = data["region"]
                confidence = data["confidence"]
                logger.info(f"  {element}: region={region}, confidence={confidence:.2f}")
            
            # Get metrics
            logger.info("Extracting metrics...")
            metrics = interface.get_metrics()
            logger.info(f"Metrics: {metrics}")
```

These tests verify:
- Connection to the game interface
- UI element detection and recognition
- Metric extraction from visual elements
- Window management and focus control

### 3. External API Integration

Tests that verify integration with external APIs and services:

- `test_api.py`: Tests integration with the game's API interface
- `test_ollama.py`: Tests integration with the Ollama vision model service

```python
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
```

These tests verify:
- API connection and authentication
- Request/response handling
- Error handling for network issues
- Data parsing and validation

## End-to-End Workflow Testing

The codebase implements end-to-end testing through batch files that simulate complete user workflows:

```
┌─────────────────────────────────────────────┐
│       all_in_one_setup_and_train.bat        │
└─────────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────┐
│           Environment Setup                 │
│                                             │
│  - Conda environment creation               │
│  - Package installation                     │
│  - GPU detection and configuration          │
└─────────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────┐
│           Component Testing                 │
│                                             │
│  - Configuration testing                    │
│  - CS2Environment testing                   │
│  - Focus and screenshot testing             │
└─────────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────┐
│           External Dependencies              │
│                                             │
│  - Ollama service connectivity              │
│  - Vision model availability                │
│  - Directory structure verification         │
└─────────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────┐
│           Training Execution                │
│                                             │
│  - Environment initialization               │
│  - Agent creation and configuration         │
│  - Training loop execution                  │
└─────────────────────────────────────────────┘
```

This approach provides several benefits:
- Validates the complete system from setup to execution
- Tests interactions between multiple components
- Verifies environment configuration and dependencies
- Identifies integration issues that might not appear in isolated tests

## Integration Test Challenges

The integration testing approach faces several challenges:

1. **External Dependencies**: Many integration tests require external dependencies like:
   - The actual CS2 game running
   - The Ollama service running with the correct model
   - CUDA and GPU support properly configured

2. **State Management**: Integration tests need to handle complex state:
   - Game state needs to be reset between tests
   - Environment configuration must be consistent
   - External services may have changing state

3. **Reproducibility**: Integration tests can be difficult to reproduce due to:
   - Timing issues with external services
   - Variations in system performance
   - Non-deterministic game behavior

## Integration Test Environment Isolation

To mitigate integration challenges, the codebase implements environment isolation strategies:

```python
# Test environment configuration that minimizes external dependencies
config = {
    "environment": {
        "type": "CS2Environment",
        "observation_space": {
            "include_visual": True,
            "include_metrics": True,
            "metrics": ["population", "happiness", "budget_balance"]
        },
        "action_space": {
            "zone": ["residential", "commercial", "industrial"],
            "infrastructure": ["road", "water", "electricity"],
            "budget": ["increase_tax", "decrease_tax"]
        }
    },
    "use_fallback_mode": True  # Enable simulation mode to avoid game dependency
}
```

This approach allows integration tests to run in isolated environments by:
- Using fallback modes to simulate external dependencies
- Configuring test-specific parameters
- Mocking external service responses when needed
- Providing predictable initial states

## Related Sections
- [Introduction](01_testing_intro.md)
- [Testing Architecture Overview](02_testing_architecture.md)
- [Unit Testing Framework](03_unit_testing.md)
- [Simulation Environment](05_simulation_environment.md)
- [Test Automation and CI/CD](07_test_automation.md) 