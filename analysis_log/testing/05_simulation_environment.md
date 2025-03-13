# Simulation Environment

The CS2 reinforcement learning agent utilizes a simulation environment to enable development, testing, and validation without requiring the actual game to be running. This simulation environment plays a critical role in the testing infrastructure by providing a controlled, reproducible environment for testing agent behavior.

## Fallback Mode Architecture

The core of the simulation environment is the fallback mode implemented in the CS2Environment class:

```python
def _setup_fallback_mode(self):
    """Set up fallback mode when the game is not available."""
    self.logger.warning("Game connection failed. Using fallback simulation mode.")
    self.in_fallback_mode = True
    self.fallback_metrics = {
        "population": 0,
        "happiness": 50.0,
        "budget_balance": 10000.0,
        "traffic": 50.0,
        "noise_pollution": 0.0,
        "air_pollution": 0.0
    }
    self.interface = None  # Clear the interface as it's not needed
```

This architecture provides several key components:

1. **State Representation**: The fallback mode maintains a simplified representation of the game state, including key metrics like population, happiness, and budget.

2. **Action Simulation**: When actions are taken, the fallback mode simulates their effects on the game state:

```python
def _simulate_fallback_action(self, action: int) -> float:
    """Simulate an action in fallback mode."""
    action_type = self._get_action_type(action)
    
    if action_type == "zone":
        # Zoning actions grow population
        self.fallback_metrics["population"] += max(5, int(self.fallback_metrics["population"] * self.fallback_growth_rate))
        self.fallback_metrics["happiness"] -= self.fallback_happiness_decay
        self.fallback_metrics["budget_balance"] += self.fallback_budget_rate
        
        # More traffic with more population
        if self.fallback_metrics["population"] > 1000:
            self.fallback_metrics["traffic"] = min(100, self.fallback_metrics["traffic"] + 0.5)
            
        reward = 0.05
    # ... other action types ...
    return reward
```

3. **Observation Generation**: The fallback mode generates observations that mimic the structure of real game observations:

```python
def _get_fallback_observation(self):
    """Get observation in fallback mode."""
    observation = {}
    
    # Add metric components
    for metric, value in self.fallback_metrics.items():
        observation[metric] = np.array([value], dtype=np.float32)
    
    # Add dummy visual component if required
    if self.include_visual:
        observation["visual"] = np.zeros((84, 84, 3), dtype=np.uint8)
    
    return observation
```

## Simulation Fidelity

The simulation environment balances simulation fidelity with simplicity:

1. **Simplified Dynamics**: The simulation implements simplified versions of the game's dynamics:
   - Linear population growth based on zoning
   - Simple happiness decay mechanisms
   - Basic budget dynamics

2. **Deterministic Behavior**: Unlike the real game, the simulation provides deterministic responses to actions, making test results reproducible.

3. **Configurable Parameters**: The simulation behavior can be configured through parameters:
   ```python
   self.fallback_growth_rate = config.get("fallback.growth_rate", 0.05)
   self.fallback_budget_rate = config.get("fallback.budget_rate", -100.0)
   self.fallback_happiness_decay = config.get("fallback.happiness_decay", -0.1)
   ```

## Testing with the Simulation Environment

The simulation environment facilitates several testing approaches:

### 1. Component Testing

Tests can use the simulation environment to verify component behavior without game dependencies:

```python
def test_create():
    """Test creating a CS2Environment instance with a minimal configuration."""
    try:
        # Create a minimal configuration with fallback mode enabled
        config = {
            "environment": { /* ... */ },
            "use_fallback_mode": True  # Enable fallback simulation
        }
        
        # Create the environment
        env = CS2Environment(config)
        
        # Verify environment functionality
        observation, info = env.reset()
        action = env.action_space.sample()
        next_observation, reward, terminated, truncated, info = env.step(action)
        
        return True
    except Exception as e:
        logging.error(f"Failed to create CS2Environment instance: {e}")
        return False
```

### 2. Agent Training Testing

The simulation enables testing the agent training process:

```python
def test_agent_training():
    """Test agent training with the simulation environment."""
    # Configure environment with fallback mode
    env_config = {
        "environment": { /* ... */ },
        "use_fallback_mode": True
    }
    
    # Create environment
    env = CS2Environment(env_config)
    
    # Configure agent
    agent_config = { /* ... */ }
    agent = PPOAgent(agent_config)
    
    # Run training for a few steps
    for i in range(100):
        observation, info = env.reset()
        done = False
        while not done:
            action = agent.select_action(observation)
            next_observation, reward, terminated, truncated, info = env.step(action)
            agent.store_transition(observation, action, reward, next_observation, terminated)
            done = terminated or truncated
            observation = next_observation
        
        # Update agent
        agent.update()
    
    # Verify agent learned something
    return agent.evaluate(env, episodes=5) > 0
```

### 3. Regression Testing

The simulation enables regression testing of agent behavior:

```python
def test_agent_regression(agent_path, threshold=0.5):
    """Test that an agent meets a performance threshold in simulation."""
    # Load agent
    agent = PPOAgent.load(agent_path)
    
    # Create simulation environment
    env = CS2Environment({"use_fallback_mode": True})
    
    # Evaluate agent
    mean_reward = agent.evaluate(env, episodes=10)
    
    # Check against threshold
    return mean_reward >= threshold
```

## Simulation Limitations

The simulation environment has several limitations:

1. **Simplified Dynamics**: The simulation cannot capture the full complexity of the game dynamics, potentially leading to agents that perform well in simulation but poorly in the real game.

2. **Missing Visual Elements**: The simulation provides dummy visual observations, which limits testing of vision-based components.

3. **Limited Interactions**: The simulation only supports basic interactions, missing many of the complexities of the real game interface.

4. **No UI Representation**: The simulation lacks a UI representation, making it unsuitable for testing UI interaction components.

## Simulation Evolution

The simulation environment evolves over time to better match the real game:

```python
class EnhancedSimulationEnvironment(CS2Environment):
    """Enhanced simulation environment with more realistic dynamics."""
    
    def __init__(self, config):
        super().__init__(config)
        
        # Load real game data to calibrate simulation
        self.real_data = self._load_real_game_data()
        
        # Initialize enhanced simulation parameters
        self._initialize_enhanced_simulation()
    
    def _initialize_enhanced_simulation(self):
        """Initialize enhanced simulation based on real game data."""
        # Calculate growth rates from real data
        self.fallback_growth_rate = self._calculate_growth_rate(self.real_data)
        
        # Calculate budget dynamics from real data
        self.fallback_budget_rate = self._calculate_budget_rate(self.real_data)
        
        # Calculate happiness dynamics from real data
        self.fallback_happiness_decay = self._calculate_happiness_decay(self.real_data)
    
    def _simulate_fallback_action(self, action):
        """Enhanced simulation with more realistic dynamics."""
        # Use enhanced simulation logic based on real game data
        # ...
```

This evolution process improves the fidelity of testing over time.

## Related Sections
- [Introduction](01_testing_intro.md)
- [Testing Architecture Overview](02_testing_architecture.md)
- [Unit Testing Framework](03_unit_testing.md)
- [Integration Testing](04_integration_testing.md)
- [Performance Testing](06_performance_testing.md) 