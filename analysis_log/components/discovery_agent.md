# Discovery Agent Implementation Analysis

*Last updated: March 14, 2025 - Initial documentation*

**Tags:** #agent #discovery #implementation #reinforcement-learning #ppo

## Overview
The Discovery Agent is a specialized reinforcement learning agent designed for exploring and discovering UI elements and navigation paths in the game environment. It focuses on menu navigation, interface discovery, and understanding the available interactions within the game, serving as a foundation for more complex strategic agents.

## Implementation Architecture

### Core Components
The Discovery Agent is implemented as a class that wraps a Proximal Policy Optimization (PPO) model from the Stable Baselines 3 library. The key components include:

1. **Model Configuration**: The agent uses a configuration-driven approach to set up the PPO model parameters.
2. **Training Framework**: Built-in methods for episodic and continuous training with checkpoint saving.
3. **Prediction Interface**: Methods for action selection from observations.
4. **State Tracking**: Mechanisms to track discovered elements and training progress.

## Key Implementation Details

### Initialization and Setup
```python
def __init__(self, environment: gym.Env, config: Dict[str, Any]):
    """Initialize the discovery agent."""
    self.env = environment
    self.config = config
    # ... other initialization code ...
    self._setup_model()
```

The agent is initialized with a reference to the environment and a configuration dictionary that controls various aspects of its behavior. During initialization, the agent sets up:
- Logging
- File paths for models and logs
- The PPO model with appropriate parameters
- Metrics tracking for discovered elements

### Model Configuration
```python
def _setup_model(self):
    """Set up the PPO model for the agent."""
    # ... directory creation ...
    
    # Get model parameters from config
    policy_kwargs = {}
    if self.config.get("model", {}).get("use_lstm", False):
        policy_kwargs["lstm_hidden_size"] = self.config.get("model", {}).get("lstm_hidden_size", 64)
        policy_kwargs["net_arch"] = [dict(pi=[64, 64], vf=[64, 64])]
    
    # ... policy determination ...
    
    # Initialize the PPO model
    self.model = PPO(
        policy=policy,
        env=self.env,
        learning_rate=self.config.get("model", {}).get("learning_rate", 3e-4),
        # ... other parameters ...
    )
```

The model setup process includes:
- Policy network configuration (including optional LSTM for memory)
- Automatic policy type selection based on observation space
- Hyperparameter configuration from the provided config dictionary

### Training Process
```python
def train(self, total_timesteps: int, callback=None):
    """Train the agent for a specified number of timesteps."""
    callbacks = self._setup_callbacks(callback)
    
    try:
        self.model.learn(
            total_timesteps=total_timesteps,
            callback=callbacks,
            log_interval=10
        )
        
        # Save the final model
        self.save(os.path.join(self.model_dir, "final_model"))
        # ... success handling ...
        
    except Exception as e:
        # ... error handling and emergency save ...
```

The training process includes:
- Setup of training callbacks (checkpoints, etc.)
- Integration with Stable Baselines 3 learning API
- Automatic model saving at completion
- Error recovery with emergency model saving

### Episode-Based Training
```python
def train_episode(self, timesteps_per_episode: int):
    """Train the agent for a single episode."""
    # ... episode setup ...
    
    # Reset the environment
    obs, _ = self.env.reset()
    done = False
    
    # Run the episode
    for _ in range(timesteps_per_episode):
        # Select action
        action, _ = self.model.predict(obs, deterministic=False)
        
        # Execute action
        next_obs, reward, terminated, truncated, info = self.env.step(action)
        # ... episode tracking ...
        
        # Break if episode is done
        if done:
            break
    
    # ... return episode info ...
```

The episode-based training provides:
- Finer control over training duration
- Comprehensive episode metrics
- Support for episode-specific exploration strategies

### Action Prediction
```python
def predict(self, observation, deterministic=False):
    """Generate a prediction from the model for a given observation."""
    if self.model is None:
        # ... fallback to random action ...
    
    action, state = self.model.predict(observation, deterministic=deterministic)
    return action, state
```

The prediction interface:
- Handles uninitialized model scenarios gracefully
- Supports both deterministic (exploitation) and non-deterministic (exploration) prediction
- Returns both actions and internal state

### Model Persistence
```python
def save(self, path: str):
    """Save the agent's model to a file."""
    # ... implementation ...

def load(self, path: str):
    """Load the agent's model from a file."""
    # ... implementation ...
```

The agent provides straightforward methods for:
- Saving trained models to disk
- Loading pre-trained models
- Error handling for missing files or initialization issues

## Integration with Environment

The Discovery Agent is designed to work with the DiscoveryEnvironment, which provides:
- Specialized observation spaces focusing on UI elements
- Reward structures that incentivize exploration and discovery
- Information about newly discovered elements and navigation paths

The agent-environment interaction is based on the standard Gymnasium (formerly Gym) interface:
```python
# Reset at the beginning
observation, info = environment.reset()

# Step loop
action, _ = agent.predict(observation)
observation, reward, terminated, truncated, info = environment.step(action)
```

## Performance Characteristics

### Strengths
1. **Exploration Focus**: Optimized for discovering new UI elements and game mechanics
2. **Configurability**: Highly configurable through the configuration dictionary
3. **Error Resilience**: Robust error handling with emergency model saving
4. **Memory Capability**: Optional LSTM support for handling sequential dependencies

### Limitations
1. **Computational Requirements**: PPO training can be computationally intensive
2. **Exploration-Exploitation Balance**: May require tuning of exploration parameters
3. **Single-Task Focus**: Primarily designed for discovery rather than strategic gameplay

## Usage Examples

### Basic Training
```python
from src.agent.discovery_agent import DiscoveryAgent
from src.environment.discovery_env import DiscoveryEnvironment

# Create environment
env = DiscoveryEnvironment(config)

# Create and train agent
agent = DiscoveryAgent(env, config)
agent.train(total_timesteps=1000000)
```

### Loading and Using a Pre-trained Agent
```python
# Create environment
env = DiscoveryEnvironment(config)

# Create agent and load model
agent = DiscoveryAgent(env, config)
agent.load("models/discovery/best_model")

# Use for prediction
obs, _ = env.reset()
while True:
    action, _ = agent.predict(obs, deterministic=True)
    obs, reward, done, _, info = env.step(action)
    if done:
        break
```

## Relationship to Other Components

### Dependency on Configuration System
The agent relies heavily on the configuration system to control various aspects of its behavior, including:
- Model architecture and hyperparameters
- Training parameters and checkpointing
- File paths and logging

### Integration with Training Scripts
The Discovery Agent is typically used by training scripts like `train_discovery.py`, which handle:
- Environment setup
- Configuration loading
- Training loops and evaluation
- Model saving and loading

### Foundation for Strategic Agents
The discoveries made by this agent can be used by more sophisticated agents:
- The Strategic Agent can use discovered UI elements for strategic planning
- The Adaptive Agent can leverage navigation paths found by the Discovery Agent

## Future Enhancement Opportunities

1. **Enhanced Exploration Strategies**: Implement more sophisticated exploration methods like intrinsic motivation
2. **Multi-objective Training**: Support for balancing multiple objectives (discovery, navigation efficiency, etc.)
3. **Transfer Learning**: Methods to transfer knowledge from discovery to strategic agents
4. **Active Learning**: Integration with active learning approaches to guide exploration
5. **Parallelized Training**: Support for distributed training across multiple environments

## Related Documentation
- [Discovery Environment Implementation](../training/discovery_environment.md)
- [Training Scripts Overview](../training/training_scripts_overview.md)
- [Discovery-Based Training](../training/discovery_training.md)
- [Strategic Agent Analysis](strategic_agent.md)

---

*For questions or further details about the Discovery Agent implementation, please contact the project maintainers.* 