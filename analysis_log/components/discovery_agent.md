# Discovery Agent Implementation Analysis

*Last updated: 2024-03-19 - Initial creation of discovery agent analysis*

**Tags:** #agent #discovery #implementation #exploration

## Context

The Discovery Agent is a specialized agent implementation focused on exploring and discovering UI elements and basic game mechanics in Cities: Skylines 2. It operates within the DiscoveryEnvironment and serves as the foundation for the early learning phase in the adaptive agent's progression. This analysis examines the implementation of the Discovery Agent, its architecture, capabilities, and integration with the broader CS2 reinforcement learning system.

## Methodology

This analysis was performed by examining the following components:
- `src/agent/discovery_agent.py` - Core implementation of the Discovery Agent
- `src/environment/discovery_env.py` - The environment with which the agent interacts
- `training/train_discovery.py` - The training script that utilizes the Discovery Agent
- Tests and dependencies related to the Discovery Agent functionality

The analysis focuses on understanding the agent's structure, learning mechanisms, and core capabilities for game exploration and UI discovery.

## Findings

### Agent Architecture

The Discovery Agent is implemented as a reinforcement learning agent using the Proximal Policy Optimization (PPO) algorithm from Stable Baselines 3. Its core components include:

1. **Model Configuration**:
   - Uses either MultiInputPolicy (for dictionary observation spaces) or MlpPolicy (for flattened spaces)
   - Supports optional LSTM-based architectures for temporal reasoning
   - Configurable hyperparameters for learning rate, batch size, epochs, and exploration coefficients

2. **State Tracking**:
   - Maintains metrics for total timesteps, episodes, and discovered UI elements
   - Logs training progress and discoveries
   - Implements configurable checkpoint mechanisms

3. **Training Interface**:
   - Provides methods for full training runs and episodic training
   - Handles environment interactions, action selection, and reward processing
   - Implements error recovery and emergency model saving

4. **Persistence Support**:
   - Model saving and loading capabilities
   - Training checkpoint management
   - Configuration persistence for reproducibility

### Learning Process

The Discovery Agent's learning process follows these key steps:

1. **Initialization**: 
   - Setup of the PPO model with appropriate policies based on observation space type
   - Configuration of training parameters and directories

2. **Training Loop**:
   - Environment observation collection
   - Action prediction using the policy model
   - Environment step execution
   - Reward collection and processing
   - Model parameter updates using PPO

3. **Feedback Mechanisms**:
   - Logging of training progress
   - Regular checkpointing for training recovery
   - Episode statistics collection
   - Success metrics tracking

### Key Methods

The Discovery Agent implements several critical methods:

1. **`_setup_model()`**:
   - Creates the PPO model with appropriate policy type
   - Configures PPO hyperparameters based on configuration
   - Handles different observation space structures

2. **`train(total_timesteps, callback=None)`**:
   - Main training method that runs for a specified number of timesteps
   - Integrates with Stable Baselines 3 training
   - Implements checkpoint callbacks
   - Provides error handling and emergency saving

3. **`train_episode(timesteps_per_episode)`**:
   - Runs a single training episode
   - Tracks episode-specific metrics
   - Returns detailed episode information

4. **`predict(observation, deterministic=False)`**:
   - Performs action prediction for a given observation
   - Handles both deterministic (exploitation) and stochastic (exploration) modes
   - Implements fallback for uninitialized models

5. **`save(path)` / `load(path)`**:
   - Persistence methods for saving and loading trained models
   - Error handling for missing files and initialization issues

### Integration with Environment

The Discovery Agent interacts closely with the DiscoveryEnvironment, which provides:

1. **Discovery-specific Observations**:
   - UI element detection
   - Menu structure information
   - Game state metrics

2. **Specialized Action Space**:
   - UI navigation actions
   - Menu interaction actions
   - Generic game control actions

3. **Discovery-focused Rewards**:
   - Rewards for discovering new UI elements
   - Rewards for successful menu navigation
   - Rewards for tutorial progression

## Relationship to Other Components

The Discovery Agent interfaces with several other system components:

1. **Adaptive Agent**: 
   - Utilized as the foundation mode in the adaptive training system
   - Knowledge from discovery phase transferred to other modes
   - UI element discoveries persist in shared knowledge base

2. **Training Framework**:
   - Integrated with train_discovery.py for standalone training
   - Used as a component in the adaptive training pipeline

3. **Testing Infrastructure**:
   - Tests verify its operation with the DiscoveryEnvironment
   - Integration tests with the adaptive mode-switching system

4. **Vision System**:
   - Leverages the vision system for UI element detection
   - Uses vision feedback to guide exploration

## Optimization Opportunities

1. **Enhanced Exploration Strategies**:
   - Implement more sophisticated exploration policies (e.g., curiosity-driven exploration)
   - Add intrinsic motivation rewards to improve discovery efficiency
   - Develop better prioritization of unexplored areas

2. **Improved Knowledge Representation**:
   - More structured representation of discovered UI elements
   - Better encoding of action-element relationships
   - Hierarchical representation of menu structures

3. **Transfer Learning Enhancements**:
   - Improved transfer of discoveries to other agent modes
   - More efficient knowledge sharing between training sessions
   - Better persistence of learned navigation patterns

4. **Model Architecture Improvements**:
   - Test alternative neural network architectures
   - Evaluate transformer-based models for better context understanding
   - Explore hybrid models that combine RL with heuristic approaches

5. **Training Efficiency**:
   - Implement experience replay for more efficient sample use
   - Explore curriculum learning to structure the discovery process
   - Add early stopping based on discovery saturation

## Next Steps

Further investigation should focus on:

1. Analyzing the effectiveness of different PPO hyperparameter configurations for discovery tasks
2. Documenting the interaction patterns between the agent and environment in more detail
3. Evaluating the completeness of UI element discovery compared to manual exploration
4. Measuring discovery efficiency across different environment configurations
5. Testing alternative RL algorithms beyond PPO for the discovery task

## References

- [Discovery-Based Training](../training/discovery_training.md) - Analysis of the discovery training process
- [DiscoveryEnvironment Implementation](../training/discovery_environment.md) - Analysis of the environment the agent operates in
- [Adaptive Agent System](adaptive_agent.md) - Analysis of the adaptive agent that incorporates the discovery agent
- [Training Scripts Overview](../training/training_scripts_overview.md) - Overview of all training approaches 