# Discovery-Based Training Analysis

*Last updated: 2024-03-19 - Initial creation of discovery training analysis*

**Tags:** #training #agent #discovery #exploration

## Context

The discovery-based training approach is designed to enable the agent to learn game mechanics through exploration rather than predefined rules. This training method focuses on UI navigation, element discovery, and basic interaction patterns as a foundation for more complex gameplay strategies. This analysis examines the implementation of the discovery training system and its role within the CS2 reinforcement learning ecosystem.

## Methodology

This analysis was performed by examining the following components:
- `training/train_discovery.py` - Main training script for discovery-based agents
- `src/agent/discovery_agent.py` - Core implementation of the discovery agent
- `src/environment/discovery_env.py` - The discovery environment implementation
- Related batch scripts and configuration files

The analysis focuses on understanding the exploration mechanisms, reward structures, and learning approach used in discovery-based training.

## Findings

### Discovery Training Architecture

The discovery training system is built around three key components:

1. **DiscoveryEnvironment**: A specialized environment extending the VisionGuidedCS2Environment that focuses on UI exploration and element discovery
2. **DiscoveryAgent**: Agent implementation focused on learning through exploration and discovery
3. **PPO-based Learning**: Uses Proximal Policy Optimization for reinforcement learning with tailored reward functions

The system leverages computer vision for UI element detection and uses a combination of guided and random exploration to learn game mechanics.

### Training Process

The training process in `train_discovery.py` follows these key steps:

1. Configuration loading with flexible reward focus options (exploration vs. goal-oriented)
2. Environment creation with the DiscoveryEnvironment
3. Feature extraction setup using the CombinedExtractor
4. Model initialization with custom policy network architecture
5. Training loop with checkpointing and logging
6. Evaluation of discovery performance metrics

The script provides command-line arguments to customize training, including:
- `--config`: Path to the configuration file
- `--checkpoint`: Path to continue training from a previous checkpoint
- `--timesteps`: Override total timesteps in config
- `--goal-focus`: Emphasize city-building goals in rewards
- `--exploration-focus`: Emphasize exploration in rewards

### Reward Structure

The discovery training implements a multi-faceted reward structure that balances:

1. **Exploration Rewards**: Given for discovering new UI elements and menus
2. **Interaction Rewards**: Given for successful interactions with game elements
3. **Tutorial Completion Rewards**: Given for following and completing tutorials
4. **Goal-oriented Rewards**: Optional rewards tied to city-building metrics

The balance between these reward components can be adjusted through configuration, with specific command-line arguments (`--goal-focus` and `--exploration-focus`) to emphasize different learning priorities.

### Discovery Process

The discovery process implements several key mechanisms:

1. **Menu Exploration**: Systematic exploration of game menus to discover UI elements
2. **Random Action Sampling**: Occasional random actions to discover unexpected interactions
3. **Tutorial Detection**: Identification of tutorial elements to guide learning
4. **Action Sequencing**: Learning successful sequences of actions that accomplish game tasks

These mechanisms allow the agent to build a repertoire of interactions with the game environment progressively.

## Relationship to Other Components

The discovery training system interfaces with:

1. **Adaptive Agent**: Provides a foundation for the adaptive agent's discovery mode
2. **Vision System**: Uses computer vision to identify UI elements and interpret the game state
3. **Configuration System**: Uses specialized configuration for discovery-focused training
4. **Deployment Scripts**: Integrated into batch scripts for easy deployment and training

## Optimization Opportunities

1. **Improved Element Recognition**: Enhance the vision system's ability to recognize UI elements
2. **More Efficient Exploration**: Implement smarter exploration strategies to reduce redundant actions
3. **Better Knowledge Transfer**: Improve transfer of discovered knowledge to other training modes
4. **Curriculum Learning**: Implement progressive difficulty in discovery tasks
5. **Exploration Efficiency Metrics**: Better measure and optimize the efficiency of the discovery process

## Next Steps

Further investigation should focus on:

1. Analyzing the effectiveness of different reward structures in discovery training
2. Documenting the integration between discovery training and autonomous training
3. Exploring the vision-guided aspects of the discovery environment
4. Measuring the impact of discovery pre-training on overall agent performance

## References

- [Training Scripts Overview](training_scripts_overview.md) - Overview of all training approaches
- [Adaptive Agent Training](adaptive_agent_training.md) - Related adaptive training that incorporates discovery mode
- [Strategic Agent Training](strategic_agent_training.md) - Advanced training building on discovery fundamentals 