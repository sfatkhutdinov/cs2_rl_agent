# Training Scripts Overview

*Last updated: 2024-03-13 - Initial creation of training scripts overview*

**Tags:** #training #architecture #comparison #summary

## Context

The CS2 reinforcement learning agent employs multiple training approaches, each designed to address specific aspects of the agent's learning process. This document provides a comparative analysis of the different training scripts, their purposes, and how they relate to one another within the overall training ecosystem.

## Methodology

This analysis was performed by examining:
- All training scripts in the root directory
- The agent implementations they utilize
- Their configuration files and command-line parameters
- The relationships and dependencies between different training approaches

## Findings

### Training Script Classification

The CS2 RL agent includes several distinct training scripts, each with a specific focus:

1. **Basic Training Scripts**
   - `train_discovery.py` - Focuses on UI element discovery and basic interactions
   - `train_tutorial_guided.py` - Focuses on learning through guided tutorials
   - `train_vision_guided.py` - Emphasizes visual interpretation training

2. **Advanced Training Scripts**
   - `train_autonomous.py` - Trains for independent gameplay with minimal guidance
   - `train_strategic.py` - Focuses on high-level strategic decision-making

3. **Meta-Training Scripts**
   - `train_adaptive.py` - Integrates all training modes with dynamic switching

### Comparison of Training Approaches

| Training Script | Focus Area | Environment Type | Input Complexity | Action Space | Primary Learning Goal |
|-----------------|------------|------------------|-----------------|--------------|----------------------|
| train_discovery.py | UI Navigation | DiscoveryEnvironment | Low | Discrete, limited | Learn to navigate UI and basic controls |
| train_tutorial_guided.py | Basic Mechanics | TutorialEnvironment | Medium | Discrete | Learn game mechanics through guided scenarios |
| train_vision_guided.py | Visual Understanding | VisionEnvironment | High | Discrete | Interpret visual elements and act accordingly |
| train_autonomous.py | Independent Play | AutonomousEnvironment | Very High | Discrete & Continuous | Play independently with minimal guidance |
| train_strategic.py | Strategy | StrategicEnvironment | Very High | Complex, hierarchical | Make high-level strategic decisions |
| train_adaptive.py | All Areas | Multiple Environments | Varies | Varies | Adaptive learning across all domains |

### Script Execution Flow

The training scripts generally follow a similar execution pattern:

1. Parse command-line arguments and load configuration
2. Set up the environment with appropriate wrappers
3. Initialize the agent with the appropriate policy
4. Execute the training loop with callbacks for monitoring
5. Save the trained model and evaluation results

However, the specific implementation varies based on the training focus:

- **Discovery training** emphasizes exploration and simple reward signals
- **Tutorial-guided training** provides more structured learning with staged progression
- **Vision-guided training** integrates computer vision components
- **Autonomous training** focuses on independent decision-making
- **Strategic training** emphasizes long-term planning and complex goal structures
- **Adaptive training** orchestrates all of these approaches dynamically

### Batch Script Integration

Each Python training script is accompanied by one or more `.bat` scripts that:
- Set up the appropriate environment variables
- Configure paths and dependencies
- Launch the training with default or customized parameters

These batch scripts enable easier execution in Windows environments and provide standardized configuration for different training scenarios.

## Relationship to Other Components

The training scripts integrate with:

1. **Agent Module**: Each training script initializes and uses a specific agent type
2. **Environment Module**: Training scripts configure and interact with specific environment implementations
3. **Configuration System**: Scripts load and apply configuration settings
4. **Callback System**: Integration of monitoring, logging, and checkpoint capabilities

## Optimization Opportunities

1. **Unified Training Interface**: Develop a more consistent API across all training scripts
2. **Configuration Standardization**: Standardize configuration parameters and naming conventions
3. **Improved Progress Monitoring**: Implement more sophisticated visualization of training progress
4. **Enhanced Integration Testing**: Develop tests for training script interactions
5. **Documentation Improvements**: Better document the relationships between different training approaches

## Next Steps

Further investigation should focus on:

1. Analyzing the performance differences between different training approaches
2. Documenting the environment implementations in detail
3. Exploring the integration between training scripts and the evaluation system
4. Creating a visual diagram of the relationships between different training approaches

## References

- [Adaptive Agent Training](adaptive_agent_training.md) - Detailed analysis of the adaptive training approach
- [Strategic Agent Training](strategic_agent_training.md) - Detailed analysis of the strategic training approach
- [Discovery-Based Training](discovery_training.md) - Detailed analysis of the discovery-based training approach
- [Training Architecture](../architecture/comprehensive_architecture.md) - Overall architecture of the training system 