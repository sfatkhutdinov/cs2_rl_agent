# Adaptive Agent Orchestration

*Last updated: March 13, 2025 21:06 - Initial documentation of adaptive agent orchestration*

**Tags:** #architecture #agent #orchestration #adaptive #implementation

## Overview

The Adaptive Agent serves as the primary orchestrator in the CS2 RL Agent system, dynamically switching between different specialized agent modes based on performance metrics and game state. This document outlines how the Adaptive Agent coordinates the different specialized agents and manages the overall training and inference process.

## Architecture

### Orchestration Model

The Adaptive Agent implements a meta-controller pattern that:

1. Monitors performance metrics across all agent modes
2. Decides when to switch between modes based on configurable thresholds
3. Transfers knowledge between modes to maintain continuity
4. Manages the training of all specialized agents

```
┌─────────────────────────────────────────────────────────┐
│                     Adaptive Agent                      │
│                  (Meta-Controller)                      │
└───────────┬─────────────┬────────────┬─────────────┬────┘
            │             │            │             │
            ▼             ▼            ▼             ▼
┌───────────────┐ ┌─────────────┐ ┌──────────┐ ┌──────────────┐
│  Discovery    │ │  Tutorial   │ │  Vision  │ │  Autonomous  │
│    Agent      │ │    Agent    │ │  Agent   │ │    Agent     │
└───────────────┘ └─────────────┘ └──────────┘ └──────────────┘
                                                      │
                                                      ▼
                                              ┌──────────────┐
                                              │  Strategic   │
                                              │    Agent     │
                                              └──────────────┘
```

### Mode Switching Logic

The Adaptive Agent uses the following metrics to determine mode switches:

- **Discovery Mode**: UI element discovery count, confidence score
- **Tutorial Mode**: Tutorial step completion rate, confidence score
- **Vision Mode**: Visual interpretation accuracy, confidence score
- **Autonomous Mode**: Reward accumulation, game cycle completion
- **Strategic Mode**: Strategic decision quality, goal achievement rate

Mode switching thresholds are defined in `config/adaptive_config.yaml` and can be adjusted to prioritize different learning objectives.

## Implementation

### Key Components

1. **Training Mode Enum** (`src/agent/adaptive_agent.py`):
   ```python
   class TrainingMode(Enum):
       """Available training modes for the agent"""
       DISCOVERY = "discovery"    # Learn UI elements 
       TUTORIAL = "tutorial"      # Learn basic mechanisms
       VISION = "vision"          # Learn to interpret visual info
       AUTONOMOUS = "autonomous"  # Basic gameplay
       STRATEGIC = "strategic"    # Advanced strategic gameplay with goal discovery
   ```

2. **Mode Configuration Management**:
   The adaptive agent loads configurations for all modes:
   ```python
   def __init__(
       self,
       config: Dict[str, Any],
       discovery_config_path: str = "config/discovery_config.yaml",
       vision_config_path: str = "config/vision_guided_config.yaml",
       autonomous_config_path: str = "config/autonomous_config.yaml",
       tutorial_config_path: str = "config/tutorial_guided_config.yaml",
       strategic_config_path: str = "config/strategic_config.yaml"
   ):
   ```

3. **Agent Instance Management**:
   The adaptive agent creates and manages instances of all specialized agents:
   ```python
   self.agents = {
       TrainingMode.DISCOVERY: DiscoveryAgent(discovery_config, ...),
       TrainingMode.TUTORIAL: TutorialAgent(tutorial_config, ...),
       TrainingMode.VISION: VisionAgent(vision_config, ...),
       TrainingMode.AUTONOMOUS: AutonomousAgent(autonomous_config, ...),
       TrainingMode.STRATEGIC: StrategicAgent(strategic_config, ...)
   }
   ```

4. **Mode Switching Decision Function**:
   ```python
   def _should_switch_mode(self, metrics: Dict[str, Any]) -> Optional[TrainingMode]:
       """Determine if the agent should switch to a different mode based on metrics."""
       current_mode = self.current_mode
       # Logic to determine if mode switch is needed
       # Returns the new mode or None if no switch is needed
   ```

5. **Knowledge Transfer**:
   ```python
   def _transfer_knowledge(self, old_mode: TrainingMode, new_mode: TrainingMode) -> None:
       """Transfer knowledge from old mode agent to new mode agent."""
       old_agent = self.agents[old_mode]
       new_agent = self.agents[new_mode]
       
       # Transfer appropriate knowledge based on mode combination
       if self.config["knowledge_transfer"]["enabled"]:
           # Transfer logic based on mode combination
   ```

## Configuration

The adaptive agent's configuration (`config/adaptive_config.yaml`) contains the following key sections:

1. **Mode Switching Thresholds**:
   ```yaml
   mode_switching:
     min_discovery_confidence: 0.7  
     min_ui_elements: 20
     min_tutorial_steps: 5
     max_stuck_episodes: 5
     min_vision_confidence: 0.6
     min_autonomous_confidence: 0.8
     min_game_cycles: 10
   ```

2. **Knowledge Transfer Settings**:
   ```yaml
   knowledge_transfer:
     enabled: true
     transfer_ui_elements: true
     transfer_action_mappings: true
     transfer_reward_signals: true
     knowledge_retention: 0.7
   ```

3. **Path References to Mode-Specific Configs**:
   ```yaml
   mode_configs:
     discovery: "config/discovery_config.yaml"
     tutorial: "config/tutorial_guided_config.yaml" 
     vision: "config/vision_guided_config.yaml"
     autonomous: "config/autonomous_config.yaml"
     strategic: "config/strategic_config.yaml"
   ```

## Deployment

The adaptive agent is deployed using two simplified scripts:

1. **Streamlined Deployment** (`scripts/deployment/run_adaptive_agent.bat`):
   - Focuses solely on running the adaptive agent
   - Provides command-line options for configuration

2. **All-in-One Setup and Training** (`scripts/deployment/all_in_one_setup_and_train.bat`):
   - Handles environment setup, dependencies, and training
   - Configurable training parameters
   - TensorFlow patching if needed

## Performance Metrics

The adaptive agent tracks the following metrics for each mode:

- **Episode Count**: Number of episodes completed in the mode
- **Reward Average**: Rolling average of rewards in the mode
- **Confidence**: Estimated proficiency in the mode
- **Stuck Episodes**: Count of episodes with minimal progress
- **Mode Duration**: Time spent in each mode
- **Knowledge Transfer Rate**: Effectiveness of knowledge transfer between modes

## Optimization Opportunities

1. **Parallel Training**: Train multiple specialized agents in parallel and synchronize knowledge periodically
2. **Hierarchical Mode Structure**: Implement sub-modes within each major mode for finer-grained control
3. **Meta-Learning**: Apply meta-learning techniques to optimize the mode switching policy itself
4. **Ensemble Methods**: Combine predictions from multiple specialized agents for improved performance
5. **Active Learning**: Prioritize learning in modes where performance is weakest

## References

- [Adaptive Agent System](../components/adaptive_agent.md)
- [Strategic Agent Analysis](../components/strategic_agent.md)
- [Codebase Structure and Dependencies](codebase_mindmap.md)
- [Component Integration](component_integration.md)
- [Adaptive Agent Training](../training/adaptive_agent_training.md) 