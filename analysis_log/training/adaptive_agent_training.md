# Adaptive Agent Training Analysis

*Last updated: March 13, 2025 21:10 - Updated to reflect adaptive agent as central training mechanism*

**Tags:** #training #agent #architecture #adaptive #tensorflow #compatibility #orchestration

## Context

The adaptive agent is a meta-controller system designed to dynamically switch between different training modes based on performance metrics and game state feedback. As the primary orchestrator in the CS2 RL system, it coordinates all specialized agent types, eliminating the need for separate training pipelines. This analysis examines the implementation of the adaptive agent training system, its architecture, and its central role in the project.

## Methodology

This analysis was performed by examining the following components:
- `train_adaptive.py` - Main training script for the adaptive agent
- `src/agent/adaptive_agent.py` - Core implementation of the adaptive agent
- `scripts/training/train_adaptive.bat` - Batch script for running the adaptive agent training
- `scripts/deployment/run_adaptive_agent.bat` - Streamlined deployment script
- `scripts/deployment/all_in_one_setup_and_train.bat` - Comprehensive setup and training script
- `src/utils/patch_tensorflow.py` - TensorFlow compatibility patch utility
- Related configuration files and supporting modules

The analysis focuses on understanding the mode-switching mechanisms, training process, integration with different environment types, and the streamlined deployment architecture.

## Findings

### Centralized Training Architecture

Following the streamlining of the project, the adaptive agent now serves as the central training mechanism for all agent types. This eliminates the need for separate training scripts for each specialized agent, simplifying the system architecture while maintaining full functionality.

Key aspects of this centralized architecture:

1. **Single Training Entry Point**: `train_adaptive.py` handles all training needs
2. **Dynamic Agent Mode Selection**: The adaptive agent determines which specialized agent type is appropriate based on performance metrics
3. **Unified Configuration**: All agent configurations are managed through the adaptive agent's config system
4. **Knowledge Sharing**: Learning from one agent mode can be transferred to other modes

### Adaptive Agent Architecture

The adaptive agent is implemented as a meta-controller that can dynamically switch between five distinct training modes:

1. **Discovery Mode**: Focused on learning UI elements and basic interactions
2. **Tutorial Mode**: Learning basic game mechanisms through guided tutorials
3. **Vision Mode**: Training on interpretation of visual information
4. **Autonomous Mode**: Basic gameplay with limited guidance
5. **Strategic Mode**: Advanced strategic gameplay with goal discovery

The agent maintains internal metrics for each mode and implements a decision-making system to determine when to switch between modes based on performance indicators.

### Streamlined Training Process

The training process in `train_adaptive.py` follows these key steps:

1. Configuration loading and environment setup
2. Initialization of the adaptive agent with configuration for all modes
3. Training loop with periodic evaluation and mode-switching decisions
4. Metrics tracking and visualization
5. Model saving and checkpointing

The script implements progress tracking and visualization tools to monitor the agent's performance across different modes over time.

### Mode-Switching Mechanism

The mode-switching logic is a key component of the adaptive training system:

```python
def _evaluate_mode_switch(self, metrics: Dict[str, Any]) -> Optional[TrainingMode]:
    """Evaluate whether to switch training modes based on current metrics."""
    
    current_mode = self.current_mode
    
    # Check if we've met the criteria to advance from the current mode
    if current_mode == TrainingMode.DISCOVERY:
        if (metrics['confidence'] > self.config['mode_switching']['min_discovery_confidence'] and
            metrics['ui_elements_discovered'] >= self.config['mode_switching']['min_ui_elements']):
            return TrainingMode.VISION
            
    elif current_mode == TrainingMode.VISION:
        if metrics['confidence'] > self.config['mode_switching']['min_vision_confidence']:
            return TrainingMode.AUTONOMOUS
            
    # ... similar logic for other modes
    
    # Check if we're stuck in the current mode
    if metrics['stuck_episodes'] > self.config['mode_switching']['max_stuck_episodes']:
        # Try a different mode if we're stuck
        return self._select_fallback_mode()
        
    return None  # No mode switch needed
```

### Simplified Deployment

The deployment process has been streamlined with two main scripts:

1. **`scripts/deployment/run_adaptive_agent.bat`**:
   - Focused specifically on running the adaptive agent
   - Command-line options for configuration, starting mode, etc.
   - Clean interface for deployment

2. **`scripts/deployment/all_in_one_setup_and_train.bat`**:
   - Comprehensive setup including environment, dependencies, etc.
   - Simplified to focus exclusively on the adaptive agent
   - TensorFlow compatibility handling

This simplification removes the complexity of managing multiple training scripts while preserving full functionality.

### TensorFlow Compatibility

A critical component of the adaptive agent training system is the TensorFlow compatibility patch:

```python
# In train_adaptive.py
try:
    from src.utils.patch_tensorflow import apply_tensorflow_io_patch
    patch_applied = apply_tensorflow_io_patch()
    if patch_applied:
        print("Applied TensorFlow compatibility patch successfully")
except ImportError:
    print("WARNING: Could not import TensorFlow patch module")
except Exception as e:
    print(f"WARNING: Error applying TensorFlow patch: {e}")
```

This patch resolves compatibility issues between different TensorFlow versions, ensuring the training system works consistently across environments.

## Integration with Specialized Agents

The adaptive agent training system integrates with each specialized agent type:

```python
# In adaptive_agent.py
def _initialize_agent_modes(self):
    """Initialize all available agent modes."""
    # Load configurations for each mode
    discovery_config = self._load_config(self.discovery_config_path)
    tutorial_config = self._load_config(self.tutorial_config_path)
    vision_config = self._load_config(self.vision_config_path)
    autonomous_config = self._load_config(self.autonomous_config_path)
    strategic_config = self._load_config(self.strategic_config_path)
    
    # Initialize agents for each mode
    self.agents = {
        TrainingMode.DISCOVERY: DiscoveryAgent(discovery_config, ...),
        TrainingMode.TUTORIAL: TutorialAgent(tutorial_config, ...),
        TrainingMode.VISION: VisionAgent(vision_config, ...),
        TrainingMode.AUTONOMOUS: AutonomousAgent(autonomous_config, ...),
        TrainingMode.STRATEGIC: StrategicAgent(strategic_config, ...)
    }
```

This approach allows each specialized agent to be developed independently while still functioning cohesively within the adaptive framework.

## Performance Metrics and Visualization

The adaptive training system tracks comprehensive metrics across all agent modes and visualizes them to provide insights into training progress:

```python
def plot_mode_history(adaptive_agent, save_path: str) -> None:
    """Plot the agent's mode history over time."""
    plt.figure(figsize=(12, 6))
    
    # Extract timestamps and modes
    timestamps = [ts for ts, _, _ in adaptive_agent.mode_history]
    start_time = timestamps[0]
    rel_times = [(ts - start_time) / 60 for ts in timestamps]  # Minutes
    
    # Get unique modes
    mode_values = []
    seen_modes = set()
    for _, old_mode, _ in adaptive_agent.mode_history:
        if old_mode.value not in seen_modes:
            seen_modes.add(old_mode.value)
            mode_values.append(old_mode.value)
    
    # Plot mode changes
    # ... visualization code ...
    
    plt.savefig(save_path)
```

This visualization shows how the agent transitions between different modes over time, providing valuable insights into its learning process.

## Conclusion

The adaptive agent training system represents a sophisticated approach to reinforcement learning in complex environments. By dynamically switching between specialized agent modes, it can tackle different aspects of gameplay with appropriate strategies while maintaining continuity of learning.

With the streamlining of the project to focus on the adaptive agent as the primary orchestrator, the training process has been simplified without sacrificing functionality. The unified training approach eliminates the need for separate training scripts for each agent type, making the system more maintainable and cohesive.

## References

- [Adaptive Agent Orchestration](../architecture/adaptive_orchestration.md)
- [Adaptive Agent System](../components/adaptive_agent.md)
- [Codebase Structure and Dependencies](../architecture/codebase_mindmap.md)
- [Component Integration](../architecture/component_integration.md)
- [TensorFlow Compatibility Issues](#tensorflow-compatibility-issues) 