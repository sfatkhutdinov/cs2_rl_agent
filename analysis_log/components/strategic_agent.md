# Strategic Agent Analysis: Long-term Planning and Causal Reasoning

## Context
This document examines the Strategic Agent implementation to understand how the system implements long-term planning, causal modeling, and goal inference capabilities.

## Methodology
1. Analyzed the `strategic_agent.py` implementation to understand its architecture
2. Examined the `strategic_env.py` environment to understand strategic capabilities
3. Reviewed the `train_strategic.py` script to understand how training is configured
4. Explored the `strategic_config.yaml` file to understand key parameters

## Analysis Findings

### Strategic Agent Architecture
- Builds on the [Adaptive Agent](adaptive_agent.md) framework with specific focus on long-term planning
- Implements PPO algorithm with LSTM networks for temporal memory
- Designed to discover and optimize game strategies autonomously
- Code structure:

```python
class StrategicAgent:
    def __init__(self, environment: gym.Env, config: Dict[str, Any]):
        # Initialize with environment and configuration
        # Set up paths for logs and models
        # Configure knowledge bootstrapping options
        
    def _setup_model(self):
        # Set up PPO with LSTM architecture
        # Configure policy and value networks
        
    def train(self, total_timesteps: int, callback=None):
        # Train the agent with exploration phases
        # Track strategic metrics during training
        
    def predict(self, observation, deterministic=False):
        # Make decisions based on current observation
        # Consider long-term impact of actions
```

### Strategic Environment Capabilities
- Extends the Autonomous Environment with strategic capabilities:
  - Metric discovery and tracking
  - Causal modeling between actions and outcomes
  - Goal inference and prioritization
  - Intrinsic rewards for strategic exploration
- Sophisticated environment wrapper architecture:

```python
class StrategicEnvironment(gym.Wrapper):
    def __init__(self, config: Dict[str, Any]):
        # Initialize with configuration
        # Set up knowledge base and metric tracking
        # Initialize causal modeling components
        
    def step(self, action):
        # Execute action and get basic observation
        # Extract and update metrics
        # Correlate actions with metric changes
        # Calculate intrinsic strategic rewards
        
    def _discover_metrics(self, observation, current_metrics):
        # Autonomously discover new metrics in the game
        
    def _correlate_actions_with_metrics(self, action, pre_metrics, post_metrics):
        # Build causal model of how actions affect metrics
        # Update confidence in causal relationships
        
    def _extract_game_logic(self, game_message):
        # Extract game rules from text messages
        # Update knowledge base with discovered rules
```

### Causal Modeling System
- Implements a sophisticated action-effect model:
  - Tracks recent actions in a deque for capturing delayed effects
  - Correlates metric changes with past actions
  - Builds confidence scores for causal relationships
  - Handles delayed effects through temporal analysis
- Extracts game rules from observation text:
  - Parses game messages for rule information
  - Updates knowledge base with discovered rules
  - Assigns confidence values to extracted rules
- Implements a causal inference pipeline:
  - Direct action effects (immediate impacts)
  - Delayed action effects (changes over time)
  - Secondary effects (chain reactions)
  - Rule-based predictions (game logic constraints)

### Strategic Learning Process
- Three-phase training approach defined in configuration:
```yaml
strategic_learning:
  # Phase durations
  exploration_phase_steps: 500000     # Initial exploration phase
  balanced_phase_steps: 1000000       # Balanced exploration/exploitation phase
  optimization_phase_steps: 3000000   # Optimization phase
```
- Exploration phases focus on discovering:
  - Game metrics and their relationships
  - Causal links between actions and outcomes
  - Game rules and constraints
  - Goal hierarchy and importance
- Optimization phases focus on:
  - Maximizing discovered metrics
  - Applying learned causal models
  - Following inferred game rules
  - Prioritizing actions based on goal importance

### Goal Inference Capabilities
- Infers game goals based on:
  - Game feedback (positive/negative messages)
  - Trends in key metrics
  - Game rules and constraints
  - Player progression indicators
- Builds a goal hierarchy with relative importance:
  - Ranks goals based on game feedback
  - Adjusts importance based on difficulty
  - Prioritizes goals with higher rewards
  - Balances competing objectives

### Knowledge Bootstrapping
- Option to accelerate learning through prior knowledge:
```python
# Knowledge bootstrapping
self.bootstrap = config.get("strategic", {}).get("bootstrap", True)
self.bootstrap_model_path = config.get("strategic", {}).get("bootstrap_model_path", None)
```
- Can load pre-trained models as starting points
- Supports transfer learning between different game scenarios
- Imports causal models from previous training runs
- Allows manual specification of game rules

### LSTM Implementation for Temporal Memory
- Configuration for temporal memory:
```yaml
# LSTM-specific settings
lstm_hidden_size: 256      # LSTM hidden layer size
lstm_layers: 1             # Number of LSTM layers
```
- Enables the agent to:
  - Remember past game states and actions
  - Track long-term trends in metrics
  - Identify delayed effects of actions
  - Plan multi-step action sequences

## Relationship to Other Components

### Position in Agent Hierarchy
- The Strategic Agent represents the most advanced agent type in the progression:
  Discovery → Tutorial → Vision → Autonomous → **Strategic**
- It builds on capabilities of previous agent types
- Focuses on high-level planning rather than basic gameplay

### Integration with Environment System
- Uses a specialized environment wrapper (`StrategicEnvironment`)
- Extends `AutonomousCS2Environment` with strategic capabilities
- Implements custom reward functions focused on strategy
- Adds additional observation features for strategic planning

### Relationship to Adaptive Agent
- Can be used as a mode within the [Adaptive Agent](adaptive_agent.md) framework
- Training script supports both direct training and adaptive mode:
```python
parser.add_argument('--use-adaptive', action='store_true',
                  help='Use the adaptive agent as a wrapper instead of direct strategic training')
```
- Shares knowledge base with other agent modes
- Represents the most advanced mode in the adaptive progression

## Optimization Opportunities

### Enhanced Causal Modeling
- Implement Bayesian networks for more robust causal inference
- Add counterfactual reasoning for better strategy evaluation
- Incorporate causal discovery algorithms (PC, FCI)
- Implement multi-step causal planning

### Improved Temporal Reasoning
- Extend LSTM architecture with attention mechanisms
- Implement transformer-based policy for better long-range dependencies
- Add hierarchical time representation (short/medium/long-term)
- Implement explicit planning horizons

### Knowledge Transfer Enhancement
- Develop more sophisticated knowledge bootstrapping methods
- Implement meta-learning for faster adaptation
- Add curriculum learning based on scenario difficulty
- Create a shared knowledge repository across training runs

### Decision Explainability
- Add causal attribution for agent decisions
- Implement visualization of causal models
- Create natural language explanations of strategies
- Develop tools for analyzing strategic decision-making

## Next Steps
1. Analyze the performance characteristics of the strategic agent compared to other types
2. Examine how the causal modeling system affects training efficiency
3. Investigate the goal inference accuracy in various game scenarios
4. Explore optimization opportunities for the causal reasoning components

## Related Analyses
- [Adaptive Agent Analysis](adaptive_agent.md)
- [Comprehensive Codebase Architecture](../architecture/comprehensive_architecture.md)
- [Comprehensive Synthesis](../architecture/comprehensive_synthesis.md)
- [Configuration System Analysis](../architecture/configuration_system.md) 