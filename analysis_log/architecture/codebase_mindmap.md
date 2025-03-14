# CS2 RL Agent Codebase Structure and Dependencies

*Last updated: March 13, 2025 21:00 - Updated to focus on adaptive agent as primary orchestrator*

**Tags:** #architecture #documentation #codebase #reference

## Overview
This document provides a comprehensive mindmap of the CS2 RL Agent codebase, showing how different files relate to each other. The adaptive agent serves as the primary orchestrator, managing the other agent types dynamically based on performance metrics and game state.

## Core Python Modules

### Agent Module (`src/agent/`)
- **`__init__.py`** - Exports agent classes
  - **Imports:** 
    - `src.agent.discovery_agent`
    - `src.agent.tutorial_agent`
    - `src.agent.vision_agent`
    - `src.agent.autonomous_agent`
    - `src.agent.strategic_agent`
    - `src.agent.adaptive_agent`
  - **Used by:** 
    - `training/train_*.py` scripts
    - Environment implementations

- **`adaptive_agent.py`** - Main orchestrator agent
  - **Imports:** 
    - `src.utils.logger`
    - `src.environment.cs2_env`
    - `src.utils.patch_tensorflow`
  - **Used by:** 
    - `training/train_adaptive.py`
    - `scripts/training/train_adaptive.bat`
    - `scripts/training/run_adaptive_fixed.bat`

- **`discovery_agent.py`** - Used by adaptive agent for UI discovery
  - **Imports:** 
    - `src.environment.discovery_env`
    - `src.utils.logger`
  - **Used by:** 
    - `src.agent.adaptive_agent`

- **`strategic_agent.py`** - Used by adaptive agent for strategic gameplay
  - **Imports:** 
    - `src.utils.logger`
    - `src.environment.cs2_env`
  - **Used by:** 
    - `src.agent.adaptive_agent`

- **`autonomous_agent.py`** - Used by adaptive agent for autonomous gameplay
  - **Imports:** 
    - `src.utils.logger`
    - `src.environment.autonomous_env`
  - **Used by:** 
    - `src.agent.adaptive_agent`

- **`vision_agent.py`** - Used by adaptive agent for vision-based gameplay
  - **Imports:** 
    - `src.utils.logger`
    - `src.environment.cs2_env`
  - **Used by:** 
    - `src.agent.adaptive_agent`

- **`tutorial_agent.py`** - Used by adaptive agent for tutorial learning
  - **Imports:** 
    - `src.utils.logger`
    - `src.environment.cs2_env`
  - **Used by:** 
    - `src.agent.adaptive_agent`

- **`agent_factory.py`** - Creates appropriate agent instances
  - **Imports:** 
    - All agent implementations
  - **Used by:** 
    - `src.agent.adaptive_agent`

### Environment Module (`src/environment/`)
- **`__init__.py`** - Exports environment classes
  - **Imports:** 
    - `src.environment.cs2_env`
    - `src.environment.discovery_env`
  - **Used by:** 
    - Agent implementations
    - Training scripts

- **`cs2_env.py`** - Main CS2 environment
  - **Imports:** 
    - `src.interface.window_manager`
    - `src.utils.logger`
  - **Used by:** 
    - `src.agent.adaptive_agent`
    - Various agent implementations

- **`discovery_env.py`** - Discovery mode environment
  - **Imports:** 
    - `src.interface.auto_vision_interface`
    - `src.utils.logger`
  - **Used by:** 
    - `src.agent.adaptive_agent`
    - `src.agent.discovery_agent`

### Interface Module (`src/interface/`)
- **`__init__.py`** - Exports interface classes
  - **Imports:** 
    - `src.interface.window_manager`
    - `src.interface.ollama_vision_interface`
    - `src.interface.auto_vision_interface`
  - **Used by:** 
    - Environment implementations

- **`auto_vision_interface.py`** - Vision processing interface
  - **Imports:** 
    - `src.interface.base_interface`
    - `src.interface.input_enhancer`
  - **Used by:** 
    - `src.environment.discovery_env`
    - `testing/test_vision_windows.py`

- **`ollama_vision_interface.py`** - ML-based vision interface
  - **Imports:** 
    - `src.interface.base_interface`
  - **Used by:** 
    - `src.environment.autonomous_env`
    - `testing/test_ollama.py`

### Utils Module (`src/utils/`)
- **`__init__.py`** - Exports utility functions
  - **Used by:** Multiple modules

- **`logger.py`** - Logging and metrics tracking
  - **Imports:** 
    - Optional TensorFlow imports
    - Standard libraries
  - **Used by:** 
    - All agent implementations
    - All environment implementations
    - Training scripts

- **`window_utils.py`** - Window management utilities
  - **Imports:** 
    - Windows-specific modules
  - **Used by:** 
    - `src.interface.window_manager`

- **`file_utils.py`** - File operations utilities
  - **Used by:** 
    - Multiple modules

- **`patch_tensorflow.py`** - TensorFlow compatibility fixes
  - **Used by:** 
    - `training/train_adaptive.py`
    - `scripts/training/run_adaptive_fixed.bat`
    - `src.agent.adaptive_agent`

## Training Scripts

### Python Training Scripts (`training/`)
- **`train_adaptive.py`** - Primary training script for adaptive agent
  - **Imports:** 
    - `src.utils.patch_tensorflow`
    - `src.agent.adaptive_agent`
    - `src.environment.cs2_env`
  - **Used by:** 
    - `scripts/training/train_adaptive.bat`
    - `scripts/training/run_adaptive_fixed.bat`
    - `scripts/deployment/run_adaptive_agent.bat`

### Batch Scripts (`scripts/training/`)
- **`train_adaptive.bat`** - Main training script for adaptive agent
  - **Runs:** `training/train_adaptive.py`
  - **Config:** `config/adaptive_config.yaml`
  - **Related scripts:** 
    - `scripts/utils/setup_conda.bat`
    - `scripts/utils/check_gpu.bat`

- **`run_adaptive_fixed.bat`** - Training with TensorFlow fixes
  - **Runs:** 
    - `training/train_adaptive.py`
    - `src/utils/patch_tensorflow.py`
  - **Config:** `config/adaptive_config.yaml`

### Utility Scripts (`scripts/utils/`)
- **`setup_conda.bat`**
  - **Used by:** All training and deployment scripts
  - **Related:** `requirements.txt`

- **`check_gpu.bat`**
  - **Runs:** `utils/check_gpu.py`
  - **Used by:** Training scripts

- **`enable_gpu.bat`**
  - **Runs:** `utils/setup_gpu.py`
  - **Used by:** Training scripts
  - **Config:** `config/gpu_config.json`

## Testing and Evaluation

### Testing Scripts (`testing/`)
- **`test_cs2_env.py`**
  - **Imports:** `src.environment.cs2_env`
  - **Used by:** `scripts/testing/test_cs2_env.bat`

- **`test_discovery_env.py`**
  - **Imports:** `src.environment.discovery_env`
  - **Used by:** `scripts/testing/test_discovery_env.bat`

- **`test_ollama.py`**
  - **Imports:** `src.interface.ollama_vision_interface`
  - **Used by:** `scripts/testing/test_ollama.bat`

### Evaluation Scripts (`evaluation/`)
- **`evaluate.py`**
  - **Imports:** 
    - `src.agent` (various agents)
    - `src.environment` (various environments)
  - **Used by:** Analysis tools

## Configuration Files

### YAML Configuration (`config/`)
- **`adaptive_config.yaml`** - Primary configuration file
  - **Used by:** 
    - `training/train_adaptive.py`
    - `scripts/training/train_adaptive.bat`
    - `scripts/deployment/run_adaptive_agent.bat`

- **`discovery_config.yaml`**
  - **Used by:** 
    - Adaptive agent's discovery mode

- **`strategic_config.yaml`**
  - **Used by:** 
    - Adaptive agent's strategic mode

- **`autonomous_config.yaml`**
  - **Used by:** 
    - Adaptive agent's autonomous mode

- **`vision_guided_config.yaml`**
  - **Used by:** 
    - Adaptive agent's vision mode

- **`tutorial_guided_config.yaml`**
  - **Used by:** 
    - Adaptive agent's tutorial mode

### JSON Configuration
- **`gpu_config.json`**
  - **Used by:** 
    - `utils/setup_gpu.py`
    - `scripts/utils/enable_gpu.bat`

## Deployment Scripts (`scripts/deployment/`)
- **`all_in_one_setup_and_train.bat`** - Comprehensive setup and training script
  - **Runs:** 
    - Various setup and utility scripts
    - Adaptive agent training

- **`run_adaptive_agent.bat`** - Streamlined deployment script
  - **Runs:** 
    - `training/train_adaptive.py`
  - **Config:** `config/adaptive_config.yaml`

## Visual Dependency Map

```
┌─────────────────────┐      ┌─────────────────────┐      ┌─────────────────────┐
│  Adaptive Training  │      │  Adaptive Batch     │      │  Config Files       │
│  (train_adaptive.py)│◄────▶│  (scripts/*.bat)    │◄────▶│  (config/*.yaml)    │
└─────────────────────┘      └─────────────────────┘      └─────────────────────┘
          ▲                            ▲                            ▲
          │                            │                            │
          ▼                            ▼                            ▼
┌─────────────────────┐      ┌─────────────────────┐      ┌─────────────────────┐
│  Adaptive Agent     │      │  Utils Scripts      │      │  JSON Config        │
│  (adaptive_agent.py)│◄────▶│  (scripts/utils/*.bat)◄───▶│  (config/*.json)    │
└─────────────────────┘      └─────────────────────┘      └─────────────────────┘
          ▲                            ▲                            ▲
          │                            │                            │
          ▼                            ▼                            ▼
┌─────────────────────┐      ┌─────────────────────┐      ┌─────────────────────┐
│  Other Agents       │      │  Testing Scripts    │      │  Utility Module     │
│  (src/agent/*.py)   │◄────▶│  (testing/*.py)     │◄────▶│  (src/utils/*.py)   │
└─────────────────────┘      └─────────────────────┘      └─────────────────────┘
          ▲                                                         ▲
          │                                                         │
          ▼                                                         ▼
┌─────────────────────┐                                   ┌─────────────────────┐
│  Environment Module │                                   │  Deployment Scripts │
│  (src/env/*.py)     │◄──────────────────────────────────▶  (scripts/deploy/*.bat)
└─────────────────────┘                                   └─────────────────────┘
```

## Key Dependency Patterns

1. **Adaptive Agent Orchestration Flow:**
   `Adaptive Agent` → `Manages and Switches Between` → `Other Agent Types` → `Using Different Environments`

2. **Configuration Flow:**
   `Adaptive Config (yaml)` → `References Sub-agent Configs` → `Used by Adaptive Agent` → `For Dynamic Mode Switching`

3. **Training Flow:**
   `train_adaptive.py` → `Creates Adaptive Agent` → `Which Orchestrates Other Agents` → `Based on Performance Metrics`

4. **Utility Integrations:**
   - `src/utils/logger.py` tracks metrics across all agent modes
   - `src/utils/patch_tensorflow.py` ensures TensorFlow compatibility
   - Window utilities connect the interface layer to the operating system

## Conclusion

The CS2 RL Agent codebase is now streamlined around the adaptive agent as the primary orchestrator. The adaptive agent dynamically switches between different modes (discovery, tutorial, vision, autonomous, strategic) based on performance metrics and game state. All other agent implementations are retained but are now managed by the adaptive agent rather than being trained individually.

## References
- [Comprehensive Architecture](comprehensive_architecture.md)
- [Component Integration](component_integration.md)
- [Batch Scripts Reference](../tools/batch_scripts_reference.md)
- [Training Scripts Overview](../training/training_scripts_overview.md)
- [Adaptive Agent Training](../training/adaptive_agent_training.md)
- [Strategic Agent Analysis](../components/strategic_agent.md)
- [Autonomous Environment Implementation](../environment/autonomous_environment.md) 