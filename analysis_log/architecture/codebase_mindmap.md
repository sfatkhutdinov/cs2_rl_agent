# CS2 RL Agent Codebase Structure and Dependencies

*Last updated: March 13, 2025 20:31 - Updated mindmap with adaptive agent and fixed references*

**Tags:** #architecture #documentation #codebase #reference

## Overview
This document provides a comprehensive mindmap of the CS2 RL Agent codebase, showing how different files relate to each other. This visual representation helps in understanding the dependencies and relationships between Python files (.py), batch scripts (.bat), and configuration files (.yaml/.json).

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

- **`adaptive_agent.py`**
  - **Imports:** 
    - `src.utils.logger`
    - `src.environment.cs2_env`
    - `src.utils.patch_tensorflow`
  - **Used by:** 
    - `training/train_adaptive.py`
    - `scripts/training/train_adaptive.bat`
    - `scripts/training/run_adaptive_fixed.bat`

- **`discovery_agent.py`**
  - **Imports:** 
    - `src.environment.discovery_env`
    - `src.utils.logger`
  - **Used by:** 
    - `training/train_discovery.py`
    - `scripts/training/train_discovery*.bat`

- **`strategic_agent.py`**
  - **Imports:** 
    - `src.utils.logger`
    - `src.environment.cs2_env`
  - **Used by:** 
    - `training/train_strategic.py`
    - `scripts/training/train_strategic*.bat`

- **`autonomous_agent.py`**
  - **Imports:** 
    - `src.utils.logger`
    - `src.environment.autonomous_env`
  - **Used by:** 
    - `training/train_autonomous.py`
    - `scripts/training/train_autonomous.bat`

- **`vision_agent.py`**
  - **Imports:** 
    - `src.utils.logger`
    - `src.environment.cs2_env`
  - **Used by:** 
    - `training/train_vision_guided.py`
    - `scripts/training/train_vision_guided.bat`

- **`tutorial_agent.py`**
  - **Imports:** 
    - `src.utils.logger`
    - `src.environment.cs2_env`
  - **Used by:** 
    - `training/train_tutorial_guided.py`
    - `scripts/training/train_tutorial_guided.bat`

- **`agent_factory.py`**
  - **Imports:** 
    - All agent implementations
  - **Used by:** 
    - Training scripts for dynamic agent instantiation

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
    - `training/train.py`
    - `testing/test_cs2_env.py`

- **`discovery_env.py`** - Discovery mode environment
  - **Imports:** 
    - `src.interface.auto_vision_interface`
    - `src.utils.logger`
  - **Used by:** 
    - `training/train_discovery.py`
    - `testing/test_discovery_env.py`

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
- **`train.py`** - Generic training script
  - **Imports:** 
    - `src.agent` (various agents)
    - `src.environment.cs2_env`
    - `src.utils.logger`
  - **Used by:** 
    - `scripts/training/train_*.bat`

- **`train_adaptive.py`** - Adaptive agent training
  - **Imports:** 
    - `src.utils.patch_tensorflow`
    - `src.agent.adaptive_agent`
    - `src.environment.cs2_env`
  - **Used by:** 
    - `scripts/training/train_adaptive.bat`
    - `scripts/training/run_adaptive_fixed.bat`

- **`train_discovery.py`** - Discovery agent training
  - **Imports:** 
    - `src.agent.discovery_agent`
    - `src.environment.discovery_env`
  - **Used by:** 
    - `scripts/training/train_discovery*.bat`

- **`train_strategic.py`** - Strategic agent training
  - **Imports:** 
    - `src.agent.strategic_agent`
    - `src.environment.cs2_env`
  - **Used by:** 
    - `scripts/training/train_strategic*.bat`

- **`train_autonomous.py`** - Autonomous agent training
  - **Imports:** 
    - `src.agent.autonomous_agent`
    - `src.environment.autonomous_env`
  - **Used by:** 
    - `scripts/training/train_autonomous.bat`

- **`train_vision_guided.py`** - Vision-guided agent training
  - **Imports:** 
    - `src.agent.vision_agent`
    - `src.environment.cs2_env`
  - **Used by:** 
    - `scripts/training/train_vision_guided.bat`

- **`train_tutorial_guided.py`** - Tutorial agent training
  - **Imports:** 
    - `src.agent.tutorial_agent`
    - `src.environment.cs2_env`
  - **Used by:** 
    - `scripts/training/train_tutorial_guided.bat`

### Batch Scripts (`scripts/training/`)
- **`train_adaptive.bat`**
  - **Runs:** `training/train_adaptive.py`
  - **Config:** `config/adaptive_config.yaml`
  - **Related scripts:** 
    - `scripts/utils/setup_conda.bat`
    - `scripts/utils/check_gpu.bat`

- **`run_adaptive_fixed.bat`**
  - **Runs:** 
    - `training/train_adaptive.py`
    - `src/utils/patch_tensorflow.py`
  - **Config:** `config/adaptive_config.yaml`

- **`train_discovery.bat`** / **`train_discovery_with_focus.bat`**
  - **Runs:** `training/train_discovery.py`
  - **Config:** `config/discovery_config.yaml`

- **`train_strategic_agent.bat`**
  - **Runs:** `training/train_strategic.py`
  - **Config:** `config/strategic_config.yaml`

- **`train_autonomous.bat`**
  - **Runs:** `training/train_autonomous.py`
  - **Config:** `config/autonomous_config.yaml`

- **`train_vision_guided.bat`**
  - **Runs:** `training/train_vision_guided.py`
  - **Config:** `config/vision_guided_config.yaml`

- **`train_tutorial_guided.bat`**
  - **Runs:** `training/train_tutorial_guided.py`
  - **Config:** `config/tutorial_guided_config.yaml`

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
- **`adaptive_config.yaml`**
  - **Used by:** 
    - `training/train_adaptive.py`
    - `scripts/training/train_adaptive.bat`

- **`discovery_config.yaml`**
  - **Used by:** 
    - `training/train_discovery.py`
    - `scripts/training/train_discovery*.bat`

- **`strategic_config.yaml`**
  - **Used by:** 
    - `training/train_strategic.py`
    - `scripts/training/train_strategic*.bat`

- **`autonomous_config.yaml`**
  - **Used by:** 
    - `training/train_autonomous.py`
    - `scripts/training/train_autonomous.bat`

- **`vision_guided_config.yaml`**
  - **Used by:** 
    - `training/train_vision_guided.py`
    - `scripts/training/train_vision_guided.bat`

- **`tutorial_guided_config.yaml`**
  - **Used by:** 
    - `training/train_tutorial_guided.py`
    - `scripts/training/train_tutorial_guided.bat`

### JSON Configuration
- **`gpu_config.json`**
  - **Used by:** 
    - `utils/setup_gpu.py`
    - `scripts/utils/enable_gpu.bat`

## Deployment Scripts (`scripts/deployment/`)
- **`all_in_one_setup_and_train.bat`**
  - **Runs:** 
    - Various setup and utility scripts
    - Training scripts based on mode selection
  - **Config:** Various config files

- **`run_discovery_agent.bat`**
  - **Runs:** 
    - `utils/setup_gpu.py`
    - Discovery agent in inference mode
  - **Config:** `config/discovery_config.yaml`

- **`run_all_simple.bat`**
  - **Runs:** Simplified version of agents
  - **Used for:** Quick demonstrations

## Visual Dependency Map

```
┌─────────────────────┐      ┌─────────────────────┐      ┌─────────────────────┐
│   Training Scripts  │      │  Training Batch     │      │  Config Files       │
│   (training/*.py)   │◄────▶│  (scripts/*.bat)    │◄────▶│  (config/*.yaml)    │
└─────────────────────┘      └─────────────────────┘      └─────────────────────┘
          ▲                            ▲                            ▲
          │                            │                            │
          ▼                            ▼                            ▼
┌─────────────────────┐      ┌─────────────────────┐      ┌─────────────────────┐
│   Agent Module      │      │  Utils Scripts      │      │  JSON Config        │
│   (src/agent/*.py)  │◄────▶│  (scripts/utils/*.bat)◄───▶│  (config/*.json)    │
└─────────────────────┘      └─────────────────────┘      └─────────────────────┘
          ▲                            ▲                            ▲
          │                            │                            │
          ▼                            ▼                            ▼
┌─────────────────────┐      ┌─────────────────────┐      ┌─────────────────────┐
│  Environment Module │      │  Testing Scripts    │      │  Utility Module     │
│  (src/env/*.py)     │◄────▶│  (testing/*.py)     │◄────▶│  (src/utils/*.py)   │
└─────────────────────┘      └─────────────────────┘      └─────────────────────┘
          ▲                                                         ▲
          │                                                         │
          ▼                                                         ▼
┌─────────────────────┐                                   ┌─────────────────────┐
│  Interface Module   │                                   │  Deployment Scripts │
│  (src/interface/*.py)◄──────────────────────────────────▶  (scripts/deploy/*.bat)
└─────────────────────┘                                   └─────────────────────┘
```

## Enhanced Visual Dependency Map

For a more detailed representation, consider using a dedicated diagramming tool like [Mermaid](https://mermaid-js.github.io/) or [PlantUML](https://plantuml.com/) to create an interactive visualization that better represents the complex relationships between components.

Example PlantUML representation (for illustration - would need to be generated with proper tools):

```
@startuml
package "Agent Module" {
  [adaptive_agent.py]
  [discovery_agent.py]
  [strategic_agent.py]
  [autonomous_agent.py]
  [vision_agent.py]
  [tutorial_agent.py]
  [agent_factory.py]
}

package "Environment Module" {
  [cs2_env.py]
  [discovery_env.py]
  [autonomous_env.py]
}

package "Interface Module" {
  [auto_vision_interface.py]
  [ollama_vision_interface.py]
  [window_manager.py]
}

package "Utils Module" {
  [logger.py]
  [patch_tensorflow.py]
  [window_utils.py]
  [file_utils.py]
}

package "Training Scripts" {
  [train.py]
  [train_adaptive.py]
  [train_discovery.py]
  [train_strategic.py]
  [train_autonomous.py]
  [train_vision_guided.py]
  [train_tutorial_guided.py]
}

package "Config Files" {
  [adaptive_config.yaml]
  [discovery_config.yaml]
  [strategic_config.yaml]
  [autonomous_config.yaml]
  [vision_guided_config.yaml]
  [tutorial_guided_config.yaml]
  [gpu_config.json]
}

[adaptive_agent.py] --> [logger.py]
[adaptive_agent.py] --> [cs2_env.py]
[adaptive_agent.py] --> [patch_tensorflow.py]

[train_adaptive.py] --> [adaptive_agent.py]
[train_adaptive.py] --> [patch_tensorflow.py]

[adaptive_config.yaml] --> [train_adaptive.py]

' Additional relationships would be defined here...
@enduml
```

## Key Dependency Patterns

1. **Training Pipeline Flow:**
   `Batch Scripts (.bat)` → `Training Scripts (.py)` → `Agent Module` → `Environment Module` → `Interface Module`

2. **Configuration Flow:**
   `Config Files (.yaml/.json)` → `Used by Training Scripts` → `Passed to Agent/Environment initialization`

3. **Utility Integrations:**
   - `src/utils/logger.py` is used by almost all components
   - `src/utils/patch_tensorflow.py` is critical for the adaptive training pipeline
   - Window utilities connect the interface layer to the operating system

4. **Testing Integrations:**
   - Testing scripts directly import implementation modules
   - Batch scripts automate the testing process
   - Config files provide test parameters

## Conclusion

The CS2 RL Agent codebase follows a modular design with clear separation of concerns between:
- Agent implementations (learning algorithms)
- Environment implementations (game interaction)
- Interface components (vision and input)
- Utilities (shared functionality)
- Scripts (automation)
- Configuration (parameters)

This structure allows for independent development and testing of components while maintaining a coherent system architecture.

## References
- [Comprehensive Architecture](comprehensive_architecture.md)
- [Component Integration](component_integration.md)
- [Batch Scripts Reference](../tools/batch_scripts_reference.md)
- [Training Scripts Overview](../training/training_scripts_overview.md)
- [Adaptive Agent Training](../training/adaptive_agent_training.md)
- [Strategic Agent Analysis](../components/strategic_agent.md)
- [Autonomous Environment Implementation](../environment/autonomous_environment.md) 