# Batch Scripts Reference

*Last updated: March 14, 2025 - Initial documentation*

**Tags:** #tools #scripts #documentation #deployment #utility

## Overview
This document provides a comprehensive reference for all batch scripts (.bat files) used in the CS2 RL Agent project. These scripts automate various tasks related to setup, training, testing, deployment, and utility functions.

## Script Categories

### Deployment Scripts
Located in `scripts/deployment/`, these scripts are used for deploying and running the agent in various configurations.

| Script | Purpose | Key Parameters |
|--------|---------|----------------|
| all_in_one_setup_and_train.bat | Complete setup and training process | None |
| run_all_in_one.bat | Runs the agent with all components | None |
| run_all_simple.bat | Runs the agent in simplified mode | None |
| run_discovery_agent.bat | Runs the agent in discovery mode | None |
| run_discovery_agent_with_focus.bat | Runs the agent in discovery mode with window focus | None |
| run_discovery.bat | Simple discovery mode execution | None |
| run_discovery_fixed.bat | Discovery mode with fixed parameters | None |
| run_discovery_fixed2.bat | Alternative fixed parameters for discovery | None |
| run_discovery_fixed3.bat | Third configuration for discovery mode | None |
| run_vision_test.bat | Tests the vision interface | None |

### Training Scripts
Located in `scripts/training/`, these scripts are used to train different agent models.

| Script | Purpose | Key Parameters |
|--------|---------|----------------|
| train_adaptive.bat | Trains the adaptive agent | None |
| train_autonomous.bat | Trains the autonomous agent | None |
| train_discovery.bat | Trains the discovery-based agent | None |
| train_discovery_with_focus.bat | Trains discovery agent with window focus | None |
| train_strategic.bat | Trains the strategic agent | None |
| train_strategic_agent.bat | Extended training for strategic agent | None |
| train_tutorial_guided.bat | Trains with tutorial guidance | None |
| train_vision_guided.bat | Trains with vision guidance | None |

### Testing Scripts
Located in `scripts/testing/`, these scripts are used to test various components of the system.

| Script | Purpose | Key Parameters |
|--------|---------|----------------|
| test_adaptive_modes.bat | Tests adaptive agent mode switching | None |
| test_config.bat | Tests configuration loading | Config file path |
| test_cs2_env.bat | Tests the CS2 environment | None |
| test_discovery_env.bat | Tests the discovery environment | None |
| test_ollama.bat | Tests the Ollama vision interface | None |

### Utility Scripts
Located in `scripts/utils/`, these scripts provide various utility functions.

| Script | Purpose | Key Parameters |
|--------|---------|----------------|
| add_cuda.bat | Adds CUDA support to environment | None |
| auto_detect.bat | Automatic detection of game windows | None |
| calibrate_vision.bat | Calibrates the vision system | None |
| capture_templates.bat | Captures vision templates | None |
| check_conda.bat | Checks Conda installation | None |
| check_gpu.bat | Checks GPU availability | None |
| enable_gpu.bat | Enables GPU support | None |
| generate_templates.bat | Generates vision templates | None |
| setup_conda.bat | Sets up Conda environment | None |
| setup_ollama.bat | Sets up Ollama vision model | None |
| setup_venv.bat | Sets up virtual environment | None |
| verify_discovery_config.bat | Verifies discovery configuration | None |

## Script Dependencies and Relationships

### Setup Sequence
The recommended setup sequence is:
1. `setup_conda.bat` or `setup_venv.bat` - Set up Python environment
2. `check_gpu.bat` - Verify GPU availability
3. `enable_gpu.bat` - Enable GPU support if available
4. `setup_ollama.bat` - Set up vision model

### Training Workflow
A typical training workflow includes:
1. Environment setup (as above)
2. `test_config.bat` - Verify configuration
3. One of the training scripts (e.g., `train_adaptive.bat`)
4. Model evaluation using Python scripts

### Deployment Workflow
A typical deployment workflow includes:
1. Environment setup (as above)
2. `calibrate_vision.bat` - Calibrate vision system
3. `run_vision_test.bat` - Test vision system
4. One of the deployment scripts (e.g., `run_discovery_agent.bat`)

## Implementation Details

### Common Script Patterns
Most scripts follow these common patterns:
- Setting the working directory to the project root
- Activating the appropriate environment
- Running Python scripts with appropriate parameters
- Providing user feedback about operation status

### Error Handling
Scripts implement error handling through:
- Checking for prerequisites
- Verifying environment variables
- Capturing and reporting error codes
- Providing informative error messages

## Usage Examples

### Setting Up Environment
```
scripts\utils\setup_conda.bat
scripts\utils\check_gpu.bat
scripts\utils\enable_gpu.bat
```

### Training a Model
```
scripts\utils\setup_conda.bat
scripts\training\train_adaptive.bat
```

### Running the Agent
```
scripts\utils\setup_conda.bat
scripts\deployment\run_discovery_agent.bat
```

## Best Practices for Script Management

1. **Version Control**: Keep scripts under version control
2. **Documentation**: Update this reference when adding or modifying scripts
3. **Error Messages**: Use descriptive error messages
4. **Parameterization**: Use parameters for configurable values
5. **Testing**: Test scripts before deployment

## Related Documentation
- [Deployment Processes](../testing/deployment_processes.md)
- [Training Scripts Overview](../training/training_scripts_overview.md)
- [Testing Infrastructure](../testing/testing_infrastructure.md)

---

*For questions about specific scripts, contact the project maintainers.* 