# CS2 Reinforcement Learning Agent

A reinforcement learning agent for Counter-Strike 2 that uses multiple training modes and computer vision to learn and play the game.

## Directory Structure

```
cs2_rl_agent/
│
├── scripts/                  # Executable scripts
│   ├── training/             # Training script batch files
│   ├── testing/              # Testing script batch files  
│   ├── utils/                # Utility script batch files
│   └── deployment/           # Deployment script batch files
│
├── training/                 # Training Python scripts
│   ├── train_adaptive.py     # Adaptive agent training
│   ├── train_strategic.py    # Strategic agent training
│   ├── train_discovery.py    # Discovery mode training
│   ├── train.py              # General training script
│   └── ...
│
├── testing/                  # Testing Python scripts
│   ├── test_cs2_env.py       # CS2 environment tests
│   ├── test_config.py        # Configuration tests
│   ├── test_api.py           # API testing script
│   ├── test_vision_windows.py # Vision interface tests
│   └── ...
│
├── evaluation/               # Evaluation scripts
│   └── evaluate.py           # Model evaluation script
│
├── utils/                    # Utility Python scripts
│   ├── setup_gpu.py          # GPU setup utilities
│   ├── check_gpu.py          # GPU checking utilities
│   └── ...
│
├── config/                   # Configuration files
│   ├── adaptive_config.yaml  # Adaptive agent configuration
│   ├── strategic_config.yaml # Strategic agent configuration
│   └── ...
│
├── docs/                     # Documentation
│   ├── ANACONDA_SETUP.md     # Anaconda setup guide
│   ├── WINDOWS_SETUP.md      # Windows setup guide
│   └── ...
│
├── src/                      # Core source code
│   ├── agent/                # Agent implementations
│   ├── actions/              # Action system
│   ├── environment/          # Environment implementations
│   ├── utils/                # Core utilities
│   └── interface/            # Interface components
│
├── analysis_log/             # Analysis documentation
│
├── images/                   # Image files
│
├── models/                   # Trained models
│
├── data/                     # Training data
│
├── logs/                     # Log files
│
└── tensorboard/              # TensorBoard logs
```
## Setup

1. Install Anaconda or Miniconda
2. Run the setup script: `scripts/utils/setup_conda.bat`
3. Install dependencies: `pip install -r requirements.txt`
4. Configure GPU (optional): `scripts/utils/enable_gpu.bat`

## Training

To train the agent, use one of the following scripts:

- **Adaptive Agent**: `scripts/training/train_adaptive.bat` - Primary training script (recommended)

The adaptive agent serves as the central orchestrator that manages all specialized agent types:
- Discovery Mode
- Tutorial Mode
- Vision Mode
- Autonomous Mode
- Strategic Mode

This centralized approach eliminates the need for separate training pipelines while maintaining the full functionality of all specialized agent types.

## Testing

To test the agent, use one of the following scripts:

- **CS2 Environment**: `scripts/testing/test_cs2_env.bat`
- **Configuration**: `scripts/testing/test_config.bat`
- **Discovery Environment**: `scripts/testing/test_discovery_env.bat`
- **Adaptive Modes**: `scripts/testing/test_adaptive_modes.bat`

## Deployment

For deployment, use one of the following scripts:

- **All-in-One Setup and Train**: `scripts/deployment/all_in_one_setup_and_train.bat` - Complete setup and training
- **Run Adaptive Agent**: `scripts/deployment/run_adaptive_agent.bat` - Focused deployment script

These streamlined deployment scripts provide a simplified interface while maintaining full functionality through the adaptive agent's orchestration capabilities.

## Documentation

For detailed documentation, see the following:

- [Anaconda Setup Guide](docs/ANACONDA_SETUP.md)
- [Windows Setup Guide](docs/WINDOWS_SETUP.md)
- [All-in-One Guide](docs/ALL_IN_ONE_GUIDE.md)
- [Autonomous Agent Guide](docs/AUTONOMOUS_AGENT.md)
- [Auto Detection Guide](docs/AUTO_DETECTION.md)

For code analysis and architecture documentation, see the [analysis_log](analysis_log/main.md) directory.

## License

This project is licensed under the MIT License - see the LICENSE file for details. 
