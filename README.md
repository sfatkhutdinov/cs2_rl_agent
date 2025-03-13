# Cities: Skylines 2 Reinforcement Learning Agent

This repository contains code for training a reinforcement learning agent to play Cities: Skylines 2 using a discovery-based approach with the Granite 3.2 Vision model for game understanding.

## Prerequisites

- Windows 10/11
- Cities: Skylines 2 installed
- Anaconda or Miniconda
- [Ollama](https://ollama.ai/) installed and running

## Quick Start

For the fastest way to get started, run the all-in-one setup and training batch file:

```bash
all_in_one_setup_and_train.bat
```

This script will:
1. Check for Anaconda and set up the environment
2. Install all required packages
3. Configure GPU support if available
4. Set up Ollama and the vision model
5. Run all necessary tests
6. Start the training process

You can customize the training with command-line parameters:
```bash
all_in_one_setup_and_train.bat [timesteps] [mode] [focus]
```

For more information about options:
```bash
all_in_one_setup_and_train.bat help
```

## Manual Setup

If you prefer to set up manually:

1. Clone this repository
2. Create and activate the Anaconda environment:
   ```bash
   setup_conda.bat
   ```
   This will create a conda environment named 'cs2_agent' with all required packages.

3. If needed, activate the environment manually:
   ```bash
   conda activate cs2_agent
   ```

4. Run the Ollama setup script to ensure the vision model is installed:
   ```bash
   setup_ollama.bat
   ```

For more details on the Anaconda setup, see [ANACONDA_SETUP.md](ANACONDA_SETUP.md).

## Usage

### Discovery-Based Training

This mode uses the Granite 3.2 Vision model to autonomously discover and interact with the game's UI elements.

1. Start Cities: Skylines 2 and make sure it's visible on your screen
2. Run the discovery training script:
   ```bash
   train_discovery_with_focus.bat
   ```

The script will:
- Focus on the game window
- Detect UI elements using the vision model
- Attempt to learn game mechanics through exploration
- Save models to the `models` directory

### Configuration

The agent's behavior is controlled by configuration files in the `config` directory:

- `discovery_config.yaml`: Settings for discovery-based training
- `tutorial_guided_config.yaml`: Settings for tutorial-guided training

## Troubleshooting

### Common Issues

- **Window Focus Problems**: If the agent doesn't seem to be interacting with the game, make sure the game window is visible and not minimized.

- **Ollama Connection**: If you see connection errors, ensure Ollama is running by opening a terminal and typing:
  ```bash
  ollama serve
  ```

- **Vision Model Issues**: If the vision model isn't working correctly, try reinstalling it manually:
  ```bash
  ollama pull granite3.2-vision:latest
  ```

- **GPU Not Detected**: If your GPU isn't being detected, run:
  ```bash
  check_gpu.bat
  ```

## Project Structure

- `src/`: Source code
  - `environment/`: Game environment classes
  - `agent/`: RL agent implementations
  - `utils/`: Utility functions
- `config/`: Configuration files
- `models/`: Saved model checkpoints
- `logs/`: Training logs

## License

This project is licensed under the MIT License - see the LICENSE file for details. 