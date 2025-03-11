# Cities: Skylines 2 Reinforcement Learning Agent

This repository contains code for training a reinforcement learning agent to play Cities: Skylines 2 using a discovery-based approach with the Granite 3.2 Vision model for game understanding.

## Prerequisites

- Windows 10/11
- Cities: Skylines 2 installed
- Python 3.9+ with venv
- [Ollama](https://ollama.ai/) installed and running

## Setup

1. Clone this repository
2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   venv\Scripts\activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Run the Ollama setup script to ensure the vision model is installed:
   ```bash
   setup_ollama.bat
   ```

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