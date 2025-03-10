# Cities: Skylines 2 RL Agent

A reinforcement learning agent designed to learn and optimize city management in Cities: Skylines 2.

## Project Overview

This project builds a complete reinforcement learning (RL) agent capable of learning to play and manage a city in Cities: Skylines 2. The agent optimizes key urban metrics including:
- Population growth
- Citizen satisfaction
- Financial balance
- Traffic efficiency

## Architecture

The project is structured into several key components:

1. **Game Interface** - Connects to Cities: Skylines 2 via:
   - Bridge Mod API integration (primary method)
   - Computer vision fallback (using screen capture and OCR)

2. **Environment Wrapper** - Formats game state into a standardized RL environment following OpenAI Gym patterns

3. **RL Agent** - Implements state-of-the-art reinforcement learning algorithms (PPO/DQN)

4. **Training Pipeline** - Manages the training process, hyperparameter tuning, and evaluation

## Project Structure

```
cs2_rl_agent/
├── bridge_mod/         # C# bridge mod for direct game integration
├── data/               # Storage for collected game data and datasets
├── models/             # Saved model checkpoints
├── logs/               # Training logs and metrics
├── notebooks/          # Jupyter notebooks for exploration and visualization
├── src/                # Source code
│   ├── agent/          # RL algorithm implementations
│   ├── config/         # Configuration files
│   ├── environment/    # Game environment wrapper
│   ├── interface/      # Game interface (API/computer vision)
│   └── utils/          # Utility functions
└── tests/              # Unit and integration tests
```

## Installation

1. Clone this repository
2. Install the required dependencies:
```
pip install -r requirements.txt
```
3. Install the bridge mod:
   - Copy the contents of the `bridge_mod` folder to your Cities: Skylines mods directory
   - Enable the "RL Agent Bridge" mod in the game's Content Manager
   - See `bridge_mod/README.md` for detailed installation instructions

## Usage

### Training the Agent

```
python -m src.train --config configs/default.yaml
```

### Evaluating the Agent

```
python -m src.evaluate --model models/trained_agent.pt
```

### Viewing Results

Training metrics and visualizations can be found in the `logs/` directory.

## Game Integration

The agent can interact with Cities: Skylines 2 in two ways:

1. **Bridge Mod (Recommended)**: A C# mod that exposes a REST API for direct game interaction. This provides accurate game state information and precise control.

2. **Computer Vision (Fallback)**: Uses screen capture and OCR to interact with the game when the bridge mod is not available. This is less reliable but works without modifying the game.

## Requirements

- Python 3.8+
- Cities: Skylines 2
- CUDA-compatible GPU (recommended for faster training)

## License

MIT License

## Acknowledgements

This project draws inspiration from research in reinforcement learning, urban planning optimization, and game AI. 