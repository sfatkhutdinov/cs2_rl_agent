# All-in-One Setup and Training Guide

The `all_in_one_setup_and_train.bat` script provides a complete solution for setting up and running the CS2 RL Agent with minimal manual steps. This document explains how to use it effectively.

## Prerequisites

Before running the script, ensure:

1. **Anaconda or Miniconda** is installed and added to your PATH
2. **Ollama** is installed and running (`ollama serve`)
3. **Cities: Skylines 2** is installed and can be launched

## Basic Usage

Simply run the script with no parameters to use all default settings:

```
all_in_one_setup_and_train.bat
```

This will:
- Train for 1000 timesteps
- Use discovery mode
- Auto-focus the game window

## Command-Line Parameters

Customize the training process with these parameters:

```
all_in_one_setup_and_train.bat [timesteps] [mode] [focus]
```

### Parameters

1. **timesteps** - Number of training timesteps
   - Default: 1000
   - Example: `all_in_one_setup_and_train.bat 5000`

2. **mode** - Training mode
   - Default: discovery
   - Options:
     - `discovery`: Uses vision model to discover and interact with UI
     - `tutorial`: Follows the in-game tutorial
     - `vision`: Vision-guided approach
     - `autonomous`: Fully autonomous approach
   - Example: `all_in_one_setup_and_train.bat 2000 vision`

3. **focus** - Whether to auto-focus the game window
   - Default: true
   - Options: true, false
   - Example: `all_in_one_setup_and_train.bat 1000 discovery false`

## Example Commands

```
# Use default settings (1000 timesteps, discovery mode, auto-focus)
all_in_one_setup_and_train.bat

# Train for 5000 timesteps in discovery mode
all_in_one_setup_and_train.bat 5000

# Train for 2000 timesteps in vision mode
all_in_one_setup_and_train.bat 2000 vision

# Train for 3000 timesteps in autonomous mode without auto-focusing
all_in_one_setup_and_train.bat 3000 autonomous false
```

## Troubleshooting

Common issues and solutions:

### "Anaconda is not recognized"
- Make sure Anaconda is installed and added to your PATH
- Restart your terminal after installation
- You may need to edit your system's Environment Variables

### "Ollama is not running"
- Install Ollama from https://ollama.ai/
- Run `ollama serve` in a separate terminal window before starting the script

### "Game window not focused correctly"
- Ensure the game is running and visible before starting training
- If auto-focus isn't working, try setting it to false and manually focus the window

### Package Installation Errors
- If some packages fail to install, the script will ask if you want to continue
- Try running `setup_conda.bat` separately first to troubleshoot package issues 