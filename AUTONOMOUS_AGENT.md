# Autonomous Agent for Cities: Skylines 2

This document explains the autonomous agent feature that allows reinforcement learning to play Cities: Skylines 2 with minimal human guidance. The agent is designed to start with a blank slate and learn through exploration, discovering the game's interface and gradually improving its city-building skills.

## Overview

The autonomous agent combines reinforcement learning with menu exploration capabilities, automatically detecting UI elements and learning optimal actions through trial and error. It requires **no manual configuration of UI coordinates** and adapts to different screen resolutions and game versions.

### Key Features

- **Fully autonomous exploration**: The agent discovers menus, buttons, and interactive elements without pre-programming
- **Automatic UI detection**: Uses OCR and computer vision to identify clickable elements
- **Memory system**: Remembers discovered UI elements and their functions
- **Curriculum learning**: Starts with heavy exploration and gradually shifts to optimization
- **Self-improvement**: Gets better over time through reinforcement learning

## How It Works

The autonomous agent system consists of several interconnected components:

1. **Menu Explorer**: Systematically explores the game interface to discover interactive elements
2. **Auto Vision Interface**: Detects UI elements using OCR and template matching
3. **Autonomous Environment**: Wraps the base environment with exploration capabilities
4. **Memory Buffer**: Stores discovered UI elements and their purposes
5. **RL Agent**: Uses PPO algorithm with LSTM memory for sequential decision making

### Exploration Process

The agent follows a structured exploration process:

1. **Initial phase**: Clicks in different screen regions to discover main menus
2. **Main menu phase**: Systematically explores discovered menus to find submenus
3. **Submenu phase**: Interacts with submenu options to learn their effects
4. **Advanced phase**: Balances exploration with optimization based on rewards

During exploration, the agent:
- Records new UI elements it discovers
- Tests the effects of clicking different elements
- Receives small rewards for discovering new functions
- Gradually builds a mental model of the game interface

### Learning Process

As the agent explores, it simultaneously learns:

1. **Which actions yield positive rewards** (population growth, happiness, etc.)
2. **Which sequences of actions work together** (zoning → services → infrastructure)
3. **How to navigate menus efficiently** to perform specific actions
4. **Long-term planning strategies** for city development

The learning process uses curriculum learning with three phases:
- **Initial Exploration** (first 1M steps): Heavy emphasis on discovery
- **Balanced Learning** (next 2M steps): Equal focus on exploration and city metrics
- **Focused Optimization** (final 7M steps): Primarily optimizing city performance

## Setup Instructions

### Prerequisites

- Cities: Skylines 2 installed and working
- Python environment set up as described in WINDOWS_SETUP.md
- Game window visible and not minimized
- Screen resolution of at least 1280x720 (higher recommended)

### Running the Autonomous Agent

1. Start with a new, empty city in Cities: Skylines 2
2. Ensure the game window is visible (not minimized)
3. Run the provided batch script:
   ```
   train_autonomous.bat
   ```
4. Wait as the agent begins exploration (this is a long process!)

### Training Settings

The autonomous agent uses settings from `config/autonomous_agent.yaml`. You can modify this file to adjust:

- Exploration frequency and randomness
- Network architecture and learning parameters
- Reward functions and weights
- Training duration and checkpoint frequency

## Expected Behavior

### Initial Phases

In the first few hours, the agent will:
- Click on different parts of the screen seemingly randomly
- Gradually discover main menus and remember them
- Make frequent mistakes and take inefficient actions
- Create small, basic city elements with no clear plan

**This phase may look chaotic - this is normal and expected!**

### Middle Phases

After several hours/days of training, the agent will:
- Navigate menus more purposefully
- Show preferences for certain city-building patterns
- Create more organized road layouts and zoning
- React to city problems (though not always effectively)

### Advanced Phases

With extensive training (multiple days), the agent may:
- Navigate menus efficiently to perform specific actions
- Create reasonably functional city layouts
- Manage basic city services and infrastructure
- Respond appropriately to problems like traffic or pollution

## Monitoring Progress

You can monitor agent progress through:

1. **TensorBoard**: View learning curves and rewards
   ```
   tensorboard --logdir=tensorboard/autonomous
   ```

2. **Log files**: Check detailed exploration and learning logs in:
   ```
   logs/autonomous/
   ```

3. **Saved models**: Find checkpoints of agent progress in:
   ```
   models/autonomous/
   ```

## Limitations

Be aware of the following limitations:

- **Learning is slow**: Meaningful results may take days of continuous training
- **Exploration is random**: Early behavior will be chaotic and inefficient
- **City design quality**: The agent optimizes for metrics, not aesthetics
- **Complex interactions**: Some game features may remain undiscovered
- **Resource usage**: Training requires significant CPU and GPU resources

## Troubleshooting

### Agent doesn't click on menus

Possible causes:
- OCR detection issues: Try adjusting `ocr_confidence` in the configuration
- Game UI changed: Update to latest version of the code
- Screen resolution issues: Ensure resolution is at least 1280x720

### Agent gets stuck in loops

Possible causes:
- Reward function encouraging repetitive behavior
- Limited exploration of key interfaces
- Agent needs more training time to discover efficient actions

Solution: Try increasing the exploration frequency in configuration

### Performance degrades over time

Possible causes:
- Agent has found a local optimum
- City has grown too complex for agent's current capabilities
- Environment has become unstable

Solution: Try restarting training with a slightly modified configuration

## Future Improvements

The autonomous agent system could be enhanced with:

- **Memory of past cities**: Transfer knowledge from previous training runs
- **Imitation learning**: Combining autonomous exploration with human demonstrations
- **Multi-task learning**: Training on different city scenarios simultaneously
- **Meta-learning**: Learning how to adapt to different city types quickly

## Contributing

If you'd like to contribute to improving the autonomous agent:

1. Test with different game configurations and report results
2. Suggest improvements to the exploration or learning algorithms
3. Contribute enhancements to the code via pull requests

## Conclusion

The autonomous agent represents a significant step toward truly self-learning AI for complex simulation games. While it has limitations, it demonstrates the potential for agents to learn without extensive human programming or guidance.

With enough training time, the agent can develop from random clicking to purposeful city planning - showcasing the power of reinforcement learning combined with exploration. 