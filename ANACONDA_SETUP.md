# Anaconda Setup for CS2 RL Agent

This project now uses Anaconda for managing Python environments and dependencies instead of venv. This change provides better package management, especially for GPU support and machine learning libraries.

## Initial Setup

1. **Install Anaconda or Miniconda**:
   - Download and install from [Anaconda's website](https://www.anaconda.com/download/)
   - Make sure to add Anaconda to your PATH during installation

2. **Run the setup script**:
   ```
   setup_conda.bat
   ```
   This will:
   - Create a new conda environment named `cs2_agent`
   - Install all required packages with their latest versions
   - Configure GPU support when available

3. **Manual activation** (if needed):
   ```
   conda activate cs2_agent
   ```

## Benefits of Anaconda

- **Better dependency resolution**: Conda handles package conflicts better than pip
- **Improved GPU support**: Conda simplifies installing GPU-enabled versions of PyTorch and TensorFlow
- **Environment isolation**: Complete isolation from your system Python
- **Binary packages**: Many packages are pre-compiled, reducing build errors

## Troubleshooting

### Common Issues

- **"conda is not recognized"**: Restart your terminal after installation, or add Anaconda to your PATH
- **Package conflicts**: Try installing problematic packages one at a time
- **GPU not detected**: Run `check_gpu.bat` to diagnose GPU issues

### Switching Back to venv

If you need to use the original venv setup:
1. Run `setup_venv.bat` to recreate the venv environment
2. Update batch files with `python update_scripts_to_venv.py` (you'll need to create this script if needed)

## Package Management

- **Add new packages**:
  ```
  conda install packagename  # For conda packages
  pip install packagename    # For packages not in conda
  ```

- **Update all packages**:
  ```
  conda update --all
  ``` 