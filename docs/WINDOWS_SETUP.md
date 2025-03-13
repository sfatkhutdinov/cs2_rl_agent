# Cities: Skylines 2 RL Agent - Windows Setup Guide

This guide provides step-by-step instructions for setting up and running the Cities: Skylines 2 RL Agent on Windows using the vision interface.

## Requirements

- Windows 10 or 11
- Python 3.8 or newer
- Cities: Skylines 2 game
- Administrator rights (for installing software)

## Installation Steps

### 1. Install Python Dependencies

Open Command Prompt as Administrator and run:

```
cd path\to\cs2_rl_agent
pip install -r requirements.txt
```

### 2. Install Tesseract OCR

The vision interface uses Tesseract OCR to read text from the game screen.

1. Download the Tesseract installer from: https://github.com/UB-Mannheim/tesseract/wiki
2. Run the installer (recommended: use the default installation path)
3. Add Tesseract to your PATH:
   - Right-click on "This PC" or "My Computer"
   - Select "Properties"
   - Click "Advanced system settings"
   - Click "Environment Variables"
   - Under "System variables", find and select "Path", then click "Edit"
   - Click "New" and add the Tesseract installation path (e.g., `C:\Program Files\Tesseract-OCR`)
   - Click "OK" on all dialogs

### 3. Calibrate the Vision Interface

The vision interface needs to know where UI elements are on your screen. Run the calibration tool:

1. Start Cities: Skylines 2 and load a city
2. Double-click on `calibrate_vision.bat` in the `cs2_rl_agent` folder
3. Follow the on-screen instructions
4. Note the coordinates of key UI elements (population, happiness, budget, etc.)
5. Edit `src\interface\vision_interface.py` to update the UI element positions

Example UI element definition:
```python
self.ui_elements = {
    "population": {"region": (100, 50, 200, 80)},  # Update these coordinates
    "happiness": {"region": (300, 50, 400, 80)},   # based on your screen
    # ...etc
}
```

### 4. Test the Vision Interface

Once calibrated, you can test if the vision interface works correctly:

1. Start Cities: Skylines 2 and load a city
2. Double-click on `run_vision_test.bat` in the `cs2_rl_agent` folder
3. The test script will attempt to connect to the game, read metrics, and perform simple actions

If the test fails, check the error messages and ensure:
- The game is running and a city is loaded
- The game window is not minimized
- You've properly calibrated the UI element positions

### 5. Run the RL Agent

After confirming the vision interface works, you can run the full RL agent:

```
python -m src.train --config src\config\default.yaml
```

Make sure the config file has `interface.type` set to `"vision"`.

## Troubleshooting

### OCR Not Reading Text Correctly

- Make sure Tesseract is properly installed and in your PATH
- Verify the UI element coordinates are correct
- Try increasing the screen capture resolution
- Ensure the game UI is not obstructed

### Mouse Actions Not Working

- Make sure the game window is in focus
- Try running scripts as Administrator
- Disable any software that might interfere with mouse input

### Performance Issues

- Lower the capture_fps setting in the config file
- Use a smaller screen region
- Close unnecessary applications
- Make sure your GPU drivers are up to date

## Advanced: Custom UI Layouts

If you're using custom UI mods or a different resolution:

1. Update the screen region in the config file:
```yaml
vision:
  screen_region: [0, 0, 1920, 1080]  # Change to your resolution
```

2. Re-run the calibration tool to find new UI element positions

## Support

If you encounter any issues, please:
1. Check if the issue is covered in this guide
2. Look for solutions in the project's GitHub issues
3. Create a new issue with detailed information about your problem 