# Auto-Detection Vision Interface for Cities: Skylines 2 RL Agent

This guide explains how to use the automatic UI element detection feature which eliminates the need to manually specify UI coordinates.

## Overview

The auto-detection interface uses computer vision techniques to automatically locate UI elements in the game:

1. **OCR-based detection**: Scans the screen for text labels like "Population", "Budget", etc.
2. **Template matching**: Uses saved images of UI elements to locate them on screen

## Quick Start

1. Start Cities: Skylines 2 and load a city
2. Run the auto-detection tool:
   ```
   python auto_detect.py
   ```
   Or on Windows, double-click `auto_detect.bat`

## How It Works

The auto-detection interface:

1. Takes a screenshot of the game window
2. Processes the image using OCR to find text elements
3. Identifies UI elements like population counters, budget displays, and control buttons
4. Records the positions of these elements for future interaction
5. Updates these positions automatically as needed

## Detection Methods

### OCR-Based Detection (Default)

This method uses Optical Character Recognition to find text on the screen:

- Pros: Works immediately without setup
- Cons: Can be less accurate depending on UI appearance and text clarity

To run with OCR detection:
```
python auto_detect.py --method ocr
```

### Template-Based Detection

This method uses image templates of UI elements:

- Pros: More accurate, works well with non-text elements
- Cons: Requires initial setup to capture templates

To capture templates:
```
python auto_detect.py --capture-templates
```
Or on Windows, double-click `capture_templates.bat`

To run with template detection:
```
python auto_detect.py --method template
```

## Training the RL Agent with Auto-Detection

To use auto-detection with the RL agent, modify the config file:

```yaml
interface:
  type: "auto_vision"
  vision:
    detection_method: "ocr"  # or "template"
```

Or run training with:
```
python -m src.train --interface-type auto_vision
```

## Troubleshooting

### OCR Not Detecting UI Elements

- Make sure the game is in focus and UI elements are visible
- Try using the template-based approach instead
- Adjust the screen resolution or UI scale in the game

### Templates Not Matching

- Recapture templates using `auto_detect.py --capture-templates`
- Make sure to capture clean images of each UI element
- If the game UI changes (e.g., with mods), you'll need to recapture templates

### Actions Not Working

- Make sure the game window is in focus
- Try running as administrator
- The auto-detection may need to be recalibrated if game UI changes

## Advanced Configuration

You can fine-tune the auto-detection in the config file:

```yaml
interface:
  vision:
    detection_method: "ocr"  # or "template"
    ocr_confidence: 0.7     # Increase for stricter matching
    cache_validity: 60      # How long to cache UI positions (seconds)
```

## Benefits

- **No manual coordinates**: You don't need to manually specify UI element positions
- **UI independence**: Works even if UI layout changes
- **Resolution independence**: Works across different screen resolutions
- **Adaptability**: Automatically adjusts to UI changes during gameplay 