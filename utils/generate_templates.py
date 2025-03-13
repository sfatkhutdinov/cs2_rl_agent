import os
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

def create_template_directory():
    """Create the templates directory if it doesn't exist."""
    template_dir = os.path.join("data", "templates")
    os.makedirs(template_dir, exist_ok=True)
    return template_dir

def create_text_template(text, filename, template_dir, size=(200, 50)):
    """Create a template image with text."""
    # Create a white image
    img = Image.new('RGB', size, color='white')
    draw = ImageDraw.Draw(img)
    
    try:
        # Try to use Arial font
        font = ImageFont.truetype("arial.ttf", 32)
    except:
        # Fallback to default font
        font = ImageFont.load_default()
    
    # Get text size
    text_bbox = draw.textbbox((0, 0), text, font=font)
    text_width = text_bbox[2] - text_bbox[0]
    text_height = text_bbox[3] - text_bbox[1]
    
    # Calculate center position
    x = (size[0] - text_width) // 2
    y = (size[1] - text_height) // 2
    
    # Draw text
    draw.text((x, y), text, fill='black', font=font)
    
    # Save the image
    filepath = os.path.join(template_dir, filename)
    img.save(filepath)
    print(f"Created template: {filepath}")

def create_button_template(text, filename, template_dir, size=(200, 50)):
    """Create a template image of a button with text."""
    # Create a white image
    img = Image.new('RGB', size, color='white')
    draw = ImageDraw.Draw(img)
    
    # Draw button background
    button_margin = 5
    draw.rectangle([button_margin, button_margin, size[0]-button_margin, size[1]-button_margin], 
                  fill='lightgray', outline='gray')
    
    try:
        # Try to use Arial font
        font = ImageFont.truetype("arial.ttf", 24)
    except:
        # Fallback to default font
        font = ImageFont.load_default()
    
    # Get text size
    text_bbox = draw.textbbox((0, 0), text, font=font)
    text_width = text_bbox[2] - text_bbox[0]
    text_height = text_bbox[3] - text_bbox[1]
    
    # Calculate center position
    x = (size[0] - text_width) // 2
    y = (size[1] - text_height) // 2
    
    # Draw text
    draw.text((x, y), text, fill='black', font=font)
    
    # Save the image
    filepath = os.path.join(template_dir, filename)
    img.save(filepath)
    print(f"Created template: {filepath}")

def main():
    """Generate all required template files."""
    template_dir = create_template_directory()
    
    # Create label templates
    create_text_template("Population", "population_label.png", template_dir)
    create_text_template("Happiness", "happiness_label.png", template_dir)
    create_text_template("Budget", "budget_label.png", template_dir)
    create_text_template("Traffic", "traffic_label.png", template_dir)
    
    # Create button templates
    create_button_template("Speed", "speed_button.png", template_dir)
    create_button_template("Menu", "menu_button.png", template_dir)
    
    print("\nTemplate generation complete!")
    print("Note: These are placeholder templates. You may need to replace them with actual")
    print("screenshots from your game for better detection accuracy.")

if __name__ == "__main__":
    main() 