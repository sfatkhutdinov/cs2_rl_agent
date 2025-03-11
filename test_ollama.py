import requests
import json
import sys
import os
from PIL import Image
import base64
import io

def test_ollama_text():
    """Test basic Ollama text generation."""
    print("Testing basic Ollama text generation...")
    
    url = "http://localhost:11434/api/generate"
    data = {
        "model": "granite3.2-vision:latest",
        "prompt": "Hello, can you help me analyze game UI?",
        "stream": False
    }
    
    try:
        response = requests.post(url, json=data)
        if response.status_code == 200:
            result = response.json()
            print("Success! Ollama responded:")
            print(result.get('response', '')[:100] + "..." if len(result.get('response', '')) > 100 else result.get('response', ''))
            return True
        else:
            print(f"Error: Received status code {response.status_code}")
            print(response.text)
            return False
    except Exception as e:
        print(f"Exception: {str(e)}")
        return False

def encode_image(image_path):
    """Encode image to base64."""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def test_ollama_vision(image_path=None):
    """Test Ollama vision model with an image."""
    print("\nTesting Ollama vision capabilities...")
    
    # If no image is provided, create a test image
    if not image_path or not os.path.exists(image_path):
        print("No image provided, creating a test image...")
        img = Image.new('RGB', (100, 100), color = (73, 109, 137))
        
        # Save to a temporary file
        image_path = "test_image.png"
        img.save(image_path)
        print(f"Created test image at {image_path}")
    
    # Encode the image
    try:
        image_base64 = encode_image(image_path)
    except Exception as e:
        print(f"Error encoding image: {str(e)}")
        return False
    
    url = "http://localhost:11434/api/generate"
    data = {
        "model": "granite3.2-vision:latest",
        "prompt": """
        Look at this image and return a JSON object with the following structure:
        {
            "description": "brief description of what you see",
            "elements": ["list", "of", "key", "elements"],
            "colors": ["main", "colors", "present"]
        }
        Return ONLY the JSON object, no other text.
        """,
        "stream": False,
        "images": [image_base64]
    }
    
    try:
        response = requests.post(url, json=data)
        if response.status_code == 200:
            result = response.json()
            print("Success! Vision model responded:")
            response_text = result.get('response', '')
            print(response_text[:500] + "..." if len(response_text) > 500 else response_text)
            
            # Try to extract JSON from the response
            try:
                json_start = response_text.find('{')
                json_end = response_text.rfind('}') + 1
                if json_start >= 0 and json_end > json_start:
                    json_str = response_text[json_start:json_end]
                    parsed_json = json.loads(json_str)
                    print("\nSuccessfully parsed JSON from response:")
                    print(json.dumps(parsed_json, indent=2))
                else:
                    print("\nCould not find JSON in response")
            except Exception as e:
                print(f"\nFailed to parse JSON: {str(e)}")
            
            return True
        else:
            print(f"Error: Received status code {response.status_code}")
            print(response.text)
            return False
    except Exception as e:
        print(f"Exception: {str(e)}")
        return False

if __name__ == "__main__":
    print("=== Ollama Integration Test ===")
    
    # Check if an image path was provided
    image_path = None
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
        print(f"Using provided image: {image_path}")
    
    text_success = test_ollama_text()
    vision_success = test_ollama_vision(image_path)
    
    if text_success and vision_success:
        print("\n✅ All tests passed! Ollama integration is working correctly.")
    else:
        print("\n❌ Some tests failed. Please check the errors above.") 