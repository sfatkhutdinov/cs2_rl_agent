import sys
import time
import logging
import numpy as np
from PIL import Image, ImageGrab

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("test_focus")

def test_screenshot_capture():
    """Test basic screenshot capturing"""
    logger.info("Testing screenshot capture")
    
    try:
        # Try with PIL ImageGrab
        logger.info("Capturing with PIL ImageGrab")
        start_time = time.time()
        screenshot = ImageGrab.grab()
        duration = time.time() - start_time
        
        if screenshot is None:
            logger.error("Failed to capture screenshot with PIL")
            return False
            
        # Convert to numpy
        np_screenshot = np.array(screenshot)
        
        # Check validity
        if np_screenshot is None or np_screenshot.size == 0:
            logger.error("Screenshot conversion failed")
            return False
            
        logger.info(f"PIL screenshot captured successfully in {duration:.3f} seconds")
        logger.info(f"Screenshot shape: {np_screenshot.shape}")
        
        # Save for verification
        Image.fromarray(np_screenshot).save("test_screenshot.png")
        logger.info("Screenshot saved to test_screenshot.png")
        
        return True
    except Exception as e:
        logger.error(f"Error in test_screenshot_capture: {e}")
        return False

def test_time_module():
    """Test the time module functionality"""
    logger.info("Testing time module")
    
    try:
        start_time = time.time()
        time.sleep(0.5)
        end_time = time.time()
        
        duration = end_time - start_time
        logger.info(f"Time module works correctly. Sleep duration: {duration:.3f} seconds")
        return True
    except Exception as e:
        logger.error(f"Error in test_time_module: {e}")
        return False

if __name__ == "__main__":
    logger.info("Running focus and screenshot tests")
    
    # Test time module
    time_result = test_time_module()
    logger.info(f"Time module test {'PASSED' if time_result else 'FAILED'}")
    
    # Test screenshot
    screenshot_result = test_screenshot_capture()
    logger.info(f"Screenshot test {'PASSED' if screenshot_result else 'FAILED'}")
    
    # Overall result
    if time_result and screenshot_result:
        logger.info("All tests PASSED")
        sys.exit(0)
    else:
        logger.error("Some tests FAILED")
        sys.exit(1) 