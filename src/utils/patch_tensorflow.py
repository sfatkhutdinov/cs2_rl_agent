#!/usr/bin/env python
"""
Patch script to fix TensorFlow compatibility issues.
"""

import os
import sys
import importlib.util
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger("TensorFlowPatch")

def is_tensorflow_installed():
    """Check if TensorFlow is installed."""
    return importlib.util.find_spec("tensorflow") is not None

def apply_tensorflow_io_patch():
    """
    Apply a patch to provide a dummy tensorflow.io module if missing.
    This fixes the AttributeError: module 'tensorflow' has no attribute 'io' 
    that occurs with some TensorFlow versions.
    """
    try:
        import tensorflow as tf
        
        # Check if tf.io exists
        if not hasattr(tf, 'io'):
            logger.info("tensorflow.io not found, applying patch...")
            
            # Create a dummy io module
            class DummyGFile:
                @staticmethod
                def join(*args):
                    return os.path.join(*args)
            
            class DummyIO:
                class gfile:
                    join = DummyGFile.join
            
            # Attach it to the tensorflow module
            tf.io = DummyIO()
            logger.info("Applied tensorflow.io patch successfully")
            return True
        else:
            logger.info("tensorflow.io module exists, no patch needed")
            return False
    except ImportError:
        logger.error("Cannot import tensorflow to apply patch")
        return False
    except Exception as e:
        logger.error(f"Error applying tensorflow.io patch: {e}")
        return False

def check_tensorflow_version():
    """Check TensorFlow version and report any issues."""
    try:
        import tensorflow as tf
        version = tf.__version__
        logger.info(f"TensorFlow version: {version}")
        
        # Check for known problematic versions
        if version.startswith('2.13'):
            logger.warning("TensorFlow 2.13.x may have compatibility issues with PyTorch")
            logger.warning("Consider downgrading to 2.10.x or using the patches provided")
        
        return version
    except ImportError:
        logger.error("TensorFlow not installed")
        return None
    except Exception as e:
        logger.error(f"Error checking TensorFlow version: {e}")
        return None

def main():
    """Apply necessary patches for TensorFlow compatibility."""
    logger.info("Checking TensorFlow installation...")
    
    if not is_tensorflow_installed():
        logger.error("TensorFlow is not installed. Please install it first.")
        return 1
    
    version = check_tensorflow_version()
    if not version:
        logger.error("Could not determine TensorFlow version.")
        return 1
    
    # Apply patches based on version
    patched = apply_tensorflow_io_patch()
    
    if patched:
        logger.info("TensorFlow patched successfully")
    else:
        logger.info("No patches were needed or applied")
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 