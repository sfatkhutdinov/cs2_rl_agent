"""
CUDA Enabler for PyTorch and TensorFlow
Import this module before importing PyTorch or TensorFlow to enable CUDA
"""

import os
import sys
import ctypes
import platform
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger("CUDA Enabler")

# Set environment variables
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
os.environ["TF_GPU_ALLOCATOR"] = "cuda_malloc_async"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"
logger.info("Set environment variables for GPU usage")

# CUDA paths to check on Windows
CUDA_PATHS = [
    r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8\bin",
    r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.7\bin",
    r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.6\bin",
    r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.0\bin",
    r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.1\bin"
]

# DLLs that need to be preloaded
CUDA_DLLS = [
    "cudart64_110.dll",
    "cublas64_11.dll",
    "cublasLt64_11.dll",
    "cufft64_11.dll",
    "curand64_11.dll",
    "cusolver64_11.dll",
    "cusparse64_11.dll",
    "cudnn64_8.dll"
]

def add_cuda_to_path():
    """Add CUDA directories to PATH"""
    if platform.system() != "Windows":
        return
        
    for cuda_path in CUDA_PATHS:
        if os.path.exists(cuda_path) and cuda_path not in os.environ["PATH"]:
            os.environ["PATH"] = cuda_path + os.pathsep + os.environ["PATH"]
            logger.info(f"Added {cuda_path} to PATH")

def preload_cuda_dlls():
    """Preload CUDA DLLs"""
    if platform.system() != "Windows":
        return []
        
    loaded = []
    for dll in CUDA_DLLS:
        try:
            ctypes.CDLL(dll)
            loaded.append(dll)
            logger.info(f"Loaded {dll}")
        except OSError:
            logger.warning(f"Could not load {dll}")
    
    return loaded

def is_cuda_available():
    """Check if CUDA is available with PyTorch"""
    try:
        import torch
        cuda_available = torch.cuda.is_available()
        if cuda_available:
            device_name = torch.cuda.get_device_name(0)
            logger.info(f"PyTorch CUDA available: {device_name}")
        else:
            logger.warning("PyTorch CUDA not available")
        return cuda_available
    except ImportError:
        logger.warning("PyTorch not installed")
        return False
    except Exception as e:
        logger.error(f"Error checking PyTorch CUDA: {e}")
        return False

def is_tensorflow_gpu_available():
    """Check if TensorFlow can access GPU"""
    try:
        import tensorflow as tf
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            logger.info(f"TensorFlow GPU available: {gpus}")
            return True
        else:
            logger.warning("TensorFlow GPU not available")
            return False
    except ImportError:
        logger.warning("TensorFlow not installed")
        return False
    except Exception as e:
        logger.error(f"Error checking TensorFlow GPU: {e}")
        return False

# Add CUDA to PATH
add_cuda_to_path()

# Preload CUDA DLLs
preloaded_dlls = preload_cuda_dlls()
if preloaded_dlls:
    logger.info(f"Preloaded {len(preloaded_dlls)} CUDA DLLs")

# Create a flag to indicate if CUDA enablement was successful
cuda_enabled = bool(preloaded_dlls or is_cuda_available() or is_tensorflow_gpu_available())

logger.info(f"CUDA enablement {'successful' if cuda_enabled else 'failed'}")

# This function can be called to explicitly check CUDA after importing frameworks
def verify_cuda():
    """Verify CUDA is working with ML frameworks"""
    torch_cuda = is_cuda_available()
    tf_cuda = is_tensorflow_gpu_available()
    
    if torch_cuda or tf_cuda:
        logger.info("CUDA is working with at least one ML framework")
        return True
    else:
        logger.warning("CUDA is not working with any ML framework")
        return False

# If this module is run directly, print diagnostic information
if __name__ == "__main__":
    logger.info("=" * 50)
    logger.info("CUDA Enabler Diagnostic Information")
    logger.info("=" * 50)
    
    # Check CUDA paths
    logger.info("\nCUDA Paths:")
    for path in CUDA_PATHS:
        if os.path.exists(path):
            logger.info(f"  Found: {path}")
    
    # Verify CUDA
    logger.info("\nVerifying CUDA...")
    is_working = verify_cuda()
    
    logger.info("\nTo enable CUDA in your code, add this line BEFORE importing PyTorch or TensorFlow:")
    logger.info("import enable_cuda")     if torch.cuda.is_available(): 
        print(f"GPU Device: {torch.cuda.get_device_name(0)}") 
except ImportError: 
    print("PyTorch not installed") 
try: 
    import tensorflow as tf 
    print(f"TensorFlow GPU devices: {tf.config.list_physical_devices('GPU')}") 
except ImportError: 
    print("TensorFlow not installed") 
