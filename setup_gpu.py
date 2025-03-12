import os
import sys
import subprocess
import platform
import json
import importlib.util
from pathlib import Path

def check_nvidia_smi():
    """Check if nvidia-smi is available and working"""
    try:
        output = subprocess.check_output(["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"])
        print("NVIDIA GPU detected:")
        print(output.decode("utf-8").strip())
        return True
    except (subprocess.SubprocessError, FileNotFoundError):
        print("Error: nvidia-smi failed. NVIDIA driver might not be installed or working properly.")
        return False

def check_cuda_version():
    """Check CUDA version from pytorch"""
    try:
        import torch
        if torch.cuda.is_available():
            print(f"PyTorch CUDA available: {torch.cuda.is_available()}")
            print(f"PyTorch CUDA version: {torch.version.cuda}")
            print(f"PyTorch cuDNN version: {torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else 'Not available'}")
            print(f"GPU Device: {torch.cuda.get_device_name(0)}")
            print(f"GPU Count: {torch.cuda.device_count()}")
            print(f"Current device: {torch.cuda.current_device()}")
            return True
        else:
            print("PyTorch CUDA not available despite having NVIDIA GPU.")
            print("Possible causes:")
            print("1. Incompatible CUDA version")
            print("2. PyTorch was installed without CUDA support")
            print("3. GPU drivers need to be updated")
            return False
    except ImportError:
        print("PyTorch not installed. Cannot check CUDA compatibility.")
        return False

def check_tensorflow_gpu():
    """Check if TensorFlow can see the GPU"""
    try:
        import tensorflow as tf
        gpus = tf.config.list_physical_devices('GPU')
        print(f"TensorFlow version: {tf.__version__}")
        print(f"TensorFlow GPU devices: {gpus}")
        if gpus:
            print("TensorFlow can access the GPU.")
            for gpu in gpus:
                print(f"  {gpu}")
            return True
        else:
            print("TensorFlow cannot see any GPU.")
            print("Possible causes:")
            print("1. TensorFlow was not installed with GPU support")
            print("2. Incompatible CUDA/cuDNN versions")
            print("3. GPU drivers need to be updated")
            return False
    except ImportError:
        print("TensorFlow not installed. Cannot check GPU compatibility.")
        return False

def setup_environment_variables():
    """Set up environment variables for better GPU performance"""
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
    os.environ["TF_GPU_ALLOCATOR"] = "cuda_malloc_async"
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"

    print("Environment variables set:")
    print(f"CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES', 'Not set')}")
    print(f"TF_FORCE_GPU_ALLOW_GROWTH: {os.environ.get('TF_FORCE_GPU_ALLOW_GROWTH', 'Not set')}")
    print(f"TF_GPU_ALLOCATOR: {os.environ.get('TF_GPU_ALLOCATOR', 'Not set')}")
    print(f"PYTORCH_CUDA_ALLOC_CONF: {os.environ.get('PYTORCH_CUDA_ALLOC_CONF', 'Not set')}")

def create_gpu_config():
    """Create a GPU configuration file"""
    config = {
        "gpu_enabled": True,
        "cuda_device": 0,
        "tensorflow_memory_growth": True,
        "pytorch_memory_config": "max_split_size_mb:512",
        "environment_variables": {
            "CUDA_VISIBLE_DEVICES": "0",
            "TF_FORCE_GPU_ALLOW_GROWTH": "true",
            "TF_GPU_ALLOCATOR": "cuda_malloc_async"
        }
    }
    
    with open("gpu_config.json", "w") as f:
        json.dump(config, f, indent=4)
    
    print("Created GPU configuration file: gpu_config.json")

def check_requirements():
    """Check for required packages and versions"""
    required_packages = [
        "torch>=2.0.0",
        "tensorflow>=2.12.0",
        "stable-baselines3>=2.0.0"
    ]
    
    missing = []
    for req in required_packages:
        pkg_name = req.split(">=")[0]
        min_version = req.split(">=")[1] if ">=" in req else None
        
        spec = importlib.util.find_spec(pkg_name)
        if spec is None:
            missing.append(req)
            continue
            
        if min_version:
            try:
                module = importlib.import_module(pkg_name)
                version = getattr(module, "__version__", "0.0.0")
                if version < min_version:
                    missing.append(f"{pkg_name}>={min_version} (found {version})")
            except:
                missing.append(req + " (version check failed)")
    
    if missing:
        print("Missing or outdated packages:")
        for pkg in missing:
            print(f"  - {pkg}")
        return False
    
    print("All required packages are installed with compatible versions.")
    return True

def setup_cuda_paths():
    """Set up CUDA paths for Windows"""
    if platform.system() == "Windows":
        # Try to find CUDA installation
        cuda_paths = [
            r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8",
            r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.7",
            r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.6",
            r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.0",
            r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.1"
        ]
        
        for cuda_path in cuda_paths:
            if os.path.exists(cuda_path):
                # Add CUDA bin to PATH
                bin_path = os.path.join(cuda_path, "bin")
                if bin_path not in os.environ["PATH"]:
                    os.environ["PATH"] = bin_path + os.pathsep + os.environ["PATH"]
                
                # Set CUDA_PATH
                os.environ["CUDA_PATH"] = cuda_path
                
                print(f"Found CUDA installation at: {cuda_path}")
                print(f"Updated PATH with: {bin_path}")
                return True
        
        print("Could not find CUDA installation in standard locations.")
        return False
    
    return True  # Non-Windows platforms handled differently

def main():
    """Main function to check and configure GPU support"""
    print("=" * 80)
    print(" GPU Setup and Diagnostics Tool ".center(80, "="))
    print("=" * 80)
    
    setup_environment_variables()
    
    print("\nChecking NVIDIA driver...")
    has_nvidia = check_nvidia_smi()
    
    if not has_nvidia:
        print("\nNVIDIA GPU not detected or driver not working.")
        print("Please install the latest NVIDIA drivers from:")
        print("https://www.nvidia.com/Download/index.aspx")
        return 1
    
    print("\nSetting up CUDA paths...")
    setup_cuda_paths()
    
    print("\nChecking PyTorch CUDA support...")
    pytorch_gpu = check_cuda_version()
    
    print("\nChecking TensorFlow GPU support...")
    tf_gpu = check_tensorflow_gpu()
    
    print("\nChecking required packages...")
    has_requirements = check_requirements()
    
    if pytorch_gpu or tf_gpu:
        print("\nCreating GPU configuration...")
        create_gpu_config()
        
        print("\nGPU setup complete!")
        print("At least one of PyTorch or TensorFlow can use the GPU.")
        print("You can now run your ML applications with GPU acceleration.")
        return 0
    else:
        print("\nGPU setup failed!")
        print("Neither PyTorch nor TensorFlow can use the GPU despite NVIDIA drivers being installed.")
        print("Please try reinstalling PyTorch and TensorFlow with GPU support.")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 