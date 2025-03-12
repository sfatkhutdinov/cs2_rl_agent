import os
import sys
import subprocess
import platform
import site
import shutil
from pathlib import Path

def check_cuda_paths():
    """Check and print all CUDA-related paths"""
    print("Checking CUDA paths...")
    
    # Check environment variables
    cuda_path = os.environ.get("CUDA_PATH", "Not set")
    cuda_home = os.environ.get("CUDA_HOME", "Not set")
    
    print(f"CUDA_PATH: {cuda_path}")
    print(f"CUDA_HOME: {cuda_home}")
    
    # Check if in PATH
    path = os.environ.get("PATH", "")
    cuda_in_path = False
    cuda_paths = []
    
    for p in path.split(os.pathsep):
        if "cuda" in p.lower():
            cuda_in_path = True
            cuda_paths.append(p)
    
    if cuda_in_path:
        print("CUDA found in PATH:")
        for p in cuda_paths:
            print(f"  {p}")
    else:
        print("CUDA not found in PATH")
    
    return cuda_in_path

def install_pytorch_cuda():
    """Reinstall PyTorch with CUDA support"""
    print("Reinstalling PyTorch with CUDA support...")
    
    try:
        # Uninstall existing PyTorch
        subprocess.check_call([sys.executable, "-m", "pip", "uninstall", "-y", "torch", "torchvision", "torchaudio"])
        
        # Install with CUDA support
        subprocess.check_call([
            sys.executable, "-m", "pip", "install",
            "torch==2.0.1", "torchvision==0.15.2", "torchaudio==2.0.2",
            "--index-url", "https://download.pytorch.org/whl/cu118"
        ])
        
        print("PyTorch with CUDA reinstalled successfully")
        return True
    except subprocess.SubprocessError as e:
        print(f"Error reinstalling PyTorch: {e}")
        return False

def install_tensorflow_gpu():
    """Install TensorFlow with GPU support"""
    print("Reinstalling TensorFlow with GPU support...")
    
    try:
        # Uninstall existing TensorFlow
        subprocess.check_call([sys.executable, "-m", "pip", "uninstall", "-y", "tensorflow", "tensorflow-gpu"])
        
        # Install TensorFlow
        subprocess.check_call([sys.executable, "-m", "pip", "install", "tensorflow==2.13.0"])
        
        print("TensorFlow reinstalled successfully")
        return True
    except subprocess.SubprocessError as e:
        print(f"Error reinstalling TensorFlow: {e}")
        return False

def fix_cuda_dll_issues():
    """Fix CUDA DLL issues by adding DLLs to the Python path"""
    if platform.system() != "Windows":
        print("DLL fixes only apply to Windows systems")
        return
        
    # Potential CUDA paths
    cuda_paths = [
        r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8\bin",
        r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.7\bin",
        r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.6\bin",
        r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.0\bin",
        r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.1\bin"
    ]
    
    # Python DLL directory
    python_dll_dir = os.path.join(sys.exec_prefix, "DLLs")
    if not os.path.exists(python_dll_dir):
        os.makedirs(python_dll_dir, exist_ok=True)
        
    # DLLs that might be required
    required_dlls = [
        "cudart64_110.dll",
        "cublas64_11.dll",
        "cublasLt64_11.dll",
        "cufft64_11.dll",
        "curand64_11.dll",
        "cusolver64_11.dll",
        "cusparse64_11.dll",
        "cudnn64_8.dll"
    ]
    
    dll_found = False
    
    # Try to copy the DLLs from CUDA installation
    for cuda_path in cuda_paths:
        if os.path.exists(cuda_path):
            print(f"Found CUDA bin directory: {cuda_path}")
            for dll in required_dlls:
                dll_path = os.path.join(cuda_path, dll)
                if os.path.exists(dll_path):
                    target_path = os.path.join(python_dll_dir, dll)
                    try:
                        shutil.copy2(dll_path, target_path)
                        print(f"Copied {dll} to {python_dll_dir}")
                        dll_found = True
                    except (shutil.SameFileError, OSError) as e:
                        print(f"Error copying {dll}: {e}")
    
    if dll_found:
        print("DLL files copied to Python directory")
    else:
        print("Could not find CUDA DLL files")

def check_gpu_with_torch():
    """Check if PyTorch can access the GPU"""
    try:
        import torch
        if torch.cuda.is_available():
            device_count = torch.cuda.device_count()
            device_name = torch.cuda.get_device_name(0) if device_count > 0 else "Unknown"
            print(f"PyTorch CUDA is available. Found {device_count} devices.")
            print(f"Device: {device_name}")
            return True
        else:
            print("PyTorch CUDA is not available.")
            return False
    except ImportError:
        print("PyTorch is not installed.")
        return False
    except Exception as e:
        print(f"Error checking PyTorch CUDA: {e}")
        return False

def check_gpu_with_tensorflow():
    """Check if TensorFlow can access the GPU"""
    try:
        import tensorflow as tf
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            print(f"TensorFlow found {len(gpus)} GPU(s):")
            for gpu in gpus:
                print(f"  {gpu}")
            return True
        else:
            print("TensorFlow did not find any GPUs.")
            return False
    except ImportError:
        print("TensorFlow is not installed.")
        return False
    except Exception as e:
        print(f"Error checking TensorFlow GPU: {e}")
        return False

def create_custom_loader():
    """Create a custom DLL loader script"""
    loader_code = """
import os
import sys
import ctypes
from pathlib import Path

def load_cuda_dlls():
    # Load CUDA DLLs manually
    cuda_paths = [
        r"C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\bin",
        r"C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.7\\bin",
        r"C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.6\\bin",
        r"C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.0\\bin",
        r"C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.1\\bin"
    ]
    
    # Add CUDA paths to system PATH
    for path in cuda_paths:
        if os.path.exists(path) and path not in os.environ["PATH"]:
            os.environ["PATH"] = path + os.pathsep + os.environ["PATH"]
            print(f"Added {path} to PATH")
    
    # DLLs to preload
    dlls = [
        "cudart64_110.dll",
        "cublas64_11.dll",
        "cublasLt64_11.dll",
        "cufft64_11.dll",
        "curand64_11.dll",
        "cusolver64_11.dll",
        "cusparse64_11.dll",
        "cudnn64_8.dll"
    ]
    
    # Try to load each DLL
    loaded = []
    for dll in dlls:
        try:
            ctypes.CDLL(dll)
            loaded.append(dll)
            print(f"Loaded {dll}")
        except OSError:
            print(f"Could not load {dll}")
    
    return loaded

if __name__ == "__main__":
    loaded_dlls = load_cuda_dlls()
    print(f"Loaded {len(loaded_dlls)} CUDA DLLs")
    
    # Import ML frameworks
    try:
        import torch
        print(f"PyTorch CUDA available: {torch.cuda.is_available()}")
    except ImportError:
        print("PyTorch not installed")
    
    try:
        import tensorflow as tf
        print(f"TensorFlow GPUs: {tf.config.list_physical_devices('GPU')}")
    except ImportError:
        print("TensorFlow not installed")
"""
    
    # Write the loader script
    with open("cuda_loader.py", "w") as f:
        f.write(loader_code)
    
    print("Created cuda_loader.py - Run this before importing PyTorch or TensorFlow")

def main():
    print("=" * 80)
    print(" CUDA Fix Tool ".center(80, "="))
    print("=" * 80)
    
    # Check CUDA paths
    has_cuda_paths = check_cuda_paths()
    
    # Check PyTorch GPU access
    print("\nChecking PyTorch GPU access...")
    pytorch_gpu = check_gpu_with_torch()
    
    # Check TensorFlow GPU access
    print("\nChecking TensorFlow GPU access...")
    tensorflow_gpu = check_gpu_with_tensorflow()
    
    if pytorch_gpu and tensorflow_gpu:
        print("\nBoth PyTorch and TensorFlow can access the GPU.")
        print("No fixes needed.")
        return 0
    
    print("\nApplying fixes...")
    
    # Fix 1: Fix CUDA DLL issues
    print("\nFix 1: Fixing CUDA DLL issues...")
    fix_cuda_dll_issues()
    
    # Fix 2: Reinstall PyTorch with CUDA support
    if not pytorch_gpu:
        print("\nFix 2: Reinstalling PyTorch with CUDA support...")
        install_pytorch_cuda()
    
    # Fix 3: Reinstall TensorFlow
    if not tensorflow_gpu:
        print("\nFix 3: Reinstalling TensorFlow...")
        install_tensorflow_gpu()
    
    # Create custom loader
    print("\nCreating custom CUDA loader script...")
    create_custom_loader()
    
    print("\nAll fixes applied. Please restart your Python environment.")
    print("If issues persist, run the generated cuda_loader.py before importing PyTorch or TensorFlow.")
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 