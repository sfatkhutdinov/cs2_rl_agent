"""
GPU Checker - Simple diagnostic tool for PyTorch and TensorFlow GPU detection
"""

import os
import sys
import subprocess
import platform

def print_separator(title):
    """Print a separator with title"""
    print("\n" + "=" * 80)
    print(f" {title} ".center(78, "="))
    print("=" * 80)

def check_system_info():
    """Print basic system information"""
    print_separator("System Information")
    print(f"Python version: {sys.version.split()[0]}")
    print(f"Platform: {platform.platform()}")
    print(f"OS: {platform.system()} {platform.version()}")
    
    # Check for environment variables
    for var in ["CUDA_VISIBLE_DEVICES", "TF_FORCE_GPU_ALLOW_GROWTH", "PYTORCH_CUDA_ALLOC_CONF"]:
        value = os.environ.get(var, "Not set")
        print(f"{var}: {value}")

def check_nvidia_smi():
    """Check if nvidia-smi is available and working"""
    print_separator("NVIDIA Driver")
    try:
        output = subprocess.check_output(["nvidia-smi"], stderr=subprocess.STDOUT, text=True)
        print("NVIDIA driver working:")
        print(output.splitlines()[0])  # Print the first line with version info
        print(output.splitlines()[1])  # Print the second line
        return True
    except (subprocess.SubprocessError, FileNotFoundError):
        print("❌ Error: nvidia-smi failed. NVIDIA driver might not be installed or working properly.")
        print("   Please install the latest NVIDIA drivers from: https://www.nvidia.com/Download/index.aspx")
        return False

def check_pytorch():
    """Check if PyTorch can see the GPU"""
    print_separator("PyTorch GPU Support")
    try:
        import torch
        print(f"PyTorch version: {torch.__version__}")
        
        if torch.cuda.is_available():
            device_count = torch.cuda.device_count()
            device_name = torch.cuda.get_device_name(0) if device_count > 0 else "Unknown"
            
            print("✅ PyTorch CUDA is available!")
            print(f"   CUDA version: {torch.version.cuda}")
            print(f"   cuDNN version: {torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else 'Not available'}")
            print(f"   Device count: {device_count}")
            print(f"   Device name: {device_name}")
            
            # Try a simple tensor operation on GPU
            try:
                x = torch.tensor([1.0, 2.0, 3.0], device='cuda')
                y = x * 2
                print(f"   Simple GPU tensor test: {y.cpu().numpy()}")
                print("   ✅ GPU computation successful!")
            except Exception as e:
                print(f"   ❌ GPU computation failed: {e}")
                
            return True
        else:
            print("❌ PyTorch CUDA is not available.")
            print("   Possible causes:")
            print("   1. PyTorch was installed without CUDA support")
            print("   2. CUDA/cuDNN version mismatch")
            print("   3. GPU drivers need to be updated")
            print("\n   To install PyTorch with CUDA support, run:")
            print("   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118")
            return False
    except ImportError:
        print("❌ PyTorch is not installed.")
        print("   To install PyTorch with CUDA support, run:")
        print("   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118")
        return False
    except Exception as e:
        print(f"❌ Error checking PyTorch CUDA: {e}")
        return False

def check_tensorflow():
    """Check if TensorFlow can see the GPU"""
    print_separator("TensorFlow GPU Support")
    try:
        import tensorflow as tf
        print(f"TensorFlow version: {tf.__version__}")
        
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            print("✅ TensorFlow GPU support is available!")
            print(f"   Number of GPUs detected: {len(gpus)}")
            for i, gpu in enumerate(gpus):
                print(f"   GPU #{i+1}: {gpu}")
            
            # Try a simple tensor operation on GPU
            try:
                with tf.device('/GPU:0'):
                    x = tf.constant([1.0, 2.0, 3.0])
                    y = x * 2
                    print(f"   Simple GPU tensor test: {y.numpy()}")
                    print("   ✅ GPU computation successful!")
            except Exception as e:
                print(f"   ❌ GPU computation failed: {e}")
                
            return True
        else:
            print("❌ TensorFlow cannot detect any GPUs.")
            print("   Possible causes:")
            print("   1. TensorFlow was installed without GPU support")
            print("   2. CUDA/cuDNN version mismatch")
            print("   3. GPU drivers need to be updated")
            print("\n   For TensorFlow GPU support, you need compatible CUDA and cuDNN versions")
            return False
    except ImportError:
        print("❌ TensorFlow is not installed.")
        print("   To install TensorFlow, run:")
        print("   pip install tensorflow")
        return False
    except Exception as e:
        print(f"❌ Error checking TensorFlow GPU: {e}")
        return False

def check_path_for_cuda():
    """Check if CUDA is in the system PATH"""
    print_separator("CUDA in PATH")
    path = os.environ.get("PATH", "")
    cuda_paths = [p for p in path.split(os.pathsep) if "cuda" in p.lower()]
    
    if cuda_paths:
        print("✅ CUDA found in PATH:")
        for p in cuda_paths:
            print(f"   {p}")
        return True
    else:
        print("❌ CUDA not found in PATH")
        print("   This might prevent libraries from finding CUDA runtime files")
        return False

def print_summary(nvidia_ok, pytorch_ok, tensorflow_ok, path_ok):
    """Print a summary of the checks"""
    print_separator("Summary")
    
    if nvidia_ok and pytorch_ok and tensorflow_ok:
        print("✅ GPU SUPPORT IS FULLY AVAILABLE!")
        print("   Both PyTorch and TensorFlow can use your GPU.")
    elif nvidia_ok and (pytorch_ok or tensorflow_ok):
        print("✅ GPU SUPPORT IS PARTIALLY AVAILABLE!")
        print(f"   PyTorch GPU support: {'Yes' if pytorch_ok else 'No'}")
        print(f"   TensorFlow GPU support: {'Yes' if tensorflow_ok else 'No'}")
    else:
        print("❌ GPU SUPPORT IS NOT AVAILABLE!")
        print("   Please check the details above to troubleshoot.")
    
    print("\nRecommended actions:")
    if not nvidia_ok:
        print("1. Install or update NVIDIA drivers")
    if not path_ok:
        print("2. Add CUDA bin directory to your PATH environment variable")
    if not pytorch_ok:
        print("3. Reinstall PyTorch with CUDA support:")
        print("   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118")
    if not tensorflow_ok:
        print("4. Reinstall TensorFlow:")
        print("   pip install tensorflow")

if __name__ == "__main__":
    print("\nGPU Checker - Diagnosing GPU support for Machine Learning\n")
    
    # Set environment variables that might help
    os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    
    check_system_info()
    nvidia_ok = check_nvidia_smi()
    pytorch_ok = check_pytorch()
    tensorflow_ok = check_tensorflow()
    path_ok = check_path_for_cuda()
    
    print_summary(nvidia_ok, pytorch_ok, tensorflow_ok, path_ok) 