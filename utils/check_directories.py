import os
import logging

def ensure_directory_structure():
    """
    Ensure all required directories exist for the project.
    Creates any missing directories.
    """
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    
    # Required directories
    directories = [
        "src",
        "src/agent",
        "src/environment",
        "src/interface",
        "src/utils",
        "config",
        "models",
        "logs",
        "debug",
        "debug/vision"
    ]
    
    # Create missing directories
    for directory in directories:
        if not os.path.exists(directory):
            os.makedirs(directory, exist_ok=True)
            logging.info(f"Created directory: {directory}")
        else:
            logging.info(f"Directory exists: {directory}")
    
    # Create empty __init__.py files for Python packages if they don't exist
    python_packages = [
        "src/__init__.py",
        "src/agent/__init__.py",
        "src/environment/__init__.py",
        "src/interface/__init__.py",
        "src/utils/__init__.py",
    ]
    
    for package_init in python_packages:
        if not os.path.exists(package_init):
            with open(package_init, "w") as f:
                f.write("# Make directory a Python package\n")
            logging.info(f"Created package file: {package_init}")
    
    logging.info("Directory structure check completed")

if __name__ == "__main__":
    ensure_directory_structure() 