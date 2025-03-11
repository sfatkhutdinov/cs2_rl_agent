import os
import shutil
from typing import List, Optional

def ensure_dir(dir_path: str) -> None:
    """
    Ensure that a directory exists, creating it if necessary.
    
    Args:
        dir_path: Path to directory
    """
    if not os.path.exists(dir_path):
        os.makedirs(dir_path, exist_ok=True)

def clean_dir(dir_path: str, preserve: Optional[List[str]] = None) -> None:
    """
    Clean a directory by removing all files and subdirectories except those listed in preserve.
    
    Args:
        dir_path: Path to directory
        preserve: List of file/directory names to preserve
    """
    preserve = preserve or []
    
    if not os.path.exists(dir_path):
        return
    
    for item in os.listdir(dir_path):
        item_path = os.path.join(dir_path, item)
        if item not in preserve:
            if os.path.isfile(item_path):
                os.remove(item_path)
            elif os.path.isdir(item_path):
                shutil.rmtree(item_path)

def safe_copy_file(src: str, dst: str, overwrite: bool = False) -> bool:
    """
    Safely copy a file from source to destination.
    
    Args:
        src: Source file path
        dst: Destination file path
        overwrite: Whether to overwrite existing file
        
    Returns:
        True if file was copied, False otherwise
    """
    if not os.path.exists(src):
        return False
    
    if os.path.exists(dst) and not overwrite:
        return False
    
    # Make sure destination directory exists
    dst_dir = os.path.dirname(dst)
    ensure_dir(dst_dir)
    
    # Copy file
    shutil.copy2(src, dst)
    return True 