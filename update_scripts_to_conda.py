import os
import glob
import re

def update_batch_file(file_path):
    """Update a batch file to use conda instead of venv"""
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Replace venv activation with conda activation
    content = re.sub(
        r'call venv\\Scripts\\activate(?:\.bat)?',
        'call conda activate cs2_agent',
        content
    )
    
    # Update any pip install commands to use conda when appropriate
    content = re.sub(
        r'pip install -r requirements\.txt',
        'REM Using conda environment with pre-installed packages',
        content
    )
    
    # Save the updated content
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(content)
    
    print(f"Updated: {file_path}")

def main():
    """Find all batch files and update them to use conda"""
    # Get all .bat files in the current directory
    batch_files = glob.glob("*.bat")
    
    for batch_file in batch_files:
        # Skip the setup files as we've created a new one
        if batch_file in ["setup_venv.bat", "setup_conda.bat"]:
            continue
        
        update_batch_file(batch_file)

if __name__ == "__main__":
    main()
    print("Batch file update complete!") 