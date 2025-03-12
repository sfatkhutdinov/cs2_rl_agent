@echo off
echo Running GPU Checker to diagnose GPU availability for ML...
echo.

REM Activate virtual environment if it exists
if exist venv\Scripts\activate.bat (
    call venv\Scripts\activate.bat
)

REM Set environment variables that might help
set CUDA_VISIBLE_DEVICES=0
set TF_FORCE_GPU_ALLOW_GROWTH=true
set PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

python check_gpu.py

echo.
echo GPU check complete. Press any key to exit...
pause > nul 