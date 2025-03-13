@echo off
echo Setting up Python virtual environment for Cities: Skylines 2 RL Agent
echo ---------------------------------------------------

REM Check if Python is installed
python --version > nul 2>&1
if errorlevel 1 (
    echo Python is not installed or not in PATH
    echo Please install Python 3.8 or higher
    pause
    exit /b 1
)

REM Create virtual environment if it doesn't exist
if not exist venv (
    echo Creating virtual environment...
    python -m venv venv
) else (
    echo Virtual environment already exists
)

REM Activate virtual environment
call venv\Scripts\activate.bat

REM Upgrade pip
python -m pip install --upgrade pip

REM Install requirements
echo Installing required packages...
pip install -r requirements.txt

REM Install PyTorch with CUDA if available
echo Checking for CUDA support...
nvidia-smi > nul 2>&1
if errorlevel 1 (
    echo CUDA not found, installing CPU version of PyTorch
    pip install torch torchvision torchaudio
) else (
    echo CUDA found, installing GPU version of PyTorch
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
)

REM Check Tesseract installation
echo Checking Tesseract OCR installation...
tesseract --version > nul 2>&1
if errorlevel 1 (
    echo WARNING: Tesseract OCR is not installed or not in PATH
    echo Please install Tesseract OCR from: https://github.com/UB-Mannheim/tesseract/wiki
    echo After installation, add it to your PATH environment variable
    echo Recommended installation path: C:\Program Files\Tesseract-OCR
)

echo.
echo Setup complete! You can now run train_autonomous.bat
pause 