@echo off
echo Activating virtual environment...
call venv\Scripts\activate

echo Checking if Ollama is running...
curl -s http://localhost:11434/api/tags > nul
if %ERRORLEVEL% NEQ 0 (
    echo ERROR: Ollama is not running. Please start Ollama first.
    pause
    exit /b 1
)

echo Installing required dependencies...
pip install -r requirements.txt

echo Checking directory structure...
python check_directories.py

echo Testing configuration...
python test_config.py config/discovery_config.yaml
if %ERRORLEVEL% NEQ 0 (
    echo Configuration test failed. Please fix the issues before running training.
    pause
    exit /b 1
)

echo Warming up the Granite model...
curl -s -X POST http://localhost:11434/api/generate -d "{\"model\":\"granite3.2-vision:latest\",\"prompt\":\"Hello, are you ready to analyze game screens?\",\"stream\":false}" > nul

echo Starting discovery-based training...
echo.
echo IMPORTANT: Please make sure Cities: Skylines 2 is running and visible on your screen.
echo The script will now focus on the game window.
echo.
timeout /t 5

echo Focusing on game window and starting training...
python train_discovery.py --config config/discovery_config.yaml

pause 