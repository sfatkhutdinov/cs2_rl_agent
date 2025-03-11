@echo off
echo Activating virtual environment...
call venv\Scripts\activate

echo Checking if Ollama is running...
curl -s http://localhost:11434/api/version > nul
if %ERRORLEVEL% NEQ 0 (
    echo WARNING: Ollama does not appear to be running. Please start it before continuing.
    echo You can start Ollama by running the 'ollama serve' command.
    timeout /t 5
)

echo Starting discovery-based training...
python train_discovery.py --config config/discovery_config.yaml --log-level INFO

echo Training complete.
pause 