@echo off
echo Activating virtual environment...
call venv\Scripts\activate

echo Running Ollama integration test...
python test_ollama.py

echo Test complete.
pause 