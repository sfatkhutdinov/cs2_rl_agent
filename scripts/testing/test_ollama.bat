@echo off
echo Activating virtual environment...
call conda activate cs2_agent

echo Running Ollama integration test...
python test_ollama.py

echo Test complete.
pause 