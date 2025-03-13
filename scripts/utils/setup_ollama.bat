@echo off
echo Checking if Ollama is running...
curl -s http://localhost:11434/api/tags > nul
if %ERRORLEVEL% NEQ 0 (
    echo ERROR: Ollama is not running. Please start Ollama first.
    pause
    exit /b 1
)

echo Checking if granite3.2-vision model is installed...
curl -s http://localhost:11434/api/tags | findstr "granite3.2-vision" > nul
if %ERRORLEVEL% NEQ 0 (
    echo Granite 3.2 Vision model not found. Installing...
    curl -s -X POST http://localhost:11434/api/pull -d "{\"model\":\"granite3.2-vision:latest\"}" 
    if %ERRORLEVEL% NEQ 0 (
        echo Failed to install Granite 3.2 Vision model.
        pause
        exit /b 1
    )
    echo Granite 3.2 Vision model installed successfully.
) else (
    echo Granite 3.2 Vision model is already installed.
)

echo Warming up the model with a test query...
echo.
curl -s -X POST http://localhost:11434/api/generate -d "{\"model\":\"granite3.2-vision:latest\",\"prompt\":\"Hello, are you ready to analyze game screens?\",\"stream\":false}" > nul

echo.
echo Ollama setup completed successfully.
echo The Granite 3.2 Vision model is ready for use.
echo.
echo Next steps:
echo 1. Make sure Cities: Skylines 2 is running and visible
echo 2. Run train_discovery_with_focus.bat to start training
echo.
pause 