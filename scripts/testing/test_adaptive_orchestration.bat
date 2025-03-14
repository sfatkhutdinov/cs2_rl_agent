@echo off
REM Test Adaptive Agent Orchestration
REM This script runs the adaptive orchestration test to verify the agent's ability to switch modes correctly
REM Last updated: March 13, 2025 21:17

echo ===== Testing Adaptive Agent Orchestration =====
echo This test verifies that the adaptive agent correctly switches between specialized modes

REM Setup environment
call scripts\utils\setup_conda.bat
if %ERRORLEVEL% neq 0 (
    echo Error setting up conda environment
    exit /b %ERRORLEVEL%
)

REM Check for required dependencies
python -c "import numpy; import tensorflow" 2>nul
if %ERRORLEVEL% neq 0 (
    echo Installing required dependencies...
    pip install numpy tensorflow
)

echo Running adaptive orchestration test...

REM Default to mock environment for testing
set MOCK_ENV=--mock-env
set TEST_DURATION=60

REM Parse command line arguments
if "%1"=="--real-env" (
    set MOCK_ENV=
    echo Using real environment for testing
)

if not "%2"=="" (
    set TEST_DURATION=%2
    echo Setting test duration to %TEST_DURATION% seconds
)

REM Apply TensorFlow patch to avoid compatibility issues
python src\utils\patch_tensorflow.py

REM Run the test
python testing\test_adaptive_orchestration.py --config config\adaptive_config.yaml --test-duration %TEST_DURATION% %MOCK_ENV%

if %ERRORLEVEL% neq 0 (
    echo TEST FAILED with error code %ERRORLEVEL%
    exit /b %ERRORLEVEL%
)

echo Adaptive orchestration test completed successfully.
echo.
echo For more detailed testing, you can run:
echo   scripts\testing\test_adaptive_orchestration.bat --real-env 300
echo.
echo This will run the test with the real environment for 5 minutes.

exit /b 0 