@echo off
echo Activating virtual environment...
call conda activate cs2_agent

echo Testing configuration...
python test_config.py config/discovery_config.yaml

if %ERRORLEVEL% EQU 0 (
    echo Configuration test passed! You can now run the training script.
) else (
    echo Configuration test failed. Please fix the issues before running training.
)

pause 