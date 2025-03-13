@echo off
echo Activating virtual environment...
call conda activate cs2_agent

echo Running DiscoveryEnvironment import test...
python test_discovery_env.py

if %ERRORLEVEL% EQU 0 (
    echo Test passed successfully!
) else (
    echo Test failed. Please check the error messages above.
)

pause 