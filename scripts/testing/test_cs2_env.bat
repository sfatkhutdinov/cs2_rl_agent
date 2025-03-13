@echo off
echo Activating virtual environment...
call conda activate cs2_agent

echo Testing CS2Environment...
python ..\..\testing\test_cs2_env.py

if %ERRORLEVEL% EQU 0 (
    echo Test passed successfully!
) else (
    echo Test failed! Please check the error messages above.
)

pause 