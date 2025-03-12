@echo off
echo Activating virtual environment...
call venv\Scripts\activate

echo Testing CS2Environment...
python test_cs2_env.py

if %ERRORLEVEL% EQU 0 (
    echo Test passed successfully!
) else (
    echo Test failed! Please check the error messages above.
)

pause 