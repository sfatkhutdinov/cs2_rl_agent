@echo off
echo Generating template files for Cities: Skylines 2 RL Agent
echo ---------------------------------------------------

REM Try to activate virtual environment
if exist venv\Scripts\activate.bat (
    call venv\Scripts\activate.bat
) else if exist .venv\Scripts\activate.bat (
    call .venv\Scripts\activate.bat
) else if exist env\Scripts\activate.bat (
    call env\Scripts\activate.bat
) else (
    echo Virtual environment not found. Using system Python.
)

echo Creating template files...
python generate_templates.py

echo.
echo Template generation complete!
echo You may want to replace these placeholder templates with actual screenshots
echo from your game for better detection accuracy.
pause 