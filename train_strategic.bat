@echo off
setlocal

echo -------------------------------------------
echo Cities: Skylines 2 Strategic Agent Training
echo -------------------------------------------

:: Check Python environment
call check_conda.bat
if %ERRORLEVEL% neq 0 (
    echo Error: Conda environment not properly set up
    exit /b 1
)

:: Set PYTHONPATH to include the current directory
set PYTHONPATH=%CD%;%PYTHONPATH%

:: Create directories if they don't exist
if not exist logs mkdir logs
if not exist models mkdir models
if not exist models\strategic mkdir models\strategic
if not exist tensorboard mkdir tensorboard
if not exist tensorboard\strategic mkdir tensorboard\strategic

echo Starting Strategic Agent training...

:: Parse command line arguments
set TIMESTEPS=5000000
set USE_ADAPTIVE=false
set BOOTSTRAP=false
set CHECKPOINT=

:parse_args
if "%~1"=="" goto :start_training
if /i "%~1"=="--timesteps" (
    set TIMESTEPS=%~2
    shift
    shift
    goto :parse_args
)
if /i "%~1"=="--use-adaptive" (
    set USE_ADAPTIVE=true
    shift
    goto :parse_args
)
if /i "%~1"=="--knowledge-bootstrap" (
    set BOOTSTRAP=true
    shift
    goto :parse_args
)
if /i "%~1"=="--load-checkpoint" (
    set CHECKPOINT=%~2
    shift
    shift
    goto :parse_args
)
shift
goto :parse_args

:start_training
echo Configuration:
echo - Training steps: %TIMESTEPS%
echo - Use adaptive agent: %USE_ADAPTIVE%
echo - Knowledge bootstrapping: %BOOTSTRAP%
if not "%CHECKPOINT%"=="" echo - Resume from checkpoint: %CHECKPOINT%

echo.
echo Training will now begin. This process may take several hours.
echo Press Ctrl+C to stop training at any time.
echo.

:: Build command with appropriate flags
set CMD=python train_strategic.py --timesteps %TIMESTEPS%

if "%USE_ADAPTIVE%"=="true" (
    set CMD=%CMD% --use-adaptive
)

if "%BOOTSTRAP%"=="true" (
    set CMD=%CMD% --knowledge-bootstrap
)

if not "%CHECKPOINT%"=="" (
    set CMD=%CMD% --load-checkpoint "%CHECKPOINT%"
)

:: Run the training
echo Running command: %CMD%
%CMD%

echo.
if %ERRORLEVEL% equ 0 (
    echo Strategic agent training completed successfully!
) else (
    echo Strategic agent training encountered an error.
)

endlocal 