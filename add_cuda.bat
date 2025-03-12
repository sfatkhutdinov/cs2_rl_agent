@echo off
echo Adding CUDA directories to system PATH...

REM Try to find CUDA installation
set FOUND_CUDA=0
for %%v in (11.8 11.7 12.0 12.1 12.2 12.3) do (
    if exist "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v%%v\bin" (
        setx PATH "%PATH%;C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v%%v\bin" /M
        echo Added CUDA v%%v to PATH
        set FOUND_CUDA=1
        goto :found
    )
)

:found
if %FOUND_CUDA%==0 (
    echo No CUDA installation found.
    echo Please install CUDA Toolkit from https://developer.nvidia.com/cuda-downloads
) else (
    echo CUDA added to PATH successfully.
    echo Please restart your computer for changes to take effect.
)
pause