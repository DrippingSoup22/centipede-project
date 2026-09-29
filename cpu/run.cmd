@echo off
setlocal
if not defined CENTIPEDE_PYTHON set "CENTIPEDE_PYTHON=%USERPROFILE%\.venvs\Centipede\Scripts\python.exe"
if not exist "%CENTIPEDE_PYTHON%" (
    echo Python not found: %CENTIPEDE_PYTHON%
    echo Set CENTIPEDE_PYTHON to the Centipede environment's python.exe.
    exit /b 1
)
"%CENTIPEDE_PYTHON%" -m centipede %*
exit /b %ERRORLEVEL%
