@echo off
set PYTHON_EXE=%~dp0Scripts\python.exe

if not exist "%PYTHON_EXE%" (
  echo Project Python interpreter not found at "%PYTHON_EXE%"
  exit /b 1
)

"%PYTHON_EXE%" "%~dp0app.py"
