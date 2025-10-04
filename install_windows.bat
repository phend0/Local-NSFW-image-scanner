@echo off
echo NSFW Image Scanner - Windows Setup
echo ===================================
echo.
echo This script will install the required dependencies for the NSFW Image Scanner.
echo.
pause

echo Checking Python installation...
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ERROR: Python is not installed or not in PATH.
    echo Please install Python 3.8+ from https://python.org
    pause
    exit /b 1
)

echo Python found. Installing dependencies...
echo.

pip install --upgrade pip
pip install -r requirements.txt

echo.
echo Installation complete!
echo.
echo To run the application, double-click nsfw_image_scanner.py
echo or run: python nsfw_image_scanner.py
echo.
pause
