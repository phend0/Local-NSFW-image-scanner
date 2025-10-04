#!/bin/bash

echo "NSFW Image Scanner - Linux/Mac Setup"
echo "====================================="
echo

echo "This script will install the required dependencies for the NSFW Image Scanner."
echo
read -p "Press Enter to continue..."

echo "Checking Python installation..."
if ! command -v python3 &> /dev/null; then
    echo "ERROR: Python 3 is not installed."
    echo "Please install Python 3.8+ using your package manager:"
    echo "  Ubuntu/Debian: sudo apt update && sudo apt install python3 python3-pip python3-tk"
    echo "  CentOS/RHEL: sudo yum install python3 python3-pip tkinter"
    echo "  macOS: brew install python-tk"
    exit 1
fi

echo "Python found. Installing dependencies..."
echo

# Check if we're in a virtual environment, if not, suggest creating one
if [[ -z "$VIRTUAL_ENV" ]]; then
    echo "Recommendation: Use a virtual environment for cleaner dependency management"
    echo "Would you like to create one? (y/n)"
    read -r create_venv
    if [[ $create_venv == "y" || $create_venv == "Y" ]]; then
        python3 -m venv nsfw_scanner_env
        source nsfw_scanner_env/bin/activate
        echo "Virtual environment created and activated"
    fi
fi

python3 -m pip install --upgrade pip
python3 -m pip install -r requirements.txt

echo
echo "Installation complete!"
echo

if [[ -n "$VIRTUAL_ENV" ]]; then
    echo "To run the application:"
    echo "1. Activate the virtual environment: source nsfw_scanner_env/bin/activate"
    echo "2. Run the application: python3 nsfw_image_scanner.py"
else
    echo "To run the application: python3 nsfw_image_scanner.py"
fi

echo
