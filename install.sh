#!/bin/bash

# Script to set up the Python environment for GNN Edit Paths project

echo "Setting up Python environment for GNN Edit Paths..."

# Detect operating system
if [[ "$OSTYPE" == "msys" || "$OSTYPE" == "win32" ]]; then
    echo "Detected Windows operating system"
    ACTIVATE_CMD="venv\\Scripts\\activate"
    # Check if Python 3.12 is installed
    if ! command -v python3.12 &> /dev/null; then
        echo "Python 3.12 could not be found. Please install Python 3.12 and try again."
        exit 1
    fi
else
    echo "Detected Unix-like operating system (Linux/macOS)"
    ACTIVATE_CMD="source venv/bin/activate"
    # Check if Python 3.12 is installed
    if ! command -v python3.12 &> /dev/null; then
        echo "Python 3.12 could not be found. Please install Python 3.12 and try again."
        exit 1
    fi
    # Use python3.12 command on Unix-like systems
    alias python=python3.12
fi

# Create virtual environment
echo "Creating virtual environment..."
python -m venv venv

# Activate virtual environment
echo "Activating virtual environment..."
if [[ "$OSTYPE" == "msys" || "$OSTYPE" == "win32" ]]; then
    source venv/Scripts/activate
else
    source venv/bin/activate
fi

# Install required packages
echo "Installing required packages..."
# Install all dependencies from requirements.txt
# install torch
pip install torch~=2.7.1 --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt


echo "Installation complete! The virtual environment has been set up with all required dependencies."
echo "To activate the environment in the future, run: $ACTIVATE_CMD"
