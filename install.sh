#!/bin/bash

# Script to set up the Python environment for GNN Edit Paths project

echo "Setting up Python environment for GNN Edit Paths..."

# Detect operating system
if [[ "$OSTYPE" == "msys" || "$OSTYPE" == "win32" ]]; then
    echo "Detected Windows operating system"
    ACTIVATE_CMD="venv\\Scripts\\activate"
    # Check for Python 3.12, 3.11, or 3.10
    if command -v python3.12 &> /dev/null; then
        PYTHON_CMD="python3.12"
        echo "Using Python 3.12"
    elif command -v python3.11 &> /dev/null; then
        PYTHON_CMD="python3.11"
        echo "Python 3.12 not found. Using Python 3.11 instead."
    elif command -v python3.10 &> /dev/null; then
        PYTHON_CMD="python3.10"
        echo "Python 3.12 and 3.11 not found. Using Python 3.10 instead."
    else
        echo "Python 3.12, 3.11, or 3.10 could not be found. Please install one of these Python versions and try again."
        exit 1
    fi
else
    echo "Detected Unix-like operating system (Linux/macOS)"
    ACTIVATE_CMD="source venv/bin/activate"
    # Check for Python 3.12, 3.11, or 3.10
    if command -v python3.12 &> /dev/null; then
        PYTHON_CMD="python3.12"
        echo "Using Python 3.12"
    elif command -v python3.11 &> /dev/null; then
        PYTHON_CMD="python3.11"
        echo "Python 3.12 not found. Using Python 3.11 instead."
    elif command -v python3.10 &> /dev/null; then
        PYTHON_CMD="python3.10"
        echo "Python 3.12 and 3.11 not found. Using Python 3.10 instead."
    else
        echo "Python 3.12, 3.11, or 3.10 could not be found. Please install one of these Python versions and try again."
        exit 1
    fi
    # Use the detected Python version on Unix-like systems
    alias python=$PYTHON_CMD
fi

# Create virtual environment
echo "Creating virtual environment..."
$PYTHON_CMD -m venv venv

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
