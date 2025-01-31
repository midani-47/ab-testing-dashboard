#!/bin/bash

# Function to check if command exists
command_exists () {
    type "$1" &> /dev/null ;
}

echo "Setting up AB Testing Dashboard..."

# Check Python version
if command_exists python3; then
    PYTHON_CMD=python3
elif command_exists python; then
    PYTHON_CMD=python
else
    echo "Error: Python is not installed. Please install Python 3.8 or higher."
    exit 1
fi

# Check Python version meets minimum requirements
PYTHON_VERSION=$($PYTHON_CMD -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
if (( $(echo "$PYTHON_VERSION < 3.8" | bc -l) )); then
    echo "Error: Python version must be 3.8 or higher. Current version: $PYTHON_VERSION"
    exit 1
fi

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    $PYTHON_CMD -m venv venv
fi

# Activate virtual environment
if [[ "$OSTYPE" == "msys" || "$OSTYPE" == "win32" ]]; then
    source venv/Scripts/activate
else
    source venv/bin/activate
fi

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# Install requirements
echo "Installing requirements..."
pip install -r requirements.txt

echo "Setup complete! You can now run the dashboard with: streamlit run app.py"