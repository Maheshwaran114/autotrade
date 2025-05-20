#!/bin/bash
# Run the Bank Nifty Options Trading System

# Ensure we're in the project root directory
cd "$(dirname "$0")"

# Activate the virtual environment if it exists
if [ -d "venv" ]; then
    echo "Activating virtual environment..."
    source venv/bin/activate
else
    echo "Virtual environment not found. Creating one..."
    python -m venv venv
    source venv/bin/activate
    pip install -r requirements.txt
fi

# Set Python path to include the src directory
export PYTHONPATH="$PYTHONPATH:$(pwd)"

# Run the application
echo "Starting Bank Nifty Options Trading System..."
python src/main.py "$@"
