#!/bin/bash

# Script to set up and activate the Python virtual environment
# and install required dependencies for the autotrade project

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
source venv/bin/activate

# Install basic dependencies for testing
pip install flask==2.0.1 werkzeug==2.0.1 pytest==6.2.5 flake8==3.9.2 python-dotenv==0.19.1 psycopg2-binary==2.9.1

# Install numpy and pandas (optional - comment out if causing issues)
pip install numpy==1.21.2 pandas==1.3.3

echo ""
echo "Virtual environment is set up and activated."
echo "You can run the tests with: python -m pytest tests/test_app.py -v"
echo ""
echo "To install additional dependencies like scikit-learn, you may need to:"
echo "pip install scikit-learn==1.0"
echo ""
echo "Note: Some dependencies like upstox-python-api==2.0.0 and scikit-learn may require additional setup."
