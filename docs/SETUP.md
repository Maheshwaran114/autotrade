# Bank Nifty Options Trading System - Setup Guide

This document provides instructions on how to set up and run the Bank Nifty Options Trading System.

## Prerequisites

- Python 3.8 or higher
- Git (for version control)
- Basic knowledge of options trading concepts

## Installation

### 1. Clone the repository

```bash
git clone <repository-url>
cd autotrade
```

### 2. Create a virtual environment

Create a Python virtual environment to isolate dependencies:

```bash
# Create a virtual environment
python -m venv venv

# Activate the virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
```

### 3. Install dependencies

Install the required Python packages:

```bash
pip install -r requirements.txt
```

## Configuration

Before running the system, you need to configure it:

1. Create your broker configuration file by copying the template:

```bash
cp config/broker_config.template.json config/broker_config.json
```

2. Edit `config/broker_config.json` and add your broker API credentials.

## Running the System

### Run the Dashboard

The dashboard provides a complete interface to the trading system:

```bash
python -m src.dashboard.app
```

This will start the dashboard application which initializes all system components and performs a system check.

### Run Individual Components

You can also run individual components for testing:

```bash
# Test the market data fetcher
python -m src.data_ingest.fetch_data

# Test the day classifier
python -m src.ml_models.day_classifier

# Test the delta-theta strategy
python -m src.strategies.delta_theta

# Test the gamma scalping strategy
python -m src.strategies.gamma_scalping

# Test the order manager
python -m src.execution.order_manager
```

## Running Tests

Run all tests:

```bash
python -m unittest discover tests
```

Run a specific test:

```bash
python -m unittest tests.test_smoke
```

## Project Structure

```
autotrade/
├── config/                   # Configuration files
├── docs/                     # Documentation
├── src/                      # Source code
│   ├── dashboard/            # Dashboard UI
│   ├── data_ingest/          # Market data fetching
│   ├── execution/            # Order execution
│   ├── ml_models/            # ML models for market analysis
│   └── strategies/           # Trading strategies
├── tests/                    # Unit and integration tests
├── scripts/                  # Utility scripts
├── requirements.txt          # Dependencies
└── README.md                 # Project overview
```

## Development Workflow

1. Activate the virtual environment
2. Make your code changes
3. Run tests to ensure everything works
4. Commit your changes with a descriptive message
5. Push to the repository

## Troubleshooting

### Import Errors

If you encounter import errors when running individual modules, ensure that the Python path includes the project root:

```bash
# From the project root
export PYTHONPATH=$PYTHONPATH:$(pwd)
# Or on Windows
set PYTHONPATH=%PYTHONPATH%;%cd%
```

### API Connection Issues

If you encounter issues connecting to your broker's API:

1. Verify your API credentials in `config/broker_config.json`
2. Check your internet connection
3. Ensure your broker's API service is operational
4. Check for any IP restrictions from your broker

## Contributing

1. Create a branch for your feature or bugfix
2. Make your changes
3. Run tests and ensure they pass
4. Submit a pull request with a clear description of your changes
