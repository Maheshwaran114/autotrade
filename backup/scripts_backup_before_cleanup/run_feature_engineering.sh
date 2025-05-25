#!/bin/bash

# Script to run the feature engineering pipeline

# Set working directory to project root
cd "$(dirname "$0")/../.." || exit

# Create necessary directories if they don't exist
mkdir -p data/processed
mkdir -p models
mkdir -p reports/feature_analysis

echo "Starting Feature Engineering Pipeline..."

# Step 1: Run the feature engineering test (generates sample data if needed)
echo "Step 1: Testing feature engineering..."
python src/features/test_features.py

# Check if the previous command was successful
if [ $? -ne 0 ]; then
    echo "Error: Feature engineering test failed."
    exit 1
fi

# Step 2: Run the integration example
echo "Step 2: Running integration example..."
python src/features/integration_example.py

# Check if the previous command was successful
if [ $? -ne 0 ]; then
    echo "Error: Integration example failed."
    exit 1
fi

# Step 3: Run the feature generation script
echo "Step 3: Generating and analyzing features..."
python src/features/generate_features.py

# Check if the previous command was successful
if [ $? -ne 0 ]; then
    echo "Error: Feature generation failed."
    exit 1
fi

echo "Feature engineering pipeline completed successfully."
echo "Generated features are saved in data/processed/"
echo "Feature analysis reports are available in reports/feature_analysis/"
echo "Model and selected features are saved in models/"

exit 0
