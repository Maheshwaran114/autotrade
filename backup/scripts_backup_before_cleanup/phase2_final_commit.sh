#!/bin/bash

# Script to perform the final commit for Phase 2 of the Bank Nifty Options Trading System

echo "===== Phase 2 Final Commit ====="
echo "Preparing to commit all Phase 2 changes..."

# Make sure we're in the right directory
cd "$(dirname "$0")/.." || exit

# Check if there are changes to be committed
if ! git diff --quiet || ! git diff --cached --quiet || [ -n "$(git ls-files --others --exclude-standard)" ]; then
    echo "Changes detected. Proceeding with commit..."
    
    # Add all changes
    git add .
    
    # Commit with the specified message
    git commit -m "feat(phase2): data ingest, labeling, features, classifiers, backtest integration"
    
    echo "Successfully committed all Phase 2 changes!"
    echo "Commit message: feat(phase2): data ingest, labeling, features, classifiers, backtest integration"
else
    echo "No changes detected. Nothing to commit."
fi

echo "===== Phase 2 Complete ====="
