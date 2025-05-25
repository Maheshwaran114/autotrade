#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Test script for feature engineering module.
This script demonstrates the basic usage of the feature engineering pipeline.
"""

import os
import sys
import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Add project root to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.features.feature_engineering import FeatureEngineer
from src.features.feature_selection import FeatureSelector

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def create_sample_data(n_samples=100):
    """
    Create sample data for testing if actual data not available
    
    Args:
        n_samples: Number of samples to generate
        
    Returns:
        DataFrame with sample data
    """
    # Generate dates
    start_date = pd.Timestamp('2025-01-01')
    dates = [start_date + pd.Timedelta(days=i) for i in range(n_samples)]
    
    # Filter for weekdays only
    dates = [date for date in dates if date.dayofweek < 5]
    n_weekdays = len(dates)
    
    # Generate sample data
    base_price = 48000
    daily_volatility = 500
    
    # Create a price series with some persistence
    returns = np.random.normal(0.0002, 0.015, n_weekdays)  # Small positive drift
    
    # Add some autocorrelation to returns
    for i in range(1, len(returns)):
        returns[i] = 0.7 * returns[i] + 0.3 * returns[i-1]
    
    # Generate price series
    closes = [base_price]
    for ret in returns:
        closes.append(closes[-1] * (1 + ret))
    closes = closes[1:]  # Remove initial price
    
    # Generate OHLC
    opens = [close * (1 + np.random.normal(0, 0.003)) for close in closes]
    highs = [max(open_price, close) * (1 + abs(np.random.normal(0, 0.005))) 
             for open_price, close in zip(opens, closes)]
    lows = [min(open_price, close) * (1 - abs(np.random.normal(0, 0.005)))
            for open_price, close in zip(opens, closes)]
    
    # Generate volumes
    volumes = [int(np.random.normal(1000000, 200000)) for _ in range(n_weekdays)]
    
    # Create DataFrame
    df = pd.DataFrame({
        'date': dates,
        'open': opens,
        'high': highs,
        'low': lows,
        'close': closes,
        'volume': volumes
    })
    
    # Add first 30 min data
    df['first_30min_high'] = df['high'] * np.random.uniform(0.95, 0.99, n_weekdays)
    df['first_30min_low'] = df['low'] * np.random.uniform(1.01, 1.05, n_weekdays)
    
    # Add implied volatility (simulated)
    df['implied_volatility'] = 15 + np.random.normal(0, 3, n_weekdays)
    for i in range(1, n_weekdays):
        df.loc[i, 'implied_volatility'] = df.loc[i-1, 'implied_volatility'] * 0.9 + \
                                          df.loc[i, 'implied_volatility'] * 0.1
    
    # Add labels (for testing)
    df['label'] = np.random.choice(
        ['Trend', 'RangeBound', 'Event', 'MildBias', 'Momentum'],
        size=n_weekdays,
        p=[0.2, 0.25, 0.1, 0.3, 0.15]
    )
    
    return df


def main():
    """Main function to test feature engineering"""
    try:
        labeled_data_path = "data/processed/labeled_days.parquet"
        
        # Check if labeled data exists
        if os.path.exists(labeled_data_path):
            logger.info(f"Loading labeled data from {labeled_data_path}")
            df = pd.read_parquet(labeled_data_path)
        else:
            # Create sample data if actual data not available
            logger.info("Labeled data not found. Creating sample data for testing.")
            df = create_sample_data(100)
            os.makedirs("data/processed", exist_ok=True)
            df.to_parquet("data/processed/sample_labeled_days.parquet")
            labeled_data_path = "data/processed/sample_labeled_days.parquet"
        
        logger.info(f"Data shape: {df.shape}")
        logger.info(f"Columns: {df.columns.tolist()}")
        
        # Create feature engineer
        engineer = FeatureEngineer()
        
        # Compute features
        logger.info("Computing features...")
        features_df = engineer.compute_features(df)
        logger.info(f"Features shape: {features_df.shape}")
        logger.info(f"Feature columns: {features_df.columns.tolist()}")
        
        # Display some features
        logger.info("Sample features:")
        print(features_df.head())
        
        # Create and test feature selector
        if 'label' in features_df.columns:
            logger.info("Testing feature selection...")
            X = features_df.drop(columns=['label', 'day_of_week'], errors='ignore')
            y = features_df['label']
            
            selector = FeatureSelector()
            
            # Test correlation-based selection
            corr_features = selector.correlation_based_selection(X, 0.8)
            logger.info(f"Selected {len(corr_features)} features after correlation filtering")
            
            # Test importance-based selection
            importance_features, importance_df = selector.importance_based_selection(X, y, 0.01)
            logger.info(f"Selected {len(importance_features)} features based on importance")
            
            if len(importance_df) > 0:
                logger.info("Top 5 features by importance:")
                print(importance_df.head(5))
            
            # Save top features to a CSV file for reference
            if len(importance_df) > 0:
                os.makedirs("reports", exist_ok=True)
                importance_df.to_csv("reports/feature_importance.csv", index=False)
                logger.info("Saved feature importance to reports/feature_importance.csv")
        
        logger.info("Feature engineering test completed successfully")
        
    except Exception as e:
        logger.error(f"Error in feature engineering test: {e}", exc_info=True)
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
