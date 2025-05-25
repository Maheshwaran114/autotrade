#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Feature engineering module for Bank Nifty Options trading system.
This module computes various technical and market features from historical price data.
"""

import os
import logging
import numpy as np
import pandas as pd
import joblib
from typing import Dict, List, Optional, Union, Tuple
from datetime import datetime, timedelta
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# Configure logging
logger = logging.getLogger(__name__)


class FeatureEngineer:
    """
    Extracts and transforms features from labeled financial data
    for use in machine learning models.
    """

    def __init__(self, scaler_type: str = 'standard'):
        """
        Initialize the feature engineer.
        
        Args:
            scaler_type: Type of scaler to use ('standard' or 'minmax')
        """
        self.scaler_type = scaler_type
        self.scaler = None
        logger.info(f"FeatureEngineer initialized with {scaler_type} scaling")
    
    def fit_scaler(self, features_df: pd.DataFrame) -> None:
        """
        Fit the scaler to the feature data
        
        Args:
            features_df: DataFrame containing features to scale
        """
        if self.scaler_type == 'standard':
            self.scaler = StandardScaler()
        else:  # minmax
            self.scaler = MinMaxScaler()
            
        self.scaler.fit(features_df)
        logger.info(f"Scaler fitted on {features_df.shape[1]} features")
        
    def transform_features(self, features_df: pd.DataFrame) -> pd.DataFrame:
        """
        Scale features using the fitted scaler
        
        Args:
            features_df: Features to scale
            
        Returns:
            DataFrame with scaled features
        """
        if self.scaler is None:
            logger.warning("Scaler not fitted yet. Fitting now...")
            self.fit_scaler(features_df)
            
        scaled_data = self.scaler.transform(features_df)
        scaled_df = pd.DataFrame(
            scaled_data, 
            columns=features_df.columns,
            index=features_df.index
        )
        
        return scaled_df
    
    def compute_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Compute features from the labeled day data
        
        Args:
            data: DataFrame containing OHLCV data with date index
            
        Returns:
            DataFrame with computed features
        """
        if data.empty:
            logger.warning("Empty DataFrame provided for feature engineering")
            return pd.DataFrame()
        
        try:
            # Make a copy to avoid modifying the original
            df = data.copy()
            
            # Ensure date is datetime and set as index if not already
            if 'date' in df.columns:
                df['date'] = pd.to_datetime(df['date'])
                df.set_index('date', inplace=True)
            
            # Ensure we have required columns
            required_cols = ['open', 'high', 'low', 'close']
            missing_cols = [col for col in required_cols if col not in df.columns]
            if missing_cols:
                logger.error(f"Missing required columns: {missing_cols}")
                return pd.DataFrame()
            
            # Sort by date
            df = df.sort_index()
            
            # Feature 1: Gap % (open - prev_close)/prev_close
            df['prev_close'] = df['close'].shift(1)
            df['gap_pct'] = (df['open'] - df['prev_close']) / df['prev_close'] * 100
            
            # Feature 2: Opening range width (high-low in first 30 min)
            if 'first_30min_high' in df.columns and 'first_30min_low' in df.columns:
                df['opening_range_width'] = (df['first_30min_high'] - df['first_30min_low']) / df['open'] * 100
            else:
                # If we don't have specific columns for first 30 min, we'll use a proxy
                # Here we assume that the dataset has already aggregated this information or 
                # we need to provide a reasonable placeholder
                logger.warning("First 30 min high/low not available, using proxy")
                df['opening_range_width'] = (df['high'] - df['low']) / df['open'] * 100 * 0.4  # Rough approximation
            
            # Feature 3: Realized volatility (std of intraday returns)
            # Ideally, we would calculate this from minute data, but here we'll use a proxy from daily data
            # Using Garman-Klass volatility estimator which is better than simple high-low range
            df['log_hl'] = np.log(df['high'] / df['low'])**2
            df['log_co'] = np.log(df['close'] / df['open'])**2
            df['realized_vol'] = np.sqrt(0.5 * df['log_hl'] - (2 * np.log(2) - 1) * df['log_co']) * 100
            
            # Apply rolling window to smooth volatility
            df['realized_vol_5d'] = df['realized_vol'].rolling(window=5).mean()
            df['realized_vol_20d'] = df['realized_vol'].rolling(window=20).mean()
            
            # Feature 4: IV change % from previous day
            if 'implied_volatility' in df.columns:
                df['prev_iv'] = df['implied_volatility'].shift(1)
                df['iv_change_pct'] = (df['implied_volatility'] - df['prev_iv']) / df['prev_iv'] * 100
            else:
                logger.warning("Implied volatility data not available, skipping IV change feature")
            
            # Feature 5: Sector Strength Index (would require data for constituent banks)
            # For now, we'll use a proxy based on relative performance to broader market
            # In a real implementation, we would fetch constituent data and calculate this properly
            if 'market_index' in df.columns:
                df['sector_strength'] = (df['close'] / df['open']) / (df['market_index'] / df['market_index_open'])
            else:
                logger.warning("Market index data not available, skipping sector strength feature")
            
            # Additional features that might be useful
            
            # Price momentum features
            df['returns_1d'] = df['close'].pct_change(1) * 100
            df['returns_5d'] = df['close'].pct_change(5) * 100
            df['returns_10d'] = df['close'].pct_change(10) * 100
            
            # Volatility features
            df['range_pct'] = (df['high'] - df['low']) / df['open'] * 100
            df['range_pct_5d_avg'] = df['range_pct'].rolling(window=5).mean()
            
            # Volume-based features
            if 'volume' in df.columns:
                df['volume_change'] = df['volume'].pct_change(1) * 100
                df['volume_5d_avg'] = df['volume'].rolling(window=5).mean()
                df['volume_ratio'] = df['volume'] / df['volume_5d_avg']
            
            # Mean reversion features
            df['close_5d_ma'] = df['close'].rolling(window=5).mean()
            df['close_20d_ma'] = df['close'].rolling(window=20).mean()
            df['ma_ratio_5_20'] = df['close_5d_ma'] / df['close_20d_ma']
            df['distance_from_5d_ma'] = (df['close'] - df['close_5d_ma']) / df['close_5d_ma'] * 100
            
            # Day of week features (could be useful for certain patterns)
            df['day_of_week'] = df.index.dayofweek
            # One-hot encode day of week
            for i in range(5):  # 0-4 for Monday-Friday
                df[f'day_{i}'] = (df['day_of_week'] == i).astype(int)
            
            # Expiry week feature (helpful for options trading)
            if 'is_expiry_week' in df.columns:
                df['is_expiry_week'] = df['is_expiry_week'].astype(int)
            else:
                # As proxy, assuming last Thursday of month is expiry
                df['is_month_end_week'] = ((df.index.day > 22) & (df.index.dayofweek == 3)).astype(int)
            
            # Drop rows with NaN values (typically the first few rows due to lagged features)
            df = df.dropna()
            
            # Log feature creation results
            logger.info(f"Created {df.shape[1]} features from {df.shape[0]} days of data")
            
            return df
            
        except Exception as e:
            logger.error(f"Error during feature engineering: {e}")
            return pd.DataFrame()
    
    def prepare_ml_datasets(
        self, 
        labeled_data_path: str, 
        output_dir: str = "data/processed", 
        train_test_split: bool = True,
        test_size: float = 0.2,
        scale_features: bool = True
    ) -> Dict[str, pd.DataFrame]:
        """
        Prepare machine learning datasets from labeled data
        
        Args:
            labeled_data_path: Path to labeled data
            output_dir: Directory to save processed data
            train_test_split: Whether to split data into train and test sets
            test_size: Proportion of data to use for testing
            scale_features: Whether to scale features
        
        Returns:
            Dict containing X and y DataFrames
        """
        try:
            # Load labeled data
            df = pd.read_parquet(labeled_data_path)
            logger.info(f"Loaded labeled data from {labeled_data_path}: {df.shape}")
            
            # Compute features
            features_df = self.compute_features(df)
            if features_df.empty:
                logger.error("Failed to compute features")
                return {}
            
            # Extract labels
            if 'label' not in features_df.columns:
                logger.error("No 'label' column found in labeled data")
                return {}
            
            y = features_df['label']
            
            # Select feature columns (exclude label and any other non-feature columns)
            non_feature_cols = ['label', 'day_of_week']  # Add any other columns to exclude
            X = features_df.drop(columns=non_feature_cols, errors='ignore')
            
            # Scale features if requested
            if scale_features:
                X = self.transform_features(X)
            
            result = {'X': X, 'y': y}
            
            # Split into train and test sets if requested
            if train_test_split:
                from sklearn.model_selection import train_test_split as sk_split
                
                # Use time-based split (earlier data for training, later for testing)
                split_idx = int(X.shape[0] * (1 - test_size))
                
                X_train = X.iloc[:split_idx]
                X_test = X.iloc[split_idx:]
                y_train = y.iloc[:split_idx]
                y_test = y.iloc[split_idx:]
                
                result.update({
                    'X_train': X_train,
                    'X_test': X_test,
                    'y_train': y_train,
                    'y_test': y_test
                })
                
                logger.info(f"Split data into train ({X_train.shape[0]} samples) and test ({X_test.shape[0]} samples)")
            
            # Save processed data
            os.makedirs(output_dir, exist_ok=True)
            
            # Save full datasets
            joblib.dump(X, f"{output_dir}/features.pkl")
            joblib.dump(y, f"{output_dir}/labels.pkl")
            
            # Save train/test splits if created
            if train_test_split:
                joblib.dump(X_train, f"{output_dir}/X_train.pkl")
                joblib.dump(X_test, f"{output_dir}/X_test.pkl")
                joblib.dump(y_train, f"{output_dir}/y_train.pkl")
                joblib.dump(y_test, f"{output_dir}/y_test.pkl")
            
            logger.info(f"Saved processed data to {output_dir}")
            
            # Return the datasets
            return result
            
        except Exception as e:
            logger.error(f"Error preparing ML datasets: {e}")
            return {}
    
    def feature_importance(self, model, feature_names: List[str]) -> pd.DataFrame:
        """
        Extract feature importance from trained model
        
        Args:
            model: Trained model with feature_importances_ attribute
            feature_names: List of feature names
            
        Returns:
            DataFrame with feature names and importance scores
        """
        try:
            # Check if model has feature importances
            if not hasattr(model, 'feature_importances_'):
                logger.warning("Model does not have feature_importances_ attribute")
                return pd.DataFrame()
            
            # Get feature importances
            importances = model.feature_importances_
            
            # Create DataFrame
            importance_df = pd.DataFrame({
                'feature': feature_names,
                'importance': importances
            })
            
            # Sort by importance
            importance_df = importance_df.sort_values('importance', ascending=False)
            
            return importance_df
            
        except Exception as e:
            logger.error(f"Error extracting feature importance: {e}")
            return pd.DataFrame()


# For testing purposes
if __name__ == "__main__":
    # Configure logging for testing
    logging.basicConfig(level=logging.INFO)
    
    # Create feature engineer
    engineer = FeatureEngineer()
    
    # Test with sample data
    try:
        # Try to load actual labeled data
        labeled_data_path = "data/processed/labeled_days.parquet"
        if os.path.exists(labeled_data_path):
            print(f"Testing with actual labeled data from {labeled_data_path}")
            datasets = engineer.prepare_ml_datasets(labeled_data_path)
            
            if datasets:
                print(f"Features shape: {datasets['X'].shape}")
                print(f"Labels shape: {datasets['y'].shape}")
                print("\nFeature columns:")
                print(datasets['X'].columns.tolist())
                print("\nSample features:")
                print(datasets['X'].head())
                print("\nLabel distribution:")
                print(datasets['y'].value_counts())
                
        else:
            # Create sample data for testing
            print("No labeled data found. Creating sample data for testing...")
            sample_data = pd.DataFrame({
                'date': pd.date_range(start='2025-01-01', periods=100),
                'open': np.random.normal(48000, 500, 100),
                'high': np.random.normal(48200, 600, 100),
                'low': np.random.normal(47800, 600, 100),
                'close': np.random.normal(48100, 500, 100),
                'volume': np.random.normal(1000000, 200000, 100),
                'label': np.random.choice(['Trend', 'RangeBound', 'Event', 'MildBias', 'Momentum'], 100)
            })
            
            # Add first 30 min data
            sample_data['first_30min_high'] = sample_data['high'] * 0.9
            sample_data['first_30min_low'] = sample_data['low'] * 1.1
            
            # Compute features
            features_df = engineer.compute_features(sample_data)
            print(f"Features shape: {features_df.shape}")
            print("Sample features:")
            print(features_df.head())
            
    except Exception as e:
        print(f"Test failed: {e}")
