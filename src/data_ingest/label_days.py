# src/data_ingest/label_days.py
"""
Module for labeling trading days based on various market characteristics.
This module classifies each trading day into specific regime types.
"""

import os
import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta

# Configure logging
logger = logging.getLogger(__name__)

class DayLabeler:
    """Labels trading days based on price action and volatility characteristics"""
    
    def __init__(self):
        """Initialize the day labeler"""
        logger.info("Day labeler initialized")
        self.labels = {
            "Trend": "Strong directional move with sustained momentum",
            "RangeBound": "Price oscillates within a well-defined range",
            "Event": "High volatility with sharp price movements (earnings, news)",
            "MildBias": "Slight directional bias but with limited follow-through",
            "Momentum": "Strong momentum in one direction with occasional pullbacks"
        }
    
    def load_data(self, banknifty_path: str, options_path: Optional[str] = None) -> Tuple[pd.DataFrame, Optional[pd.DataFrame]]:
        """
        Load Bank Nifty and options data
        
        Args:
            banknifty_path: Path to Bank Nifty historical data
            options_path: Path to options data (optional)
            
        Returns:
            Tuple of (banknifty_df, options_df)
        """
        try:
            # Load Bank Nifty data
            if banknifty_path.endswith('.parquet'):
                banknifty_df = pd.read_parquet(banknifty_path)
            else:
                banknifty_df = pd.read_csv(banknifty_path)
            
            logger.info(f"Loaded Bank Nifty data: {len(banknifty_df)} rows")
            
            # Load options data if provided
            options_df = None
            if options_path:
                if options_path.endswith('.parquet'):
                    options_df = pd.read_parquet(options_path)
                else:
                    options_df = pd.read_csv(options_path)
                logger.info(f"Loaded options data: {len(options_df)} rows")
            
            return banknifty_df, options_df
            
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            return pd.DataFrame(), None
    
    def aggregate_to_daily(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Aggregate intraday data to daily OHLCV data
        
        Args:
            df: DataFrame with minute-level data
            
        Returns:
            DataFrame with daily aggregated data
        """
        if df.empty:
            return pd.DataFrame()
        
        try:
            # Ensure datetime format
            if 'date' in df.columns:
                df['date'] = pd.to_datetime(df['date'])
            
            # Extract date component for aggregation
            df['day'] = df['date'].dt.date
            
            # Aggregate by day
            daily = df.groupby('day').agg({
                'open': 'first',
                'high': 'max',
                'low': 'min',
                'close': 'last',
                'volume': 'sum'
            }).reset_index()
            
            # Rename day column back to date
            daily = daily.rename(columns={'day': 'date'})
            
            # Ensure date is datetime
            daily['date'] = pd.to_datetime(daily['date'])
            
            logger.info(f"Aggregated to {len(daily)} daily records")
            return daily
            
        except Exception as e:
            logger.error(f"Error aggregating to daily: {e}")
            return pd.DataFrame()
    
    def calculate_features(self, 
                          banknifty_df: pd.DataFrame, 
                          options_df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """
        Calculate features for day classification
        
        Args:
            banknifty_df: Bank Nifty historical data
            options_df: Options data (optional)
            
        Returns:
            DataFrame with calculated features
        """
        if banknifty_df.empty:
            logger.warning("Empty DataFrame provided")
            return pd.DataFrame()
        
        try:
            # Check if data is minute-level or daily
            is_intraday = 'day' not in banknifty_df.columns and len(banknifty_df) > 0 and 'date' in banknifty_df.columns
            
            # Aggregate to daily if minute-level
            if is_intraday and len(banknifty_df['date'].dt.date.unique()) > 1:
                daily_df = self.aggregate_to_daily(banknifty_df)
            else:
                daily_df = banknifty_df.copy()
            
            # Sort by date
            daily_df = daily_df.sort_values('date')
            
            # Calculate daily returns
            daily_df['prev_close'] = daily_df['close'].shift(1)
            daily_df['open_to_close_return'] = (daily_df['close'] - daily_df['open']) / daily_df['open'] * 100
            daily_df['close_to_close_return'] = (daily_df['close'] - daily_df['prev_close']) / daily_df['prev_close'] * 100
            daily_df['gap'] = (daily_df['open'] - daily_df['prev_close']) / daily_df['prev_close'] * 100
            daily_df['high_low_range'] = (daily_df['high'] - daily_df['low']) / daily_df['open'] * 100
            
            # Calculate volatility (20-day rolling standard deviation of returns)
            daily_df['volatility_20d'] = daily_df['close_to_close_return'].rolling(20).std()
            
            # Calculate additional features from intraday data if available
            if is_intraday:
                # Get first 30 minutes of each day
                first_30min_data = []
                for date in daily_df['date']:
                    day_data = banknifty_df[(banknifty_df['date'].dt.date == date.date())]
                    first_30min = day_data.iloc[:30] if len(day_data) >= 30 else day_data
                    
                    if not first_30min.empty:
                        first_30min_range = (first_30min['high'].max() - first_30min['low'].min()) / first_30min['open'].iloc[0] * 100
                        first_30min_data.append({
                            'date': date,
                            'first_30min_range': first_30min_range
                        })
                
                if first_30min_data:
                    first_30min_df = pd.DataFrame(first_30min_data)
                    daily_df = daily_df.merge(first_30min_df, on='date', how='left')
            
            # Calculate VWAP if volume is available
            if 'volume' in banknifty_df.columns:
                banknifty_df['vwap'] = (banknifty_df['close'] * banknifty_df['volume']).cumsum() / banknifty_df['volume'].cumsum()
                
                # Map VWAP to daily data
                if is_intraday:
                    vwap_data = []
                    for date in daily_df['date']:
                        day_data = banknifty_df[(banknifty_df['date'].dt.date == date.date())]
                        if not day_data.empty:
                            last_vwap = day_data['vwap'].iloc[-1]
                            close_vs_vwap = (day_data['close'].iloc[-1] - last_vwap) / last_vwap * 100
                            vwap_data.append({
                                'date': date,
                                'close_vs_vwap': close_vs_vwap
                            })
                    
                    if vwap_data:
                        vwap_df = pd.DataFrame(vwap_data)
                        daily_df = daily_df.merge(vwap_df, on='date', how='left')
            
            # Add implied volatility feature if options data is available
            if options_df is not None and not options_df.empty:
                try:
                    # Ensure date format is consistent
                    if 'date' not in options_df.columns:
                        # Try to extract date from the options data
                        if 'timestamp' in options_df.columns:
                            options_df['date'] = pd.to_datetime(options_df['timestamp']).dt.date
                        elif 'expiry_date' in options_df.columns:
                            # Use a reference date like the snapshot date
                            if 'snapshot_date' in options_df.columns:
                                options_df['date'] = pd.to_datetime(options_df['snapshot_date'])
                    
                    # If we have implied volatility in the options data, use it
                    if 'implied_volatility' in options_df.columns:
                        iv_data = []
                        for date in daily_df['date']:
                            day_options = options_df[options_df['date'].dt.date == date.date()]
                            if not day_options.empty:
                                avg_iv = day_options['implied_volatility'].mean()
                                iv_data.append({
                                    'date': date,
                                    'avg_iv': avg_iv
                                })
                        
                        if iv_data:
                            iv_df = pd.DataFrame(iv_data)
                            daily_df = daily_df.merge(iv_df, on='date', how='left')
                            
                            # Calculate IV percentile (20-day lookback)
                            daily_df['iv_percentile'] = daily_df['avg_iv'].rolling(20).apply(
                                lambda x: percentile_rank(x.iloc[-1], x[:-1])
                            )
                except Exception as e:
                    logger.error(f"Error processing options data: {e}")
            
            logger.info(f"Calculated features for {len(daily_df)} days")
            return daily_df
            
        except Exception as e:
            logger.error(f"Error calculating features: {e}")
            return pd.DataFrame()
    
    def classify_days(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Classify trading days into different regimes
        
        Args:
            df: DataFrame with calculated features
            
        Returns:
            DataFrame with day classifications
        """
        if df.empty:
            return pd.DataFrame()
        
        try:
            # Make a copy to avoid modifying the original
            result = df.copy()
            
            # Initialize classification column
            result['day_type'] = None
            
            # Define thresholds
            trend_return_threshold = 1.0  # 1% move open to close
            range_threshold = 0.5  # 0.5% open to close for range-bound
            high_vol_threshold = 2.0  # 2% high to low range for high volatility
            
            # Apply classification rules
            
            # 1. Trend days - strong directional move
            trend_mask = (abs(result['open_to_close_return']) > trend_return_threshold) & \
                         (result['high_low_range'] > 1.0) & \
                         (
                            # Open near low/high and close near high/low
                            ((result['open_to_close_return'] > 0) & 
                             (result['open'] - result['low']) / (result['high'] - result['low']) < 0.3 & 
                             (result['high'] - result['close']) / (result['high'] - result['low']) < 0.3) | 
                            ((result['open_to_close_return'] < 0) & 
                             (result['high'] - result['open']) / (result['high'] - result['low']) < 0.3 & 
                             (result['close'] - result['low']) / (result['high'] - result['low']) < 0.3)
                         )
            result.loc[trend_mask, 'day_type'] = 'Trend'
            
            # 2. Range-bound days - small open to close range
            range_mask = (abs(result['open_to_close_return']) < range_threshold) & \
                         (result['high_low_range'] > 0.8) & \
                         (result['day_type'].isnull())
            result.loc[range_mask, 'day_type'] = 'RangeBound'
            
            # 3. Event days - high volatility, significant price swings
            event_mask = (result['high_low_range'] > high_vol_threshold) & \
                         (result['day_type'].isnull())
            result.loc[event_mask, 'day_type'] = 'Event'
            
            # 4. Mild Bias - slight directional move
            mild_bias_mask = (abs(result['open_to_close_return']) > range_threshold) & \
                            (abs(result['open_to_close_return']) < trend_return_threshold) & \
                            (result['day_type'].isnull())
            result.loc[mild_bias_mask, 'day_type'] = 'MildBias'
            
            # 5. Momentum - significant directional move but with pullbacks
            momentum_mask = (abs(result['open_to_close_return']) > trend_return_threshold * 0.7) & \
                           (result['day_type'].isnull())
            result.loc[momentum_mask, 'day_type'] = 'Momentum'
            
            # Any remaining unclassified days are labeled as MildBias
            result.loc[result['day_type'].isnull(), 'day_type'] = 'MildBias'
            
            # Calculate the distribution of day types
            type_counts = result['day_type'].value_counts()
            total_days = len(result)
            
            for day_type, count in type_counts.items():
                percentage = count / total_days * 100
                logger.info(f"{day_type}: {count} days ({percentage:.1f}%)")
            
            return result
            
        except Exception as e:
            logger.error(f"Error classifying days: {e}")
            return df
    
    def run_labeling_pipeline(self, 
                           banknifty_path: str = "data/processed/banknifty.parquet", 
                           options_path: str = "data/processed/options.parquet") -> pd.DataFrame:
        """
        Run the complete day labeling pipeline
        
        Args:
            banknifty_path: Path to Bank Nifty data
            options_path: Path to options data
            
        Returns:
            DataFrame with labeled days
        """
        try:
            # 1. Load data
            banknifty_df, options_df = self.load_data(banknifty_path, options_path)
            if banknifty_df.empty:
                logger.error("Failed to load Bank Nifty data")
                return pd.DataFrame()
            
            # 2. Calculate features
            features_df = self.calculate_features(banknifty_df, options_df)
            if features_df.empty:
                logger.error("Failed to calculate features")
                return pd.DataFrame()
            
            # 3. Classify days
            labeled_df = self.classify_days(features_df)
            if labeled_df.empty:
                logger.error("Failed to classify days")
                return pd.DataFrame()
            
            # 4. Save results
            output_path = "data/processed/labeled_days.parquet"
            labeled_df.to_parquet(output_path, index=False)
            logger.info(f"Saved labeled days to {output_path}")
            
            return labeled_df
            
        except Exception as e:
            logger.error(f"Error in labeling pipeline: {e}")
            return pd.DataFrame()


def percentile_rank(value: float, array: np.ndarray) -> float:
    """
    Calculate the percentile rank of a value within an array
    
    Args:
        value: Value to find rank for
        array: Array of values
        
    Returns:
        Percentile rank (0-100)
    """
    if len(array) == 0:
        return 50.0
    
    # Count values less than our value
    less_than_count = sum(1 for x in array if x < value)
    return less_than_count / len(array) * 100


# For testing purposes
if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    
    labeler = DayLabeler()
    
    # Check if the processed files exist
    banknifty_path = "data/processed/banknifty.parquet"
    options_path = "data/processed/options.parquet"
    
    if not os.path.exists(banknifty_path):
        logger.warning(f"Bank Nifty data not found at {banknifty_path}")
        
        # Use sample data for testing
        sample_data = pd.DataFrame({
            'date': pd.date_range(start='2025-01-01', periods=100),
            'open': np.random.normal(48000, 500, 100),
            'high': np.random.normal(48200, 600, 100),
            'low': np.random.normal(47800, 600, 100),
            'close': np.random.normal(48100, 550, 100),
            'volume': np.random.randint(800000, 1500000, 100)
        })
        
        # Ensure high >= open, close, low
        for i in range(len(sample_data)):
            values = [sample_data.loc[i, 'open'], sample_data.loc[i, 'close'], sample_data.loc[i, 'low']]
            sample_data.loc[i, 'high'] = max([sample_data.loc[i, 'high']] + values)
        
        # Ensure low <= open, close, high
        for i in range(len(sample_data)):
            values = [sample_data.loc[i, 'open'], sample_data.loc[i, 'close'], sample_data.loc[i, 'high']]
            sample_data.loc[i, 'low'] = min([sample_data.loc[i, 'low']] + values)
        
        # Create the directory if it doesn't exist
        os.makedirs("data/processed", exist_ok=True)
        
        # Save sample data
        sample_data.to_parquet(banknifty_path, index=False)
        logger.info(f"Created sample data at {banknifty_path}")
    
    # Run the labeling pipeline
    labeled_df = labeler.run_labeling_pipeline(banknifty_path, options_path if os.path.exists(options_path) else None)
    
    if not labeled_df.empty:
        print("\nDay Type Distribution:")
        print(labeled_df['day_type'].value_counts())
        
        print("\nSample of labeled days:")
        print(labeled_df[['date', 'open', 'high', 'low', 'close', 'open_to_close_return', 'high_low_range', 'day_type']].head())
