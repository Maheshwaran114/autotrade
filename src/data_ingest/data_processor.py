#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Data preprocessing module for Bank Nifty Options trading system.
This module is responsible for cleaning, transforming, and preparing data for feature engineering.
"""

import os
import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Union, Tuple
from datetime import datetime, timedelta

# Configure logging
logger = logging.getLogger(__name__)


class DataProcessor:
    """Processes raw financial data for use in machine learning models"""

    def __init__(self):
        """Initialize the data processor."""
        logger.info("DataProcessor initialized")
    
    def process_historical_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Process raw historical price data
        
        Args:
            data: DataFrame containing raw price data
            
        Returns:
            Processed DataFrame ready for feature engineering
        """
        if data.empty:
            logger.warning("Empty DataFrame provided for processing")
            return pd.DataFrame()
        
        try:
            # Make a copy to avoid modifying the original
            df = data.copy()
            
            # Convert date/time column to datetime if not already
            if 'date' in df.columns:
                df['date'] = pd.to_datetime(df['date'])
            
            # Ensure OHLC data is numeric
            for col in ['open', 'high', 'low', 'close']:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # Handle missing values
            df = self._handle_missing_values(df)
            
            # Remove outliers
            df = self._remove_outliers(df)
            
            # Add basic time features
            df = self._add_time_features(df)
            
            # Sort by date
            if 'date' in df.columns:
                df = df.sort_values('date')
            
            logger.info(f"Successfully processed data: {df.shape[0]} rows, {df.shape[1]} columns")
            return df
            
        except Exception as e:
            logger.error(f"Error during data processing: {e}")
            return data
    
    def _handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Handle missing values in the dataset
        
        Args:
            df: DataFrame to process
            
        Returns:
            DataFrame with missing values handled
        """
        # Count missing values
        missing_count = df.isnull().sum()
        
        if missing_count.sum() > 0:
            logger.info(f"Found {missing_count.sum()} missing values")
            
            # Forward fill for time-series data (use previous value)
            df = df.ffill()
            
            # If any missing values remain, use backward fill
            df = df.bfill()
            
            # If still having missing numeric values, replace with column mean
            for col in df.select_dtypes(include=[np.number]).columns:
                if df[col].isnull().any():
                    mean_val = df[col].mean()
                    df[col] = df[col].fillna(mean_val)
        
        return df
    
    def _remove_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Remove or cap outliers in OHLC data
        
        Args:
            df: DataFrame to process
            
        Returns:
            DataFrame with outliers handled
        """
        # Only process common OHLC columns
        ohlc_cols = [col for col in ['open', 'high', 'low', 'close'] if col in df.columns]
        
        if not ohlc_cols:
            return df
        
        # Use IQR method to detect and handle outliers
        for col in ohlc_cols:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            
            # Define bounds
            lower_bound = Q1 - 3 * IQR
            upper_bound = Q3 + 3 * IQR
            
            # Count outliers
            outliers_count = ((df[col] < lower_bound) | (df[col] > upper_bound)).sum()
            
            if outliers_count > 0:
                logger.info(f"Found {outliers_count} outliers in {col}")
                
                # Cap the outliers rather than removing rows
                df[col] = np.where(df[col] < lower_bound, lower_bound, df[col])
                df[col] = np.where(df[col] > upper_bound, upper_bound, df[col])
        
        return df
    
    def _add_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add basic time features to the dataset
        
        Args:
            df: DataFrame to process
            
        Returns:
            DataFrame with additional time features
        """
        if 'date' not in df.columns:
            return df
        
        # Ensure date is datetime
        df['date'] = pd.to_datetime(df['date'])
        
        # Extract time components
        df['day_of_week'] = df['date'].dt.dayofweek
        df['day_of_month'] = df['date'].dt.day
        df['month'] = df['date'].dt.month
        df['year'] = df['date'].dt.year
        
        # Add is_month_start and is_month_end
        df['is_month_start'] = df['date'].dt.is_month_start.astype(int)
        df['is_month_end'] = df['date'].dt.is_month_end.astype(int)
        
        # Add quarter
        df['quarter'] = df['date'].dt.quarter
        
        return df
    
    def process_option_chain(self, 
                           option_chain: Dict, 
                           add_greeks: bool = False,
                           risk_free_rate: float = 0.05) -> pd.DataFrame:
        """
        Process option chain data into a structured DataFrame
        
        Args:
            option_chain: Dict containing option chain data
            add_greeks: Whether to calculate option Greeks
            risk_free_rate: Risk-free rate for options calculations
            
        Returns:
            Processed DataFrame with options data
        """
        try:
            # Extract key information
            spot_price = option_chain.get('spot_price')
            expiry_date_str = option_chain.get('expiry_date')
            
            if not spot_price or not expiry_date_str:
                logger.error("Missing spot price or expiry date in option chain")
                return pd.DataFrame()
            
            # Convert expiry to datetime
            try:
                expiry_date = pd.to_datetime(expiry_date_str)
            except:
                logger.error(f"Could not parse expiry date: {expiry_date_str}")
                return pd.DataFrame()
            
            # Extract calls and puts
            calls = option_chain.get('options', {}).get('calls', [])
            puts = option_chain.get('options', {}).get('puts', [])
            
            # Process calls
            calls_df = pd.DataFrame(calls)
            if not calls_df.empty:
                calls_df['option_type'] = 'CE'
            
            # Process puts
            puts_df = pd.DataFrame(puts)
            if not puts_df.empty:
                puts_df['option_type'] = 'PE'
            
            # Combine
            df = pd.concat([calls_df, puts_df], ignore_index=True)
            
            if df.empty:
                logger.warning("No options data found in option chain")
                return pd.DataFrame()
            
            # Add spot price and time to expiry
            df['spot_price'] = spot_price
            
            # Calculate days to expiry
            current_date = datetime.now()
            if 'expiry' in df.columns:
                # Convert to datetime if it's not already
                df['expiry'] = pd.to_datetime(df['expiry'])
                df['days_to_expiry'] = (df['expiry'] - current_date).dt.days
            else:
                # Use the provided expiry date
                days_to_expiry = (expiry_date - current_date).days
                df['days_to_expiry'] = days_to_expiry
                df['expiry'] = expiry_date
            
            # Calculate moneyness
            if 'strike' in df.columns:
                df['moneyness'] = (df['strike'] - spot_price) / spot_price
            
            # Add options Greeks if requested
            if add_greeks:
                df = self._calculate_greeks(df, risk_free_rate)
            
            logger.info(f"Successfully processed option chain with {len(df)} options")
            return df
            
        except Exception as e:
            logger.error(f"Error processing option chain: {e}")
            return pd.DataFrame()
    
    def _calculate_greeks(self, df: pd.DataFrame, risk_free_rate: float = 0.05) -> pd.DataFrame:
        """
        Calculate option Greeks (simplified versions)
        
        Args:
            df: DataFrame with option data
            risk_free_rate: Risk-free interest rate
            
        Returns:
            DataFrame with added Greeks columns
        """
        try:
            from scipy.stats import norm
            
            # We need these columns for calculation
            required_cols = ['spot_price', 'strike', 'days_to_expiry']
            if not all(col in df.columns for col in required_cols):
                logger.error("Missing required columns for Greeks calculation")
                return df
            
            # Time to expiry in years
            df['T'] = df['days_to_expiry'] / 365.0
            
            # Volatility (using historical or implied if available, otherwise a placeholder)
            if 'implied_volatility' in df.columns:
                df['sigma'] = df['implied_volatility']
            else:
                # Using a simplistic approximation based on moneyness
                # In reality, this should be calculated from market data
                df['sigma'] = 0.20 + 0.05 * abs(df['moneyness'])
            
            # Calculate d1 and d2 for Black-Scholes
            df['d1'] = (np.log(df['spot_price'] / df['strike']) + 
                      (risk_free_rate + 0.5 * df['sigma']**2) * df['T']) / (df['sigma'] * np.sqrt(df['T']))
            df['d2'] = df['d1'] - df['sigma'] * np.sqrt(df['T'])
            
            # Normal distribution functions
            df['N_d1'] = norm.cdf(df['d1'])
            df['N_d2'] = norm.cdf(df['d2'])
            df['n_d1'] = norm.pdf(df['d1'])  # Density function
            
            # Calculate Greeks for calls
            calls = df[df['option_type'] == 'CE']
            if not calls.empty:
                # Delta - rate of change of option price with respect to underlying price
                df.loc[df['option_type'] == 'CE', 'delta'] = df.loc[df['option_type'] == 'CE', 'N_d1']
                
                # Gamma - rate of change of delta with respect to underlying price
                df.loc[df['option_type'] == 'CE', 'gamma'] = df.loc[df['option_type'] == 'CE', 'n_d1'] / (
                    df.loc[df['option_type'] == 'CE', 'spot_price'] * df.loc[df['option_type'] == 'CE', 'sigma'] * 
                    np.sqrt(df.loc[df['option_type'] == 'CE', 'T'])
                )
                
                # Theta - rate of change of option price with respect to time
                df.loc[df['option_type'] == 'CE', 'theta'] = -(
                    df.loc[df['option_type'] == 'CE', 'spot_price'] * df.loc[df['option_type'] == 'CE', 'sigma'] * 
                    df.loc[df['option_type'] == 'CE', 'n_d1']) / (2 * np.sqrt(df.loc[df['option_type'] == 'CE', 'T'])) - 
                    risk_free_rate * df.loc[df['option_type'] == 'CE', 'strike'] * 
                    np.exp(-risk_free_rate * df.loc[df['option_type'] == 'CE', 'T']) * 
                    df.loc[df['option_type'] == 'CE', 'N_d2']
                ) / 365.0  # Convert to daily
                
                # Vega - rate of change of option price with respect to volatility
                df.loc[df['option_type'] == 'CE', 'vega'] = (
                    df.loc[df['option_type'] == 'CE', 'spot_price'] * 
                    np.sqrt(df.loc[df['option_type'] == 'CE', 'T']) * 
                    df.loc[df['option_type'] == 'CE', 'n_d1']
                ) / 100  # Scale to represent change per 1% change in vol
            
            # Calculate Greeks for puts
            puts = df[df['option_type'] == 'PE']
            if not puts.empty:
                # Delta for puts
                df.loc[df['option_type'] == 'PE', 'delta'] = df.loc[df['option_type'] == 'PE', 'N_d1'] - 1
                
                # Gamma for puts (same formula as calls)
                df.loc[df['option_type'] == 'PE', 'gamma'] = df.loc[df['option_type'] == 'PE', 'n_d1'] / (
                    df.loc[df['option_type'] == 'PE', 'spot_price'] * df.loc[df['option_type'] == 'PE', 'sigma'] * 
                    np.sqrt(df.loc[df['option_type'] == 'PE', 'T'])
                )
                
                # Theta for puts
                df.loc[df['option_type'] == 'PE', 'theta'] = -(
                    df.loc[df['option_type'] == 'PE', 'spot_price'] * df.loc[df['option_type'] == 'PE', 'sigma'] * 
                    df.loc[df['option_type'] == 'PE', 'n_d1']) / (2 * np.sqrt(df.loc[df['option_type'] == 'PE', 'T'])) + 
                    risk_free_rate * df.loc[df['option_type'] == 'PE', 'strike'] * 
                    np.exp(-risk_free_rate * df.loc[df['option_type'] == 'PE', 'T']) * 
                    (1 - df.loc[df['option_type'] == 'PE', 'N_d2'])
                ) / 365.0  # Convert to daily
                
                # Vega for puts (same formula as calls)
                df.loc[df['option_type'] == 'PE', 'vega'] = (
                    df.loc[df['option_type'] == 'PE', 'spot_price'] * 
                    np.sqrt(df.loc[df['option_type'] == 'PE', 'T']) * 
                    df.loc[df['option_type'] == 'PE', 'n_d1']
                ) / 100  # Scale to represent change per 1% change in vol
            
            # Clean up intermediate calculation columns
            df = df.drop(columns=['d1', 'd2', 'N_d1', 'N_d2', 'n_d1'], errors='ignore')
            
            logger.info("Successfully calculated option Greeks")
            return df
            
        except Exception as e:
            logger.error(f"Error calculating Greeks: {e}")
            return df
    
    def save_processed_data(self, data: pd.DataFrame, filename: str) -> bool:
        """
        Save processed data to CSV in the processed directory
        
        Args:
            data: DataFrame to save
            filename: Name of the output file
            
        Returns:
            bool: True if successfully saved
        """
        try:
            # Create directory if it doesn't exist
            os.makedirs("data/processed", exist_ok=True)
            
            # Save to CSV
            filepath = f"data/processed/{filename}"
            data.to_csv(filepath, index=False)
            logger.info(f"Processed data successfully saved to {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save processed data: {e}")
            return False


# For testing purposes
if __name__ == "__main__":
    # Configure logging for testing
    logging.basicConfig(level=logging.INFO)
    
    processor = DataProcessor()
    
    # Create some sample data for testing
    sample_data = pd.DataFrame({
        'date': pd.date_range(start='2025-01-01', periods=10),
        'open': [48000, 48100, 48050, 47900, 48200, 48500, 48400, 48300, 48100, 48000],
        'high': [48200, 48300, 48200, 48100, 48500, 48700, 48600, 48500, 48300, 48200],
        'low': [47900, 47950, 47900, 47800, 48050, 48300, 48200, 48100, 47950, 47900],
        'close': [48100, 48050, 47950, 48100, 48450, 48400, 48300, 48150, 48000, 48100],
        'volume': [1000000, 1200000, 950000, 1100000, 1300000, 1500000, 1200000, 1100000, 950000, 1050000]
    })
    
    # Process the sample data
    processed_data = processor.process_historical_data(sample_data)
    
    # Show sample of processed data
    print("Sample of processed data:")
    print(processed_data.head())
    
    # Check added time features
    print("\nAdded time features:")
    time_columns = ['day_of_week', 'day_of_month', 'month', 'year', 'is_month_start', 'is_month_end', 'quarter']
    print(processed_data[['date'] + time_columns].head())
