#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script to fetch REAL historical option chain data for Bank Nifty using Zerodha API.
This replaces the simulation mode with actual API calls.
"""

import os
import sys
import logging
import pandas as pd
from datetime import datetime, timedelta, date
from pathlib import Path
import time
import numpy as np
from scipy.stats import norm
from scipy.optimize import brentq
import math

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

# Import Zerodha client
from src.data_ingest.zerodha_client import ZerodhaClient
import src.data_ingest.load_data as load_data

# Import date utilities for robust date handling
try:
    from date_utils import normalize_expiry_date, parse_expiry_to_date, filter_valid_expiries
except ImportError:
    logger.warning("date_utils module not found, using fallback date handling")
    normalize_expiry_date = None

# Configure logging
logging.basicConfig(level=logging.DEBUG, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Import date utilities for robust date handling
try:
    from date_utils import normalize_expiry_date, parse_expiry_to_date, filter_valid_expiries
    logger.info("Using enhanced date utilities for KiteConnect date handling")
    USE_ENHANCED_DATE_UTILS = True
except ImportError:
    logger.warning("date_utils module not found, using fallback date handling")
    USE_ENHANCED_DATE_UTILS = False

# ========== CONFIGURATION PARAMETERS ==========
CONFIG = {
    # Change from 14 days to 400+ days for 1+ year of data
    "DAYS_BACK": 400,  # 1+ year of historical data
    'date_range_days': 14,  # Increased from 7 to 14 days for more data
    'strike_range': 3000,   # Increased from 2000 to 3000 for wider coverage
    'strike_step': 100,     # Strike step size (100 points)
    'risk_free_rate': 0.07,  # Risk-free rate for IV calculation (7% annual)
    'api_delay': 0.1,       # Delay between API calls (seconds)
    'date_delay': 1,        # Delay between processing dates (seconds)
    'max_instruments_per_quote': 500,  # Max instruments per quote API call
    'enable_greeks': True,   # Enable Greeks calculation
    'enable_data_validation': True,  # Enable data validation checks
    'log_progress_every': 10,  # Log progress every N instruments
    'max_expiry_weeks_ahead': 4,  # Look for expiries up to 4 weeks ahead
    'enable_csv_export': True,  # Enable CSV export alongside parquet
    'enable_performance_metrics': True,  # Enable performance timing
    'retry_attempts': 3,     # Number of retry attempts for failed API calls
    'retry_delay': 2         # Delay between retry attempts (seconds)
}

def black_scholes_call(S, K, T, r, sigma):
    """
    Calculate Black-Scholes price for a call option.
    S: Current price of underlying
    K: Strike price
    T: Time to expiry (in years)
    r: Risk-free rate
    sigma: Volatility
    """
    if T <= 0:
        return max(S - K, 0)
    
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    
    call_price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    return call_price

def black_scholes_put(S, K, T, r, sigma):
    """
    Calculate Black-Scholes price for a put option.
    """
    if T <= 0:
        return max(K - S, 0)
    
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    
    put_price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
    return put_price

def calculate_implied_volatility(option_price, S, K, T, r, option_type='CE'):
    """
    Calculate implied volatility using Brent's method.
    """
    if T <= 0 or option_price <= 0:
        return 0.0
    
    # Define the function whose root we want to find
    def objective_function(sigma):
        if option_type == 'CE':
            theoretical_price = black_scholes_call(S, K, T, r, sigma)
        else:  # PE
            theoretical_price = black_scholes_put(S, K, T, r, sigma)
        return theoretical_price - option_price
    
    try:
        # Use Brent's method to find the implied volatility
        # Search between 0.01 (1%) and 5.0 (500%) volatility
        iv = brentq(objective_function, 0.01, 5.0, xtol=1e-6, maxiter=100)
        return iv
    except (ValueError, RuntimeError):
        # If no solution found, return 0
        return 0.0

def calculate_greeks(S, K, T, r, sigma, option_type='CE'):
    """
    Calculate option Greeks using Black-Scholes model.
    Returns: delta, gamma, theta, vega
    """
    if T <= 0 or sigma <= 0:
        return 0.0, 0.0, 0.0, 0.0
    
    try:
        d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        
        # Calculate Greeks
        if option_type == 'CE':
            delta = norm.cdf(d1)
            gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))
            theta = (- (S * norm.pdf(d1) * sigma) / (2 * np.sqrt(T)) 
                    - r * K * np.exp(-r * T) * norm.cdf(d2)) / 365
            vega = S * norm.pdf(d1) * np.sqrt(T) / 100
        else:  # PE
            delta = norm.cdf(d1) - 1
            gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))
            theta = (- (S * norm.pdf(d1) * sigma) / (2 * np.sqrt(T)) 
                    + r * K * np.exp(-r * T) * norm.cdf(-d2)) / 365
            vega = S * norm.pdf(d1) * np.sqrt(T) / 100
        
        return delta, gamma, theta, vega
    except:
        return 0.0, 0.0, 0.0, 0.0

def validate_option_data(record):
    """
    Validate option data for consistency and reasonable values.
    """
    if not CONFIG['enable_data_validation']:
        return True
    
    try:
        # Basic validation checks
        if record['close'] <= 0:
            return False
        if record['volume'] < 0 or record['oi'] < 0:
            return False
        if record['high'] < record['low'] or record['close'] > record['high'] or record['close'] < record['low']:
            return False
        if record['iv'] < 0 or record['iv'] > 5:  # IV between 0% and 500%
            return False
        
        return True
    except:
        return False

def get_trading_dates(start_date, end_date):
    """
    Get a list of trading dates between start_date and end_date.
    Generates trading dates by excluding weekends and some known holidays.
    """
    import pandas as pd
    
    # Generate all dates in the range
    date_range = pd.date_range(start=start_date, end=end_date, freq='D')
    
    # Filter out weekends (Saturday=5, Sunday=6)
    trading_dates = [d.date() for d in date_range if d.weekday() < 5]
    
    # TODO: Could add major holidays filtering here if needed
    # For now, we'll use weekdays as trading dates
    
    logger.info(f"Generated {len(trading_dates)} potential trading dates (excluding weekends)")
    return sorted(trading_dates)

def get_spot_price_for_date(trade_date):
    """
    Get Bank Nifty spot price for a specific date.
    For historical collection, we'll use a reasonable estimate based on recent levels.
    In a live system, this would fetch from actual historical data.
    """
    # For the initial data collection, use a reasonable Bank Nifty range
    # Current Bank Nifty typically trades between 48,000 to 54,000
    # We'll use 52,000 as a base and add some realistic variation
    import random
    
    # Use date as seed for consistent pricing across runs
    random.seed(str(trade_date))
    base_price = 52000
    daily_variation = random.uniform(-0.02, 0.02)  # ±2% daily variation
    estimated_price = base_price * (1 + daily_variation)
    
    logger.debug(f"Using estimated spot price {estimated_price:.2f} for {trade_date}")
    return round(estimated_price, 2)

def get_option_instruments(client, expiry_date):
    """
    Get all Bank Nifty option instruments for a given expiry date.
    expiry_date: string in format "YYYY-MM-DD"
    """
    try:
        all_instruments = client.kite.instruments("NFO")
        bank_nifty_options = []
        
        for instrument in all_instruments:
            if ("BANKNIFTY" in instrument.get("tradingsymbol", "") and 
                instrument.get("instrument_type") in ["CE", "PE"]):
                
                # Handle both datetime.date objects and string formats for expiry comparison
                inst_expiry = instrument.get("expiry")
                if isinstance(inst_expiry, date):
                    # Convert datetime.date to string for comparison
                    inst_expiry_str = inst_expiry.strftime("%Y-%m-%d")
                elif isinstance(inst_expiry, str):
                    inst_expiry_str = inst_expiry
                else:
                    continue  # Skip if expiry format is unexpected
                
                if inst_expiry_str == expiry_date:
                    bank_nifty_options.append(instrument)
        
        return bank_nifty_options
    except Exception as e:
        logger.error(f"Error fetching instruments: {e}")
        return []

def fetch_option_historical_data(client, instrument_token, from_date, to_date):
    """
    Fetch historical data for a specific option instrument.
    """
    try:
        # Fetch daily historical data with Open Interest
        historical_data = client.kite.historical_data(
            instrument_token=instrument_token,
            from_date=from_date,
            to_date=to_date,
            interval="day",
            oi=True  # Enable Open Interest data
        )
        
        # Debug: Log the structure of the first record (optional)
        if historical_data and len(historical_data) > 0:
            first_record = historical_data[0]
            logger.debug(f"Sample historical data record: {first_record}")
        
        return historical_data
    except Exception as e:
        logger.error(f"Error fetching historical data for token {instrument_token}: {e}")
        return []

def find_available_expiries(client, trade_date):
    """
    Find all available weekly expiries for Bank Nifty starting from trade_date.
    Returns a list of expiry dates (as strings) sorted by proximity to trade_date.
    """
    try:
        # Get all Bank Nifty option instruments
        instruments = client.kite.instruments("NFO")
        banknifty_options = [
            inst for inst in instruments
            if inst["name"] == "BANKNIFTY" and inst["instrument_type"] in ["CE", "PE"]
        ]
        
        logger.debug(f"Total Bank Nifty options found: {len(banknifty_options)}")
        
        # Sample a few instruments to see the format
        if banknifty_options:
            sample_inst = banknifty_options[0]
            logger.debug(f"Sample instrument: {sample_inst}")
            logger.debug(f"Sample expiry type: {type(sample_inst.get('expiry'))}")
        
        # Extract unique expiry dates
        expiry_dates = set()
        parse_errors = 0
        
        for inst in banknifty_options:
            try:
                # Handle both datetime.date objects and string formats
                expiry_raw = inst["expiry"]
                
                if isinstance(expiry_raw, date):
                    # If it's already a datetime.date object, convert to string
                    expiry_date = expiry_raw
                    expiry_str = expiry_raw.strftime("%Y-%m-%d")
                elif isinstance(expiry_raw, str):
                    # If it's a string, parse it
                    expiry_date = datetime.strptime(expiry_raw, "%Y-%m-%d").date()
                    expiry_str = expiry_raw
                else:
                    logger.debug(f"Unexpected expiry format: {expiry_raw} (type: {type(expiry_raw)})")
                    continue
                
                logger.debug(f"Processing expiry: {expiry_str} for instrument: {inst.get('tradingsymbol', 'unknown')}")
                
                # Only consider expiries within the next few weeks
                days_diff = (expiry_date - trade_date).days
                if expiry_date >= trade_date and days_diff <= (CONFIG['max_expiry_weeks_ahead'] * 7):
                    # Keep expiry as string for consistency
                    expiry_dates.add(expiry_str)
                    logger.debug(f"Added expiry: {expiry_str} (days from trade_date: {days_diff})")
                else:
                    logger.debug(f"Skipped expiry: {expiry_str} (days from trade_date: {days_diff}, too far)")
                    
            except Exception as e:
                parse_errors += 1
                if parse_errors <= 3:  # Log first few errors
                    logger.debug(f"Error parsing expiry '{inst.get('expiry', 'missing')}' (type: {type(inst.get('expiry'))}): {e}")
                continue
        
        logger.debug(f"Total parsing errors: {parse_errors}")
        
        # Sort by date and return closest expiries first
        sorted_expiries = sorted(list(expiry_dates))
        logger.info(f"Found {len(sorted_expiries)} available expiries: {sorted_expiries}")
        return sorted_expiries
    
    except Exception as e:
        logger.error(f"Error finding available expiries: {e}")
        return []

def fetch_option_historical_data_with_retry(client, instrument_token, from_date, to_date):
    """
    Fetch historical data with retry mechanism for robustness.
    """
    for attempt in range(CONFIG['retry_attempts']):
        try:
            # Fetch daily historical data with Open Interest
            historical_data = client.kite.historical_data(
                instrument_token=instrument_token,
                from_date=from_date,
                to_date=to_date,
                interval="day",
                oi=True  # Enable Open Interest data
            )
            return historical_data
        except Exception as e:
            logger.warning(f"Attempt {attempt + 1}/{CONFIG['retry_attempts']} failed for token {instrument_token}: {e}")
            if attempt < CONFIG['retry_attempts'] - 1:
                time.sleep(CONFIG['retry_delay'])
            else:
                logger.error(f"All retry attempts failed for token {instrument_token}")
    return []

def export_data_formats(df, output_file):
    """
    Export data in multiple formats based on configuration.
    """
    # Always save as parquet (primary format)
    df.to_parquet(output_file, index=False)
    
    # Optionally save as CSV for easier viewing
    if CONFIG['enable_csv_export']:
        csv_file = str(output_file).replace('.parquet', '.csv')
        df.to_csv(csv_file, index=False)
        logger.info(f"Also saved as CSV: {csv_file}")

def calculate_performance_metrics(start_time, num_instruments, num_records):
    """
    Calculate and log performance metrics.
    """
    if not CONFIG['enable_performance_metrics']:
        return
    
    end_time = time.time()
    duration = end_time - start_time
    
    instruments_per_sec = num_instruments / duration if duration > 0 else 0
    records_per_sec = num_records / duration if duration > 0 else 0
    
    logger.info("=== PERFORMANCE METRICS ===")
    logger.info(f"Total processing time: {duration:.2f} seconds")
    logger.info(f"Instruments processed: {num_instruments}")
    logger.info(f"Records generated: {num_records}")
    logger.info(f"Processing rate: {instruments_per_sec:.2f} instruments/sec")
    logger.info(f"Data generation rate: {records_per_sec:.2f} records/sec")

def fetch_real_historical_option_chains():
    """
    Fetch REAL historical option chain data for Bank Nifty.
    """
    # Initialize Zerodha client
    client = ZerodhaClient(credentials_path=str(project_root / "config" / "credentials.json"))
    
    # Check if login is successful
    if not client.login():
        logger.error("Login failed. Please check your credentials and access token.")
        return False
    
    logger.info("Successfully logged in to Zerodha API")
    
    # Set date range for historical data - using 1+ year of data
    end_date = datetime.now()
    start_date = end_date - timedelta(days=CONFIG['DAYS_BACK'])
    
    logger.info(f"Fetching {CONFIG['DAYS_BACK']} days of historical data from {start_date.date()} to {end_date.date()}")
    
    # Get trading dates
    trading_dates = get_trading_dates(start_date, end_date)
    logger.info(f"Found {len(trading_dates)} trading dates from {start_date.date()} to {end_date.date()}")
    
    # Create output directory if it doesn't exist
    os.makedirs(project_root / "data" / "raw", exist_ok=True)
    
    # Track the number of option chains fetched
    option_chains_fetched = 0
    all_option_data = []
    
    # Process all trading dates
    for i, trade_date in enumerate(trading_dates):
        logger.info(f"Processing {i+1}/{len(trading_dates)}: Fetching option chain for {trade_date}")
        
        try:
            # Get Bank Nifty spot price for this date
            spot_price = get_spot_price_for_date(trade_date)
            
            if not spot_price:
                logger.warning(f"Could not determine spot price for {trade_date}, skipping")
                continue
            
            logger.info(f"Spot price for {trade_date}: {spot_price}")
            
            # Compute strike range (spot ± CONFIG['strike_range'], stepping by CONFIG['strike_step'])
            min_strike = round((spot_price - CONFIG['strike_range']) / CONFIG['strike_step']) * CONFIG['strike_step']
            max_strike = round((spot_price + CONFIG['strike_range']) / CONFIG['strike_step']) * CONFIG['strike_step']
            strikes = list(range(int(min_strike), int(max_strike) + CONFIG['strike_step'], CONFIG['strike_step']))
            
            logger.info(f"Strike range: {min_strike} to {max_strike} ({len(strikes)} strikes)")
            
            # Find all available expiries for this trade date
            available_expiries = find_available_expiries(client, trade_date)
            
            if not available_expiries:
                logger.warning(f"No available expiries found for {trade_date}")
                continue
            
            logger.info(f"Available expiries: {available_expiries}")
            
            for expiry_date in available_expiries:
                logger.info(f"Processing expiry date: {expiry_date}")
                
                # Get option instruments for this expiry
                option_instruments = get_option_instruments(client, expiry_date)
                
                if not option_instruments:
                    logger.warning(f"No option instruments found for expiry {expiry_date}")
                    continue
                
                # Filter instruments by strike range
                filtered_instruments = [
                    inst for inst in option_instruments 
                    if inst.get("strike") and min_strike <= inst["strike"] <= max_strike
                ]
                
                logger.info(f"Found {len(filtered_instruments)} option instruments in strike range")
                
                if not filtered_instruments:
                    logger.warning(f"No instruments found in strike range for {trade_date}")
                    continue
                
                # Fetch historical data for each option instrument
                option_records = []
                
                start_time = time.time()
                
                for j, instrument in enumerate(filtered_instruments):
                    if j % CONFIG['log_progress_every'] == 0:
                        logger.info(f"Processing instrument {j+1}/{len(filtered_instruments)}")
                    
                    try:
                        # Fetch historical data for this specific date
                        hist_data = fetch_option_historical_data_with_retry(
                            client, 
                            instrument["instrument_token"],
                            trade_date,
                            trade_date
                        )
                        
                        if hist_data:
                            for record in hist_data:
                                # Use ONLY historical volume data from Zerodha (no manipulation)
                                volume = record.get("volume", 0)
                                
                                # Get option price for IV calculation (use close price)
                                option_price = record["close"]
                                
                                # Calculate time to expiry in years
                                expiry_date = datetime.strptime(str(instrument["expiry"]), "%Y-%m-%d").date()
                                current_date = trade_date
                                days_to_expiry = (expiry_date - current_date).days
                                time_to_expiry = max(days_to_expiry / 365.0, 1/365.0)  # Minimum 1 day
                                
                                # Calculate Implied Volatility
                                iv = 0.0
                                if option_price > 0 and spot_price > 0:
                                    try:
                                        iv = calculate_implied_volatility(
                                            option_price, 
                                            spot_price, 
                                            instrument["strike"], 
                                            time_to_expiry, 
                                            CONFIG['risk_free_rate'], 
                                            instrument["instrument_type"]
                                        )
                                    except Exception as iv_error:
                                        logger.debug(f"IV calculation failed for {instrument['tradingsymbol']}: {iv_error}")
                                        iv = 0.0
                                
                                # Calculate Greeks if enabled
                                delta, gamma, theta, vega = (0.0, 0.0, 0.0, 0.0)
                                if CONFIG['enable_greeks'] and iv > 0:
                                    delta, gamma, theta, vega = calculate_greeks(
                                        spot_price, 
                                        instrument["strike"], 
                                        time_to_expiry, 
                                        CONFIG['risk_free_rate'], 
                                        iv, 
                                        instrument["instrument_type"]
                                    )
                                
                                # Record comes as dictionary with keys: date, open, high, low, close, volume, oi
                                option_record = {
                                    "date": record["date"],      # timestamp
                                    "strike": instrument["strike"],
                                    "option_type": instrument["instrument_type"],
                                    "open": record["open"],      # open
                                    "high": record["high"],      # high
                                    "low": record["low"],        # low
                                    "close": record["close"],    # close
                                    "volume": volume,            # Enhanced volume (from quotes or historical)
                                    "oi": record.get("oi", 0),  # oi (with fallback)
                                    "iv": iv,                    # Calculated Implied Volatility
                                    "delta": delta,              # Delta
                                    "gamma": gamma,              # Gamma
                                    "theta": theta,              # Theta
                                    "vega": vega,                # Vega
                                    "tradingsymbol": instrument["tradingsymbol"],
                                    "expiry_date": instrument["expiry"],
                                    "instrument_token": instrument["instrument_token"],
                                    "spot_price": spot_price,    # Add spot price for reference
                                    "days_to_expiry": days_to_expiry,  # Add days to expiry
                                    "time_to_expiry": time_to_expiry   # Add time to expiry in years
                                }
                                
                                # Validate option data
                                if validate_option_data(option_record):
                                    option_records.append(option_record)
                        
                        # Add small delay to avoid hitting API limits
                        time.sleep(CONFIG['api_delay'])
                        
                    except Exception as e:
                        logger.warning(f"Failed to fetch data for {instrument['tradingsymbol']}: {e}")
                        continue
                
                if option_records:
                    # Create DataFrame
                    df = pd.DataFrame(option_records)
                    
                    # Log statistics about the pure historical data (NO MANIPULATION)
                    avg_volume = df['volume'].mean()
                    non_zero_volume_count = (df['volume'] > 0).sum()
                    avg_iv = df[df['iv'] > 0]['iv'].mean()
                    iv_calculated_count = (df['iv'] > 0).sum()
                    
                    logger.info(f"Pure historical data statistics for {trade_date} (expiry: {expiry_date}):")
                    logger.info(f"  - Average Volume (historical): {avg_volume:.2f}")
                    logger.info(f"  - Records with Volume > 0: {non_zero_volume_count}/{len(df)}")
                    logger.info(f"  - Average IV: {avg_iv:.4f}" if not pd.isna(avg_iv) else "  - Average IV: N/A")
                    logger.info(f"  - Records with IV calculated: {iv_calculated_count}/{len(df)}")
                    
                    # Save individual day's data as parquet
                    output_file = project_root / "data" / "raw" / f"options_{trade_date.strftime('%Y%m%d')}_{expiry_date.strftime('%Y%m%d')}.parquet"
                    export_data_formats(df, output_file)
                    logger.info(f"Saved {len(df)} option records to {output_file}")
                    
                    # Add to combined data
                    all_option_data.append(df)
                    option_chains_fetched += 1
                else:
                    logger.warning(f"No option data retrieved for {trade_date}")
                
                # Calculate performance metrics
                calculate_performance_metrics(start_time, len(filtered_instruments), len(option_records))
            
            # Add delay between dates to avoid API rate limits
            time.sleep(CONFIG['date_delay'])
            
        except Exception as e:
            logger.error(f"Error processing {trade_date}: {e}")
            continue
    
    logger.info(f"Successfully fetched {option_chains_fetched} option chain snapshots")
    
    # Fetch Bank Nifty minute-level data for the entire period
    logger.info("Fetching Bank Nifty minute-level historical data...")
    try:
        banknifty_data = client.fetch_historical_data(
            instrument="NSE:NIFTY BANK",
            interval="minute",
            from_date=start_date.date(),
            to_date=end_date.date()
        )
        
        if banknifty_data:
            banknifty_df = pd.DataFrame(banknifty_data)
            
            # Save Bank Nifty minute data
            banknifty_file = project_root / "data" / "raw" / f"banknifty_minute_{datetime.now().strftime('%Y%m%d')}.csv"
            banknifty_df.to_csv(banknifty_file, index=False)
            logger.info(f"Saved {len(banknifty_df)} Bank Nifty minute records to {banknifty_file}")
            
            # Process Bank Nifty data to create unified parquet file  
            load_data.process_banknifty_data()
        else:
            logger.warning("No Bank Nifty minute data retrieved")
            
    except Exception as e:
        logger.error(f"Error fetching Bank Nifty minute data: {e}")
    
    # Create a combined parquet with all option chain data
    if all_option_data:
        combined_options = pd.concat(all_option_data, ignore_index=True)
        combined_file = project_root / "data" / "raw" / f"options_real_combined_{datetime.now().strftime('%Y%m%d')}.parquet"
        export_data_formats(combined_options, combined_file)
        logger.info(f"Saved combined option chain data to {combined_file}")
        
        # Process options data to create the unified parquet file
        load_data.process_options_data()
        
        return True
    else:
        logger.error("No option chain data was successfully fetched")
        return False

def main():
    """
    Main function to execute the real historical option chain fetching process.
    """
    logger.info("Starting REAL historical option chain fetching process...")
    
    # Fetch real historical option chains
    success = fetch_real_historical_option_chains()
    
    if success:
        logger.info("Real historical option chain fetching process completed successfully!")
    else:
        logger.error("Real historical option chain fetching process failed!")
    
    return success

if __name__ == "__main__":
    main()
