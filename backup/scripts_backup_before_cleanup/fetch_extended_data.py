#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Extended data collection script for Bank Nifty Options Trading System.
This script fetches 1+ year of minute-level Bank Nifty data and option chain snapshots.
"""

import os
import sys
import logging
import pandas as pd
import argparse
from datetime import datetime, timedelta
from pathlib import Path
from tqdm import tqdm

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

# Import Zerodha client classes
from src.data_ingest.zerodha_client import ZerodhaClient
import src.data_ingest.load_data as load_data

# Configure logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def fetch_extended_banknifty_data(client, interval="minute", months=13):
    """
    Fetch 1+ year of Bank Nifty data at the specified interval.
    
    Args:
        client: ZerodhaClient instance
        interval: Data interval ('minute', '5minute', 'day', etc.)
        months: Number of months of historical data to fetch
        
    Returns:
        bool: True if successful
    """
    logger.info(f"Fetching {months} months of {interval}-level Bank Nifty data...")
    
    # Get Bank Nifty instrument token
    banknifty_token = client.get_instrument_token("NIFTY BANK", "NSE")
    
    if not banknifty_token:
        logger.error("Failed to get Bank Nifty instrument token.")
        return False
    
    # Due to Zerodha API limits, we need to fetch data in chunks
    # Zerodha typically limits historical data to 60 days for minute-level data
    end_date = datetime.now()
    all_data = []
    
    # Create date ranges for chunked requests (60-day chunks)
    chunk_size = 60  # days
    total_days = months * 30
    chunks = [(total_days - i - chunk_size, total_days - i) for i in range(0, total_days, chunk_size)]
    
    # Fetch data in chunks
    for days_back_start, days_back_end in tqdm(chunks, desc="Fetching data chunks"):
        start_date = end_date - timedelta(days=days_back_start)
        to_date = end_date - timedelta(days=days_back_end)
        
        logger.info(f"Fetching Bank Nifty data from {start_date.date()} to {to_date.date()}")
        
        # Add a small delay between requests to avoid rate limiting
        historical_data = client.fetch_historical_data(
            symbol=banknifty_token,
            interval=interval,
            from_date=start_date,
            to_date=to_date
        )
        
        if not historical_data.empty:
            all_data.append(historical_data)
        
    # Combine all chunks
    if not all_data:
        logger.error("No historical data retrieved in any chunk.")
        return False
    
    # Combine all data frames
    combined_data = pd.concat(all_data, ignore_index=True)
    combined_data = combined_data.drop_duplicates(subset=['date'])
    combined_data = combined_data.sort_values(by='date')
    
    # Create output directory if it doesn't exist
    os.makedirs(project_root / "data" / "raw", exist_ok=True)
    
    # Save data to CSV file
    output_file = project_root / "data" / "raw" / f"banknifty_{datetime.now().strftime('%Y%m%d')}.csv"
    combined_data.to_csv(output_file, index=False)
    logger.info(f"Successfully saved {len(combined_data)} records of Bank Nifty data to {output_file}.")
    
    # Print sample data
    logger.info(f"Data spans from {combined_data['date'].min()} to {combined_data['date'].max()}")
    logger.info(f"Sample data:\n{combined_data.head()}")
    
    return True


def fetch_option_chain_snapshots(client, days=30, strikes_range=2000):
    """
    Fetch Bank Nifty option chain snapshots for the current expiry.
    
    Args:
        client: ZerodhaClient instance
        days: Number of days to fetch option data for
        strikes_range: Range around spot price to include (±strikes_range)
        
    Returns:
        bool: True if successful
    """
    logger.info(f"Fetching Bank Nifty option chain snapshots (±{strikes_range} around spot)...")
    
    # Get current Bank Nifty spot price
    spot_price = client.get_ltp("NIFTY BANK", "NSE")
    
    if not spot_price:
        logger.error("Failed to get Bank Nifty spot price.")
        return False
    
    logger.info(f"Current Bank Nifty spot price: {spot_price}")
    
    # Calculate strike range
    min_strike = spot_price - strikes_range
    max_strike = spot_price + strikes_range
    
    # Fetch option chain using the direct symbol approach
    try:
        # Get NFO instruments and filter for Bank Nifty options
        instruments = client.kite.instruments("NFO")
        current_date = datetime.now()
        
        # Find the nearest weekly expiry
        expiry_dates = []
        for instrument in instruments:
            if "BANKNIFTY" in instrument.get("tradingsymbol", "") and instrument.get("expiry"):
                expiry_dates.append(instrument.get("expiry"))
        
        # Get unique expiry dates and find the nearest
        unique_expiry_dates = list(set(expiry_dates))
        unique_expiry_dates.sort()
        
        if not unique_expiry_dates:
            logger.error("Could not find any expiry dates for Bank Nifty options")
            return False
        
        nearest_expiry = unique_expiry_dates[0]
        logger.info(f"Fetching option chain for expiry: {nearest_expiry}")
        
        # Filter instruments for the nearest expiry and within strike range
        filtered_options = []
        for instrument in instruments:
            if ("BANKNIFTY" in instrument.get("tradingsymbol", "") and 
                instrument.get("expiry") == nearest_expiry and
                instrument.get("strike") is not None and
                min_strike <= instrument.get("strike") <= max_strike):
                filtered_options.append(instrument)
        
        logger.info(f"Found {len(filtered_options)} option instruments in strike range")
        
        # Get LTP for all filtered options
        if filtered_options:
            trading_symbols = [f"NFO:{opt['tradingsymbol']}" for opt in filtered_options]
            
            # Due to Zerodha API limits, we may need to chunk the requests
            chunk_size = 500  # Zerodha typically limits quote requests
            option_data = []
            
            for i in range(0, len(trading_symbols), chunk_size):
                chunk = trading_symbols[i:i + chunk_size]
                try:
                    quotes = client.kite.quote(chunk)
                    
                    # Process quotes
                    for symbol, quote in quotes.items():
                        symbol = symbol.replace("NFO:", "")
                        instrument_info = next((opt for opt in filtered_options 
                                             if opt["tradingsymbol"] == symbol), None)
                        
                        if instrument_info:
                            option_data.append({
                                "tradingsymbol": symbol,
                                "strike": instrument_info.get("strike"),
                                "expiry_date": instrument_info.get("expiry"),
                                "last_price": quote.get("last_price"),
                                "volume": quote.get("volume"),
                                "oi": quote.get("oi"),
                                "option_type": "CE" if symbol.endswith("CE") else "PE",
                                "timestamp": current_date
                            })
                except Exception as e:
                    logger.error(f"Error fetching quotes chunk {i}: {e}")
            
            # Create DataFrame
            options_df = pd.DataFrame(option_data)
            
            if not options_df.empty:
                # Create output directory if it doesn't exist
                os.makedirs(project_root / "data" / "raw", exist_ok=True)
                
                # Save data to CSV file
                output_file = project_root / "data" / "raw" / f"options_{datetime.now().strftime('%Y%m%d')}.csv"
                options_df.to_csv(output_file, index=False)
                logger.info(f"Successfully saved option chain data to {output_file}.")
                
                # Print summary
                ce_count = len(options_df[options_df['option_type'] == 'CE'])
                pe_count = len(options_df[options_df['option_type'] == 'PE'])
                logger.info(f"Total options: {len(options_df)} (Calls: {ce_count}, Puts: {pe_count})")
                
                return True
    
    except Exception as e:
        logger.error(f"Failed to fetch option chain: {e}")
    
    return False


def main():
    """
    Main function to execute the extended data fetching process.
    """
    parser = argparse.ArgumentParser(description="Fetch extended historical data for Bank Nifty")
    parser.add_argument("--interval", default="minute", choices=["minute", "5minute", "15minute", "30minute", "60minute", "day"],
                        help="Data interval (default: minute)")
    parser.add_argument("--months", type=int, default=13,
                        help="Number of months of historical data to fetch (default: 13)")
    parser.add_argument("--strikes-range", type=int, default=2000,
                        help="Range around spot price for option chain (default: 2000)")
    args = parser.parse_args()
    
    logger.info("Starting extended Bank Nifty data fetching process...")
    
    # Initialize client
    client = ZerodhaClient(credentials_path=str(project_root / "config" / "credentials.json"))
    
    # Check if login is successful
    if not client.login():
        logger.error("Login failed. Please check your credentials and access token.")
        return
    
    # Step 1: Fetch extended Bank Nifty historical data
    fetch_extended_banknifty_data(client, interval=args.interval, months=args.months)
    
    # Step 2: Fetch option chain data
    fetch_option_chain_snapshots(client, strikes_range=args.strikes_range)
    
    # Step 3: Process the raw data into parquet format
    logger.info("Processing raw data into parquet format...")
    banknifty_path, banknifty_df = load_data.process_banknifty_data()
    options_path, options_df = load_data.process_options_data()
    
    if banknifty_path:
        logger.info(f"Bank Nifty data processed and saved to {banknifty_path}")
    
    if options_path:
        logger.info(f"Options data processed and saved to {options_path}")
    
    logger.info("Extended data fetching process complete!")


if __name__ == "__main__":
    main()
