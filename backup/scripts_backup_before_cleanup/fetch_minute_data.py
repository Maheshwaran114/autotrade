#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script to fetch Bank Nifty minute-level historical data.
This script specifically obtains 1+ year of minute-level Bank Nifty data.
"""

import os
import sys
import logging
import pandas as pd
import time
from datetime import datetime, timedelta
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

# Import Zerodha client
from src.data_ingest.zerodha_client import ZerodhaClient
import src.data_ingest.load_data as load_data

# Configure logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def fetch_minute_level_data(client):
    """
    Fetch 1+ year of minute-level Bank Nifty data using chunked requests to handle API limits.
    
    Args:
        client: ZerodhaClient instance
        
    Returns:
        bool: True if successful
    """
    logger.info("Fetching 1+ year of minute-level Bank Nifty data...")
    
    # Get Bank Nifty instrument token
    banknifty_token = client.get_instrument_token("NIFTY BANK", "NSE")
    
    if not banknifty_token:
        logger.error("Failed to get Bank Nifty instrument token.")
        return False
    
    logger.info(f"Bank Nifty instrument token: {banknifty_token}")
    
    # The Zerodha API has limits on historical data requests:
    # - minute: 60 days per request
    # - day: 2000 days per request
    #
    # Since we need 1+ year (365+ days) of minute-level data, we'll fetch in 60-day chunks
    
    # Set up date range
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365 + 30)  # 1 year + 1 month
    
    logger.info(f"Fetching data from {start_date.date()} to {end_date.date()}")
    
    # Create chunks of 60 days each
    chunks = []
    current_start = start_date
    
    while current_start < end_date:
        current_end = current_start + timedelta(days=59)  # 60 days including start date
        
        if current_end > end_date:
            current_end = end_date
            
        chunks.append((current_start, current_end))
        current_start = current_end + timedelta(days=1)
    
    logger.info(f"Created {len(chunks)} 60-day chunks for data fetch")
    
    # Fetch data for each chunk
    all_data = []
    
    for i, (chunk_start, chunk_end) in enumerate(chunks):
        logger.info(f"Fetching chunk {i+1}/{len(chunks)}: {chunk_start.date()} to {chunk_end.date()}")
        
        try:
            historical_data = client.fetch_historical_data(
                symbol=banknifty_token,
                interval="minute",
                from_date=chunk_start,
                to_date=chunk_end
            )
            
            if not historical_data.empty:
                logger.info(f"Fetched {len(historical_data)} records for chunk {i+1}")
                all_data.append(historical_data)
            else:
                logger.warning(f"No data retrieved for chunk {i+1}")
                
            # Add a delay to avoid hitting API rate limits
            time.sleep(1)
            
        except Exception as e:
            logger.error(f"Error fetching data for chunk {i+1}: {e}")
    
    # Combine all chunks
    if not all_data:
        logger.error("No historical data retrieved in any chunk.")
        return False
    
    combined_data = pd.concat(all_data, ignore_index=True)
    
    # Remove duplicates and sort
    combined_data = combined_data.drop_duplicates(subset=['date'])
    combined_data = combined_data.sort_values(by='date')
    
    # Create output directory if it doesn't exist
    os.makedirs(project_root / "data" / "raw", exist_ok=True)
    
    # Save data to CSV file
    output_file = project_root / "data" / "raw" / f"banknifty_minute_{datetime.now().strftime('%Y%m%d')}.csv"
    combined_data.to_csv(output_file, index=False)
    
    logger.info(f"Successfully saved {len(combined_data)} minute-level records to {output_file}")
    logger.info(f"Data spans from {combined_data['date'].min()} to {combined_data['date'].max()}")
    
    return True

def main():
    """Main function to execute the data fetching process."""
    logger.info("Starting Bank Nifty minute-level data fetching process...")
    
    # Initialize client
    client = ZerodhaClient(credentials_path=str(project_root / "config" / "credentials.json"))
    
    # Check if login is successful
    if not client.login():
        logger.error("Login failed. Please check your credentials and access token.")
        return
    
    # Fetch minute-level data
    success = fetch_minute_level_data(client)
    
    if success:
        # Process the raw data into parquet format
        logger.info("Processing raw data into parquet format...")
        banknifty_path, banknifty_df = load_data.process_banknifty_data()
        
        if banknifty_path:
            logger.info(f"Bank Nifty data processed and saved to {banknifty_path}")
        
        logger.info("Data fetching process complete!")
    else:
        logger.error("Failed to fetch minute-level data.")

if __name__ == "__main__":
    main()
