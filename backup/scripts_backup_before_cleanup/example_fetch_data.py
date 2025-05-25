#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Example script for fetching Bank Nifty historical data from Zerodha.
This script demonstrates how to use the ZerodhaClient and ZerodhaDataFetcher
classes to authenticate and fetch data from Zerodha's Kite API.
"""

import os
import sys
import logging
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

# Import Zerodha client classes
from src.data_ingest.zerodha_client import ZerodhaClient
from src.data_ingest.zerodha_fetcher import ZerodhaDataFetcher

# Configure logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def setup_credentials():
    """
    Check if credentials file exists, if not create a template.
    """
    credentials_path = project_root / "config" / "credentials.json"
    template_path = project_root / "config" / "credentials.template.json"
    
    if not credentials_path.exists():
        if template_path.exists():
            logger.warning(f"Credentials file not found! Please copy {template_path} to {credentials_path} and update with your Zerodha API credentials.")
        else:
            logger.warning("Credentials file not found! Please create config/credentials.json with your Zerodha API credentials.")
        
        logger.info("Required format for credentials.json:")
        logger.info("""
        {
            "api_key": "YOUR_ZERODHA_API_KEY",
            "api_secret": "YOUR_ZERODHA_API_SECRET",
            "access_token": "OPTIONAL_SAVED_ACCESS_TOKEN"
        }
        """)
        return False
    
    return True


def fetch_banknifty_data():
    """
    Fetch and save Bank Nifty historical data using ZerodhaClient.
    """
    logger.info("Initializing Zerodha client...")
    zerodha_client = ZerodhaClient(credentials_path=str(project_root / "config" / "credentials.json"))
    
    # Check if login is successful
    if not zerodha_client.kite:
        logger.error("Failed to initialize Kite client. Please check your credentials.")
        return False
    
    # Try to login with existing token
    if not zerodha_client.login():
        logger.error("Login failed. You need to generate a new access token.")
        logger.info(f"Please visit: {zerodha_client.get_login_url()}")
        logger.info("After login, you will be redirected to a URL containing a request token.")
        logger.info("Extract the request token from the URL and run this script again with:")
        logger.info("python example_fetch_data.py REQUEST_TOKEN")
        return False
    
    logger.info("Login successful! Fetching Bank Nifty data...")
    
    # Get Bank Nifty instrument token
    banknifty_token = zerodha_client.get_instrument_token("NIFTY BANK", "NSE")
    
    if not banknifty_token:
        logger.error("Failed to get Bank Nifty instrument token.")
        return False
    
    # Set date range for historical data
    end_date = datetime.now()
    start_date = end_date - timedelta(days=180)  # Last 6 months
    
    # Fetch historical data
    historical_data = zerodha_client.fetch_historical_data(
        symbol=banknifty_token,
        interval="day",
        from_date=start_date,
        to_date=end_date
    )
    
    if historical_data.empty:
        logger.error("No historical data retrieved.")
        return False
    
    # Create output directory if it doesn't exist
    os.makedirs(project_root / "data" / "raw", exist_ok=True)
    
    # Save data to CSV file
    output_file = project_root / "data" / "raw" / f"banknifty_{datetime.now().strftime('%Y%m%d')}.csv"
    historical_data.to_csv(output_file, index=False)
    logger.info(f"Successfully saved Bank Nifty data to {output_file}.")
    
    # Print sample data
    logger.info(f"Total records: {len(historical_data)}")
    logger.info(f"Sample data:\n{historical_data.head()}")
    
    return True


def fetch_option_chain():
    """
    Fetch and save Bank Nifty option chain data using ZerodhaClient.
    """
    logger.info("Initializing Zerodha client...")
    zerodha_client = ZerodhaClient(credentials_path=str(project_root / "config" / "credentials.json"))
    
    # Check if login is successful
    if not zerodha_client.login():
        logger.error("Login failed. Cannot fetch option chain.")
        return False
    
    logger.info("Login successful! Fetching Bank Nifty option chain...")
    
    # Fetch option chain (nearest expiry)
    option_chain = zerodha_client.fetch_option_chain(index_symbol="BANKNIFTY")
    
    if not option_chain:
        logger.error("Failed to fetch option chain.")
        return False
    
    # Extract calls and puts
    calls = option_chain.get('calls', [])
    puts = option_chain.get('puts', [])
    
    if not calls or not puts:
        logger.error("No options data in the option chain.")
        return False
    
    # Create DataFrames for calls and puts
    calls_df = pd.DataFrame(calls)
    puts_df = pd.DataFrame(puts)
    
    # Add option type
    calls_df['option_type'] = 'CE'
    puts_df['option_type'] = 'PE'
    
    # Combine into a single DataFrame
    options_df = pd.concat([calls_df, puts_df], ignore_index=True)
    
    # Create output directory if it doesn't exist
    os.makedirs(project_root / "data" / "raw", exist_ok=True)
    
    # Save data to CSV file
    output_file = project_root / "data" / "raw" / f"options_{datetime.now().strftime('%Y%m%d')}.csv"
    options_df.to_csv(output_file, index=False)
    logger.info(f"Successfully saved option chain data to {output_file}.")
    
    # Print summary
    logger.info(f"Total options: {len(options_df)}")
    logger.info(f"Calls: {len(calls_df)}, Puts: {len(puts_df)}")
    
    return True


def main():
    """
    Main function to execute the data fetching process.
    """
    logger.info("Starting Bank Nifty data fetching process...")
    
    # Check credentials setup
    if not setup_credentials():
        return
    
    # Process command-line arguments for request token
    if len(sys.argv) > 1 and sys.argv[1]:
        request_token = sys.argv[1]
        logger.info(f"Request token provided: {request_token}")
        
        # Initialize client and generate session
        zerodha_client = ZerodhaClient(credentials_path=str(project_root / "config" / "credentials.json"))
        if zerodha_client.login(request_token):
            logger.info("Successfully generated access token and saved to credentials.")
        else:
            logger.error("Failed to generate access token.")
            return
    
    # Fetch historical data
    fetch_banknifty_data()
    
    # Fetch option chain data
    fetch_option_chain()
    
    # Process the raw data into parquet format
    logger.info("Processing raw data into parquet format...")
    import src.data_ingest.load_data as load_data
    banknifty_path, banknifty_df = load_data.process_banknifty_data()
    options_path, options_df = load_data.process_options_data()
    
    if banknifty_path:
        logger.info(f"Bank Nifty data processed and saved to {banknifty_path}")
    
    if options_path:
        logger.info(f"Options data processed and saved to {options_path}")
    
    logger.info("Data fetching process complete!")


if __name__ == "__main__":
    main()
