#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Optimized script to fetch option chain data for Bank Nifty.
"""

import os
import sys
import logging
import pandas as pd
from datetime import datetime
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

# Import Zerodha client classes
from src.data_ingest.zerodha_client import ZerodhaClient

# Configure logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    """
    Fetch Bank Nifty option chain data using ZerodhaClient.
    """
    logger.info("Initializing Zerodha client...")
    client = ZerodhaClient(credentials_path=str(project_root / "config" / "credentials.json"))
    
    # Check if login is successful
    if not client.login():
        logger.error("Login failed. Cannot fetch option chain.")
        return False
    
    logger.info("Login successful! Fetching Bank Nifty option chain...")
    
    # Get current Bank Nifty spot price
    spot_price = client.get_ltp("NIFTY BANK", "NSE")
    
    if not spot_price:
        logger.error("Failed to get Bank Nifty spot price.")
        return False
    
    logger.info(f"Current Bank Nifty spot price: {spot_price}")
    
    # Calculate strike range (±2000 around spot)
    strikes_range = 2000
    min_strike = int((spot_price - strikes_range) // 100) * 100  # Round to nearest 100
    max_strike = int((spot_price + strikes_range) // 100) * 100 + 100  # Round to nearest 100
    
    logger.info(f"Strike range: {min_strike} to {max_strike}")
    
    # Get all instruments
    try:
        logger.info("Fetching all NFO instruments...")
        all_instruments = client.kite.instruments("NFO")
        logger.info(f"Fetched {len(all_instruments)} NFO instruments")
        
        # Filter for Bank Nifty options
        bank_nifty_options = []
        for instrument in all_instruments:
            if ("BANKNIFTY" in instrument.get("tradingsymbol", "") and 
                instrument.get("strike") is not None and
                min_strike <= instrument.get("strike") <= max_strike):
                bank_nifty_options.append(instrument)
        
        logger.info(f"Found {len(bank_nifty_options)} Bank Nifty options in strike range")
        
        # Extract all expiry dates
        expiry_dates = list(set(opt["expiry"] for opt in bank_nifty_options if opt.get("expiry")))
        expiry_dates.sort()
        
        if not expiry_dates:
            logger.error("No option expiries found")
            return False
        
        # Get nearest expiry
        nearest_expiry = expiry_dates[0]
        logger.info(f"Nearest expiry: {nearest_expiry}")
        
        # Filter for nearest expiry
        filtered_options = [opt for opt in bank_nifty_options 
                          if opt.get("expiry") == nearest_expiry]
        
        logger.info(f"Found {len(filtered_options)} options for nearest expiry")
        
        # Create a list of symbols to fetch quotes for
        trading_symbols = [f"NFO:{opt['tradingsymbol']}" for opt in filtered_options]
        
        # Fetch in smaller chunks to avoid rate limiting
        chunk_size = 50
        option_data = []
        
        for i in range(0, len(trading_symbols), chunk_size):
            chunk = trading_symbols[i:i+chunk_size]
            logger.info(f"Fetching quotes for chunk {i//chunk_size + 1}/{(len(trading_symbols) + chunk_size - 1)//chunk_size}")
            
            try:
                quotes = client.kite.quote(chunk)
                
                for symbol, quote in quotes.items():
                    instrument_symbol = symbol.replace("NFO:", "")
                    instrument_info = next((opt for opt in filtered_options 
                                         if opt["tradingsymbol"] == instrument_symbol), None)
                    
                    if instrument_info:
                        option_data.append({
                            "tradingsymbol": instrument_symbol,
                            "strike": instrument_info.get("strike"),
                            "expiry_date": instrument_info.get("expiry"),
                            "last_price": quote.get("last_price"),
                            "volume": quote.get("volume"),
                            "oi": quote.get("oi"),
                            "option_type": "CE" if instrument_symbol.endswith("CE") else "PE",
                            "timestamp": datetime.now()
                        })
            except Exception as e:
                logger.error(f"Error fetching quotes for chunk: {e}")
        
        if not option_data:
            logger.error("No option data retrieved")
            return False
        
        # Create DataFrame
        df = pd.DataFrame(option_data)
        
        # Create output directory if it doesn't exist
        os.makedirs(project_root / "data" / "raw", exist_ok=True)
        
        # Save data to CSV file
        output_file = project_root / "data" / "raw" / f"options_{datetime.now().strftime('%Y%m%d')}.csv"
        df.to_csv(output_file, index=False)
        logger.info(f"Successfully saved {len(df)} option records to {output_file}")
        
        # Print summary
        ce_count = len(df[df['option_type'] == 'CE'])
        pe_count = len(df[df['option_type'] == 'PE'])
        logger.info(f"Call options: {ce_count}, Put options: {pe_count}")
        
        return True
    
    except Exception as e:
        logger.error(f"Error in option chain fetching: {e}")
        return False

if __name__ == "__main__":
    main()
