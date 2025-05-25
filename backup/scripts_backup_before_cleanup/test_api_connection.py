#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Test script to verify Zerodha API connection and fetch a single option's historical data.
"""

import sys
from pathlib import Path
from datetime import datetime, timedelta
import logging

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from src.data_ingest.zerodha_client import ZerodhaClient

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_api_connection():
    """Test if we can connect to Zerodha API and fetch historical data."""
    
    # Initialize client
    client = ZerodhaClient(credentials_path=str(project_root / "config" / "credentials.json"))
    
    # Test login
    if not client.login():
        logger.error("Login failed")
        return False
    
    logger.info("Login successful!")
    
    # Get instruments
    try:
        logger.info("Fetching NFO instruments...")
        instruments = client.kite.instruments("NFO")
        logger.info(f"Found {len(instruments)} NFO instruments")
        
        # Find a Bank Nifty option
        bank_nifty_options = [
            inst for inst in instruments 
            if "BANKNIFTY" in inst.get("tradingsymbol", "") and 
            inst.get("instrument_type") in ["CE", "PE"]
        ]
        
        logger.info(f"Found {len(bank_nifty_options)} Bank Nifty options")
        
        if bank_nifty_options:
            # Pick the first one for testing
            test_option = bank_nifty_options[0]
            logger.info(f"Testing with option: {test_option['tradingsymbol']}")
            
            # Try to fetch historical data
            try:
                from_date = datetime.now() - timedelta(days=7)
                to_date = datetime.now()
                
                logger.info(f"Fetching historical data from {from_date.date()} to {to_date.date()}")
                hist_data = client.kite.historical_data(
                    instrument_token=test_option["instrument_token"],
                    from_date=from_date,
                    to_date=to_date,
                    interval="day"
                )
                
                logger.info(f"Successfully fetched {len(hist_data)} records")
                if hist_data:
                    logger.info(f"Sample record: {hist_data[0]}")
                
                return True
                
            except Exception as e:
                logger.error(f"Error fetching historical data: {e}")
                return False
        else:
            logger.error("No Bank Nifty options found")
            return False
            
    except Exception as e:
        logger.error(f"Error fetching instruments: {e}")
        return False

if __name__ == "__main__":
    success = test_api_connection()
    print(f"API test {'PASSED' if success else 'FAILED'}")
