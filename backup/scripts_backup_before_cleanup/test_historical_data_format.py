#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Test script to check the format of historical data returned by Zerodha API
"""

import os
import sys
import logging
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
import json

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

# Import Zerodha client
from src.data_ingest.zerodha_client import ZerodhaClient

# Configure logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_historical_data_format():
    """Test the format of historical data returned by Zerodha API"""
    
    # Initialize client
    client = ZerodhaClient(credentials_path=str(project_root / "config" / "credentials.json"))
    
    if not client.login():
        logger.error("Login failed")
        return
    
    logger.info("Successfully logged in")
    
    # Get Bank Nifty instruments first
    try:
        instruments = client.kite.instruments("NFO")
        bank_nifty_options = []
        
        for instrument in instruments:
            if ("BANKNIFTY" in instrument.get("tradingsymbol", "") and 
                instrument.get("expiry") and
                instrument.get("instrument_type") in ["CE", "PE"]):
                bank_nifty_options.append(instrument)
                
        logger.info(f"Found {len(bank_nifty_options)} Bank Nifty option instruments")
        
        if bank_nifty_options:
            # Test with the first instrument
            test_instrument = bank_nifty_options[0]
            logger.info(f"Testing with instrument: {test_instrument['tradingsymbol']}")
            logger.info(f"Instrument token: {test_instrument['instrument_token']}")
            
            # Try to fetch historical data
            from_date = datetime.now() - timedelta(days=5)
            to_date = datetime.now() - timedelta(days=1)
            
            historical_data = client.kite.historical_data(
                instrument_token=test_instrument['instrument_token'],
                from_date=from_date,
                to_date=to_date,
                interval="day"
            )
            
            logger.info(f"Historical data type: {type(historical_data)}")
            logger.info(f"Historical data length: {len(historical_data) if historical_data else 0}")
            
            if historical_data:
                logger.info("Sample record structure:")
                logger.info(json.dumps(historical_data[0], indent=2, default=str))
                
                # Check if we have multiple records
                if len(historical_data) > 1:
                    logger.info("Second record:")
                    logger.info(json.dumps(historical_data[1], indent=2, default=str))
                    
                # Show all available keys
                if isinstance(historical_data[0], dict):
                    logger.info(f"Available keys: {list(historical_data[0].keys())}")
            else:
                logger.warning("No historical data returned")
                
    except Exception as e:
        logger.error(f"Error: {e}")

if __name__ == "__main__":
    test_historical_data_format()
