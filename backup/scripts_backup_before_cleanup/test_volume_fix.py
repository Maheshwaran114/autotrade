#!/usr/bin/env python3
"""
Test script to verify Bank Nifty volume fix using front-month futures.
This script tests the fixed implementation on a single day to ensure it works.
"""

import sys
import os
import logging
from pathlib import Path
from datetime import datetime, date, timedelta

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

# Import the client and updated functions
from src.data_ingest.zerodha_client import ZerodhaClient

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_front_month_volume_fix():
    """Test the front-month futures volume fix on a recent trading day"""
    
    logger.info("🧪 Testing Bank Nifty volume fix using front-month futures")
    
    # Initialize client
    client = ZerodhaClient(credentials_path=str(project_root / "config" / "credentials.json"))
    
    if not client.login():
        logger.error("❌ Authentication failed")
        return False
    
    logger.info("✅ Successfully authenticated with Zerodha API")
    
    # Test on a recent trading day
    test_date = date(2024, 12, 20)  # A recent trading day
    logger.info(f"📅 Testing on {test_date}")
    
    # Test 1: Check if we can find front-month futures
    try:
        # Get all NFO instruments
        nfo_instruments = client.kite.instruments("NFO")
        if not nfo_instruments:
            logger.error("❌ Failed to fetch NFO instruments")
            return False
            
        import pandas as pd
        nfo_df = pd.DataFrame(nfo_instruments)
        
        # Filter for Bank Nifty futures
        banknifty_futures = nfo_df[
            (nfo_df['name'] == 'BANKNIFTY') & 
            (nfo_df['instrument_type'] == 'FUT')
        ].copy()
        
        if banknifty_futures.empty:
            logger.error("❌ No Bank Nifty futures contracts found")
            return False
        
        logger.info(f"✅ Found {len(banknifty_futures)} Bank Nifty futures contracts")
        
        # Convert expiry to date and find valid contracts
        banknifty_futures['expiry_date'] = pd.to_datetime(banknifty_futures['expiry']).dt.date
        valid_contracts = banknifty_futures[banknifty_futures['expiry_date'] >= test_date].copy()
        
        if valid_contracts.empty:
            logger.warning(f"⚠️ No valid futures contracts for {test_date}")
            return False
        
        # Get front-month contract
        valid_contracts = valid_contracts.sort_values('expiry_date')
        front_contract = valid_contracts.iloc[0]
        
        logger.info(f"📊 Front-month contract: {front_contract['tradingsymbol']} (expires: {front_contract['expiry_date']})")
        
        # Test 2: Try to fetch sample futures data
        logger.info("📈 Testing futures data fetch...")
        
        futures_data = client.fetch_historical_data(
            instrument=front_contract['instrument_token'],
            interval="minute", 
            from_date=test_date,
            to_date=test_date
        )
        
        if futures_data and len(futures_data) > 0:
            logger.info(f"✅ Successfully fetched {len(futures_data)} futures records")
            
            # Check data structure
            sample = futures_data[0]
            logger.info(f"📊 Sample futures record keys: {list(sample.keys())}")
            
            # Check if volume data exists
            if 'volume' in sample:
                volumes = [record['volume'] for record in futures_data[:5]]  # First 5 records
                logger.info(f"📊 Sample volumes: {volumes}")
                
                total_volume = sum(record.get('volume', 0) for record in futures_data)
                logger.info(f"📊 Total volume for {test_date}: {total_volume:,}")
                
                if total_volume > 0:
                    logger.info("✅ Volume fix working correctly - found non-zero futures volume!")
                    return True
                else:
                    logger.warning("⚠️ All volume records are zero")
                    
            else:
                logger.error("❌ No volume field in futures data")
                
        else:
            logger.error("❌ No futures data fetched")
            
        # Test 3: Try to fetch index data for comparison
        logger.info("🏦 Testing index data fetch...")
        
        index_data = client.fetch_historical_data(
            instrument="NSE:NIFTY BANK",
            interval="minute",
            from_date=test_date,
            to_date=test_date
        )
        
        if index_data and len(index_data) > 0:
            logger.info(f"✅ Successfully fetched {len(index_data)} index records")
            
            sample_index = index_data[0]
            logger.info(f"📊 Sample index record keys: {list(sample_index.keys())}")
            
            if 'volume' in sample_index:
                index_volumes = [record['volume'] for record in index_data[:5]]
                logger.info(f"📊 Sample index volumes: {index_volumes}")
                
                index_total = sum(record.get('volume', 0) for record in index_data)
                logger.info(f"📊 Total index volume: {index_total:,}")
                
        return True
        
    except Exception as e:
        logger.error(f"❌ Test failed: {e}")
        import traceback
        logger.debug(traceback.format_exc())
        return False

if __name__ == "__main__":
    success = test_front_month_volume_fix()
    if success:
        logger.info("🎉 Volume fix test completed successfully")
    else:
        logger.error("❌ Volume fix test failed")
        sys.exit(1)
