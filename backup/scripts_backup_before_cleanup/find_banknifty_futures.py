#!/usr/bin/env python3
"""
Bank Nifty Futures Discovery Script

This script finds the current Bank Nifty futures contract and tests volume data extraction.
"""

import os
import sys
from datetime import datetime, timedelta
import pandas as pd
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from src.data_ingest.zerodha_client import ZerodhaClient
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def find_banknifty_futures(client):
    """Find Bank Nifty futures instruments."""
    try:
        logger.info("🔍 Searching for Bank Nifty futures instruments...")
        
        # Get NFO instruments (futures and options)
        nfo_instruments = client.kite.instruments("NFO")
        
        # Filter for Bank Nifty futures
        banknifty_futures = []
        for instrument in nfo_instruments:
            if (instrument.get('name', '').upper() == 'BANKNIFTY' and 
                instrument.get('instrument_type') == 'FUT'):
                banknifty_futures.append(instrument)
        
        logger.info(f"Found {len(banknifty_futures)} Bank Nifty futures contracts:")
        
        for future in banknifty_futures:
            logger.info(f"  📊 {future['tradingsymbol']} | Token: {future['instrument_token']} | Expiry: {future['expiry']} | Lot Size: {future['lot_size']}")
        
        # Get the current month contract (nearest expiry)
        if banknifty_futures:
            current_contract = min(banknifty_futures, key=lambda x: x['expiry'])
            logger.info(f"✅ Current month contract: {current_contract['tradingsymbol']} (Token: {current_contract['instrument_token']})")
            
            return current_contract
                
        return None
        
    except Exception as e:
        logger.error(f"Error finding Bank Nifty futures: {e}")
        return None


def test_volume_data(client, contract):
    """Test fetching volume data for the Bank Nifty futures contract."""
    logger.info(f"🧪 Testing volume data for {contract['tradingsymbol']} (Token: {contract['instrument_token']})")
    
    try:
        # Test fetching historical data with volume
        from datetime import date, timedelta
        
        test_data = client.kite.historical_data(
            instrument_token=contract['instrument_token'],
            from_date=date.today() - timedelta(days=5),
            to_date=date.today(),
            interval="minute"
        )
        
        if test_data:
            logger.info(f"📈 Retrieved {len(test_data)} minute bars")
            
            # Convert to DataFrame for analysis
            df = pd.DataFrame(test_data)
            df.columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            
            # Volume statistics
            total_volume = df['volume'].sum()
            avg_volume = df['volume'].mean()
            max_volume = df['volume'].max()
            non_zero_count = (df['volume'] > 0).sum()
            
            logger.info(f"📊 Volume Statistics:")
            logger.info(f"   Total volume: {total_volume:,}")
            logger.info(f"   Average volume per minute: {avg_volume:.2f}")
            logger.info(f"   Max volume in a minute: {max_volume:,}")
            logger.info(f"   Minutes with volume > 0: {non_zero_count}/{len(df)}")
            
            # Show last few records
            logger.info(f"📈 Last 3 records:")
            for _, row in df.tail(3).iterrows():
                logger.info(f"   {row['timestamp']}: Volume = {row['volume']:,}")
            
            return True
        else:
            logger.warning("No test data received")
            return False
            
    except Exception as e:
        logger.error(f"Error testing volume data: {e}")
        return False


def main():
    """Main function to discover Bank Nifty futures and test volume data."""
    logger.info("=" * 60)
    logger.info("Bank Nifty Futures Discovery Script")
    logger.info("=" * 60)
    
    # Initialize client first
    try:
        client = ZerodhaClient()
        logger.info("Successfully connected to Zerodha API")
    except Exception as e:
        logger.error(f"Failed to connect to Zerodha API: {e}")
        return
    
    # Find Bank Nifty futures contract
    contract = find_banknifty_futures(client)
    
    if contract is None:
        logger.error("❌ Failed to find Bank Nifty futures contract")
        return
    
    # Test volume data
    success = test_volume_data(client, contract)
    
    if success:
        logger.info("=" * 60)
        logger.info("✅ SUCCESS: Bank Nifty futures volume data is available!")
        logger.info(f"📋 Contract Details:")
        logger.info(f"   Trading Symbol: {contract['tradingsymbol']}")
        logger.info(f"   Instrument Token: {contract['instrument_token']}")
        logger.info(f"   Expiry: {contract['expiry']}")
        logger.info(f"   Lot Size: {contract['lot_size']}")
        logger.info("=" * 60)
    else:
        logger.error("❌ FAILED: Could not fetch volume data for Bank Nifty futures")

if __name__ == "__main__":
    main()
