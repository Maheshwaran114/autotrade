#!/usr/bin/env python3
"""
Test script to discover how to get historical Bank Nifty futures contracts
for different time periods in the past year.
"""

import logging
import sys
from pathlib import Path
from datetime import datetime, date, timedelta
import pandas as pd

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from src.data_ingest.zerodha_client import ZerodhaClient

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_historical_futures_availability():
    """Test availability of Bank Nifty futures data for different historical periods."""
    
    # Initialize client
    client = ZerodhaClient(credentials_path=str(project_root / "config" / "credentials.json"))
    
    if not client.login():
        logger.error("❌ Authentication failed")
        return False
    
    logger.info("✅ Connected to Zerodha API")
    
    # Get all Bank Nifty futures contracts
    try:
        nfo_instruments = client.kite.instruments("NFO")
        banknifty_futures = [
            inst for inst in nfo_instruments 
            if inst.get('name', '').upper() == 'BANKNIFTY' and inst.get('instrument_type') == 'FUT'
        ]
        
        logger.info(f"📊 Found {len(banknifty_futures)} Bank Nifty futures contracts")
        
        # Sort by expiry to see all available contracts
        sorted_futures = sorted(banknifty_futures, key=lambda x: x['expiry'])
        
        logger.info("📅 Available Bank Nifty futures contracts:")
        for contract in sorted_futures:
            logger.info(f"  - {contract['tradingsymbol']} (Token: {contract['instrument_token']}) Expiry: {contract['expiry']}")
            
        # Test data availability for different time periods
        test_dates = [
            date(2024, 6, 1),   # 1 year ago
            date(2024, 9, 1),   # 8 months ago  
            date(2024, 12, 1),  # 5 months ago
            date(2025, 3, 1),   # 2 months ago
            date(2025, 5, 1),   # Current month
        ]
        
        logger.info("🔍 Testing data availability for different periods...")
        
        for test_date in test_dates:
            logger.info(f"\n📅 Testing data for {test_date}...")
            
            # Find the contract that was active during this date
            active_contracts = [
                contract for contract in sorted_futures
                if contract['expiry'] >= test_date
            ]
            
            if not active_contracts:
                logger.warning(f"  ⚠️ No active contracts found for {test_date}")
                continue
            
            # Use the nearest expiry contract for that date
            nearest_contract = min(active_contracts, key=lambda x: x['expiry'])
            
            logger.info(f"  📊 Testing contract: {nearest_contract['tradingsymbol']} (Expiry: {nearest_contract['expiry']})")
            
            try:
                # Test fetching 1 day of data
                historical_data = client.kite.historical_data(
                    instrument_token=nearest_contract['instrument_token'],
                    from_date=test_date,
                    to_date=test_date + timedelta(days=1),
                    interval="minute"
                )
                
                if historical_data:
                    logger.info(f"  ✅ Got {len(historical_data)} records for {test_date}")
                    
                    # Check volume data
                    if len(historical_data) > 0:
                        sample_record = historical_data[0]
                        logger.info(f"  📊 Sample record: {sample_record}")
                        
                        if len(sample_record) >= 6:  # Should have volume
                            total_volume = sum(record[5] if len(record) > 5 else 0 for record in historical_data)
                            logger.info(f"  📈 Total volume for day: {total_volume:,}")
                        
                else:
                    logger.warning(f"  ⚠️ No data available for {test_date}")
                    
            except Exception as e:
                logger.error(f"  ❌ Error fetching data for {test_date}: {e}")
                
        return True
        
    except Exception as e:
        logger.error(f"❌ Failed to get futures contracts: {e}")
        return False

if __name__ == "__main__":
    logger.info("=" * 70)
    logger.info("Testing Historical Bank Nifty Futures Data Availability")
    logger.info("=" * 70)
    
    success = test_historical_futures_availability()
    
    if success:
        logger.info("✅ Test completed successfully")
    else:
        logger.error("❌ Test failed")
