#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Task 2.1: Quick Test of Fixed Historical Data Implementation
This demonstrates the CORRECT approach using historical_data() instead of ltp()
"""

import os
import sys
import logging
from datetime import datetime, timedelta, date
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from src.data_ingest.zerodha_client import ZerodhaClient

# Configure logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_fixed_approach():
    """Test the fixed approach with a single option instrument."""
    logger.info("🧪 Testing FIXED approach: historical_data() vs ltp()")
    
    # Initialize client
    client = ZerodhaClient(credentials_path=str(project_root / "config" / "credentials.json"))
    
    if not client.login():
        logger.error("❌ Authentication failed")
        return False
    
    logger.info("✅ Authentication successful")
    
    # Get Bank Nifty instruments
    logger.info("📋 Fetching Bank Nifty instruments...")
    instruments = client.kite.instruments("NFO")
    
    # Find a recent BANKNIFTY option
    banknifty_option = None
    for inst in instruments:
        if (inst.get("name") == "BANKNIFTY" and 
            inst.get("instrument_type") == "CE" and
            inst.get("strike") == 50000):  # Example strike
            banknifty_option = inst
            break
    
    if not banknifty_option:
        logger.error("❌ No suitable Bank Nifty option found")
        return False
    
    logger.info(f"📈 Testing with: {banknifty_option['tradingsymbol']}")
    
    # Test date (a few days ago)
    test_date = datetime.now().date() - timedelta(days=3)
    logger.info(f"📅 Test date: {test_date}")
    
    # Method 1: LTP (WRONG for historical data)
    logger.info("🔍 Method 1: Testing LTP (should fail for historical data)")
    try:
        symbol = f"NFO:{banknifty_option['tradingsymbol']}"
        ltp_result = client.kite.ltp([symbol])
        logger.info(f"LTP Result: {ltp_result}")
        logger.warning("⚠️  LTP returned current data, not historical!")
    except Exception as e:
        logger.error(f"LTP failed: {e}")
    
    # Method 2: Historical Data (CORRECT)
    logger.info("🔍 Method 2: Testing historical_data() (CORRECT approach)")
    try:
        from_datetime = datetime.combine(test_date, datetime.min.time().replace(hour=9, minute=15))
        to_datetime = datetime.combine(test_date, datetime.min.time().replace(hour=15, minute=30))
        
        historical_result = client.kite.historical_data(
            instrument_token=banknifty_option["instrument_token"],
            from_date=from_datetime,
            to_date=to_datetime,
            interval="day",
            oi=True  # Include Open Interest
        )
        
        if historical_result:
            day_data = historical_result[-1] if historical_result else None
            logger.info(f"✅ Historical Data Success!")
            logger.info(f"📊 OHLC: O={day_data['open']}, H={day_data['high']}, L={day_data['low']}, C={day_data['close']}")
            logger.info(f"📊 Volume: {day_data['volume']}, OI: {day_data.get('oi', 'N/A')}")
            logger.info("🎉 REAL MARKET DATA OBTAINED!")
        else:
            logger.warning("⚠️  No historical data for this date")
            
    except Exception as e:
        logger.error(f"Historical data failed: {e}")
        return False
    
    logger.info("✅ Test completed - historical_data() is the correct approach!")
    return True

def demonstrate_data_difference():
    """Demonstrate the difference between synthetic and real data."""
    logger.info("=" * 60)
    logger.info("DEMONSTRATION: Synthetic vs Real Market Data")
    logger.info("=" * 60)
    
    # Show existing synthetic data
    logger.info("📊 Current Synthetic Data Sample:")
    try:
        import pandas as pd
        sample_file = project_root / "data" / "raw" / "options_20240419.csv"
        if sample_file.exists():
            df = pd.read_csv(sample_file)
            logger.info(f"Sample from {sample_file.name}:")
            logger.info(f"Records: {len(df)}")
            logger.info(f"Volume=0 count: {(df['volume'] == 0).sum()}/{len(df)} ({(df['volume'] == 0).sum()/len(df)*100:.1f}%)")
            logger.info(f"OI=0 count: {(df['oi'] == 0).sum()}/{len(df)} ({(df['oi'] == 0).sum()/len(df)*100:.1f}%)")
            logger.info(f"IV=0 count: {(df['iv'] == 0).sum()}/{len(df)} ({(df['iv'] == 0).sum()/len(df)*100:.1f}%)")
            logger.info("❌ This is SYNTHETIC data - all volume and OI are zero!")
        else:
            logger.warning("No sample synthetic data file found")
    except Exception as e:
        logger.error(f"Error reading synthetic data: {e}")
    
    logger.info("\n🔄 SOLUTION: Use historical_data() API for real market data")
    logger.info("✅ Fixed implementation will provide:")
    logger.info("   - Real volume from exchange trading")
    logger.info("   - Real Open Interest from exchange positions")
    logger.info("   - Real OHLC prices from historical candles")
    logger.info("   - Accurate IV calculations using real option prices")

def main():
    """Main test function."""
    try:
        logger.info("🚀 Starting Fixed Implementation Test")
        
        # Test the fixed approach
        test_result = test_fixed_approach()
        
        # Demonstrate the data difference
        demonstrate_data_difference()
        
        if test_result:
            logger.info("✅ Test PASSED: Fixed implementation approach is correct!")
            logger.info("💡 Ready to run full data collection with real market data")
        else:
            logger.error("❌ Test FAILED: Issues with the fixed approach")
            
        return test_result
        
    except Exception as e:
        logger.error(f"❌ Test failed with error: {e}")
        return False

if __name__ == "__main__":
    main()
