#!/usr/bin/env python3
"""
Test the fixed implementation with OI data for a single day.
"""

import sys
import os
from pathlib import Path
import pandas as pd
from datetime import date, datetime
import time
import logging

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from src.data_ingest.zerodha_client import ZerodhaClient

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import the functions from our fixed implementation
exec(open(project_root / "scripts" / "task2_1_fixed_historical_implementation.py").read())

def test_oi_fix_single_day():
    """Test the OI fix for a single day."""
    
    print("🧪 Testing OI fix for single day data collection...")
    
    # Initialize client
    client = ZerodhaClient()
    
    if not client.kite or not client.access_token:
        print("❌ Client not authenticated")
        return False
    
    print("✅ Client authenticated")
    
    # Test for 2025-05-16 (we know this has data)
    test_date = date(2025, 5, 16)
    
    # Get expiry data
    print("📅 Getting expiry information...")
    expiry_info = get_banknifty_weekly_expiries(client)
    
    if not expiry_info["weekly_expiries"]:
        print("❌ No weekly expiries found")
        return False
    
    # Get the next expiry for our test date
    next_expiry = get_next_weekly_expiry_for_date(test_date, expiry_info)
    
    if not next_expiry:
        print("❌ No expiry found for test date")
        return False
    
    print(f"📅 Using expiry: {next_expiry}")
    
    # Define strike range around spot price (we know it was around 55350)
    spot_price = 55350
    min_strike = int((spot_price - 1000) // 100) * 100  # Smaller range for testing
    max_strike = int((spot_price + 1000) // 100 + 1) * 100
    strike_range = range(min_strike, max_strike + 1, 100)
    
    print(f"📊 Testing {len(strike_range)} strikes around spot {spot_price}")
    
    # Fetch option chain with OI fix
    print("🔍 Fetching option chain with OI fix...")
    option_records = fetch_option_chain_snapshot_historical(client, next_expiry, strike_range, test_date)
    
    if not option_records:
        print("❌ No option records fetched")
        return False
    
    print(f"✅ Fetched {len(option_records)} option records")
    
    # Check if OI data is present and non-zero
    records_with_oi = [r for r in option_records if r.get('oi', 0) > 0]
    total_oi = sum(r.get('oi', 0) for r in option_records)
    
    print(f"📊 Records with OI > 0: {len(records_with_oi)}/{len(option_records)}")
    print(f"📊 Total OI across all options: {total_oi:,}")
    
    # Show sample records
    print("\n📋 Sample records:")
    for i, record in enumerate(option_records[:5]):  # Show first 5
        print(f"   {i+1}. {record['symbol']} | LTP: {record['ltp']} | Volume: {record['volume']:,} | OI: {record['oi']:,}")
    
    if total_oi > 0:
        print("\n🎉 SUCCESS: OI fix is working! Historical OI data is now being collected.")
        return True
    else:
        print("\n❌ FAILURE: OI values are still zero.")
        return False

if __name__ == "__main__":
    success = test_oi_fix_single_day()
    
    if success:
        print("\n🎯 The OI fix is confirmed working. Ready to re-run full data collection!")
    else:
        print("\n💥 The OI fix needs further investigation.")
