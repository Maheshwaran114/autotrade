#!/usr/bin/env python
"""Simple test to verify API and run basic data collection"""

import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from src.data_ingest.zerodha_client import ZerodhaClient
import pandas as pd
from datetime import datetime, timedelta

print("🔍 Testing API connection...")

# Initialize client
client = ZerodhaClient(credentials_path=str(project_root / "config" / "credentials.json"))

# Test login
if client.login():
    print("✅ API authentication successful!")
    
    # Test a simple data fetch
    print("📊 Testing data fetch...")
    try:
        # Try to get instrument list (simple test)
        instruments = client.kite.instruments("NSE")
        print(f"✅ Successfully retrieved {len(instruments)} NSE instruments")
        
        # Test Bank Nifty spot data fetch
        print("🏦 Testing Bank Nifty data...")
        end_date = datetime.now()
        start_date = end_date - timedelta(days=5)  # Just 5 days for testing
        
        banknifty_data = client.fetch_historical_data(
            instrument="NSE:NIFTY BANK",
            interval="day",  # Start with daily data for testing
            from_date=start_date.date(),
            to_date=end_date.date()
        )
        
        if banknifty_data:
            print(f"✅ Successfully retrieved {len(banknifty_data)} Bank Nifty records")
            print("Sample record:", banknifty_data[0] if banknifty_data else "None")
        else:
            print("⚠️ No Bank Nifty data retrieved")
            
    except Exception as e:
        print(f"❌ Error testing data fetch: {e}")
        
else:
    print("❌ API authentication failed!")
    sys.exit(1)

print("\n✅ Basic tests completed!")
