#!/usr/bin/env python3
"""
Test script to check the format of historical data returned by Kite Connect API with oi=1 parameter.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from src.data_ingest.zerodha_client import ZerodhaClient
from datetime import date
import pandas as pd

def test_historical_data_format():
    """Test the format of historical data with OI parameter."""
    
    # Initialize client
    client = ZerodhaClient()
    
    if not client.kite or not client.access_token:
        print("❌ Client not authenticated. Please check credentials.")
        return
        
    print("✅ Client authenticated successfully")
    
    # Get NFO instruments to find a Bank Nifty option
    print("📥 Loading NFO instruments...")
    nfo_instruments = client.kite.instruments("NFO")
    nfo_df = pd.DataFrame(nfo_instruments)
    
    # Find a Bank Nifty option that should have been active on 2025-05-16
    banknifty_opts = nfo_df[
        (nfo_df['name'] == 'BANKNIFTY') & 
        (nfo_df['instrument_type'] == 'CE') &
        (nfo_df['strike'] == 55400.0) &
        (pd.to_datetime(nfo_df['expiry']).dt.date == date(2025, 5, 29))
    ]
    
    if banknifty_opts.empty:
        print("❌ No suitable test option found")
        return
        
    test_option = banknifty_opts.iloc[0]
    print(f"🧪 Testing with option: {test_option['tradingsymbol']} (token: {test_option['instrument_token']})")
    
    # Test historical data WITHOUT oi parameter
    print("\n🔍 Testing historical data WITHOUT oi=1...")
    try:
        historical_data_no_oi = client.kite.historical_data(
            instrument_token=test_option['instrument_token'],
            from_date=date(2025, 5, 16),
            to_date=date(2025, 5, 16),
            interval="day"
        )
        print("📊 Data without OI:")
        print(f"   Type: {type(historical_data_no_oi)}")
        if historical_data_no_oi:
            print(f"   Length: {len(historical_data_no_oi)}")
            print(f"   First record: {historical_data_no_oi[0]}")
            print(f"   First record type: {type(historical_data_no_oi[0])}")
            if isinstance(historical_data_no_oi[0], (list, tuple)):
                print(f"   Record length: {len(historical_data_no_oi[0])}")
                print(f"   Elements: {historical_data_no_oi[0]}")
    except Exception as e:
        print(f"❌ Error fetching data without OI: {e}")
    
    # Test historical data WITH oi parameter
    print("\n🔍 Testing historical data WITH oi=1...")
    try:
        historical_data_with_oi = client.kite.historical_data(
            instrument_token=test_option['instrument_token'],
            from_date=date(2025, 5, 16),
            to_date=date(2025, 5, 16),
            interval="day",
            oi=1
        )
        print("📊 Data with OI:")
        print(f"   Type: {type(historical_data_with_oi)}")
        if historical_data_with_oi:
            print(f"   Length: {len(historical_data_with_oi)}")
            print(f"   First record: {historical_data_with_oi[0]}")
            print(f"   First record type: {type(historical_data_with_oi[0])}")
            if isinstance(historical_data_with_oi[0], (list, tuple)):
                print(f"   Record length: {len(historical_data_with_oi[0])}")
                print(f"   Elements: {historical_data_with_oi[0]}")
    except Exception as e:
        print(f"❌ Error fetching data with OI: {e}")

if __name__ == "__main__":
    test_historical_data_format()
