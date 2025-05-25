#!/usr/bin/env python3
"""
Test script to verify the OI fix works for a single option.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from src.data_ingest.zerodha_client import ZerodhaClient
from datetime import date
import pandas as pd
import time

def test_single_option_oi():
    """Test OI data collection for a single option to verify the fix."""
    
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
    
    # Test the fixed historical data collection
    trade_date = date(2025, 5, 16)
    instrument_token = test_option['instrument_token']
    symbol = test_option['tradingsymbol']
    strike = test_option['strike']
    option_type = test_option['instrument_type']
    
    print(f"\n🔍 Fetching historical data with OI for {symbol} on {trade_date}...")
    
    try:
        # This is the same logic as in our fixed implementation
        historical_data = client.kite.historical_data(
            instrument_token=instrument_token,
            from_date=trade_date,
            to_date=trade_date,
            interval="day",
            oi=1  # CRITICAL FIX: Add oi=1 to get Open Interest data
        )
        
        if historical_data and len(historical_data) > 0:
            # Get the EOD data
            eod_data = historical_data[0]  # Should be only one record for EOD
            
            print("📊 Raw historical data response:")
            print(f"   {eod_data}")
            
            # Extract data using same logic as our implementation
            record = {
                "symbol": f"NFO:{symbol}",
                "strike": float(strike),
                "instrument_type": option_type,
                "expiry": date(2025, 5, 29),
                "date": trade_date,
                "last_price": eod_data.get("close", 0.0),
                "ltp": eod_data.get("close", 0.0),
                "open": eod_data.get("open", 0.0),
                "high": eod_data.get("high", 0.0),
                "low": eod_data.get("low", 0.0),
                "close": eod_data.get("close", 0.0),
                "volume": eod_data.get("volume", 0),
                "oi": eod_data.get("oi", 0),  # This should now have real OI data
                "bid": 0.0,  # Historical data doesn't have bid/ask
                "ask": 0.0,
            }
            
            print("\n✅ Processed record:")
            print(f"   Symbol: {record['symbol']}")
            print(f"   Strike: {record['strike']}")
            print(f"   Type: {record['instrument_type']}")
            print(f"   Date: {record['date']}")
            print(f"   LTP: {record['ltp']}")
            print(f"   Volume: {record['volume']:,}")
            print(f"   OI: {record['oi']:,} ← THIS SHOULD NOT BE ZERO!")
            
            if record['oi'] > 0:
                print("🎉 SUCCESS: OI data is now properly fetched!")
                return True
            else:
                print("❌ FAILURE: OI is still zero")
                return False
                
        else:
            print(f"⚠️  No historical data found for {symbol} on {trade_date}")
            return False
    
    except Exception as e:
        print(f"❌ Error fetching historical data: {e}")
        return False

if __name__ == "__main__":
    success = test_single_option_oi()
    if success:
        print("\n🎯 The OI fix is working! Now let's run it on our full dataset.")
    else:
        print("\n💥 The OI fix needs more work.")
