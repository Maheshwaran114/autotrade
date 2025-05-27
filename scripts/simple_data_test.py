#!/usr/bin/env python3
"""
Simple test for Zerodha API data availability on May 27th, 2024
"""

import sys
from pathlib import Path
from datetime import datetime, date
import pandas as pd

# Add project root to Python path
sys.path.append('.')
from src.data_ingest.zerodha_client import ZerodhaClient

def main():
    test_date = date(2024, 5, 27)
    days_ago = (datetime.now().date() - test_date).days
    
    print(f"🧪 Testing Zerodha API for {test_date} ({days_ago} days ago)")
    
    # Initialize client
    client = ZerodhaClient(credentials_path='config/credentials.json')
    if not client.login():
        print("❌ Authentication failed")
        return
    
    print("✅ Authenticated successfully")
    
    # Test 1: Bank Nifty Index
    print("\n📊 Test 1: Bank Nifty Index Data")
    try:
        instruments = client.kite.instruments("NSE")
        df = pd.DataFrame(instruments)
        bnf_idx = df[(df['name'] == 'NIFTY BANK') & (df['segment'] == 'INDICES')]
        
        if not bnf_idx.empty:
            token = bnf_idx.iloc[0]['instrument_token']
            print(f"Found Bank Nifty token: {token}")
            
            data = client.kite.historical_data(token, test_date, test_date, "minute")
            if data:
                print(f"✅ Bank Nifty Index: {len(data)} records available")
                print(f"Sample: Open={data[0]['open']}, Close={data[0]['close']}, Volume={data[0].get('volume', 'N/A')}")
            else:
                print("❌ Bank Nifty Index: No data")
        else:
            print("❌ Bank Nifty Index: Instrument not found")
    except Exception as e:
        print(f"❌ Bank Nifty Index error: {e}")
    
    # Test 2: Bank Nifty Futures Volume
    print("\n📈 Test 2: Bank Nifty Futures Volume")
    try:
        nfo_instruments = client.kite.instruments("NFO")
        nfo_df = pd.DataFrame(nfo_instruments)
        
        # Find futures
        futures = nfo_df[(nfo_df['name'] == 'BANKNIFTY') & (nfo_df['instrument_type'] == 'FUT')].copy()
        futures['expiry_date'] = pd.to_datetime(futures['expiry']).dt.date
        
        # Find active future for May 27, 2024
        active = futures[futures['expiry_date'] > test_date].sort_values('expiry_date')
        
        if not active.empty:
            future = active.iloc[0]
            print(f"Found future: {future['tradingsymbol']} (expiry: {future['expiry_date']})")
            
            data = client.kite.historical_data(future['instrument_token'], test_date, test_date, "minute")
            if data:
                volumes = [r.get('volume', 0) for r in data]
                total_vol = sum(volumes)
                non_zero = sum(1 for v in volumes if v > 0)
                print(f"✅ Futures Volume: {len(data)} records, Total Volume: {total_vol:,}")
                print(f"Non-zero volume: {non_zero}/{len(data)} ({non_zero/len(data)*100:.1f}%)")
            else:
                print("❌ Futures Volume: No data")
        else:
            print("❌ Futures Volume: No active futures found")
    except Exception as e:
        print(f"❌ Futures Volume error: {e}")
    
    # Test 3: Bank Nifty Options
    print("\n🎯 Test 3: Bank Nifty Options")
    try:
        # Find options
        options = nfo_df[(nfo_df['name'] == 'BANKNIFTY') & (nfo_df['instrument_type'].isin(['CE', 'PE']))].copy()
        options['expiry_date'] = pd.to_datetime(options['expiry']).dt.date
        
        # Find closest expiry after test date
        future_expiries = options[options['expiry_date'] > test_date]['expiry_date'].unique()
        
        if len(future_expiries) > 0:
            closest_expiry = min(future_expiries)
            print(f"Testing options with expiry: {closest_expiry}")
            
            # Get options for this expiry
            expiry_opts = options[options['expiry_date'] == closest_expiry]
            
            # Test 3 sample options
            strikes = sorted(expiry_opts['strike'].unique())
            test_strikes = strikes[len(strikes)//2-1:len(strikes)//2+2]  # Middle 3 strikes
            
            success_count = 0
            total_tested = 0
            
            for strike in test_strikes[:3]:  # Test only 3 strikes to save time
                ce_opt = expiry_opts[(expiry_opts['strike'] == strike) & (expiry_opts['instrument_type'] == 'CE')]
                
                if not ce_opt.empty:
                    opt = ce_opt.iloc[0]
                    total_tested += 1
                    
                    try:
                        data = client.kite.historical_data(opt['instrument_token'], test_date, test_date, "day", oi=1)
                        if data and len(data) > 0:
                            record = data[0]
                            success_count += 1
                            print(f"✅ {opt['tradingsymbol']}: Close={record.get('close', 0)}, Vol={record.get('volume', 0)}, OI={record.get('oi', 0)}")
                        else:
                            print(f"❌ {opt['tradingsymbol']}: No data")
                    except Exception as e:
                        print(f"❌ {opt['tradingsymbol']}: Error - {str(e)[:50]}...")
            
            if total_tested > 0:
                success_rate = (success_count / total_tested) * 100
                print(f"\nOptions Summary: {success_count}/{total_tested} successful ({success_rate:.1f}%)")
            else:
                print("❌ No options tested")
        else:
            print("❌ No future option expiries found")
    except Exception as e:
        print(f"❌ Options error: {e}")
    
    print(f"\n🏁 Test completed for {test_date}")

if __name__ == "__main__":
    main()
