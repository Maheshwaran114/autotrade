#!/usr/bin/env python3
"""
Quick test script to check Zerodha API data availability for May 27th, 2024
This will help us understand actual data limits for:
1. Bank Nifty futures volume data
2. Bank Nifty options data
"""

import sys
from pathlib import Path
from datetime import datetime, date
import pandas as pd

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from src.data_ingest.zerodha_client import ZerodhaClient

def test_data_availability():
    """Test data availability for May 27th, 2024"""
    
    # Test date: May 27th, 2024 (exactly 1 year ago)
    test_date = date(2024, 5, 27)
    current_date = datetime.now().date()
    days_diff = (current_date - test_date).days
    
    print(f"🧪 Testing Zerodha API Data Availability")
    print(f"📅 Test Date: {test_date} ({days_diff} days ago)")
    print("=" * 60)
    
    # Initialize client
    client = ZerodhaClient(credentials_path=str(project_root / "config" / "credentials.json"))
    
    if not client.login():
        print("❌ Authentication failed")
        return False
    
    print("✅ Successfully authenticated with Zerodha API")
    print()
    
    # Test 1: Bank Nifty Index Data (should be available)
    print("🏦 TEST 1: Bank Nifty Index Data")
    print("-" * 40)
    try:
        # Get Bank Nifty instrument token
        instruments = client.kite.instruments("NSE")
        instruments_df = pd.DataFrame(instruments)
        
        # Find Bank Nifty index
        bank_nifty = instruments_df[
            (instruments_df['name'] == 'NIFTY BANK') & 
            (instruments_df['segment'] == 'INDICES')
        ]
        
        if not bank_nifty.empty:
            bank_nifty_token = bank_nifty.iloc[0]['instrument_token']
            print(f"✅ Found Bank Nifty Index token: {bank_nifty_token}")
            
            # Test historical data for May 27th, 2024
            historical_data = client.kite.historical_data(
                instrument_token=bank_nifty_token,
                from_date=test_date,
                to_date=test_date,
                interval="minute"
            )
            
            if historical_data:
                print(f"✅ Bank Nifty Index data AVAILABLE for {test_date}")
                print(f"📊 Records: {len(historical_data)}")
                print(f"📊 Sample: OHLC = {historical_data[0]['open']}, {historical_data[0]['high']}, {historical_data[0]['low']}, {historical_data[0]['close']}")
                print(f"📊 Volume: {historical_data[0].get('volume', 'N/A')}")
            else:
                print(f"❌ Bank Nifty Index data NOT AVAILABLE for {test_date}")
        else:
            print("❌ Bank Nifty Index instrument not found")
            
    except Exception as e:
        print(f"❌ Error testing Bank Nifty Index: {e}")
    
    print()
    
    # Test 2: Bank Nifty Futures Volume Data
    print("📈 TEST 2: Bank Nifty Futures Volume Data")
    print("-" * 45)
    try:
        # Get NFO instruments for futures
        nfo_instruments = client.kite.instruments("NFO")
        nfo_df = pd.DataFrame(nfo_instruments)
        
        # Find Bank Nifty futures around May 2024
        bank_nifty_futures = nfo_df[
            (nfo_df['name'] == 'BANKNIFTY') & 
            (nfo_df['instrument_type'] == 'FUT')
        ].copy()
        
        if not bank_nifty_futures.empty:
            # Convert expiry to date for filtering
            bank_nifty_futures['expiry_date'] = pd.to_datetime(bank_nifty_futures['expiry']).dt.date
            
            # Find futures that were active around May 27, 2024
            # (expiry should be after May 27, 2024)
            active_futures = bank_nifty_futures[
                bank_nifty_futures['expiry_date'] > test_date
            ].sort_values('expiry_date')
            
            if not active_futures.empty:
                front_future = active_futures.iloc[0]
                print(f"✅ Found front-month future: {front_future['tradingsymbol']} (expiry: {front_future['expiry_date']})")
                
                # Test historical data for futures volume
                historical_data = client.kite.historical_data(
                    instrument_token=front_future['instrument_token'],
                    from_date=test_date,
                    to_date=test_date,
                    interval="minute"
                )
                
                if historical_data:
                    volume_data = [record.get('volume', 0) for record in historical_data]
                    total_volume = sum(volume_data)
                    non_zero_volume = sum(1 for v in volume_data if v > 0)
                    
                    print(f"✅ Bank Nifty Futures volume data AVAILABLE for {test_date}")
                    print(f"📊 Records: {len(historical_data)}")
                    print(f"📊 Total Volume: {total_volume:,}")
                    print(f"📊 Non-zero volume records: {non_zero_volume}/{len(historical_data)} ({non_zero_volume/len(historical_data)*100:.1f}%)")
                    print(f"📊 Sample volume: {volume_data[:5]}")
                else:
                    print(f"❌ Bank Nifty Futures volume data NOT AVAILABLE for {test_date}")
            else:
                print(f"❌ No active Bank Nifty futures found for {test_date}")
        else:
            print("❌ Bank Nifty Futures instruments not found")
            
    except Exception as e:
        print(f"❌ Error testing Bank Nifty Futures: {e}")
    
    print()
    
    # Test 3: Bank Nifty Options Data
    print("🎯 TEST 3: Bank Nifty Options Data")
    print("-" * 40)
    try:
        # Find Bank Nifty options around May 2024
        bank_nifty_options = nfo_df[
            (nfo_df['name'] == 'BANKNIFTY') & 
            (nfo_df['instrument_type'].isin(['CE', 'PE']))
        ].copy()
        
        if not bank_nifty_options.empty:
            # Convert expiry to date
            bank_nifty_options['expiry_date'] = pd.to_datetime(bank_nifty_options['expiry']).dt.date
            
            # Find options expiry closest to (but after) May 27, 2024
            future_expiries = bank_nifty_options[
                bank_nifty_options['expiry_date'] > test_date
            ]['expiry_date'].unique()
            
            if len(future_expiries) > 0:
                closest_expiry = min(future_expiries)
                print(f"✅ Found closest option expiry: {closest_expiry}")
                
                # Get options for this expiry
                expiry_options = bank_nifty_options[
                    bank_nifty_options['expiry_date'] == closest_expiry
                ]
                
                # Test a few sample options (ATM strikes)
                strikes = sorted(expiry_options['strike'].unique())
                middle_idx = len(strikes) // 2
                test_strikes = strikes[middle_idx-2:middle_idx+3] if len(strikes) >= 5 else strikes[:3]
                
                print(f"📊 Testing {len(test_strikes)} sample strikes: {test_strikes}")
                
                successful_options = 0
                total_tested = 0
                
                for strike in test_strikes:
                    # Test both CE and PE
                    for opt_type in ['CE', 'PE']:
                        option = expiry_options[
                            (expiry_options['strike'] == strike) & 
                            (expiry_options['instrument_type'] == opt_type)
                        ]
                        
                        if not option.empty:
                            option_record = option.iloc[0]
                            total_tested += 1
                            
                            try:
                                # Test historical data for this option
                                historical_data = client.kite.historical_data(
                                    instrument_token=option_record['instrument_token'],
                                    from_date=test_date,
                                    to_date=test_date,
                                    interval="day",
                                    oi=1  # Include Open Interest
                                )
                                
                                if historical_data and len(historical_data) > 0:
                                    data = historical_data[0]
                                    successful_options += 1
                                    print(f"✅ {option_record['tradingsymbol']}: Close={data.get('close', 0)}, Volume={data.get('volume', 0)}, OI={data.get('oi', 0)}")
                                else:
                                    print(f"❌ {option_record['tradingsymbol']}: No data available")
                                    
                            except Exception as e:
                                print(f"❌ {option_record['tradingsymbol']}: Error - {e}")
                
                success_rate = (successful_options / total_tested * 100) if total_tested > 0 else 0
                print(f"\n📊 Options Data Test Results:")
                print(f"   - Successful: {successful_options}/{total_tested} ({success_rate:.1f}%)")
                
                if successful_options > 0:
                    print(f"✅ Bank Nifty Options data PARTIALLY/FULLY AVAILABLE for {test_date}")
                else:
                    print(f"❌ Bank Nifty Options data NOT AVAILABLE for {test_date}")
                    
            else:
                print(f"❌ No future option expiries found for {test_date}")
        else:
            print("❌ Bank Nifty Options instruments not found")
            
    except Exception as e:
        print(f"❌ Error testing Bank Nifty Options: {e}")
    
    print()
    print("=" * 60)
    print("🏁 Data Availability Test Completed")
    
    return True

if __name__ == "__main__":
    test_data_availability()
