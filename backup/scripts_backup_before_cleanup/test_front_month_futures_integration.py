#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Test script for Front-Month Futures Volume Integration
This validates the industry standard approach before full data collection.
"""

import os
import sys
import logging
import pandas as pd
from datetime import datetime, timedelta, date
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

# Import Zerodha client
from src.data_ingest.zerodha_client import ZerodhaClient

# Configure logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def api_call_with_retry(func, *args, max_retries=3, base_delay=1, **kwargs):
    """API call with exponential backoff retry logic"""
    import time
    import numpy as np
    
    for attempt in range(max_retries):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            error_str = str(e).lower()
            is_retryable = any(keyword in error_str for keyword in [
                'timeout', 'connection', 'network', 'rate limit', 'too many requests'
            ])
            
            if not is_retryable or attempt == max_retries - 1:
                logger.error(f"❌ API call failed after {attempt + 1} attempts: {e}")
                raise e
            
            delay = base_delay * (2 ** attempt) + np.random.uniform(0, 1)
            logger.warning(f"⚠️  Retrying in {delay:.1f}s (attempt {attempt + 1}/{max_retries}): {e}")
            time.sleep(delay)

def get_front_future_for_date(client, trading_date: date):
    """Get front-month Bank Nifty futures contract for a given date"""
    try:
        logger.info(f"🔍 Finding front-month futures contract for {trading_date}")
        
        # Get all NFO instruments (where Bank Nifty futures are traded)
        nfo_instruments = api_call_with_retry(client.kite.instruments, "NFO")
        nfo_df = pd.DataFrame(nfo_instruments)
        
        # Filter for Bank Nifty futures
        banknifty_futures = nfo_df[
            (nfo_df['name'] == 'BANKNIFTY') & 
            (nfo_df['instrument_type'] == 'FUT')
        ].copy()
        
        if banknifty_futures.empty:
            logger.error("❌ No Bank Nifty futures contracts found")
            return None
        
        # Convert expiry to date
        banknifty_futures['expiry_date'] = pd.to_datetime(banknifty_futures['expiry']).dt.date
        
        # Filter valid contracts (expiry >= trading_date)
        valid_contracts = banknifty_futures[
            banknifty_futures['expiry_date'] >= trading_date
        ].copy()
        
        if valid_contracts.empty:
            logger.warning(f"⚠️ No valid futures contracts for {trading_date}")
            return None
        
        # Get front-month (nearest expiry)
        front_contract = valid_contracts.sort_values('expiry_date').iloc[0]
        
        contract_info = {
            'instrument_token': front_contract['instrument_token'],
            'tradingsymbol': front_contract['tradingsymbol'],
            'expiry': front_contract['expiry_date'],
            'exchange': front_contract['exchange']
        }
        
        logger.info(f"✅ Front-month: {contract_info['tradingsymbol']} (expires: {contract_info['expiry']})")
        return contract_info
        
    except Exception as e:
        logger.error(f"❌ Error finding front-month futures: {e}")
        return None

def test_front_month_integration():
    """Test the front-month futures volume integration"""
    logger.info("🧪 Testing Front-Month Futures Volume Integration")
    logger.info("=" * 60)
    
    try:
        # Initialize client
        client = ZerodhaClient()
        
        # Test dates - a few historical dates and current
        test_dates = [
            date(2024, 5, 25),  # Historical date 1
            date(2024, 8, 15),  # Historical date 2  
            date(2024, 12, 20), # Historical date 3
            date.today()        # Current date
        ]
        
        for test_date in test_dates:
            logger.info(f"\n📅 Testing date: {test_date}")
            
            # Get front-month contract
            front_contract = get_front_future_for_date(client, test_date)
            
            if front_contract:
                logger.info(f"📊 Contract: {front_contract['tradingsymbol']}")
                logger.info(f"🏷️  Token: {front_contract['instrument_token']}")
                logger.info(f"⏰ Expires: {front_contract['expiry']}")
                
                # Test fetching volume data for one day
                try:
                    logger.info(f"📈 Testing volume data fetch...")
                    futures_data = api_call_with_retry(
                        client.kite.historical_data,
                        instrument_token=front_contract['instrument_token'],
                        from_date=test_date,
                        to_date=test_date,
                        interval="minute",
                        oi=1
                    )
                    
                    if futures_data:
                        futures_df = pd.DataFrame(futures_data)
                        futures_df.columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume', 'oi']
                        
                        total_volume = futures_df['volume'].sum()
                        avg_volume = futures_df['volume'].mean()
                        records_count = len(futures_df)
                        
                        logger.info(f"✅ Volume data: {records_count} records")
                        logger.info(f"📊 Total volume: {total_volume:,.0f}")
                        logger.info(f"📊 Avg volume/min: {avg_volume:,.0f}")
                        
                        if total_volume > 0:
                            logger.info("✅ Volume data is valid!")
                        else:
                            logger.warning("⚠️ Volume is zero - may be holiday/weekend")
                            
                    else:
                        logger.warning("⚠️ No futures data available for this date")
                        
                except Exception as e:
                    logger.warning(f"⚠️ Could not fetch volume data: {e}")
                    
            else:
                logger.warning(f"⚠️ No front-month contract found for {test_date}")
            
            logger.info("-" * 40)
    
    except Exception as e:
        logger.error(f"❌ Test failed: {e}")
        import traceback
        logger.debug(f"Full traceback: {traceback.format_exc()}")

def test_combined_data_approach():
    """Test combining index OHLC with futures volume"""
    logger.info("\n🔗 Testing Combined Index + Futures Volume Approach")
    logger.info("=" * 60)
    
    try:
        client = ZerodhaClient()
        test_date = date(2024, 6, 20)  # A business day
        
        logger.info(f"📅 Testing combined approach for {test_date}")
        
        # Get front-month contract
        front_contract = get_front_future_for_date(client, test_date)
        
        if not front_contract:
            logger.error("❌ No front-month contract available")
            return
        
        # Fetch index data (OHLC)
        logger.info("📈 Fetching Bank Nifty index OHLC data...")
        try:
            index_data = client.fetch_historical_data(
                instrument="NSE:NIFTY BANK",
                interval="minute",
                from_date=test_date,
                to_date=test_date
            )
        except:
            # Alternative method if the above fails
            index_data = api_call_with_retry(
                client.kite.historical_data,
                instrument_token=260105,  # Bank Nifty token
                from_date=test_date,
                to_date=test_date,
                interval="minute"
            )
        
        # Fetch futures data (volume)
        logger.info(f"📊 Fetching futures volume from {front_contract['tradingsymbol']}...")
        futures_data = api_call_with_retry(
            client.kite.historical_data,
            instrument_token=front_contract['instrument_token'],
            from_date=test_date,
            to_date=test_date,
            interval="minute",
            oi=1
        )
        
        if index_data and futures_data:
            # Create DataFrames
            index_df = pd.DataFrame(index_data)
            futures_df = pd.DataFrame(futures_data)
            
            # Set column names
            if len(index_df.columns) == 6:
                index_df.columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
            else:
                index_df.columns = ['timestamp', 'open', 'high', 'low', 'close']
                index_df['volume'] = 0
                
            futures_df.columns = ['timestamp', 'f_open', 'f_high', 'f_low', 'f_close', 'volume', 'oi']
            
            # Convert timestamps
            index_df['timestamp'] = pd.to_datetime(index_df['timestamp'])
            futures_df['timestamp'] = pd.to_datetime(futures_df['timestamp'])
            
            # Combine data
            combined_df = index_df.merge(
                futures_df[['timestamp', 'volume']],
                on='timestamp',
                how='left',
                suffixes=('', '_futures')
            )
            
            # Use futures volume
            combined_df['volume'] = combined_df['volume_futures'].fillna(0)
            combined_df.drop('volume_futures', axis=1, inplace=True)
            
            logger.info(f"✅ Combined data successfully!")
            logger.info(f"📊 Index records: {len(index_df)}")
            logger.info(f"📊 Futures records: {len(futures_df)}")
            logger.info(f"📊 Combined records: {len(combined_df)}")
            logger.info(f"📊 Total volume: {combined_df['volume'].sum():,.0f}")
            logger.info(f"📊 Avg volume/min: {combined_df['volume'].mean():,.0f}")
            
            # Show sample
            logger.info("\n📋 Sample combined data:")
            print(combined_df[['timestamp', 'open', 'high', 'low', 'close', 'volume']].head())
            
            logger.info("✅ Front-month futures volume integration working correctly!")
            
        else:
            logger.error("❌ Failed to fetch index or futures data")
            
    except Exception as e:
        logger.error(f"❌ Combined test failed: {e}")
        import traceback
        logger.debug(f"Full traceback: {traceback.format_exc()}")

if __name__ == "__main__":
    logger.info("🧪 Front-Month Futures Volume Integration Test")
    logger.info("Industry Standard Approach Validation")
    logger.info("=" * 80)
    
    # Test 1: Front-month contract discovery
    test_front_month_integration()
    
    # Test 2: Combined data approach
    test_combined_data_approach()
    
    logger.info("\n✅ Test completed!")
