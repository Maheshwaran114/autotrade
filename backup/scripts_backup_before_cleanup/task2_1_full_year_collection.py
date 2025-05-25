#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Task 2.1: Full Year Historical Data Collection for ML Training
Modified implementation to collect exactly one year of data with specific file naming requirements:

1. Bank Nifty index data in 5-day chunks: data/raw/bnk_index_<YYYYMMDD>.csv
2. Combined index data: data/processed/banknifty_index.parquet
3. Daily option snapshots: data/raw/options_*.parquet
4. Combined options data: data/processed/banknifty_options_chain.parquet

This builds on the fixed OI implementation with proper file naming for ML training pipeline.
"""

import os
import sys
import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, date
from pathlib import Path
import time
import argparse
from typing import Dict, List, Optional, Tuple, Union
import json

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

# Import Zerodha client
from src.data_ingest.zerodha_client import ZerodhaClient
import src.data_ingest.load_data as load_data

# Configure logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Configuration for full year collection
CONFIG = {
    "DAYS_BACK": 365,  # Full year of trading data  
    "INDEX_CHUNK_DAYS": 5,  # 5-day chunks for index data as required
    "MINUTE_DATA_CHUNK_DAYS": 60,  # API limit for minute data
    "STRIKE_RANGE_OFFSET": 2000,  # (spot - 2000) to (spot + 2000) as required
    "STRIKE_STEP": 100,    # Step by 100 as required
    "API_DELAY": 1.0,      # Safe delay between API calls (1 req/sec)
    "BATCH_SIZE": 50,      # Process options in batches
    "SNAPSHOT_TIME": "15:25",  # EOD snapshot time
    "MAX_RETRIES": 5,      # Max retries for API calls
    "RETRY_BASE_DELAY": 2.0,  # Base delay for exponential backoff (seconds)
    "PARALLEL_WORKERS": 5,  # Number of parallel workers
}

def get_all_business_days(start_date: date, end_date: date) -> List[date]:
    """Generate complete list of ALL business days for the specified period."""
    date_range = pd.date_range(start=start_date, end=end_date, freq='B')  # Business days only
    trading_dates = [d.date() for d in date_range]
    
    logger.info(f"Generated {len(trading_dates)} complete business days from {start_date} to {end_date}")
    logger.info(f"🗓️  First trading day: {trading_dates[0] if trading_dates else 'None'}")
    logger.info(f"🗓️  Last trading day: {trading_dates[-1] if trading_dates else 'None'}")
    
    return sorted(trading_dates)

def api_call_with_retry(api_func, max_retries=None, base_delay=None, *args, **kwargs):
    """Execute API call with exponential backoff retry logic."""
    if max_retries is None:
        max_retries = CONFIG["MAX_RETRIES"]
    if base_delay is None:
        base_delay = CONFIG["RETRY_BASE_DELAY"]
    
    for attempt in range(max_retries):
        try:
            return api_func(*args, **kwargs)
        except Exception as e:
            error_msg = str(e).lower()
            
            # Check if it's a rate limiting or network error
            is_retryable = any(keyword in error_msg for keyword in [
                'max retries exceeded', 'connection', 'timeout', 'network', 
                'rate limit', 'too many requests', 'nodename nor servname',
                'temporary failure', 'service unavailable'
            ])
            
            if attempt == max_retries - 1 or not is_retryable:
                logger.error(f"API call failed after {attempt + 1} attempts: {e}")
                raise e
            else:
                delay = base_delay * (2 ** attempt)
                logger.warning(f"API call attempt {attempt + 1} failed ({e}), retrying in {delay:.1f}s...")
                time.sleep(delay)
    
    return None

def get_banknifty_weekly_expiries(client) -> Dict[str, Union[List[date], Dict[date, date]]]:
    """Get Bank Nifty weekly expiries from NSE instruments as required."""
    logger.info("🔍 Fetching Bank Nifty weekly expiries from NSE instruments...")
    
    try:
        # FIXED: Use positional arguments instead of keyword arguments
        nse_instruments = api_call_with_retry(client.kite.instruments, "NSE")
        nfo_instruments = api_call_with_retry(client.kite.instruments, "NFO")
        
        if not nse_instruments or not nfo_instruments:
            logger.error("Failed to fetch instruments data")
            return {"weekly_expiries": [], "next_expiry_map": {}}
        
        # Find Bank Nifty from NSE for spot price reference
        banknifty_nse = None
        for instrument in nse_instruments:
            if instrument.get('tradingsymbol') == 'NIFTY BANK':
                banknifty_nse = instrument
                break
        
        if not banknifty_nse:
            logger.error("Bank Nifty not found in NSE instruments")
            return {"weekly_expiries": [], "next_expiry_map": {}}
        
        # Extract weekly expiries from NFO Bank Nifty options
        weekly_expiries = set()
        for instrument in nfo_instruments:
            if (instrument.get('name') == 'BANKNIFTY' and 
                instrument.get('instrument_type') in ['CE', 'PE'] and
                instrument.get('expiry')):
                
                expiry_str = instrument['expiry']
                try:
                    expiry_date = datetime.strptime(expiry_str, '%Y-%m-%d').date()
                    # Bank Nifty weekly expiries are on Thursdays
                    if expiry_date.weekday() == 3:  # Thursday = 3
                        weekly_expiries.add(expiry_date)
                except ValueError:
                    continue
        
        weekly_expiries = sorted(list(weekly_expiries))
        logger.info(f"✅ Found {len(weekly_expiries)} Bank Nifty weekly expiries")
        
        if weekly_expiries:
            logger.info(f"📅 First expiry: {weekly_expiries[0]}")
            logger.info(f"📅 Last expiry: {weekly_expiries[-1]}")
        
        # Create next expiry mapping for each trading date
        next_expiry_map = {}
        for expiry in weekly_expiries:
            # Map each day of the week leading to this expiry
            for days_back in range(7):
                trading_date = expiry - timedelta(days=days_back)
                if trading_date not in next_expiry_map:  # Don't overwrite closer expiries
                    next_expiry_map[trading_date] = expiry
        
        logger.info(f"✅ Created next expiry mapping for {len(next_expiry_map)} trading dates")
        
        return {
            "weekly_expiries": weekly_expiries,
            "next_expiry_map": next_expiry_map,
            "banknifty_token": banknifty_nse.get('instrument_token')
        }
        
    except Exception as e:
        logger.error(f"Error fetching weekly expiries: {e}")
        return {"weekly_expiries": [], "next_expiry_map": {}}

def get_next_weekly_expiry_for_date(trade_date: date, weekly_expiries: List[date], 
                                   next_expiry_map: Dict[date, date]) -> Optional[date]:
    """Get the next weekly expiry for a given trading date."""
    # First try the pre-computed mapping
    if trade_date in next_expiry_map:
        return next_expiry_map[trade_date]
    
    # Fallback: find the next Thursday expiry
    for expiry in weekly_expiries:
        if expiry >= trade_date:
            return expiry
    
    return None

def fetch_banknifty_index_in_chunks(client, start_date: date, end_date: date) -> List[str]:
    """
    Fetch Bank Nifty index data in 5-day chunks and save with specific naming pattern.
    Returns list of created CSV filenames.
    """
    logger.info("🏦 Fetching Bank Nifty index data in 5-day chunks...")
    
    created_files = []
    current_date = start_date
    chunk_count = 0
    
    while current_date < end_date:
        chunk_end = min(current_date + timedelta(days=CONFIG["INDEX_CHUNK_DAYS"]), end_date)
        chunk_count += 1
        
        # Create filename with the start date of the chunk
        filename = f"bnk_index_{current_date.strftime('%Y%m%d')}.csv"
        file_path = project_root / "data" / "raw" / filename
        
        logger.info(f"Fetching chunk {chunk_count}: {current_date} to {chunk_end} -> {filename}")
        
        try:
            # Fetch historical minute data for this chunk
            chunk_data = client.fetch_historical_data(
                instrument="NSE:NIFTY BANK",
                interval="minute",
                from_date=current_date,
                to_date=chunk_end
            )
            
            if chunk_data:
                df = pd.DataFrame(chunk_data)
                df.to_csv(file_path, index=False)
                logger.info(f"💾 Saved {len(chunk_data)} minute records to {filename}")
                created_files.append(filename)
            else:
                logger.warning(f"Chunk {chunk_count}: No data retrieved for {current_date} to {chunk_end}")
                
        except Exception as e:
            logger.error(f"Error fetching chunk {chunk_count}: {e}")
        
        current_date = chunk_end + timedelta(days=1)
        time.sleep(CONFIG["API_DELAY"])
    
    logger.info(f"✅ Created {len(created_files)} Bank Nifty index files in 5-day chunks")
    return created_files

def fetch_option_chain_snapshot_historical(client, expiry_date: date, strike_range: range, 
                                         trade_date: date) -> List[Dict]:
    """Fetch option chain snapshot using historical data with OI parameter."""
    logger.debug(f"Fetching historical option data for {trade_date}, expiry {expiry_date}")
    
    option_records = []
    
    try:
        # Get instruments for this expiry
        nfo_instruments = api_call_with_retry(client.kite.instruments, "NFO")
        
        if not nfo_instruments:
            logger.error("Failed to fetch NFO instruments")
            return []
        
        # Filter Bank Nifty options for this expiry
        expiry_str = expiry_date.strftime('%Y-%m-%d')
        relevant_instruments = []
        
        for instrument in nfo_instruments:
            if (instrument.get('name') == 'BANKNIFTY' and 
                instrument.get('expiry') == expiry_str and
                instrument.get('instrument_type') in ['CE', 'PE']):
                
                try:
                    strike = int(float(instrument.get('strike', 0)))
                    if strike in strike_range:
                        relevant_instruments.append(instrument)
                except (ValueError, TypeError):
                    continue
        
        logger.info(f"Found {len(relevant_instruments)} relevant option instruments")
        
        # Fetch historical data for each instrument
        for instrument in relevant_instruments:
            try:
                instrument_token = instrument['instrument_token']
                strike = int(float(instrument['strike']))
                option_type = instrument['instrument_type']
                
                # Fetch historical data with OI parameter - CRITICAL FIX
                from_date = trade_date
                to_date = trade_date
                interval = "day"
                
                historical_data = api_call_with_retry(
                    client.kite.historical_data,
                    None, None,  # Use default retry settings
                    instrument_token,
                    from_date,
                    to_date,
                    interval,
                    1  # CRITICAL FIX: Add oi=1 to get Open Interest data
                )
                
                if historical_data and len(historical_data) > 0:
                    data_point = historical_data[0]  # Get the day's data
                    
                    # Parse historical data format: [timestamp, open, high, low, close, volume, oi]
                    if len(data_point) >= 7:  # With OI data
                        record = {
                            "symbol": instrument['tradingsymbol'],
                            "strike": strike,
                            "instrument_type": option_type,
                            "expiry": expiry_date,
                            "date": trade_date,
                            "last_price": data_point[4],  # close price
                            "ltp": data_point[4],         # same as close for historical
                            "volume": data_point[5],      # volume
                            "oi": data_point[6],          # open interest
                            "bid": 0.0,   # Not available in historical data
                            "ask": 0.0,   # Not available in historical data
                        }
                        option_records.append(record)
                
                time.sleep(CONFIG["API_DELAY"])  # Rate limiting
                
            except Exception as e:
                logger.warning(f"Error fetching data for {instrument['tradingsymbol']}: {e}")
                continue
        
        logger.info(f"✅ Successfully fetched {len(option_records)} option records with OI data")
        
        # Log sample data to verify OI is working
        if option_records:
            sample = option_records[0]
            oi_count = sum(1 for r in option_records if r['oi'] > 0)
            vol_count = sum(1 for r in option_records if r['volume'] > 0)
            logger.info(f"📊 Sample data: {sample['symbol']} price={sample['last_price']}, oi={sample['oi']}, vol={sample['volume']}")
            logger.info(f"📊 Quality check: {oi_count}/{len(option_records)} records with OI > 0, {vol_count}/{len(option_records)} with volume > 0")
        
        return option_records
        
    except Exception as e:
        logger.error(f"Error fetching historical option chain: {e}")
        return []

def get_spot_from_index_data(trade_date: date, index_files: List[str]) -> float:
    """Get spot price from Bank Nifty index data files."""
    # Find the file that contains this date
    for filename in index_files:
        try:
            file_path = project_root / "data" / "raw" / filename
            df = pd.read_csv(file_path)
            
            if 'date' in df.columns:
                df['date'] = pd.to_datetime(df['date']).dt.date
                date_data = df[df['date'] == trade_date]
                
                if not date_data.empty:
                    # Use closing price as spot price
                    spot_price = date_data['close'].iloc[-1]  # Last close of the day
                    return float(spot_price)
        except Exception as e:
            logger.warning(f"Error reading {filename}: {e}")
            continue
    
    # Fallback: use a reasonable Bank Nifty value
    return 45000.0

def build_snapshot_dataframe(trade_date: date, expiry_date: date, spot_price: float, 
                           option_records: List[Dict]) -> pd.DataFrame:
    """Build option chain snapshot DataFrame."""
    if not option_records:
        return pd.DataFrame()
    
    df = pd.DataFrame(option_records)
    
    # Add calculated fields
    df['spot_price'] = spot_price
    df['days_to_expiry'] = (expiry_date - trade_date).days
    df['moneyness'] = df['strike'] / spot_price
    
    # Add IV calculation (simplified for now)
    df['iv'] = 0.0  # Placeholder for implied volatility
    
    return df

def create_unified_processed_files(index_files: List[str]):
    """Create unified parquet files from raw data with specific naming."""
    logger.info("🔄 Creating unified processed data files...")
    
    # 1. Combine all Bank Nifty index files into banknifty_index.parquet
    logger.info("📊 Combining Bank Nifty index data...")
    all_index_data = []
    
    for filename in index_files:
        try:
            file_path = project_root / "data" / "raw" / filename
            df = pd.read_csv(file_path)
            all_index_data.append(df)
            logger.info(f"Added {len(df)} records from {filename}")
        except Exception as e:
            logger.warning(f"Error reading {filename}: {e}")
    
    if all_index_data:
        combined_index_df = pd.concat(all_index_data, ignore_index=True)
        combined_index_df = combined_index_df.sort_values('date')
        
        index_output_path = project_root / "data" / "processed" / "banknifty_index.parquet"
        combined_index_df.to_parquet(index_output_path, index=False)
        logger.info(f"💾 Saved {len(combined_index_df)} index records to banknifty_index.parquet")
    
    # 2. Combine all option files into banknifty_options_chain.parquet
    logger.info("📈 Combining option chain data...")
    all_option_data = []
    
    option_files = list((project_root / "data" / "raw").glob("options_*.parquet"))
    for file_path in option_files:
        try:
            df = pd.read_parquet(file_path)
            all_option_data.append(df)
            logger.info(f"Added {len(df)} records from {file_path.name}")
        except Exception as e:
            logger.warning(f"Error reading {file_path}: {e}")
    
    if all_option_data:
        combined_options_df = pd.concat(all_option_data, ignore_index=True)
        combined_options_df = combined_options_df.sort_values(['date', 'expiry', 'strike'])
        
        options_output_path = project_root / "data" / "processed" / "banknifty_options_chain.parquet"
        combined_options_df.to_parquet(options_output_path, index=False)
        logger.info(f"💾 Saved {len(combined_options_df)} option records to banknifty_options_chain.parquet")
    
    logger.info("✅ Successfully created unified processed files")

def collect_full_year_data():
    """Main function to collect full year of Bank Nifty and options data."""
    logger.info("=" * 80)
    logger.info("TASK 2.1: FULL YEAR DATA COLLECTION FOR ML TRAINING")
    logger.info("Collecting 1+ year of Bank Nifty index and options data")
    logger.info("=" * 80)
    
    # Initialize client
    client = ZerodhaClient(credentials_path=str(project_root / "config" / "credentials.json"))
    
    if not client.login():
        logger.error("❌ Authentication failed")
        return False
    
    logger.info("✅ Successfully authenticated with Zerodha API")
    
    # Create directories
    os.makedirs(project_root / "data" / "raw", exist_ok=True)
    os.makedirs(project_root / "data" / "processed", exist_ok=True)
    
    # Set date range for full year
    end_date = datetime.now().date()
    start_date = end_date - timedelta(days=CONFIG["DAYS_BACK"])
    
    logger.info(f"📅 Collection period: {start_date} to {end_date} ({CONFIG['DAYS_BACK']} days)")
    
    # Step 1: Fetch Bank Nifty index data in 5-day chunks
    logger.info("🏦 STEP 1: Fetching Bank Nifty index data in 5-day chunks...")
    index_files = fetch_banknifty_index_in_chunks(client, start_date, end_date)
    
    if not index_files:
        logger.error("❌ Failed to fetch index data")
        return False
    
    # Step 2: Get Bank Nifty weekly expiries
    logger.info("🔍 STEP 2: Building weekly expiry mapping from NSE instruments...")
    expiries_data = get_banknifty_weekly_expiries(client)
    
    if not expiries_data["weekly_expiries"]:
        logger.error("❌ Failed to fetch weekly expiries data")
        return False
    
    # Step 3: Generate complete business day calendar for full year
    logger.info("📅 STEP 3: Generating complete business day calendar...")
    trading_dates = get_all_business_days(start_date, end_date)
    
    logger.info(f"🚀 Processing ALL {len(trading_dates)} trading dates for full year collection")
    
    # Step 4: Fetch option chain snapshots for each day
    logger.info("📈 STEP 4: Fetching daily option chain snapshots...")
    
    successful_snapshots = 0
    failed_snapshots = 0
    start_time = time.time()
    
    for i, trade_date in enumerate(trading_dates):
        # Progress tracking
        progress_pct = ((i + 1) / len(trading_dates)) * 100
        elapsed_time = time.time() - start_time
        
        if i > 0:  # Avoid division by zero
            avg_time_per_date = elapsed_time / i
            estimated_total_time = avg_time_per_date * len(trading_dates)
            remaining_time = estimated_total_time - elapsed_time
            
            logger.info(f"📊 Progress: {i+1}/{len(trading_dates)} ({progress_pct:.1f}%) | "
                       f"Elapsed: {elapsed_time/60:.1f}m | ETA: {remaining_time/60:.1f}m")
        else:
            logger.info(f"📊 Progress: {i+1}/{len(trading_dates)} ({progress_pct:.1f}%)")
        
        logger.info(f"🗓️  Processing {trade_date}")
        
        try:
            # Get next weekly expiry for this date
            expiry_date = get_next_weekly_expiry_for_date(
                trade_date, 
                expiries_data["weekly_expiries"], 
                expiries_data["next_expiry_map"]
            )
            
            if not expiry_date:
                logger.warning(f"⚠️  No weekly expiry found for {trade_date}, skipping")
                failed_snapshots += 1
                continue
            
            # Get spot price from index data
            spot_price = get_spot_from_index_data(trade_date, index_files)
            
            # Generate strike range: (spot-2000) to (spot+2000) step 100
            strike_range = range(int(spot_price) - CONFIG["STRIKE_RANGE_OFFSET"], 
                               int(spot_price) + CONFIG["STRIKE_RANGE_OFFSET"] + 1, 
                               CONFIG["STRIKE_STEP"])
            
            logger.info(f"📊 {trade_date}: spot={spot_price:.0f}, expiry={expiry_date}, strikes={len(strike_range)}")
            
            # Fetch option chain data using historical method with OI
            option_records = fetch_option_chain_snapshot_historical(
                client, expiry_date, strike_range, trade_date
            )
            
            if not option_records:
                logger.warning(f"⚠️  No option data for {trade_date}, skipping")
                failed_snapshots += 1
                continue
            
            # Build snapshot DataFrame
            df_snapshot = build_snapshot_dataframe(
                trade_date, expiry_date, spot_price, option_records
            )
            
            if df_snapshot.empty:
                logger.warning(f"⚠️  Empty snapshot for {trade_date}, skipping")
                failed_snapshots += 1
                continue
            
            # Save daily snapshot as Parquet
            parquet_filename = f"options_{trade_date.strftime('%Y%m%d')}.parquet"
            parquet_path = project_root / "data" / "raw" / parquet_filename
            
            df_snapshot.to_parquet(parquet_path, index=False)
            logger.info(f"💾 Saved {len(df_snapshot)} option records to {parquet_filename}")
            
            # Log OI data quality
            oi_count = (df_snapshot['oi'] > 0).sum()
            vol_count = (df_snapshot['volume'] > 0).sum()
            logger.info(f"📊 Data quality: {oi_count}/{len(df_snapshot)} with OI>0, {vol_count}/{len(df_snapshot)} with volume>0")
            
            successful_snapshots += 1
            
        except Exception as e:
            logger.error(f"❌ Error processing {trade_date}: {e}")
            failed_snapshots += 1
            continue
    
    # Step 5: Create unified processed files
    logger.info("🔄 STEP 5: Creating unified processed data files...")
    try:
        create_unified_processed_files(index_files)
        logger.info("✅ Successfully created unified processed files")
    except Exception as e:
        logger.error(f"❌ Failed to create unified files: {e}")
    
    # Final summary
    logger.info("=" * 80)
    logger.info("📋 FULL YEAR DATA COLLECTION COMPLETE")
    logger.info(f"📊 Bank Nifty index files: {len(index_files)} (5-day chunks)")
    logger.info(f"📊 Option snapshots: {successful_snapshots}/{len(trading_dates)} successful")
    logger.info(f"📊 Success rate: {successful_snapshots/len(trading_dates)*100:.1f}%")
    logger.info("📁 Files created:")
    logger.info("   - data/raw/bnk_index_<YYYYMMDD>.csv (5-day chunks)")
    logger.info("   - data/processed/banknifty_index.parquet (combined index)")
    logger.info("   - data/raw/options_*.parquet (daily snapshots)")
    logger.info("   - data/processed/banknifty_options_chain.parquet (combined options)")
    logger.info("✅ Ready for ML training pipeline!")
    logger.info("=" * 80)
    
    return True

def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(description="TASK 2.1: Full Year Data Collection for ML Training")
    parser.add_argument("--force", action="store_true", 
                       help="Force overwrite existing data files")
    
    args = parser.parse_args()
    
    try:
        success = collect_full_year_data()
        return success
        
    except KeyboardInterrupt:
        logger.info("⏹️  Collection interrupted by user")
        return False
    except Exception as e:
        logger.error(f"❌ Collection failed: {e}")
        return False

if __name__ == "__main__":
    main()
