#!/usr/bin/env python3
"""
Test script to verify Bank Nifty index + futures volume combination works correctly.
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

def test_combined_data_collection():
    """Test fetching Bank Nifty index OHLC + futures volume for a single day."""
    logger.info("=" * 60)
    logger.info("Testing Bank Nifty Index + Futures Volume Combination")
    logger.info("=" * 60)
    
    try:
        # Initialize client
        client = ZerodhaClient()
        logger.info("✅ Connected to Zerodha API")
        
        # Get current Bank Nifty futures contract
        logger.info("🔍 Finding Bank Nifty futures contract...")
        nfo_instruments = client.kite.instruments("NFO")
        banknifty_futures = [
            inst for inst in nfo_instruments 
            if inst.get('name', '').upper() == 'BANKNIFTY' and inst.get('instrument_type') == 'FUT'
        ]
        
        if not banknifty_futures:
            logger.error("❌ No Bank Nifty futures contracts found!")
            return False
            
        # Get current month contract (nearest expiry)
        current_contract = min(banknifty_futures, key=lambda x: x['expiry'])
        futures_token = current_contract['instrument_token']
        
        logger.info(f"📊 Using futures contract: {current_contract['tradingsymbol']} (Token: {futures_token})")
        
        # Test date: Use a recent weekday (Friday May 23, 2025 is a trading day)
        test_date = date(2025, 5, 23)  # Friday
        logger.info(f"📅 Testing with date: {test_date} (Friday - trading day)")
        
        # Fetch Bank Nifty index OHLC data
        logger.info("📈 Fetching Bank Nifty index OHLC data...")
        index_data = client.fetch_historical_data(
            instrument="NSE:NIFTY BANK",
            interval="minute",
            from_date=test_date,
            to_date=test_date
        )
        
        # Fetch Bank Nifty futures volume data
        logger.info("📊 Fetching Bank Nifty futures volume data...")
        futures_data = client.kite.historical_data(
            instrument_token=futures_token,
            from_date=test_date,
            to_date=test_date,
            interval="minute"
        )
        
        if not index_data:
            logger.error("❌ No index data received")
            return False
            
        if not futures_data:
            logger.error("❌ No futures data received")
            return False
            
        # Convert to DataFrames
        logger.info("🔄 Processing and combining data...")
        index_df = pd.DataFrame(index_data)
        futures_df = pd.DataFrame(futures_data)
        
        # Check the actual column structure
        logger.info(f"📊 Index columns: {list(index_df.columns)}")
        logger.info(f"📊 Futures columns: {list(futures_df.columns)}")
        
        logger.info(f"📊 Index data: {len(index_df)} records")
        logger.info(f"📊 Futures data: {len(futures_df)} records")
        
        # Both datasets should have: ['date', 'open', 'high', 'low', 'close', 'volume']
        # Index volume is 0, futures volume is meaningful
        
        # Rename date column to timestamp for consistency and align timestamps
        index_df = index_df.rename(columns={'date': 'timestamp'})
        futures_df = futures_df.rename(columns={'date': 'timestamp'})
        
        index_df['timestamp'] = pd.to_datetime(index_df['timestamp'])
        futures_df['timestamp'] = pd.to_datetime(futures_df['timestamp'])
        
        # Show sample data before combining
        logger.info(f"📊 Sample index data (first 2 rows):")
        for _, row in index_df.head(2).iterrows():
            logger.info(f"   {row['timestamp']}: OHLC={row['open']:.1f}/{row['high']:.1f}/{row['low']:.1f}/{row['close']:.1f}, Vol={row['volume']:.0f}")
            
        logger.info(f"📊 Sample futures data (first 2 rows):")
        for _, row in futures_df.head(2).iterrows():
            logger.info(f"   {row['timestamp']}: OHLC={row['open']:.1f}/{row['high']:.1f}/{row['low']:.1f}/{row['close']:.1f}, Vol={row['volume']:.0f}")
        
        # Replace index volume with futures volume
        # Keep index OHLC, but use futures volume
        combined_df = index_df.drop(columns=['volume']).merge(
            futures_df[['timestamp', 'volume']], 
            on='timestamp', 
            how='left'
        )
        
        # Fill any missing volume with 0
        combined_df['volume'] = combined_df['volume'].fillna(0)
        
        logger.info(f"📊 Combined data: {len(combined_df)} records")
        
        # Statistics
        total_volume = combined_df['volume'].sum()
        avg_volume = combined_df['volume'].mean()
        max_volume = combined_df['volume'].max()
        non_zero_volume = (combined_df['volume'] > 0).sum()
        
        logger.info("=" * 60)
        logger.info("📈 Volume Statistics:")
        logger.info(f"   Total volume: {total_volume:,.0f}")
        logger.info(f"   Average volume per minute: {avg_volume:.2f}")
        logger.info(f"   Max volume in a minute: {max_volume:,.0f}")
        logger.info(f"   Minutes with volume > 0: {non_zero_volume}/{len(combined_df)}")
        
        # Show sample data
        logger.info("📋 Sample combined data:")
        sample_data = combined_df.head()[['timestamp', 'open', 'high', 'low', 'close', 'volume']]
        for _, row in sample_data.iterrows():
            logger.info(f"   {row['timestamp']}: OHLC={row['open']:.1f}/{row['high']:.1f}/{row['low']:.1f}/{row['close']:.1f}, Vol={row['volume']:.0f}")
        
        # Save test file
        test_file = project_root / "data" / "raw" / f"test_combined_{test_date.strftime('%Y%m%d')}.csv"
        combined_df.to_csv(test_file, index=False)
        logger.info(f"💾 Saved test data to: {test_file}")
        
        logger.info("=" * 60)
        logger.info("✅ SUCCESS: Bank Nifty index + futures volume combination works!")
        logger.info("=" * 60)
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Test failed: {e}")
        return False

def main():
    """Main test function."""
    success = test_combined_data_collection()
    
    if success:
        logger.info("🎉 Test completed successfully! Ready to update production script.")
    else:
        logger.error("💥 Test failed. Please check the implementation.")

if __name__ == "__main__":
    main()
