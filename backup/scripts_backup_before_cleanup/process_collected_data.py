#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Process Collected Data Script
Convert raw CSV files to processed Parquet files using load_data.py
"""

import os
import sys
import logging
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

# Import data processing functions
from src.data_ingest.load_data import process_banknifty_data, process_options_data

# Configure logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    """Process all collected raw data into processed Parquet files."""
    logger.info("=" * 80)
    logger.info("PROCESSING COLLECTED DATA: CSV → PARQUET")
    logger.info("=" * 80)
    
    # Change to project root directory
    os.chdir(project_root)
    
    # Create processed directory
    os.makedirs("data/processed", exist_ok=True)
    
    # Process Bank Nifty data
    logger.info("🏦 Processing Bank Nifty minute data...")
    bnf_path, bnf_df = process_banknifty_data()
    
    if bnf_path:
        logger.info(f"✅ Bank Nifty data saved to: {bnf_path}")
        logger.info(f"📊 Records: {len(bnf_df):,}")
        logger.info(f"📅 Date range: {bnf_df['date'].min()} to {bnf_df['date'].max()}")
    else:
        logger.warning("❌ Failed to process Bank Nifty data")
    
    # Process options data
    logger.info("📈 Processing options chain data...")
    opt_path, opt_df = process_options_data()
    
    if opt_path:
        logger.info(f"✅ Options data saved to: {opt_path}")
        logger.info(f"📊 Records: {len(opt_df):,}")
        if not opt_df.empty:
            logger.info(f"📅 Date range: {opt_df['date'].min()} to {opt_df['date'].max()}")
    else:
        logger.warning("❌ No options data to process (this is expected if no options were collected)")
    
    # Summary
    logger.info("=" * 80)
    logger.info("PROCESSING SUMMARY")
    logger.info("=" * 80)
    logger.info(f"Bank Nifty records: {len(bnf_df):,}")
    logger.info(f"Options records: {len(opt_df):,}")
    logger.info("✅ Data processing completed!")

if __name__ == "__main__":
    main()
