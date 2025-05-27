#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Monitor 60-Day Data Collection Progress
Simple monitoring script to track the progress of the 60-day data collection process.
"""

import os
import sys
import time
from pathlib import Path
from datetime import datetime, timedelta

# Project root
project_root = Path(__file__).parent.parent

def count_files():
    """Count the number of data files created."""
    raw_dir = project_root / "data" / "raw"
    
    # Count Bank Nifty index files
    index_files = list(raw_dir.glob("bnk_index_*.csv"))
    
    # Count options files
    options_files = list(raw_dir.glob("options_corrected_*.parquet"))
    
    return len(index_files), len(options_files)

def estimate_total_files():
    """Estimate total files expected for 60 days."""
    # Bank Nifty index: approximately 10 files (6-day chunks)
    # Options data: approximately 42 trading days (60 calendar days minus weekends/holidays)
    return 10, 42

def main():
    """Monitor the data collection progress."""
    print("🔍 60-Day Data Collection Monitor")
    print("=" * 50)
    
    expected_index, expected_options = estimate_total_files()
    
    try:
        while True:
            current_index, current_options = count_files()
            
            # Calculate progress percentages
            index_progress = (current_index / expected_index) * 100 if expected_index > 0 else 0
            options_progress = (current_options / expected_options) * 100 if expected_options > 0 else 0
            
            # Display progress
            print(f"\n📊 Progress Update - {datetime.now().strftime('%H:%M:%S')}")
            print(f"📈 Bank Nifty Index: {current_index}/{expected_index} files ({index_progress:.1f}%)")
            print(f"📊 Options Data: {current_options}/{expected_options} files ({options_progress:.1f}%)")
            
            # Check if complete
            if current_index >= expected_index and current_options >= expected_options:
                print("\n✅ Data collection appears to be complete!")
                break
            elif current_index >= expected_index:
                print("✅ Bank Nifty index collection complete, waiting for options...")
            
            # Wait 30 seconds before next check
            time.sleep(30)
            
    except KeyboardInterrupt:
        print("\n⏹️  Monitoring stopped by user")
        current_index, current_options = count_files()
        print(f"📊 Final count: {current_index} index files, {current_options} options files")

if __name__ == "__main__":
    main()
