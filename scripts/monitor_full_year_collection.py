#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Monitor the full year data collection progress
"""

import os
import time
from pathlib import Path
from datetime import datetime

def monitor_collection():
    project_root = Path(__file__).parent.parent
    raw_data_dir = project_root / "data" / "raw"
    
    print("🔍 Monitoring Full Year Data Collection Progress")
    print("=" * 60)
    
    while True:
        # Count Bank Nifty index files
        index_files = list(raw_data_dir.glob("bnk_index_*.csv"))
        
        # Count options files
        option_files = list(raw_data_dir.glob("options_corrected_*.parquet"))
        
        # Get total file sizes
        total_size = 0
        for f in index_files + option_files:
            if f.exists():
                total_size += f.stat().st_size
        
        size_mb = total_size / (1024 * 1024)
        
        print(f"\n📊 Progress Update - {datetime.now().strftime('%H:%M:%S')}")
        print(f"📁 Bank Nifty Index Files: {len(index_files)}")
        print(f"📈 Options Files: {len(option_files)}")
        print(f"💾 Total Data Size: {size_mb:.1f} MB")
        
        if index_files:
            latest_index = max(index_files, key=lambda x: x.stat().st_mtime)
            print(f"🕐 Latest Index File: {latest_index.name} ({datetime.fromtimestamp(latest_index.stat().st_mtime).strftime('%H:%M:%S')})")
        
        if option_files:
            latest_option = max(option_files, key=lambda x: x.stat().st_mtime)
            print(f"📅 Latest Option File: {latest_option.name} ({datetime.fromtimestamp(latest_option.stat().st_mtime).strftime('%H:%M:%S')})")
        
        # Estimate progress (365 days total)
        if len(index_files) > 0:
            # Each index file covers ~5 days, so 365/5 = ~73 files expected
            expected_index_files = 73
            index_progress = min(100, (len(index_files) / expected_index_files) * 100)
            print(f"📊 Index Collection Progress: {index_progress:.1f}%")
        
        time.sleep(30)  # Update every 30 seconds

if __name__ == "__main__":
    try:
        monitor_collection()
    except KeyboardInterrupt:
        print("\n⏹️ Monitoring stopped")
