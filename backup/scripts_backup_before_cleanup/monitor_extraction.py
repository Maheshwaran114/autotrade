#!/usr/bin/env python3
"""
Background monitoring script for Task 2.1 full year data extraction
"""

import os
import time
import glob
from pathlib import Path
from datetime import datetime

def monitor_extraction_progress():
    """Monitor the progress of data extraction by counting files"""
    
    project_root = Path(__file__).parent.parent
    raw_data_dir = project_root / "data" / "raw"
    log_file = project_root / "logs" / "full_year_extraction.log"
    
    print("🔍 Task 2.1 Full Year Data Extraction Monitor")
    print("=" * 60)
    
    while True:
        try:
            # Count existing files
            index_parquet_files = len(glob.glob(str(raw_data_dir / "banknifty_index_*.parquet")))
            options_parquet_files = len(glob.glob(str(raw_data_dir / "banknifty_options_*.parquet")))
            
            # Count legacy CSV files
            index_csv_files = len(glob.glob(str(raw_data_dir / "bnk_index_*.csv")))
            options_csv_files = len(glob.glob(str(raw_data_dir / "options_*.csv"))) + len(glob.glob(str(raw_data_dir / "options_*.parquet")))
            
            # Read last few lines of log file for current status
            current_status = "Starting..."
            if log_file.exists():
                try:
                    with open(log_file, 'r') as f:
                        lines = f.readlines()
                        if lines:
                            # Get last relevant line
                            for line in reversed(lines[-10:]):
                                if "Processing" in line and "📅" in line:
                                    current_status = line.strip().split(" - ")[-1]
                                    break
                except:
                    pass
            
            # Display progress
            timestamp = datetime.now().strftime("%H:%M:%S")
            print(f"\r[{timestamp}] Files: Index Parquet:{index_parquet_files} Options Parquet:{options_parquet_files} | Legacy CSV: Index:{index_csv_files} Options:{options_csv_files} | Status: {current_status[:50]}...", end="", flush=True)
            
            time.sleep(5)  # Update every 5 seconds
            
        except KeyboardInterrupt:
            print("\n\n🛑 Monitoring stopped by user")
            break
        except Exception as e:
            print(f"\n⚠️ Monitor error: {e}")
            time.sleep(10)

if __name__ == "__main__":
    monitor_extraction_progress()
