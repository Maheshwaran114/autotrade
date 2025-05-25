#!/usr/bin/env python3

"""
Simple script to create unified processed files from the collected data.
"""

import sys
import logging
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

# Import the function
from scripts.task2_1_fixed_historical_implementation import create_unified_processed_files

# Configure logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    # Get all Bank Nifty index files
    raw_data_dir = project_root / "data" / "raw"
    minute_data_files = list(raw_data_dir.glob("bnk_index_*.csv"))
    minute_data_files = [f.name for f in sorted(minute_data_files)]
    
    logger.info(f"Found {len(minute_data_files)} Bank Nifty index files")
    for f in minute_data_files:
        logger.info(f"  - {f}")
    
    # Create unified files
    create_unified_processed_files(minute_data_files)
    
    logger.info("✅ Unified file creation completed!")

if __name__ == "__main__":
    main()
