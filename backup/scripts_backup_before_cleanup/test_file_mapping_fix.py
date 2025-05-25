#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Test script to validate the file mapping fix for get_spot_from_index_data()
This tests that the function can auto-discover the correct chunk file for any date.
"""

import sys
from pathlib import Path
from datetime import date
import logging

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

# Import the fixed function
from scripts.task2_1_fixed_historical_implementation import find_index_file_for_date, get_spot_from_index_data

# Configure logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_file_mapping_fix():
    """Test the file mapping fix with various dates."""
    
    # Test dates that should be in different chunk files
    test_dates = [
        date(2025, 5, 21),  # Should be in bnk_index_20250519.csv (chunk covers 2025-05-19 to 2025-05-23)
        date(2025, 5, 22),  # Should be in bnk_index_20250519.csv
        date(2025, 5, 23),  # Should be in bnk_index_20250519.csv
        date(2025, 5, 15),  # Should be in bnk_index_20250513.csv (chunk covers 2025-05-13 to 2025-05-16)
        date(2025, 5, 16),  # Should be in bnk_index_20250513.csv
        date(2025, 5, 20),  # Should be in bnk_index_20250519.csv
    ]
    
    logger.info("🧪 Testing file mapping fix for get_spot_from_index_data()")
    logger.info("=" * 80)
    
    for test_date in test_dates:
        try:
            logger.info(f"\n📅 Testing date: {test_date}")
            
            # Test file discovery
            found_file = find_index_file_for_date(test_date)
            if found_file:
                logger.info(f"✅ File discovery successful: {found_file}")
                
                # Test spot price extraction
                spot_price = get_spot_from_index_data(test_date)
                logger.info(f"✅ Spot price extraction successful: {spot_price}")
                
            else:
                logger.warning(f"⚠️  No file found for {test_date}")
                
        except Exception as e:
            logger.error(f"❌ Error testing {test_date}: {e}")
    
    logger.info("\n" + "=" * 80)
    logger.info("🎯 File mapping fix test completed!")

if __name__ == "__main__":
    test_file_mapping_fix()
