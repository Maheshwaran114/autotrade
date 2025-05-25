#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Simple validation of Fix #1: Business Day Calendar
"""

import sys
from pathlib import Path
from datetime import date, timedelta

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

def test_business_days():
    """Test the business day calendar fix."""
    print("Testing Fix #1: Business Day Calendar")
    print("="*40)
    
    try:
        from scripts.task2_1_fixed_historical_implementation import get_all_business_days
        
        # Test one year of business days
        end_date = date.today()
        start_date = end_date - timedelta(days=365)
        
        business_days = get_all_business_days(start_date, end_date)
        
        print(f"✅ Generated {len(business_days)} complete business days")
        print(f"📅 Range: {business_days[0]} to {business_days[-1]}")
        
        # Validate it's approximately one year of business days (260 days)
        expected_range = (250, 270)
        if expected_range[0] <= len(business_days) <= expected_range[1]:
            print(f"✅ Business day count looks correct: {len(business_days)} days")
            return True
        else:
            print(f"❌ Unexpected business day count: {len(business_days)} (expected {expected_range})")
            return False
            
    except Exception as e:
        print(f"❌ Error testing business days: {e}")
        return False

if __name__ == "__main__":
    success = test_business_days()
    print("="*40)
    if success:
        print("🎉 Business Day Calendar Test PASSED!")
    else:
        print("💥 Business Day Calendar Test FAILED!")
