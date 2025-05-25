#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Validation Script for Task 2.1 Critical Fixes
Tests all three corrected implementations:
1. Complete business day calendar (no sampling)
2. Weekly expiry logic using kite.instruments("NSE") 
3. Option chain fetching using kite.ltp() bulk quotes
"""

import os
import sys
import logging
from datetime import datetime, timedelta, date
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from src.data_ingest.zerodha_client import ZerodhaClient

# Configure logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_business_day_calendar():
    """Test Fix #1: Complete business day calendar generation."""
    logger.info("\n" + "="*60)
    logger.info("🧪 TESTING FIX #1: Complete Business Day Calendar")
    logger.info("="*60)
    
    # Import the fixed function
    from task2_1_fixed_historical_implementation import get_all_business_days
    
    # Test one year of business days
    end_date = date.today()
    start_date = end_date - timedelta(days=365)
    
    business_days = get_all_business_days(start_date, end_date)
    
    logger.info(f"✅ Generated {len(business_days)} complete business days")
    logger.info(f"📅 Range: {business_days[0]} to {business_days[-1]}")
    
    # Validate it's approximately one year of business days (260 days)
    expected_range = (250, 270)  # Reasonable range for business days in a year
    if expected_range[0] <= len(business_days) <= expected_range[1]:
        logger.info(f"✅ Business day count looks correct: {len(business_days)} days")
        return True
    else:
        logger.error(f"❌ Unexpected business day count: {len(business_days)} (expected {expected_range})")
        return False

def test_weekly_expiry_logic():
    """Test Fix #2: Weekly expiry logic using NSE instruments."""
    logger.info("\n" + "="*60)
    logger.info("🧪 TESTING FIX #2: Weekly Expiry Logic (NSE Instruments)")
    logger.info("="*60)
    
    # Initialize client
    client = ZerodhaClient(credentials_path=str(project_root / "config" / "credentials.json"))
    
    if not client.login():
        logger.error("❌ Authentication failed")
        return False
    
    logger.info("✅ Authentication successful")
    
    # Import the fixed function
    from task2_1_fixed_historical_implementation import get_banknifty_weekly_expiries, get_next_weekly_expiry_for_date
    
    # Test weekly expiry fetching
    expiry_data = get_banknifty_weekly_expiries(client)
    weekly_expiries = expiry_data.get("weekly_expiries", [])
    
    if not weekly_expiries:
        logger.error("❌ No weekly expiries found")
        return False
    
    logger.info(f"✅ Found {len(weekly_expiries)} weekly expiries")
    
    # Show first few expiries
    for i, expiry in enumerate(weekly_expiries[:5]):
        weekday_name = expiry.strftime("%A")
        logger.info(f"📅 Expiry {i+1}: {expiry} ({weekday_name})")
        
        # Validate it's a Thursday
        if expiry.weekday() != 3:
            logger.error(f"❌ Expiry {expiry} is not a Thursday (weekday={expiry.weekday()})")
            return False
    
    # Test getting next expiry for a specific date
    test_date = date.today()
    next_expiry = get_next_weekly_expiry_for_date(test_date, weekly_expiries)
    
    if next_expiry:
        logger.info(f"✅ Next weekly expiry for {test_date}: {next_expiry}")
        return True
    else:
        logger.error(f"❌ Could not find next weekly expiry for {test_date}")
        return False

def test_ltp_bulk_quotes():
    """Test Fix #3: Option chain fetching using kite.ltp() bulk quotes."""
    logger.info("\n" + "="*60) 
    logger.info("🧪 TESTING FIX #3: LTP Bulk Quotes for Option Chain")
    logger.info("="*60)
    
    # Initialize client
    client = ZerodhaClient(credentials_path=str(project_root / "config" / "credentials.json"))
    
    if not client.login():
        logger.error("❌ Authentication failed")
        return False
    
    logger.info("✅ Authentication successful")
    
    # Test symbol format and LTP fetching
    try:
        # Get next weekly expiry
        from task2_1_fixed_historical_implementation import get_banknifty_weekly_expiries
        expiry_data = get_banknifty_weekly_expiries(client)
        weekly_expiries = expiry_data.get("weekly_expiries", [])
        
        if not weekly_expiries:
            logger.error("❌ No weekly expiries available for testing")
            return False
        
        # Use the next available expiry
        next_expiry = min([exp for exp in weekly_expiries if exp >= date.today()])
        logger.info(f"📅 Testing with expiry: {next_expiry}")
        
        # Test symbol format: NSE:BANKNIFTY{expiry:%d%b%Y}{strike}{opt}
        expiry_str = next_expiry.strftime("%d%b%Y").upper()
        test_symbols = [
            f"NSE:BANKNIFTY{expiry_str}50000CE",
            f"NSE:BANKNIFTY{expiry_str}50000PE",
            f"NSE:BANKNIFTY{expiry_str}50100CE",
            f"NSE:BANKNIFTY{expiry_str}50100PE"
        ]
        
        logger.info(f"🔍 Testing LTP for symbols: {test_symbols}")
        
        # Test bulk LTP query
        ltp_data = client.kite.ltp(test_symbols)
        
        if not ltp_data:
            logger.error("❌ No LTP data received")
            return False
        
        logger.info(f"✅ Received LTP data for {len(ltp_data)} symbols")
        
        # Display sample LTP data
        for symbol, data in list(ltp_data.items())[:4]:
            last_price = data.get("last_price", 0.0)
            logger.info(f"📊 {symbol}: ₹{last_price}")
        
        # Validate we got data for our test symbols
        received_count = len(ltp_data)
        expected_count = len(test_symbols)
        
        if received_count >= expected_count // 2:  # Allow for some symbols to be invalid
            logger.info(f"✅ LTP bulk quotes working correctly ({received_count}/{expected_count})")
            return True
        else:
            logger.error(f"❌ Too few symbols returned ({received_count}/{expected_count})")
            return False
            
    except Exception as e:
        logger.error(f"❌ Error testing LTP bulk quotes: {e}")
        return False

def run_validation():
    """Run all validation tests."""
    logger.info("🔧 TASK 2.1 CRITICAL FIXES VALIDATION")
    logger.info("="*60)
    
    results = {}
    
    # Test 1: Business day calendar
    results["business_calendar"] = test_business_day_calendar()
    
    # Test 2: Weekly expiry logic  
    results["weekly_expiry"] = test_weekly_expiry_logic()
    
    # Test 3: LTP bulk quotes
    results["ltp_bulk_quotes"] = test_ltp_bulk_quotes()
    
    # Summary
    logger.info("\n" + "="*60)
    logger.info("📋 VALIDATION SUMMARY")
    logger.info("="*60)
    
    all_passed = True
    for test_name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        logger.info(f"{test_name.replace('_', ' ').title()}: {status}")
        if not passed:
            all_passed = False
    
    logger.info("="*60)
    if all_passed:
        logger.info("🎉 ALL TESTS PASSED - Fixes are working correctly!")
    else:
        logger.error("💥 SOME TESTS FAILED - Fixes need attention!")
    
    return all_passed

if __name__ == "__main__":
    success = run_validation()
    sys.exit(0 if success else 1)
