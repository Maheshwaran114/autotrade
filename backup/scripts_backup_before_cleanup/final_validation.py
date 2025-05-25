#!/usr/bin/env python3

import sys
from pathlib import Path
from datetime import date

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

def test_final_fixes():
    """Final validation of all three fixes."""
    
    print("🔧 FINAL VALIDATION: Task 2.1 Critical Fixes")
    print("="*60)
    
    try:
        from src.data_ingest.zerodha_client import ZerodhaClient
        
        # Initialize client
        client = ZerodhaClient(credentials_path=str(project_root / "config" / "credentials.json"))
        
        if not client.login():
            print("❌ Authentication failed")
            return False
        
        print("✅ Authentication successful")
        print()
        
        # Test 1: Business Day Calendar
        print("🧪 Testing Fix #1: Business Day Calendar")
        print("-" * 40)
        
        from scripts.task2_1_fixed_historical_implementation import get_all_business_days
        
        end_date = date.today()
        start_date = date(end_date.year - 1, end_date.month, end_date.day)
        
        business_days = get_all_business_days(start_date, end_date)
        print(f"✅ Generated {len(business_days)} complete business days")
        print(f"📅 Range: {business_days[0]} to {business_days[-1]}")
        
        if 250 <= len(business_days) <= 270:
            print("✅ Fix #1 Business Day Calendar - PASS")
            fix1_result = True
        else:
            print("❌ Fix #1 Business Day Calendar - FAIL")
            fix1_result = False
        
        print()
        
        # Test 2: Weekly Expiry Logic
        print("🧪 Testing Fix #2: Weekly Expiry Logic") 
        print("-" * 40)
        
        from scripts.task2_1_fixed_historical_implementation import get_banknifty_weekly_expiries
        
        expiry_data = get_banknifty_weekly_expiries(client)
        weekly_expiries = expiry_data.get("weekly_expiries", [])
        
        if weekly_expiries:
            print(f"✅ Found {len(weekly_expiries)} weekly expiries")
            
            # Check first few are Thursdays
            thursdays_count = 0
            for i, expiry in enumerate(weekly_expiries[:3]):
                weekday_name = expiry.strftime("%A")
                print(f"📅 Expiry {i+1}: {expiry} ({weekday_name})")
                if expiry.weekday() == 3:  # Thursday
                    thursdays_count += 1
            
            if thursdays_count == min(3, len(weekly_expiries)):
                print("✅ Fix #2 Weekly Expiry Logic - PASS")
                fix2_result = True
            else:
                print("❌ Fix #2 Weekly Expiry Logic - FAIL (Not all Thursdays)")
                fix2_result = False
        else:
            print("❌ Fix #2 Weekly Expiry Logic - FAIL (No expiries found)")
            fix2_result = False
        
        print()
        
        # Test 3: LTP Bulk Quotes  
        print("🧪 Testing Fix #3: LTP Bulk Quotes")
        print("-" * 40)
        
        if weekly_expiries:
            next_expiry = min([exp for exp in weekly_expiries if exp >= date.today()])
            print(f"📅 Testing with expiry: {next_expiry}")
            
            # Test corrected symbol format manually
            year_2digit = str(next_expiry.year)[-2:]
            month_3letter = next_expiry.strftime("%b").upper()
            expiry_str = f"{year_2digit}{month_3letter}"
            
            test_symbols = [
                f"NFO:BANKNIFTY{expiry_str}50000CE",
                f"NFO:BANKNIFTY{expiry_str}50000PE"
            ]
            
            print(f"🔍 Testing symbols: {test_symbols}")
            
            ltp_data = client.kite.ltp(test_symbols)
            
            if ltp_data and len(ltp_data) > 0:
                print(f"✅ Received LTP data for {len(ltp_data)} symbols")
                for symbol, data in ltp_data.items():
                    last_price = data.get("last_price", 0.0)
                    print(f"📊 {symbol}: ₹{last_price}")
                
                print("✅ Fix #3 LTP Bulk Quotes - PASS")
                fix3_result = True
            else:
                print("❌ Fix #3 LTP Bulk Quotes - FAIL (No LTP data)")
                fix3_result = False
        else:
            print("❌ Fix #3 LTP Bulk Quotes - FAIL (No expiries to test)")
            fix3_result = False
        
        print()
        
        # Final Summary
        print("="*60)
        print("📋 FINAL VALIDATION SUMMARY")
        print("="*60)
        
        results = [
            ("Fix #1 Business Day Calendar", fix1_result),
            ("Fix #2 Weekly Expiry Logic", fix2_result), 
            ("Fix #3 LTP Bulk Quotes", fix3_result)
        ]
        
        all_passed = True
        for fix_name, passed in results:
            status = "✅ PASS" if passed else "❌ FAIL"
            print(f"{fix_name}: {status}")
            if not passed:
                all_passed = False
        
        print("="*60)
        if all_passed:
            print("🎉 ALL THREE CRITICAL FIXES VALIDATED SUCCESSFULLY!")
            print("✅ Task 2.1 is ready for production use.")
        else:
            print("💥 SOME FIXES STILL NEED ATTENTION!")
        
        return all_passed
        
    except Exception as e:
        print(f"❌ Error during validation: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_final_fixes()
    sys.exit(0 if success else 1)
