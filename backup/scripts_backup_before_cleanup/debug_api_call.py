#!/usr/bin/env python3
"""
Debug script to isolate the API call issue
"""

import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from src.data_ingest.zerodha_client import ZerodhaClient

def test_api_calls():
    """Test different ways of calling the API"""
    client = ZerodhaClient(credentials_path=str(project_root / "config" / "credentials.json"))
    
    if not client.login():
        print("❌ Authentication failed")
        return
    
    print("✅ Authentication successful")
    
    # Test 1: Direct call
    try:
        print("\n🧪 Test 1: Direct call to instruments('NFO')")
        result = client.kite.instruments("NFO")
        print(f"✅ Success: Got {len(result)} instruments")
    except Exception as e:
        print(f"❌ Error: {e}")
    
    # Test 2: Call with no parameters
    try:
        print("\n🧪 Test 2: Direct call to instruments() with no params")
        result = client.kite.instruments()
        print(f"✅ Success: Got {len(result)} instruments")
    except Exception as e:
        print(f"❌ Error: {e}")
    
    # Test 3: Check what parameters instruments() expects
    try:
        print("\n🧪 Test 3: Checking instruments method signature")
        import inspect
        sig = inspect.signature(client.kite.instruments)
        print(f"Method signature: {sig}")
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    test_api_calls()
