#!/usr/bin/env python3
"""
Network connectivity test for Zerodha API
"""

import sys
from pathlib import Path
import requests
import socket
import time

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from src.data_ingest.zerodha_client import ZerodhaClient

def test_network_connectivity():
    """Test network connectivity to Zerodha API."""
    
    print("🔍 Testing network connectivity to Zerodha API...")
    
    # Test 1: DNS resolution
    print("\n1️⃣ Testing DNS resolution for api.kite.trade...")
    try:
        ip = socket.gethostbyname('api.kite.trade')
        print(f"✅ DNS resolved: api.kite.trade -> {ip}")
    except Exception as e:
        print(f"❌ DNS resolution failed: {e}")
        return False
    
    # Test 2: HTTP connectivity
    print("\n2️⃣ Testing HTTP connectivity...")
    try:
        response = requests.get('https://api.kite.trade', timeout=10)
        print(f"✅ HTTP connection successful: Status {response.status_code}")
    except Exception as e:
        print(f"❌ HTTP connection failed: {e}")
        return False
    
    # Test 3: API authentication
    print("\n3️⃣ Testing API authentication...")
    try:
        client = ZerodhaClient()
        if client.kite and client.access_token:
            # Try a simple API call
            profile = client.kite.profile()
            print(f"✅ API authentication successful: User {profile.get('user_name', 'Unknown')}")
        else:
            print("❌ API authentication failed: No valid token")
            return False
    except Exception as e:
        print(f"❌ API authentication failed: {e}")
        return False
    
    # Test 4: Instruments API call
    print("\n4️⃣ Testing instruments API call...")
    try:
        nse_instruments = client.kite.instruments("NSE")
        print(f"✅ NSE instruments call successful: {len(nse_instruments)} instruments")
    except Exception as e:
        print(f"❌ NSE instruments call failed: {e}")
        return False
    
    # Test 5: Historical data API call
    print("\n5️⃣ Testing historical data API call...")
    try:
        # Test with a known instrument token (NSE Bank Nifty)
        historical_data = client.kite.historical_data(
            instrument_token=260105,  # NSE:NIFTY BANK
            from_date="2025-05-23", 
            to_date="2025-05-23",
            interval="day"
        )
        print(f"✅ Historical data call successful: {len(historical_data)} records")
    except Exception as e:
        print(f"❌ Historical data call failed: {e}")
        return False
    
    print("\n🎉 All network connectivity tests passed!")
    print("📊 Current API delay: 1.0 seconds (optimal for 3 req/sec limit)")
    print("🚀 Network is ready for data collection")
    return True

if __name__ == "__main__":
    success = test_network_connectivity()
    if not success:
        print("\n💡 Troubleshooting steps:")
        print("   1. Check your internet connection")
        print("   2. Try changing DNS to 8.8.8.8 or 1.1.1.1") 
        print("   3. Check if any firewall/proxy is blocking api.kite.trade")
        print("   4. Wait a few minutes and try again (temporary server issue)")
        exit(1)
    else:
        print("\n✅ Network is healthy - the script should work fine!")
        exit(0)
