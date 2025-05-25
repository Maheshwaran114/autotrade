#!/usr/bin/env python3
"""Simple test to check Bank Nifty futures contracts."""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from src.data_ingest.zerodha_client import ZerodhaClient

def main():
    print("🔍 Testing Bank Nifty futures contracts...")
    
    # Initialize client
    client = ZerodhaClient(credentials_path=str(project_root / "config" / "credentials.json"))
    
    if not client.login():
        print("❌ Authentication failed")
        return
    
    print("✅ Connected to Zerodha API")
    
    # Get NFO instruments
    try:
        print("📊 Fetching NFO instruments...")
        nfo_instruments = client.kite.instruments("NFO")
        print(f"📊 Total NFO instruments: {len(nfo_instruments)}")
        
        # Filter Bank Nifty futures
        banknifty_futures = [
            inst for inst in nfo_instruments 
            if inst.get('name', '').upper() == 'BANKNIFTY' and inst.get('instrument_type') == 'FUT'
        ]
        
        print(f"📊 Bank Nifty futures found: {len(banknifty_futures)}")
        
        for contract in sorted(banknifty_futures, key=lambda x: x['expiry']):
            print(f"  - {contract['tradingsymbol']} | Expiry: {contract['expiry']} | Token: {contract['instrument_token']}")
            
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
