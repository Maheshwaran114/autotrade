#!/usr/bin/env python3
"""
Debug script to check data structure from zerodha client
"""

import sys
import pandas as pd
from datetime import datetime, date
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from src.data_ingest.zerodha_client import ZerodhaClient

def test_data_structure():
    """Test what the data structure looks like"""
    client = ZerodhaClient(credentials_path=str(project_root / "config" / "credentials.json"))
    
    if not client.login():
        print("❌ Authentication failed")
        return
    
    print("✅ Authentication successful")
    
    # Test index data fetch
    try:
        print("\n🧪 Testing index data structure...")
        test_date = date(2024, 6, 20)  # Date we know has data from test script
        
        index_data = client.fetch_historical_data(
            instrument="NSE:NIFTY BANK",
            interval="minute",
            from_date=test_date,
            to_date=test_date
        )
        
        if index_data:
            print(f"✅ Got {len(index_data)} index records")
            print(f"Sample record: {index_data[0]}")
            print(f"Keys: {list(index_data[0].keys())}")
            
            # Convert to DataFrame to see columns
            df = pd.DataFrame(index_data)
            print(f"DataFrame columns: {list(df.columns)}")
            print(f"DataFrame shape: {df.shape}")
            print(df.head())
        else:
            print("❌ No index data received")
            
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_data_structure()
