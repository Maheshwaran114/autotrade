#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Test script to verify basic functionality and then run unified data collection.
"""

import os
import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

print("🔍 Testing basic imports...")

try:
    from src.data_ingest.zerodha_client import ZerodhaClient
    print("✓ ZerodhaClient imported successfully")
except Exception as e:
    print(f"❌ ZerodhaClient import failed: {e}")
    sys.exit(1)

try:
    import pandas as pd
    import numpy as np
    print("✓ Data processing libraries imported successfully")
except Exception as e:
    print(f"❌ Data libraries import failed: {e}")
    sys.exit(1)

print("\n🏦 Testing Zerodha API connection...")

try:
    client = ZerodhaClient(credentials_path=str(project_root / "config" / "credentials.json"))
    print("✓ ZerodhaClient initialized")
    
    # Test login
    if client.login():
        print("✓ API authentication successful")
    else:
        print("❌ API authentication failed")
        sys.exit(1)
        
except Exception as e:
    print(f"❌ API connection failed: {e}")
    sys.exit(1)

print("\n📁 Checking data directories...")
os.makedirs(project_root / "data" / "raw", exist_ok=True)
os.makedirs(project_root / "data" / "processed", exist_ok=True)
print("✓ Data directories ready")

print("\n🚀 All prerequisites met! Now running unified data collection...")
print("=" * 60)

# Now import and run the main data collection
try:
    exec(open(project_root / "scripts" / "fetch_unified_historical_data.py").read())
except Exception as e:
    print(f"❌ Error running unified data collection: {e}")
    import traceback
    traceback.print_exc()
