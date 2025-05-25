#!/usr/bin/env python3
"""
Simple test to verify our volume fix implementation.
"""

import sys
import logging
from pathlib import Path
from datetime import date

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def simple_test():
    """Simple test to check basic functionality"""
    logger.info("🧪 Starting simple volume fix test")
    
    try:
        # Test import
        from src.data_ingest.zerodha_client import ZerodhaClient
        logger.info("✅ ZerodhaClient import successful")
        
        # Test client initialization
        client = ZerodhaClient(credentials_path=str(project_root / "config" / "credentials.json"))
        logger.info("✅ Client initialization successful")
        
        # Test login
        login_success = client.login()
        logger.info(f"📝 Login result: {login_success}")
        
        if login_success:
            logger.info("🎉 Authentication successful - ready to test volume fix")
            
            # Test NFO instruments fetch
            try:
                instruments = client.kite.instruments("NFO")
                logger.info(f"✅ Fetched {len(instruments)} NFO instruments")
                
                # Look for Bank Nifty futures
                import pandas as pd
                df = pd.DataFrame(instruments)
                futures = df[(df['name'] == 'BANKNIFTY') & (df['instrument_type'] == 'FUT')]
                logger.info(f"📊 Found {len(futures)} Bank Nifty futures contracts")
                
                if len(futures) > 0:
                    sample = futures.iloc[0]
                    logger.info(f"📋 Sample contract: {sample['tradingsymbol']} (token: {sample['instrument_token']})")
                    logger.info("✅ Volume fix prerequisites met!")
                    return True
                else:
                    logger.error("❌ No Bank Nifty futures found")
                    
            except Exception as e:
                logger.error(f"❌ Error fetching instruments: {e}")
        else:
            logger.error("❌ Authentication failed")
            
        return False
        
    except Exception as e:
        logger.error(f"❌ Test failed: {e}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        return False

if __name__ == "__main__":
    success = simple_test()
    exit_code = 0 if success else 1
    sys.exit(exit_code)
