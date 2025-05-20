# tests/test_smoke.py
"""
Smoke tests for the Bank Nifty Options Trading System.
Validates that all modules load correctly and basic functionality works.
"""

import unittest
import sys
import os
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Add the src directory to the path so we can import our modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


class SmokeTest(unittest.TestCase):
    """Smoke tests for the trading system"""

    def test_modules_import(self):
        """Test that all modules can be imported without errors"""
        try:
            import sys
            from pathlib import Path
            # Add src to path for imports to work properly
            project_root = Path(__file__).parent.parent
            sys.path.insert(0, str(project_root.joinpath('src')))
            
            from data_ingest.fetch_data import MarketDataFetcher
            from ml_models.day_classifier import DayClassifier
            from strategies.delta_theta import DeltaThetaStrategy
            from strategies.gamma_scalping import GammaScalpingStrategy
            from execution.order_manager import OrderManager
            from dashboard.app import DashboardApp
            
            # If we got here, all imports worked
            self.assertTrue(True)
        except ImportError as e:
            self.fail(f"Failed to import modules: {e}")
            
    def test_data_fetcher(self):
        """Test that the MarketDataFetcher works"""
        import sys
        from pathlib import Path
        project_root = Path(__file__).parent.parent
        sys.path.insert(0, str(project_root.joinpath('src')))
        
        from data_ingest.fetch_data import MarketDataFetcher
        
        fetcher = MarketDataFetcher()
        option_chain = fetcher.get_option_chain("BANKNIFTY")
        
        # Verify the option chain has the expected structure
        self.assertEqual(option_chain["underlying"], "BANKNIFTY")
        self.assertIn("spot_price", option_chain)
        self.assertIn("options", option_chain)
        self.assertIn("calls", option_chain["options"])
        self.assertIn("puts", option_chain["options"])
        
    def test_day_classifier(self):
        """Test that the DayClassifier works"""
        import sys
        from pathlib import Path
        project_root = Path(__file__).parent.parent
        sys.path.insert(0, str(project_root.joinpath('src')))
        
        from ml_models.day_classifier import DayClassifier
        
        classifier = DayClassifier()
        market_data = {
            "open": 48000.0,
            "high": 48500.0,
            "low": 47800.0,
            "close": 48200.0,
            "volume": 1000000
        }
        
        result = classifier.classify_day(market_data)
        
        # Verify the classification result has the expected structure
        self.assertIn("classification", result)
        self.assertIn("probabilities", result)
        self.assertIn("confidence", result)
        self.assertIn("timestamp", result)
        
    def test_order_manager(self):
        """Test that the OrderManager works"""
        import sys
        from pathlib import Path
        project_root = Path(__file__).parent.parent
        sys.path.insert(0, str(project_root.joinpath('src')))
        
        from execution.order_manager import OrderManager
        
        order_manager = OrderManager()
        
        # Create a test order
        order = order_manager.create_order(
            symbol="BANKNIFTY25MAY4800CE",
            order_type="LIMIT",
            direction="BUY",
            quantity=1,
            price=120.5
        )
        
        # Verify the order has the expected structure
        self.assertIn("order_id", order)
        self.assertEqual(order["symbol"], "BANKNIFTY25MAY4800CE")
        self.assertEqual(order["direction"], "BUY")
        self.assertEqual(order["status"], "CREATED")
        
        # Submit the order and check status
        submitted_order = order_manager.submit_order(order["order_id"])
        self.assertEqual(submitted_order["status"], "SUBMITTED")
        
    def test_dashboard_initialization(self):
        """Test that the Dashboard app initializes correctly"""
        import sys
        from pathlib import Path
        project_root = Path(__file__).parent.parent
        sys.path.insert(0, str(project_root.joinpath('src')))
        
        from dashboard.app import DashboardApp
        
        dashboard = DashboardApp()
        dashboard.initialize_components()
        
        # Verify components were initialized
        self.assertIsNotNone(dashboard.data_fetcher)
        self.assertIsNotNone(dashboard.day_classifier)
        self.assertIsNotNone(dashboard.delta_theta_strategy)
        self.assertIsNotNone(dashboard.gamma_strategy)
        self.assertIsNotNone(dashboard.order_manager)
        
        # Run system check
        system_check = dashboard.run_system_check()
        
        # Verify system check results
        for component, result in system_check["components"].items():
            self.assertEqual(result["status"], "OK")


if __name__ == "__main__":
    unittest.main()
