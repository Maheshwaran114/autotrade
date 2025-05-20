# src/dashboard/app.py
"""
Dashboard application for Bank Nifty trading system.
Provides visualization and control interface for the trading system.
"""

import logging
import sys
from typing import Dict, List, Optional
import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)

# Import modules for testing
try:
    import sys
    from pathlib import Path
    # Add parent directory to path
    current_dir = Path(__file__).parent
    parent_dir = current_dir.parent
    if str(parent_dir) not in sys.path:
        sys.path.insert(0, str(parent_dir))
    
    from data_ingest.fetch_data import MarketDataFetcher
    from ml_models.day_classifier import DayClassifier
    from strategies.delta_theta import DeltaThetaStrategy
    from strategies.gamma_scalping import GammaScalpingStrategy
    from execution.order_manager import OrderManager
    modules_loaded = True
except ImportError as e:
    logger.error(f"Error importing modules: {e}")
    modules_loaded = False
    

class DashboardApp:
    """Main dashboard application for the trading system"""

    def __init__(self):
        """Initialize the dashboard application"""
        logger.info("Initializing dashboard application")
        self.data_fetcher = None
        self.day_classifier = None
        self.delta_theta_strategy = None
        self.gamma_strategy = None
        self.order_manager = None
    
    def initialize_components(self):
        """Initialize all system components"""
        logger.info("Initializing system components")
        
        self.data_fetcher = MarketDataFetcher()
        self.day_classifier = DayClassifier()
        self.delta_theta_strategy = DeltaThetaStrategy()
        self.gamma_strategy = GammaScalpingStrategy()
        self.order_manager = OrderManager()
        
        logger.info("All components initialized successfully")
    
    def run_system_check(self) -> Dict:
        """
        Run a system check to verify all components are working
        
        Returns:
            Dict: System check results
        """
        results = {
            "timestamp": datetime.datetime.now().isoformat(),
            "components": {}
        }
        
        # Check data fetcher
        try:
            option_chain = self.data_fetcher.get_option_chain()
            results["components"]["data_fetcher"] = {
                "status": "OK",
                "details": f"Successfully fetched option chain for {option_chain['underlying']}"
            }
        except Exception as e:
            results["components"]["data_fetcher"] = {
                "status": "ERROR",
                "details": str(e)
            }
            
        # Check day classifier
        try:
            market_data = {"open": 48000, "high": 48500, "low": 47800, "close": 48200}
            classification = self.day_classifier.classify_day(market_data)
            results["components"]["day_classifier"] = {
                "status": "OK",
                "details": f"Day classified as {classification['classification']}"
            }
        except Exception as e:
            results["components"]["day_classifier"] = {
                "status": "ERROR",
                "details": str(e)
            }
            
        # Check strategies
        try:
            analysis = self.delta_theta_strategy.analyze_market(option_chain)
            signals = self.delta_theta_strategy.generate_signals(option_chain, analysis)
            results["components"]["delta_theta_strategy"] = {
                "status": "OK",
                "details": f"Generated {len(signals)} trading signals"
            }
        except Exception as e:
            results["components"]["delta_theta_strategy"] = {
                "status": "ERROR",
                "details": str(e)
            }
            
        # Check order manager
        try:
            positions = self.order_manager.get_positions()
            results["components"]["order_manager"] = {
                "status": "OK",
                "details": f"Retrieved {len(positions)} positions"
            }
        except Exception as e:
            results["components"]["order_manager"] = {
                "status": "ERROR",
                "details": str(e)
            }
            
        return results
    
    def start(self):
        """Start the dashboard application"""
        logger.info("Starting the dashboard application")
        
        # Initialize components
        self.initialize_components()
        
        # Run system check
        system_check = self.run_system_check()
        
        # Log results
        for component, result in system_check["components"].items():
            logger.info(f"Component {component}: {result['status']} - {result['details']}")
        
        logger.info("Dashboard application started successfully")
        
        return system_check


def main():
    """Main entry point for the dashboard application"""
    logger.info("Bank Nifty Options Trading System")
    
    if modules_loaded:
        logger.info("All modules loaded successfully")
        dashboard = DashboardApp()
        dashboard.start()
    else:
        logger.error("Failed to load all required modules")


if __name__ == "__main__":
    main()
