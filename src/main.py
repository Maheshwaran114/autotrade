# src/main.py
"""
Main entry point for the Bank Nifty Options Trading System.
Orchestrates the various components of the trading system.
"""

import argparse
import logging
import sys
import os
from typing import Dict, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("trading_system.log")
    ]
)

logger = logging.getLogger(__name__)

# Import system components
from dashboard.app import DashboardApp
from data_ingest.fetch_data import MarketDataFetcher
from ml_models.day_classifier import DayClassifier
from strategies.delta_theta import DeltaThetaStrategy
from strategies.gamma_scalping import GammaScalpingStrategy
from execution.order_manager import OrderManager


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Bank Nifty Options Trading System")
    
    parser.add_argument("--mode", choices=["live", "backtest", "paper"], 
                        default="paper", help="Trading mode")
    
    parser.add_argument("--strategy", choices=["delta_theta", "gamma_scalping", "auto"],
                        default="auto", help="Trading strategy to use")
    
    parser.add_argument("--config", type=str, default="config/config.json",
                        help="Path to configuration file")
    
    parser.add_argument("--dashboard", action="store_true", default=False,
                        help="Launch the dashboard UI")
    
    parser.add_argument("--debug", action="store_true", default=False,
                        help="Enable debug logging")
    
    return parser.parse_args()


def load_configuration(config_path: str) -> Dict:
    """
    Load configuration from a JSON file.
    
    Args:
        config_path: Path to the configuration file
        
    Returns:
        Dict: Configuration parameters
    """
    import json
    
    try:
        if os.path.exists(config_path):
            with open(config_path, "r") as f:
                return json.load(f)
        else:
            logger.warning(f"Configuration file {config_path} not found, using defaults")
            return {}
    except Exception as e:
        logger.error(f"Error loading configuration: {e}")
        return {}


def main():
    """Main entry point for the trading system"""
    args = parse_arguments()
    
    # Set logging level based on debug flag
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
        logger.debug("Debug logging enabled")
    
    logger.info(f"Starting Bank Nifty Options Trading System in {args.mode} mode")
    
    # Load configuration
    config = load_configuration(args.config)
    logger.info(f"Configuration loaded from {args.config}")
    
    # If dashboard flag is set, launch dashboard
    if args.dashboard:
        logger.info("Launching dashboard application")
        dashboard = DashboardApp()
        dashboard.start()
        return
    
    # Initialize components
    market_data = MarketDataFetcher(api_key=config.get("api_key"))
    classifier = DayClassifier(model_path=config.get("model_path"))
    
    # Select strategy based on command line or configuration
    strategy_name = args.strategy
    if strategy_name == "auto":
        strategy_name = config.get("strategy", "delta_theta")
    
    if strategy_name == "delta_theta":
        strategy = DeltaThetaStrategy(settings=config.get("delta_theta_settings"))
    else:
        strategy = GammaScalpingStrategy(settings=config.get("gamma_scalping_settings"))
    
    logger.info(f"Using {strategy_name} strategy")
    
    # Initialize order manager with broker configuration
    order_manager = OrderManager(broker_config=config.get("broker_config"))
    
    # Execute system check
    option_chain = market_data.get_option_chain()
    market_data_sample = {
        "open": option_chain["spot_price"] * 0.99,
        "high": option_chain["spot_price"] * 1.01,
        "low": option_chain["spot_price"] * 0.98,
        "close": option_chain["spot_price"],
        "volume": 1000000
    }
    
    # Analyze market conditions
    day_classification = classifier.classify_day(market_data_sample)
    logger.info(f"Day classified as: {day_classification['classification']}")
    
    # Generate trading signals
    if strategy_name == "delta_theta":
        market_analysis = strategy.analyze_market(option_chain)
        signals = strategy.generate_signals(option_chain, market_analysis)
    else:
        current_position = order_manager.get_positions()
        signals = strategy.generate_signals(option_chain, current_position)
    
    # Log generated signals
    logger.info(f"Generated {len(signals)} trading signals")
    for idx, signal in enumerate(signals):
        logger.info(f"Signal {idx+1}: {signal['action']} {signal.get('option_type', '')} "
                   f"{signal.get('strike', '')} {signal.get('quantity', '')}")
    
    # Execute trades if in live mode
    if args.mode == "live":
        logger.info("Executing trades in LIVE mode")
        # Implementation of live trading execution
    elif args.mode == "paper":
        logger.info("Paper trading mode - no actual orders will be placed")
        # Implementation of paper trading simulation
    else:
        logger.info("Backtest mode - running historical simulation")
        # Implementation of backtest logic
    
    logger.info("Trading system execution completed")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.info("Trading system stopped by user")
    except Exception as e:
        logger.exception(f"Unhandled exception: {e}")
        sys.exit(1)
