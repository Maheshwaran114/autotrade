# src/strategies/gamma_scalping.py
"""
Gamma Scalping strategy for Bank Nifty options trading.
Capitalizes on large price movements by dynamically hedging options positions.
"""

import logging
from typing import Dict, List, Optional, Union
import datetime

# Configure logging
logger = logging.getLogger(__name__)


class GammaScalpingStrategy:
    """Implementation of Gamma Scalping strategy for options trading"""

    def __init__(self, settings: Optional[Dict] = None):
        """
        Initialize the Gamma Scalping strategy.
        
        Args:
            settings: Strategy configuration parameters
        """
        self.settings = settings or {
            "rehedge_threshold": 50,  # Points movement to trigger rehedging
            "max_position_size": 5,
            "target_gamma": 0.5,
            "max_vega_exposure": 200,
            "hedge_ratio": 0.8  # Percentage of delta to hedge
        }
        logger.info("GammaScalpingStrategy initialized")
    
    def calculate_hedge_qty(self, options_position: Dict) -> int:
        """
        Calculate the required underlying quantity to hedge.
        
        Args:
            options_position: Details of current options position
            
        Returns:
            int: Quantity of underlying to buy/sell for hedging
        """
        position_delta = options_position.get("total_delta", 0.0)
        hedge_qty = int(position_delta * self.settings["hedge_ratio"] * -1)
        
        logger.info(f"Calculated hedge quantity: {hedge_qty}")
        return hedge_qty
    
    def is_rehedge_required(self, last_hedge_price: float, current_price: float) -> bool:
        """
        Determine if rehedging is required based on price movement.
        
        Args:
            last_hedge_price: Price level at last hedging action
            current_price: Current price of the underlying
            
        Returns:
            bool: True if rehedging is required
        """
        price_movement = abs(current_price - last_hedge_price)
        should_rehedge = price_movement > self.settings["rehedge_threshold"]
        
        if should_rehedge:
            logger.info(f"Rehedge required - price moved by {price_movement:.2f} points")
        
        return should_rehedge
    
    def generate_signals(self, option_chain: Dict, current_position: Dict) -> List[Dict]:
        """
        Generate trading signals for gamma scalping.
        
        Args:
            option_chain: Option chain data for the underlying
            current_position: Current portfolio position details
            
        Returns:
            List[Dict]: List of trading signals
        """
        current_price = option_chain.get("spot_price", 0.0)
        last_hedge_price = current_position.get("last_hedge_price", current_price)
        
        signals = []
        
        # Check if we need a new options position
        if not current_position.get("has_options", False):
            # Establish initial ATM/slightly OTM options position
            strike = round(current_price / 100) * 100  # Round to nearest 100
            
            # Buy straddle or strangle
            signals.append({
                "action": "BUY",
                "instrument_type": "OPTION",
                "option_type": "CALL",
                "strike": strike,
                "expiry": "2025-05-28",
                "quantity": 1,
                "reason": "Establish gamma scalping position - long call"
            })
            
            signals.append({
                "action": "BUY",
                "instrument_type": "OPTION",
                "option_type": "PUT",
                "strike": strike - 500,  # Slight strangle
                "expiry": "2025-05-28",
                "quantity": 1,
                "reason": "Establish gamma scalping position - long put"
            })
            
            # Initial delta hedge
            hedge_qty = self.calculate_hedge_qty({
                "total_delta": 0.6  # Approximated initial delta
            })
            
            signals.append({
                "action": "SELL" if hedge_qty > 0 else "BUY",
                "instrument_type": "FUTURE",
                "quantity": abs(hedge_qty),
                "reason": "Initial delta hedge"
            })
            
        # Check if rehedging is needed
        elif self.is_rehedge_required(last_hedge_price, current_price):
            hedge_qty = self.calculate_hedge_qty(current_position)
            
            # Only generate signal if there's a meaningful hedge quantity
            if abs(hedge_qty) > 0:
                signals.append({
                    "action": "SELL" if hedge_qty > 0 else "BUY",
                    "instrument_type": "FUTURE",
                    "quantity": abs(hedge_qty),
                    "reason": f"Rehedge after {current_price - last_hedge_price:.2f} point move"
                })
        
        return signals


# For testing purposes
if __name__ == "__main__":
    strategy = GammaScalpingStrategy()
    
    # Sample data
    option_chain = {
        "underlying": "BANKNIFTY",
        "spot_price": 48000.0,
        "timestamp": datetime.datetime.now().isoformat()
    }
    
    current_position = {
        "has_options": False
    }
    
    signals = strategy.generate_signals(option_chain, current_position)
    
    print(f"Generated {len(signals)} trading signals")
    for signal in signals:
        print(f"{signal['action']} {signal.get('option_type', '')} {signal.get('strike', '')} {signal.get('quantity', '')}")
