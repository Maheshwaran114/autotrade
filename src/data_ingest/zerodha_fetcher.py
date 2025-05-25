#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Zerodha Kite Connect data ingestion module for Bank Nifty Options trading system.
This module is responsible for fetching historical and real-time data from Zerodha.
"""

import os
import time
import logging
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Union, Tuple
from kiteconnect import KiteConnect

# Configure logging
logger = logging.getLogger(__name__)


class ZerodhaDataFetcher:
    """Fetches market data from Zerodha using Kite Connect API"""

    def __init__(self, api_key: str = None, api_secret: str = None, access_token: str = None):
        """
        Initialize the Zerodha data fetcher with API credentials.
        
        Args:
            api_key: Zerodha API key
            api_secret: Zerodha API secret
            access_token: Kite Connect access token (if already generated)
        """
        self.api_key = api_key or os.environ.get("KITE_API_KEY")
        self.api_secret = api_secret or os.environ.get("KITE_API_SECRET")
        self.access_token = access_token or os.environ.get("KITE_ACCESS_TOKEN")
        
        if not self.api_key or not self.api_secret:
            logger.warning("Zerodha API credentials not provided or found in environment variables")
        
        self.kite = self._initialize_kite()
        
    def _initialize_kite(self) -> Optional[KiteConnect]:
        """
        Initialize the KiteConnect object with the API key and set access token if available.
        
        Returns:
            KiteConnect object or None if initialization failed
        """
        try:
            kite = KiteConnect(api_key=self.api_key)
            
            if self.access_token:
                kite.set_access_token(self.access_token)
                logger.info("KiteConnect initialized with existing access token")
            else:
                logger.warning("Access token not provided, authentication flow required")
            
            return kite
        except Exception as e:
            logger.error(f"Failed to initialize KiteConnect: {e}")
            return None
    
    def generate_session(self, request_token: str) -> bool:
        """
        Generate a session using the request token obtained from the login flow.
        
        Args:
            request_token: Token obtained from the Kite Connect login redirect URL
            
        Returns:
            bool: True if session was successfully generated
        """
        try:
            data = self.kite.generate_session(
                request_token=request_token,
                api_secret=self.api_secret
            )
            
            self.access_token = data["access_token"]
            self.kite.set_access_token(self.access_token)
            
            # Update environment variable or save to secure storage as needed
            os.environ["KITE_ACCESS_TOKEN"] = self.access_token
            
            logger.info("Session generated successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to generate session: {e}")
            return False

    def get_historical_data(self, 
                           symbol: str,
                           exchange: str = "NSE",
                           interval: str = "day",
                           days: int = 30,
                           start_date: Optional[datetime] = None,
                           end_date: Optional[datetime] = None) -> pd.DataFrame:
        """
        Fetch historical market data from Zerodha
        
        Args:
            symbol: Symbol to fetch data for (e.g., 'NIFTY BANK')
            exchange: Exchange to fetch from (NSE, BSE, NFO, etc.)
            interval: Time interval for data (minute, day, etc.)
            days: Number of days of data to fetch (if start_date not provided)
            start_date: Start date for historical data
            end_date: End date for historical data (defaults to today)
            
        Returns:
            DataFrame containing historical price data
        """
        try:
            # If no start_date is provided, calculate based on days
            if not start_date:
                start_date = datetime.now() - timedelta(days=days)
                
            # If no end_date is provided, use today
            if not end_date:
                end_date = datetime.now()
                
            # Convert datetime objects to string in expected format
            from_date_str = start_date.strftime("%Y-%m-%d")
            to_date_str = end_date.strftime("%Y-%m-%d")
            
            # Get instrument token for the symbol
            # For simplicity, NIFTY BANK index is used here
            instrument_token = self._get_instrument_token(symbol, exchange)
            
            if not instrument_token:
                logger.error(f"Could not find instrument token for {symbol} on {exchange}")
                return pd.DataFrame()
                
            # Fetch historical data
            data = self.kite.historical_data(
                instrument_token=instrument_token,
                from_date=from_date_str,
                to_date=to_date_str,
                interval=interval
            )
            
            # Convert to DataFrame
            df = pd.DataFrame(data)
            
            # Add symbol and exchange information
            df['symbol'] = symbol
            df['exchange'] = exchange
            
            logger.info(f"Successfully fetched historical data for {symbol} ({len(df)} records)")
            return df
            
        except Exception as e:
            logger.error(f"Failed to fetch historical data: {e}")
            return pd.DataFrame()
    
    def _get_instrument_token(self, symbol: str, exchange: str) -> Optional[int]:
        """
        Get the instrument token for a given symbol and exchange
        
        Args:
            symbol: Symbol to fetch token for
            exchange: Exchange where the symbol is traded
            
        Returns:
            Instrument token (int) or None if not found
        """
        try:
            # Get all instruments and search for the specified symbol
            instruments = self.kite.instruments(exchange)
            
            # For Bank Nifty index
            if symbol.upper() in ["NIFTY BANK", "BANKNIFTY"]:
                # Handle Bank Nifty index specially
                for instrument in instruments:
                    if instrument["name"].upper() == "NIFTY BANK":
                        return instrument["instrument_token"]
            
            # For other symbols
            for instrument in instruments:
                if instrument["tradingsymbol"].upper() == symbol.upper():
                    return instrument["instrument_token"]
            
            logger.warning(f"Instrument token not found for {symbol} on {exchange}")
            return None
            
        except Exception as e:
            logger.error(f"Error fetching instrument token: {e}")
            return None
    
    def get_option_chain(self, 
                        index_name: str = "BANKNIFTY",
                        expiry_date: Optional[datetime] = None) -> Dict:
        """
        Fetch the option chain for Bank Nifty
        
        Args:
            index_name: Index name (default BANKNIFTY)
            expiry_date: Expiry date, defaults to the nearest expiry
            
        Returns:
            Dict containing option chain data
        """
        try:
            # Get all NFO instruments
            all_instruments = self.kite.instruments("NFO")
            
            # Filter for Bank Nifty options
            banknifty_options = [
                inst for inst in all_instruments 
                if index_name in inst["tradingsymbol"]
            ]
            
            # If no specific expiry provided, find the nearest one
            if not expiry_date:
                # Extract all available expiry dates
                unique_expiry_dates = list(set([
                    opt["expiry"] for opt in banknifty_options
                ]))
                
                # Sort and get the nearest expiry
                nearest_expiry = sorted(unique_expiry_dates)[0] if unique_expiry_dates else None
                expiry_date = nearest_expiry
            
            # Filter for the specific expiry
            filtered_options = [
                opt for opt in banknifty_options 
                if opt["expiry"] == expiry_date
            ]
            
            # Separate calls and puts
            calls = [opt for opt in filtered_options if opt["instrument_type"] == "CE"]
            puts = [opt for opt in filtered_options if opt["instrument_type"] == "PE"]
            
            # Get the current spot price for Bank Nifty
            spot_price = self._get_spot_price("NIFTY BANK", "NSE")
            
            # Structure the option chain
            option_chain = {
                "timestamp": datetime.now().isoformat(),
                "underlying": index_name,
                "spot_price": spot_price,
                "expiry_date": expiry_date.strftime("%Y-%m-%d") if isinstance(expiry_date, datetime) else expiry_date,
                "options": {
                    "calls": calls,
                    "puts": puts
                }
            }
            
            logger.info(f"Successfully fetched option chain for {index_name} with {len(calls)} calls and {len(puts)} puts")
            return option_chain
            
        except Exception as e:
            logger.error(f"Failed to fetch option chain: {e}")
            return {
                "timestamp": datetime.now().isoformat(),
                "underlying": index_name,
                "spot_price": None,
                "options": {
                    "calls": [],
                    "puts": []
                }
            }
    
    def _get_spot_price(self, symbol: str, exchange: str) -> Optional[float]:
        """
        Get the current spot price for a symbol
        
        Args:
            symbol: Symbol to fetch price for
            exchange: Exchange where the symbol is traded
            
        Returns:
            Current spot price or None if not available
        """
        try:
            # Get instrument token
            instrument_token = self._get_instrument_token(symbol, exchange)
            
            if not instrument_token:
                return None
                
            # Get quote
            quote = self.kite.quote(f"{exchange}:{symbol}")
            
            # Extract last price
            return quote[f"{exchange}:{symbol}"]["last_price"]
            
        except Exception as e:
            logger.error(f"Failed to get spot price for {symbol}: {e}")
            return None
    
    def save_data_to_csv(self, data: pd.DataFrame, filename: str) -> bool:
        """
        Save fetched data to a CSV file in the data/raw directory
        
        Args:
            data: DataFrame to save
            filename: Name of the file (without directory)
            
        Returns:
            bool: True if successfully saved
        """
        try:
            # Create directory if it doesn't exist
            os.makedirs("data/raw", exist_ok=True)
            
            # Save to CSV
            filepath = f"data/raw/{filename}"
            data.to_csv(filepath, index=False)
            logger.info(f"Data successfully saved to {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save data to CSV: {e}")
            return False


# For testing purposes
if __name__ == "__main__":
    # Configure logging for testing
    logging.basicConfig(level=logging.INFO)
    
    # Test with dummy credentials (replace with actual for real usage)
    fetcher = ZerodhaDataFetcher(
        api_key="your_api_key_here",
        api_secret="your_api_secret_here"
    )
    
    # For demo purposes, print a message
    print("ZerodhaDataFetcher initialized. Cannot fetch real data without valid credentials.")
    print("To use this module, provide valid Zerodha API credentials.")
