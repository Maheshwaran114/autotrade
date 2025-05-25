# src/data_ingest/zerodha_client.py
"""
Zerodha client module for the Bank Nifty Options Trading System.
This module is responsible for connecting to Zerodha's Kite API and fetching historical data.
"""

import os
import json
import logging
import pandas as pd
import time
from typing import Dict, List, Optional, Union
from datetime import datetime, timedelta, date
from kiteconnect import KiteConnect

# Configure logging
logger = logging.getLogger(__name__)

class ZerodhaClient:
    """Client for interacting with Zerodha's Kite Connect API"""

    def __init__(self, credentials_path: str = "config/credentials.json"):
        """
        Initialize Zerodha client with API credentials.
        
        Args:
            credentials_path: Path to the credentials JSON file
        """
        self.api_key = None
        self.api_secret = None
        self.kite = None
        self.access_token = None
        self.credentials_path = credentials_path
        self._load_credentials()
        self._initialize_kite()
        
    def _load_credentials(self) -> None:
        """Load API credentials from the credentials file"""
        try:
            if not os.path.exists(self.credentials_path):
                logger.error(f"Credentials file not found: {self.credentials_path}")
                return
                
            with open(self.credentials_path, 'r') as file:
                credentials = json.load(file)
                
            self.api_key = credentials.get('api_key')
            self.api_secret = credentials.get('api_secret')
            self.access_token = credentials.get('access_token')
            
            if not self.api_key or not self.api_secret:
                logger.error("API key or secret not found in credentials file")
                
        except Exception as e:
            logger.error(f"Error loading credentials: {e}")
    
    def _initialize_kite(self) -> None:
        """Initialize KiteConnect client with API key"""
        if not self.api_key:
            logger.error("Cannot initialize Kite: API key not available")
            return
            
        self.kite = KiteConnect(api_key=self.api_key)
        
        # Set access token if available
        if self.access_token:
            self.kite.set_access_token(self.access_token)
            logger.info("Kite initialized with existing access token")
        else:
            logger.info("Kite initialized without access token. Login required.")
    
    def login(self, request_token: str = None) -> bool:
        """
        Authenticate with Kite Connect using request token or saved credentials
        
        Args:
            request_token: Request token obtained after user login
            
        Returns:
            bool: True if login successful, False otherwise
        """
        if not self.kite:
            logger.error("Kite client not initialized")
            return False
        
        try:
            if request_token:
                # Generate session using request token
                data = self.kite.generate_session(request_token, api_secret=self.api_secret)
                self.access_token = data["access_token"]
                self.kite.set_access_token(self.access_token)
                
                # Save access token
                if os.path.exists(self.credentials_path):
                    with open(self.credentials_path, 'r') as file:
                        credentials = json.load(file)
                    
                    credentials['access_token'] = self.access_token
                    
                    with open(self.credentials_path, 'w') as file:
                        json.dump(credentials, file, indent=4)
                
                logger.info("Login successful")
                return True
            else:
                # Check if existing token works
                if self.access_token:
                    try:
                        # Test the token by fetching profile
                        self.kite.profile()
                        logger.info("Using existing access token")
                        return True
                    except:
                        logger.warning("Existing access token expired")
                
                logger.error("Request token required for login")
                return False
                
        except Exception as e:
            logger.error(f"Login failed: {e}")
            return False
    
    def get_login_url(self) -> str:
        """
        Get the login URL for user authentication
        
        Returns:
            str: Login URL
        """
        if not self.kite:
            logger.error("Kite client not initialized")
            return ""
        
        return self.kite.login_url()
    
    def fetch_historical_data(self, 
                            instrument: Union[str, int],
                            interval: str,
                            from_date: Union[datetime, date],
                            to_date: Union[datetime, date]) -> List[Dict]:
        """
        Fetch historical data for the given symbol and timeframe
        Handles API limits by chunking large date ranges
        
        Args:
            instrument: Symbol string (like "NSE:NIFTY BANK") or instrument token
            interval: Candle interval (minute, day, etc.)
            from_date: Start date
            to_date: End date
            
        Returns:
            List of dictionaries containing historical data
        """
        if not self.kite or not self.access_token:
            logger.error("Kite not initialized or not authenticated")
            return []
            
        try:
            # Handle instrument token vs symbol string
            if isinstance(instrument, str):
                # If it's a symbol string like "NSE:NIFTY BANK"
                instrument_token = self.get_instrument_token_by_symbol(instrument)
                if not instrument_token:
                    logger.error(f"Could not find instrument token for {instrument}")
                    return []
            else:
                instrument_token = instrument
            
            # Ensure dates are in the right format
            if isinstance(from_date, date):
                from_date = datetime.combine(from_date, datetime.min.time())
            if isinstance(to_date, date):
                to_date = datetime.combine(to_date, datetime.max.time())
            
            # Calculate date range and chunk if necessary
            total_days = (to_date - from_date).days
            max_chunk_days = 55 if interval == "minute" else 365  # Conservative limit for minute data
            
            all_data = []
            current_start = from_date
            
            while current_start < to_date:
                # Calculate chunk end date
                chunk_end = min(current_start + timedelta(days=max_chunk_days), to_date)
                
                from_str = current_start.strftime('%Y-%m-%d %H:%M:%S')
                to_str = chunk_end.strftime('%Y-%m-%d %H:%M:%S')
                
                logger.info(f"Fetching {interval} data chunk for {instrument} (token: {instrument_token}) from {from_str} to {to_str}")
                
                # Fetch the data chunk
                chunk_data = self.kite.historical_data(
                    instrument_token=instrument_token,
                    from_date=from_str,
                    to_date=to_str,
                    interval=interval
                )
                
                if chunk_data:
                    all_data.extend(chunk_data)
                    logger.info(f"Fetched {len(chunk_data)} records for chunk")
                
                # Move to next chunk
                current_start = chunk_end + timedelta(days=1)
                
                # Add delay between chunks to respect API limits
                time.sleep(0.5)
            
            logger.info(f"Total fetched {len(all_data)} records across all chunks")
            return all_data
            
        except Exception as e:
            logger.error(f"Error fetching historical data: {e}")
            return []
    
    def get_instrument_token(self, tradingsymbol: str, exchange: str = "NSE") -> Optional[int]:
        """
        Get the instrument token for a given trading symbol
        
        Args:
            tradingsymbol: Trading symbol (e.g., "BANKNIFTY")
            exchange: Exchange code (default: "NSE")
            
        Returns:
            int: Instrument token if found, None otherwise
        """
        if not self.kite:
            logger.error("Kite not initialized")
            return None
            
        try:
            instruments = self.kite.instruments(exchange)
            for instrument in instruments:
                if instrument['tradingsymbol'] == tradingsymbol:
                    return instrument['instrument_token']
            
            logger.error(f"Instrument {tradingsymbol} not found on {exchange}")
            return None
            
        except Exception as e:
            logger.error(f"Error fetching instrument token: {e}")
            return None
    
    def get_instrument_token_by_symbol(self, symbol_string: str) -> Optional[int]:
        """
        Get instrument token from symbol string like "NSE:NIFTY BANK"
        
        Args:
            symbol_string: Symbol in format "EXCHANGE:SYMBOL" (e.g., "NSE:NIFTY BANK")
            
        Returns:
            int: Instrument token if found, None otherwise
        """
        if not self.kite:
            logger.error("Kite not initialized")
            return None
            
        try:
            # Parse the symbol string
            if ':' in symbol_string:
                exchange, symbol = symbol_string.split(':', 1)
            else:
                exchange = "NSE"
                symbol = symbol_string
            
            # Get instruments for the exchange
            instruments = self.kite.instruments(exchange)
            
            # Look for matching symbol (handle variations like NIFTY BANK vs BANKNIFTY)
            for instrument in instruments:
                if (instrument['name'] == symbol or 
                    instrument['tradingsymbol'] == symbol or
                    instrument['name'].replace(' ', '') == symbol.replace(' ', '')):
                    logger.info(f"Found instrument: {instrument['name']} ({instrument['tradingsymbol']}) = {instrument['instrument_token']}")
                    return instrument['instrument_token']
            
            logger.error(f"Instrument '{symbol}' not found on {exchange}")
            return None
            
        except Exception as e:
            logger.error(f"Error getting instrument token for {symbol_string}: {e}")
            return None
    
    def fetch_option_chain(self, 
                          index_symbol: str = "BANKNIFTY", 
                          spot_price: float = None,
                          expiry_offset: int = 0) -> Dict:
        """
        Fetch option chain for Bank Nifty
        
        Args:
            index_symbol: Index symbol (default: "BANKNIFTY")
            spot_price: Current spot price, if None will be fetched
            expiry_offset: Expiry offset in weeks (0 = nearest)
            
        Returns:
            Dict containing option chain with calls and puts data
        """
        if not self.kite or not self.access_token:
            logger.error("Kite not initialized or not authenticated")
            return {}
            
        try:
            # Get all instruments
            instruments = self.kite.instruments("NFO")
            
            if spot_price is None:
                # Fetch current spot price
                quote = self.kite.quote(f"NSE:{index_symbol}")
                spot_price = quote[f"NSE:{index_symbol}"]["last_price"]
            
            # Filter for BANKNIFTY options
            options = [inst for inst in instruments if 
                      inst['name'] == index_symbol and 
                      (inst['instrument_type'] == 'CE' or inst['instrument_type'] == 'PE')]
            
            # Get unique expiry dates
            expiry_dates = sorted(list(set(opt['expiry'] for opt in options)))
            
            if not expiry_dates:
                logger.error("No expiry dates found for options")
                return {}
            
            # Choose the expiry date based on offset
            if expiry_offset < 0 or expiry_offset >= len(expiry_dates):
                logger.warning(f"Invalid expiry offset {expiry_offset}, using nearest expiry")
                expiry_offset = 0
                
            target_expiry = expiry_dates[expiry_offset]
            
            # Filter options for the target expiry
            expiry_options = [opt for opt in options if opt['expiry'] == target_expiry]
            
            # Filter for strikes around spot price
            buffer = 2000  # Filter strikes within ±2000 of spot price
            relevant_strikes = [opt for opt in expiry_options if 
                               (spot_price - buffer) <= opt['strike'] <= (spot_price + buffer)]
            
            # Separate calls and puts
            calls = [opt for opt in relevant_strikes if opt['instrument_type'] == 'CE']
            puts = [opt for opt in relevant_strikes if opt['instrument_type'] == 'PE']
            
            # Get current prices for calls
            call_tokens = {opt['instrument_token']: opt for opt in calls}
            if call_tokens:
                call_quotes = self.kite.quote(list(call_tokens.keys()))
                for token, quote_data in call_quotes.items():
                    call_tokens[int(token)]['last_price'] = quote_data['last_price']
                    call_tokens[int(token)]['volume'] = quote_data['volume']
                    call_tokens[int(token)]['oi'] = quote_data['oi']
            
            # Get current prices for puts
            put_tokens = {opt['instrument_token']: opt for opt in puts}
            if put_tokens:
                put_quotes = self.kite.quote(list(put_tokens.keys()))
                for token, quote_data in put_quotes.items():
                    put_tokens[int(token)]['last_price'] = quote_data['last_price']
                    put_tokens[int(token)]['volume'] = quote_data['volume']
                    put_tokens[int(token)]['oi'] = quote_data['oi']
            
            # Prepare option chain response
            option_chain = {
                "spot_price": spot_price,
                "expiry_date": target_expiry.strftime('%Y-%m-%d'),
                "options": {
                    "calls": list(call_tokens.values()),
                    "puts": list(put_tokens.values())
                }
            }
            
            logger.info(f"Fetched option chain with {len(calls)} calls and {len(puts)} puts")
            return option_chain
            
        except Exception as e:
            logger.error(f"Error fetching option chain: {e}")
            return {}
    
    def fetch_and_save_historical_data(self, 
                                     days: int = 365, 
                                     interval: str = "minute") -> str:
        """
        Fetch and save Bank Nifty historical data for a given period
        
        Args:
            days: Number of days to fetch data for
            interval: Data interval (minute, day, etc.)
            
        Returns:
            str: Path to the saved CSV file
        """
        try:
            # Get instrument token for Bank Nifty
            token = self.get_instrument_token("NIFTY BANK")
            if not token:
                logger.error("Failed to get instrument token for BANKNIFTY")
                return ""
            
            # Calculate date range
            to_date = datetime.now()
            from_date = to_date - timedelta(days=days)
            
            # Fetch data
            data = self.fetch_historical_data(
                symbol=token,
                interval=interval,
                from_date=from_date,
                to_date=to_date
            )
            
            if data.empty:
                logger.error("No data fetched")
                return ""
            
            # Create directory if it doesn't exist
            os.makedirs("data/raw", exist_ok=True)
            
            # Save to CSV
            today_str = datetime.now().strftime('%Y%m%d')
            filename = f"data/raw/banknifty_{today_str}.csv"
            data.to_csv(filename, index=False)
            logger.info(f"Data saved to {filename}")
            
            return filename
            
        except Exception as e:
            logger.error(f"Error saving historical data: {e}")
            return ""
    
    def fetch_and_save_option_chain(self, strike_buffer: int = 2000) -> str:
        """
        Fetch and save Bank Nifty option chain
        
        Args:
            strike_buffer: Buffer around spot price for strikes to include
            
        Returns:
            str: Path to the saved CSV file
        """
        try:
            option_chain = self.fetch_option_chain()
            
            if not option_chain:
                logger.error("Failed to fetch option chain")
                return ""
                
            # Convert option chain to DataFrame
            calls_df = pd.DataFrame(option_chain["options"]["calls"])
            puts_df = pd.DataFrame(option_chain["options"]["puts"])
            
            calls_df["option_type"] = "CE"
            puts_df["option_type"] = "PE"
            
            options_df = pd.concat([calls_df, puts_df], ignore_index=True)
            options_df["spot_price"] = option_chain["spot_price"]
            options_df["expiry_date"] = option_chain["expiry_date"]
            
            # Create directory if it doesn't exist
            os.makedirs("data/raw", exist_ok=True)
            
            # Save to CSV
            today_str = datetime.now().strftime('%Y%m%d')
            filename = f"data/raw/options_{today_str}.csv"
            options_df.to_csv(filename, index=False)
            logger.info(f"Option chain saved to {filename}")
            
            return filename
            
        except Exception as e:
            logger.error(f"Error saving option chain: {e}")
            return ""

    def get_ltp(self, symbol: str, exchange: str = "NSE") -> Optional[float]:
        """
        Get the Last Traded Price (LTP) for a symbol
        
        Args:
            symbol: Symbol to fetch LTP for (e.g., 'NIFTY BANK')
            exchange: Exchange to fetch from (NSE, BSE, etc.)
            
        Returns:
            float: Last Traded Price or None if not available
        """
        if not self.kite or not self.access_token:
            logger.error("Kite not initialized or not authenticated")
            return None
            
        try:
            # Create the instrument string
            instrument_str = f"{exchange}:{symbol}"
            
            # Fetch the quote
            quote = self.kite.quote(instrument_str)
            
            # Extract and return the LTP
            if quote and instrument_str in quote:
                return quote[instrument_str]["last_price"]
            else:
                logger.error(f"Could not get LTP for {instrument_str}")
                return None
                
        except Exception as e:
            logger.error(f"Error fetching LTP for {symbol}: {e}")
            return None


# For testing purposes
if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    
    # Create credentials file structure if not exists
    if not os.path.exists("config/credentials.json"):
        os.makedirs("config", exist_ok=True)
        default_creds = {
            "api_key": "your_api_key_here",
            "api_secret": "your_api_secret_here"
        }
        with open("config/credentials.json", "w") as f:
            json.dump(default_creds, f, indent=4)
        
        print("Created config/credentials.json template. Please update with your actual API credentials.")
    else:
        # Initialize client
        client = ZerodhaClient()
        
        if not client.access_token:
            # Generate login URL
            login_url = client.get_login_url()
            print(f"Please visit this URL to login:\n{login_url}")
            
            # Get request token from user
            request_token = input("Enter the request token from the redirect URL: ")
            
            # Login
            if client.login(request_token):
                print("Login successful")
            else:
                print("Login failed")
                exit(1)
        
        # Fetch and save historical data (last 30 days for testing)
        historical_file = client.fetch_and_save_historical_data(days=30)
        if historical_file:
            print(f"Historical data saved to {historical_file}")
        
        # Fetch and save option chain
        options_file = client.fetch_and_save_option_chain()
        if options_file:
            print(f"Option chain saved to {options_file}")
