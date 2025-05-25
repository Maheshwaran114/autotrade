# Zerodha Integration Guide

This document explains how to set up and use Zerodha integration for the Bank Nifty Options Trading System.

## Prerequisites

1. **Zerodha Trading Account**:
   - You must have an active Zerodha trading account
   - You need Equity and F&O market access for Bank Nifty options

2. **Kite Connect Developer Account**:
   - Register at [https://developers.kite.trade/](https://developers.kite.trade/)
   - Create a new app to get your API key and secret

## Setup Steps

### 1. Generate API Credentials

1. Log in to [Kite Connect Developer Dashboard](https://developers.kite.trade/apps)
2. Click "Create a new app" if you don't already have one
3. Fill in the required details:
   - App Name: "BankNiftyTrader" (or your preferred name)
   - Redirect URL: "https://127.0.0.1/callback" (for local usage)
4. After creating the app, you'll get an API key and secret

### 2. Set Up Credentials File

1. Create a file named `credentials.json` in the `config/` directory
2. Use the following format (see `credentials.template.json`):

```json
{
    "api_key": "YOUR_ZERODHA_API_KEY",
    "api_secret": "YOUR_ZERODHA_API_SECRET",
    "access_token": ""
}
```

3. Replace `YOUR_ZERODHA_API_KEY` and `YOUR_ZERODHA_API_SECRET` with your actual credentials
4. Leave `access_token` empty as it will be generated during the authentication flow

### 3. Generate Access Token

Access tokens for Zerodha expire daily, so you'll need to periodically refresh them. You can do this using the example script:

```bash
python scripts/example_fetch_data.py
```

The script will:
1. Check if you have valid credentials
2. If your token is expired, it will provide a login URL
3. Open the URL, log in to your Zerodha account
4. You'll be redirected to your callback URL with a request token parameter
5. Copy the request token and run:

```bash
python scripts/example_fetch_data.py YOUR_REQUEST_TOKEN
```

This will generate and save the access token to your credentials file.

## Using the Zerodha Client

The system provides two classes for interacting with Zerodha:

1. **`ZerodhaClient`** (in `src/data_ingest/zerodha_client.py`):
   - Low-level client for direct API interaction
   - Handles authentication and API requests
   - Use this for customized data fetching

2. **`ZerodhaDataFetcher`** (in `src/data_ingest/zerodha_fetcher.py`):
   - Higher-level interface for specific data operations
   - Provides simplified methods for common tasks
   - Used by the main system for data collection

### Example: Fetching Bank Nifty Data

```python
from src.data_ingest.zerodha_client import ZerodhaClient

# Initialize client
zerodha_client = ZerodhaClient(credentials_path="config/credentials.json")

# Login with existing token
if zerodha_client.login():
    # Get Bank Nifty instrument token
    banknifty_token = zerodha_client.get_instrument_token("NIFTY BANK", "NSE")
    
    # Fetch historical data (last 30 days)
    from datetime import datetime, timedelta
    end_date = datetime.now()
    start_date = end_date - timedelta(days=30)
    
    data = zerodha_client.fetch_historical_data(
        symbol=banknifty_token,
        interval="day",
        from_date=start_date,
        to_date=end_date
    )
    
    print(data.head())
```

### Example: Fetching Option Chain

```python
from src.data_ingest.zerodha_client import ZerodhaClient

# Initialize client
zerodha_client = ZerodhaClient(credentials_path="config/credentials.json")

# Login with existing token
if zerodha_client.login():
    # Fetch option chain (nearest expiry)
    option_chain = zerodha_client.fetch_option_chain(
        index_symbol="BANKNIFTY",
        expiry_offset=0  # 0 = nearest expiry, 1 = next expiry, etc.
    )
    
    # Access calls and puts
    calls = option_chain.get('calls', [])
    puts = option_chain.get('puts', [])
```

## Daily Data Collection Process

The system is designed to collect data on a regular basis:

1. **Historical Data**:
   - Bank Nifty index prices (OHLC)
   - Option chain data for current and upcoming expiries

2. **Storage Format**:
   - Raw data is saved as CSV files in `data/raw/`
   - Processed data is converted to Parquet format in `data/processed/`

The example script (`scripts/example_fetch_data.py`) demonstrates the complete data collection flow.

## Troubleshooting

### Common Issues

1. **Authentication Errors**:
   - Access tokens expire daily - regenerate when needed
   - Ensure API key and secret are correctly entered in credentials file

2. **Rate Limiting**:
   - Zerodha imposes rate limits - add delays between requests
   - Batch requests when possible to minimize API calls

3. **Connection Issues**:
   - Ensure internet connectivity
   - Check Zerodha API status at [https://status.kite.trade/](https://status.kite.trade/)

### Debugging

Enable detailed logging for more information:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## Resources

- [Kite Connect Documentation](https://kite.trade/docs/connect/v3/)
- [Kite Connect Python Library](https://kite.trade/docs/pykiteconnect/v3/)
