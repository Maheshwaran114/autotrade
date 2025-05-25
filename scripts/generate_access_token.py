#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Zerodha Access Token Generator and Manager

This script handles the generation and management of Zerodha API access tokens.
It can be run manually or as a scheduled task to refresh the token daily.

Usage:
1. Initial setup: python generate_access_token.py
   This will provide a login URL where you can authenticate and get a request token
   
2. With request token: python generate_access_token.py your_request_token
   This will exchange the request token for an access token and save it

3. Scheduled: python generate_access_token.py --scheduled
   Run this in scheduler mode to automatically refresh token if needed
"""

import os
import sys
import json
import time
import logging
import argparse
import webbrowser
import http.server
import socketserver
import threading
from urllib.parse import parse_qs, urlparse
from datetime import datetime, timedelta
from pathlib import Path

# Setup path to import project modules
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

# Import Zerodha client
from src.data_ingest.zerodha_client import ZerodhaClient

# Configure logging
log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
logging.basicConfig(
    level=logging.INFO,
    format=log_format,
    handlers=[
        logging.FileHandler(project_root / "logs" / "access_token_manager.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Configuration
CREDENTIALS_PATH = project_root / "config" / "credentials.json"
REDIRECT_URL = "http://localhost:8000/login"
TOKEN_HISTORY_PATH = project_root / "config" / "token_history.json"


class RequestTokenHandler(http.server.SimpleHTTPRequestHandler):
    """Handler for capturing request token from redirect URL"""
    
    def do_GET(self):
        """Handle GET request - extract request token from URL"""
        self.send_response(200)
        self.send_header('Content-type', 'text/html')
        self.end_headers()
        
        # Extract request token from URL
        query = urlparse(self.path).query
        params = parse_qs(query)
        
        if 'request_token' in params:
            token = params['request_token'][0]
            self.server.request_token = token
            
            # Display success message
            response = f"""
            <html>
            <head><title>Access Token Generated</title></head>
            <body>
                <h1>Request Token Captured Successfully!</h1>
                <p>The request token has been captured: {token}</p>
                <p>You can close this window now. The access token will be generated automatically.</p>
            </body>
            </html>
            """
        else:
            # No token found
            self.server.request_token = None
            response = f"""
            <html>
            <head><title>Error</title></head>
            <body>
                <h1>Error: No request token found</h1>
                <p>The URL does not contain a request token parameter.</p>
                <p>Please try the login process again.</p>
            </body>
            </html>
            """
        
        self.wfile.write(response.encode())
    
    def log_message(self, format, *args):
        """Silence the HTTP server logs"""
        return


def start_token_server():
    """
    Start a local HTTP server to capture the request token from the redirect URL
    
    Returns:
        tuple: (server_thread, server)
    """
    # Create server to listen for the redirect
    server = socketserver.TCPServer(("localhost", 8000), RequestTokenHandler)
    server.request_token = None  # Will be set when token is received
    
    # Run server in a separate thread
    server_thread = threading.Thread(target=server.serve_forever)
    server_thread.daemon = True
    server_thread.start()
    
    logger.info("Token capture server started at http://localhost:8000")
    return server_thread, server


def stop_token_server(server_thread, server):
    """Stop the token capture server"""
    server.shutdown()
    server_thread.join()
    logger.info("Token capture server stopped")


def load_credentials():
    """
    Load API credentials from the credentials file
    
    Returns:
        dict: Credentials dictionary with api_key, api_secret, and access_token
    """
    if not CREDENTIALS_PATH.exists():
        logger.error(f"Credentials file not found: {CREDENTIALS_PATH}")
        return None
    
    try:
        with open(CREDENTIALS_PATH, 'r') as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Failed to load credentials: {e}")
        return None


def update_credentials(credentials):
    """
    Update the credentials file with new information
    
    Args:
        credentials: Dictionary with updated credentials
    """
    try:
        with open(CREDENTIALS_PATH, 'w') as f:
            json.dump(credentials, f, indent=4)
        logger.info("Credentials file updated successfully")
    except Exception as e:
        logger.error(f"Failed to update credentials file: {e}")


def record_token_history(token, expiry):
    """
    Record the token and its expiry in a history file
    
    Args:
        token: The access token
        expiry: The expiry date as a string
    """
    # Create directory if it doesn't exist
    TOKEN_HISTORY_PATH.parent.mkdir(parents=True, exist_ok=True)
    
    # Load existing history or create new
    history = []
    if TOKEN_HISTORY_PATH.exists():
        try:
            with open(TOKEN_HISTORY_PATH, 'r') as f:
                history = json.load(f)
        except:
            pass
    
    # Add new token to history
    history.append({
        "token": token,
        "generated_at": datetime.now().isoformat(),
        "expires_at": expiry,
    })
    
    # Keep only last 30 entries
    history = history[-30:]
    
    # Save updated history
    with open(TOKEN_HISTORY_PATH, 'w') as f:
        json.dump(history, f, indent=4)


def generate_token_interactive():
    """
    Generate access token through interactive web login
    """
    # Load credentials
    credentials = load_credentials()
    if not credentials:
        sys.exit(1)
    
    # Initialize Zerodha client
    client = ZerodhaClient()
    client.api_key = credentials['api_key']
    client.api_secret = credentials['api_secret']
    client._initialize_kite()
    
    if not client.kite:
        logger.error("Failed to initialize Kite client")
        sys.exit(1)
    
    # Get login URL
    login_url = client.kite.login_url()
    logger.info(f"Please visit the following URL to log in to your Zerodha account:")
    logger.info(login_url)
    
    # Start server to catch the redirect with the request token
    server_thread, server = start_token_server()
    
    # Open browser to the login URL
    webbrowser.open(login_url)
    
    # Wait for the token to be received (up to 5 minutes)
    logger.info("Waiting for you to complete the login process...")
    timeout = time.time() + 300  # 5 minutes
    while time.time() < timeout and server.request_token is None:
        time.sleep(1)
    
    # Stop the server
    stop_token_server(server_thread, server)
    
    # Check if token was received
    request_token = server.request_token
    if not request_token:
        logger.error("No request token received. Please try again.")
        sys.exit(1)
    
    logger.info(f"Request token received: {request_token}")
    
    # Exchange request token for access token
    return exchange_request_token(client, request_token, credentials)


def exchange_request_token(client, request_token, credentials):
    """
    Exchange a request token for an access token
    
    Args:
        client: ZerodhaClient instance
        request_token: The request token from login
        credentials: Current credentials dictionary
        
    Returns:
        bool: True if successful
    """
    try:
        # Generate session and get access token
        data = client.kite.generate_session(
            request_token=request_token,
            api_secret=credentials['api_secret']
        )
        access_token = data["access_token"]
        
        # Update credentials with new access token
        credentials['access_token'] = access_token
        update_credentials(credentials)
        
        # Record in token history
        # Zerodha tokens expire at 6 AM the next day
        tomorrow = datetime.now() + timedelta(days=1)
        expiry = tomorrow.replace(hour=6, minute=0, second=0, microsecond=0).isoformat()
        record_token_history(access_token, expiry)
        
        logger.info("Access token generated and saved successfully!")
        logger.info(f"Token will expire at 6 AM on {tomorrow.strftime('%Y-%m-%d')}")
        
        return True
        
    except Exception as e:
        logger.error(f"Failed to generate access token: {e}")
        return False


def check_token_validity():
    """
    Check if the current access token is valid
    
    Returns:
        bool: True if valid, False otherwise
    """
    # Load credentials
    credentials = load_credentials()
    if not credentials or not credentials.get('access_token'):
        logger.info("No access token found in credentials")
        return False
    
    # Initialize client with current token
    client = ZerodhaClient()
    client.api_key = credentials['api_key']
    client.api_secret = credentials['api_secret']
    client.access_token = credentials['access_token']
    client._initialize_kite()
    
    if not client.kite:
        logger.error("Failed to initialize Kite client")
        return False
    
    # Test the token by trying to fetch profile
    try:
        profile = client.kite.profile()
        logger.info(f"Access token is valid. User: {profile.get('user_name', 'Unknown')}")
        return True
    except Exception as e:
        logger.info(f"Access token is invalid or expired: {e}")
        return False


def scheduled_token_refresh():
    """
    Check token validity and refresh if needed in scheduled mode
    """
    logger.info("Running scheduled token validity check...")
    
    # Check if token is still valid
    if check_token_validity():
        logger.info("Current access token is valid. No action needed.")
        return True
    
    # Token is invalid, need to refresh
    logger.warning("Access token is invalid or expired. Unable to refresh automatically.")
    logger.warning("Manual intervention required. Please run this script without the --scheduled flag.")
    
    # Send notification if configured
    try:
        notify_token_expiry()
    except:
        pass
    
    return False


def notify_token_expiry():
    """
    Send notification that token has expired
    """
    # This is a placeholder for notification functionality
    # You can implement email, push notifications, etc.
    logger.info("NOTIFICATION: Zerodha access token has expired and requires manual renewal")
    
    # TODO: Implement your preferred notification method
    # Example: Send email, SMS, Slack message, etc.


def main():
    """
    Main function to handle command-line arguments and execute token management
    """
    parser = argparse.ArgumentParser(description="Zerodha Access Token Manager")
    parser.add_argument("request_token", nargs="?", help="Request token from Zerodha login")
    parser.add_argument("--scheduled", action="store_true", help="Run in scheduled mode")
    parser.add_argument("--check", action="store_true", help="Check token validity")
    
    args = parser.parse_args()
    
    # Create logs directory if it doesn't exist
    (project_root / "logs").mkdir(parents=True, exist_ok=True)
    
    # Just check token validity
    if args.check:
        is_valid = check_token_validity()
        sys.exit(0 if is_valid else 1)
    
    # Scheduled mode - check and attempt refresh
    if args.scheduled:
        result = scheduled_token_refresh()
        sys.exit(0 if result else 1)
    
    # If request token is provided, exchange it for access token
    if args.request_token:
        # Load credentials
        credentials = load_credentials()
        if not credentials:
            sys.exit(1)
        
        # Initialize client
        client = ZerodhaClient()
        client.api_key = credentials['api_key']
        client.api_secret = credentials['api_secret']
        client._initialize_kite()
        
        # Exchange the token
        success = exchange_request_token(client, args.request_token, credentials)
        sys.exit(0 if success else 1)
    
    # Interactive mode - guide user through login process
    generate_token_interactive()


if __name__ == "__main__":
    main()
