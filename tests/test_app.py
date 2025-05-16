import sys
import os

# Add the src directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.app import app

def test_hello_route():
    """Test that the hello route returns the expected message."""
    client = app.test_client()
    response = client.get('/')
    assert response.status_code == 200
    assert b"Hello, Bank Nifty Trading System" in response.data

def test_health_check():
    """Test that the health check route returns the correct status."""
    client = app.test_client()
    response = client.get('/health')
    assert response.status_code == 200
    data = response.get_json()
    assert data["status"] == "healthy"
    assert data["service"] == "bank-nifty-trading-system"
