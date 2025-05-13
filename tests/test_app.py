import pytest
from src.app import app

@pytest.fixture
def client():
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client

def test_hello_endpoint(client):
    response = client.get('/')
    assert response.status_code == 200
    assert b'Hello, Bank Nifty Trading System' in response.data

def test_health_endpoint(client):
    response = client.get('/health')
    assert response.status_code == 200
    assert response.json == {"status": "ok"}
