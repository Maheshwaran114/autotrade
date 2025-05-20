# Project Structure

```
autotrade/
├── config/                  # Configuration files
├── docs/                    # Documentation
├── scripts/                 # Utility scripts
├── src/                     # Source code
│   ├── data_ingest/         # Market data fetching and processing
│   ├── ml_models/           # Machine learning models
│   ├── strategies/          # Trading strategy implementations
│   ├── execution/           # Order execution and management
│   └── dashboard/           # Web interface and API
└── tests/                   # Test suite
```

## Module Descriptions

### src/data_ingest/
Handles fetching, storing, and preprocessing market data from various sources including:
- Real-time market feeds
- Historical data repositories
- Technical indicators calculation

### src/ml_models/
Contains machine learning models for:
- Market trend prediction
- Day classification (trending/consolidation)
- Volatility forecasting
- Entry/exit point identification

### src/strategies/
Implements various option trading strategies:
- Delta-Theta neutral strategies
- Gamma scalping
- Volatility-based strategies
- Directional strategies

### src/execution/
Manages the execution of trades:
- Order placement
- Position management
- Risk management
- Broker API integration

### src/dashboard/
Provides a web interface for:
- System monitoring
- Strategy performance analysis
- Manual controls and overrides
- Configuration management
