# Data

## Cryptocurrency Price Data

This project uses historical cryptocurrency price data fetched from Yahoo Finance API.

### Data Sources

**Primary Cryptocurrencies:**
- Bitcoin (BTC-USD)
- Ethereum (ETH-USD)

### Data Features

**Available Columns:**
- Open: Opening price
- High: Highest price of the day
- Low: Lowest price of the day
- Close: Closing price (primary target for prediction)
- Volume: Trading volume
- Date: Trading date (index)

### Data Collection

**Method:** Yahoo Finance API via `yfinance` library

**Default Period:** 2 years of historical data

**Update Frequency:** Daily

### Data Preprocessing

1. **Feature Selection**: Close price used as primary target
2. **Scaling**: MinMaxScaler (0-1 range) for LSTM compatibility
3. **Sequence Creation**: 60-day lookback window
4. **Train/Test Split**: 80% training, 20% testing

### Time Series Structure
```
Input Sequence (60 days) → LSTM Model → Next Day Prediction
[Day 1, Day 2, ..., Day 60] → [Day 61 Price]
```

### Data Quality

- No missing values (dropna applied)
- Continuous time series data
- Real-time fetching ensures up-to-date information

### Usage Example
```python
from data_fetcher import fetch_crypto_data, prepare_data

# Fetch Bitcoin data
btc_data = fetch_crypto_data('BTC-USD', period='2y')
btc_prepared = prepare_data(btc_data)

print(f"Data shape: {btc_prepared.shape}")
print(f"Date range: {btc_prepared.index[0]} to {btc_prepared.index[-1]}")
```

### Notes

- Data is fetched in real-time, no static files required
- Internet connection needed for data retrieval
- Yahoo Finance may have rate limits for API calls