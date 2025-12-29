"""
Cryptocurrency Price Data Fetcher
Fetches historical price data using Yahoo Finance API
"""

import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta

def fetch_crypto_data(symbol, period='2y'):
    """
    Fetch cryptocurrency price data
    
    Args:
        symbol: Crypto symbol (e.g., 'BTC-USD', 'ETH-USD')
        period: Time period ('1y', '2y', '5y', 'max')
    
    Returns:
        DataFrame with price data
    """
    try:
        ticker = yf.Ticker(symbol)
        data = ticker.history(period=period)
        
        if data.empty:
            print(f"No data found for {symbol}")
            return None
            
        print(f"✅ Fetched {len(data)} days of data for {symbol}")
        return data
        
    except Exception as e:
        print(f"❌ Error fetching {symbol}: {e}")
        return None

def prepare_data(data, target_column='Close'):
    """
    Prepare data for LSTM model
    
    Args:
        data: Raw price DataFrame
        target_column: Column to predict (default: 'Close')
    
    Returns:
        Processed DataFrame
    """
    if data is None or data.empty:
        return None
    
    # Select relevant columns
    df = data[[target_column]].copy()
    
    # Remove any NaN values
    df = df.dropna()
    
    print(f"✅ Prepared {len(df)} data points")
    return df

if __name__ == "__main__":
    # Example usage
    print("Cryptocurrency Price Data Fetcher")
    print("=" * 50)
    
    # Fetch Bitcoin data
    btc_data = fetch_crypto_data('BTC-USD', period='2y')
    if btc_data is not None:
        btc_prepared = prepare_data(btc_data)
        print(f"\nBitcoin data shape: {btc_prepared.shape}")
        print(f"Date range: {btc_prepared.index[0]} to {btc_prepared.index[-1]}")
    
    # Fetch Ethereum data
    eth_data = fetch_crypto_data('ETH-USD', period='2y')
    if eth_data is not None:
        eth_prepared = prepare_data(eth_data)
        print(f"\nEthereum data shape: {eth_prepared.shape}")
        print(f"Date range: {eth_prepared.index[0]} to {eth_prepared.index[-1]}")