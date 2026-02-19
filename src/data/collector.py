"""
Data collection module for fetching stock data using yfinance API.
"""

import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
from typing import List, Optional


class StockDataCollector:
    """
    Collects historical stock data using yfinance API.
    """
    
    def __init__(self):
        """Initialize the data collector."""
        self.data_cache = {}
    
    def fetch_stock_data(
        self, 
        symbol: str, 
        period: str = "2y",
        interval: str = "1d"
    ) -> pd.DataFrame:
        """
        Fetch historical stock data for a given symbol.
        
        Args:
            symbol: Stock ticker symbol (e.g., 'AAPL')
            period: Time period for data (e.g., '1d', '5d', '1mo', '3mo', '6mo', '1y', '2y', '5y', '10y', 'ytd', 'max')
            interval: Data interval (e.g., '1m', '2m', '5m', '15m', '30m', '60m', '90m', '1h', '1d', '5d', '1wk', '1mo', '3mo')
        
        Returns:
            DataFrame with OHLCV data
        """
        try:
            ticker = yf.Ticker(symbol)
            data = ticker.history(period=period, interval=interval)
            
            if data.empty:
                raise ValueError(f"No data found for {symbol}")
            
            # Store in cache
            cache_key = f"{symbol}_{period}_{interval}"
            self.data_cache[cache_key] = data
            
            return data
        
        except Exception as e:
            print(f"Error fetching data for {symbol}: {e}")
            return pd.DataFrame()
    
    def fetch_multiple_stocks(
        self, 
        symbols: List[str], 
        period: str = "2y",
        interval: str = "1d"
    ) -> dict:
        """
        Fetch data for multiple stocks.
        
        Args:
            symbols: List of stock ticker symbols
            period: Time period for data
            interval: Data interval
        
        Returns:
            Dictionary with symbol as key and DataFrame as value
        """
        stock_data = {}
        
        for symbol in symbols:
            print(f"Fetching data for {symbol}...")
            data = self.fetch_stock_data(symbol, period, interval)
            if not data.empty:
                stock_data[symbol] = data
        
        return stock_data
    
    def get_stock_info(self, symbol: str) -> dict:
        """
        Get detailed information about a stock.
        
        Args:
            symbol: Stock ticker symbol
        
        Returns:
            Dictionary with stock information
        """
        try:
            ticker = yf.Ticker(symbol)
            return ticker.info
        except Exception as e:
            print(f"Error fetching info for {symbol}: {e}")
            return {}
    
    def save_data(self, data: pd.DataFrame, symbol: str, filepath: str):
        """
        Save stock data to CSV file.
        
        Args:
            data: DataFrame to save
            symbol: Stock symbol
            filepath: Path to save the file
        """
        try:
            data.to_csv(filepath)
            print(f"Data for {symbol} saved to {filepath}")
        except Exception as e:
            print(f"Error saving data: {e}")
    
    def load_data(self, filepath: str) -> pd.DataFrame:
        """
        Load stock data from CSV file.
        
        Args:
            filepath: Path to the CSV file
        
        Returns:
            DataFrame with stock data
        """
        try:
            data = pd.read_csv(filepath, index_col=0, parse_dates=True)
            return data
        except Exception as e:
            print(f"Error loading data: {e}")
            return pd.DataFrame()
