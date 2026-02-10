"""
Data preprocessing utilities for cleaning and preparing stock data.
"""

import pandas as pd
import numpy as np
from typing import Optional


class DataPreprocessor:
    """
    Preprocesses and cleans stock data for analysis.
    """
    
    @staticmethod
    def clean_data(data: pd.DataFrame) -> pd.DataFrame:
        """
        Clean stock data by handling missing values and duplicates.
        
        Args:
            data: Raw stock data DataFrame
        
        Returns:
            Cleaned DataFrame
        """
        # Remove duplicates
        data = data[~data.index.duplicated(keep='first')]
        
        # Forward fill missing values
        data = data.ffill()
        
        # Drop any remaining NaN values
        data = data.dropna()
        
        return data
    
    @staticmethod
    def add_technical_indicators(data: pd.DataFrame) -> pd.DataFrame:
        """
        Add technical indicators to the stock data.
        
        Args:
            data: Stock data DataFrame with OHLCV columns
        
        Returns:
            DataFrame with added technical indicators
        """
        df = data.copy()
        
        # Simple Moving Averages
        df['SMA_20'] = df['Close'].rolling(window=20).mean()
        df['SMA_50'] = df['Close'].rolling(window=50).mean()
        df['SMA_200'] = df['Close'].rolling(window=200).mean()
        
        # Exponential Moving Averages
        df['EMA_12'] = df['Close'].ewm(span=12, adjust=False).mean()
        df['EMA_26'] = df['Close'].ewm(span=26, adjust=False).mean()
        
        # MACD
        df['MACD'] = df['EMA_12'] - df['EMA_26']
        df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
        df['MACD_Histogram'] = df['MACD'] - df['MACD_Signal']
        
        # Bollinger Bands
        df['BB_Middle'] = df['Close'].rolling(window=20).mean()
        bb_std = df['Close'].rolling(window=20).std()
        df['BB_Upper'] = df['BB_Middle'] + (2 * bb_std)
        df['BB_Lower'] = df['BB_Middle'] - (2 * bb_std)
        
        # RSI (Relative Strength Index)
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        
        # Price Rate of Change
        df['ROC'] = df['Close'].pct_change(periods=10) * 100
        
        # Average True Range (ATR)
        high_low = df['High'] - df['Low']
        high_close = np.abs(df['High'] - df['Close'].shift())
        low_close = np.abs(df['Low'] - df['Close'].shift())
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = ranges.max(axis=1)
        df['ATR'] = true_range.rolling(window=14).mean()
        
        # Volume indicators
        df['Volume_SMA_20'] = df['Volume'].rolling(window=20).mean()
        df['Volume_Ratio'] = df['Volume'] / df['Volume_SMA_20']
        
        # Returns
        df['Daily_Return'] = df['Close'].pct_change()
        df['Log_Return'] = np.log(df['Close'] / df['Close'].shift(1))
        
        return df
    
    @staticmethod
    def create_features(data: pd.DataFrame, lookback: int = 10) -> pd.DataFrame:
        """
        Create feature columns for machine learning.
        
        Args:
            data: Stock data with technical indicators
            lookback: Number of days to look back for features
        
        Returns:
            DataFrame with feature columns
        """
        df = data.copy()
        
        # Lag features
        for i in range(1, lookback + 1):
            df[f'Close_Lag_{i}'] = df['Close'].shift(i)
            df[f'Volume_Lag_{i}'] = df['Volume'].shift(i)
            df[f'Return_Lag_{i}'] = df['Daily_Return'].shift(i)
        
        # Rolling statistics
        df['Close_Mean_5'] = df['Close'].rolling(window=5).mean()
        df['Close_Std_5'] = df['Close'].rolling(window=5).std()
        df['Close_Mean_10'] = df['Close'].rolling(window=10).mean()
        df['Close_Std_10'] = df['Close'].rolling(window=10).std()
        
        # Price momentum
        df['Momentum_5'] = df['Close'] - df['Close'].shift(5)
        df['Momentum_10'] = df['Close'] - df['Close'].shift(10)
        
        # Volatility
        df['Volatility_5'] = df['Daily_Return'].rolling(window=5).std()
        df['Volatility_10'] = df['Daily_Return'].rolling(window=10).std()
        
        return df
    
    @staticmethod
    def prepare_for_prediction(
        data: pd.DataFrame, 
        target_col: str = 'Close',
        prediction_days: int = 1
    ) -> tuple:
        """
        Prepare data for prediction by creating target variable.
        
        Args:
            data: Preprocessed stock data
            target_col: Column to predict
            prediction_days: Number of days ahead to predict
        
        Returns:
            Tuple of (features_df, target_series)
        """
        df = data.copy()
        
        # Create target: future price
        df['Target'] = df[target_col].shift(-prediction_days)
        
        # Drop rows with NaN values
        df = df.dropna()
        
        # Separate features and target
        target = df['Target']
        features = df.drop(['Target'], axis=1)
        
        return features, target
