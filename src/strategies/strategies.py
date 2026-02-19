"""
Trading strategies based on statistical inference.
"""

import pandas as pd
import numpy as np
from typing import Optional, Tuple


class MovingAverageCrossover:
    """
    Moving Average Crossover strategy.
    
    Generates buy signal when short-term MA crosses above long-term MA,
    and sell signal when it crosses below.
    """
    
    def __init__(self, short_window: int = 20, long_window: int = 50):
        """
        Initialize the strategy.
        
        Args:
            short_window: Period for short-term moving average
            long_window: Period for long-term moving average
        """
        self.short_window = short_window
        self.long_window = long_window
    
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Generate trading signals.
        
        Args:
            data: Stock data with Close prices
        
        Returns:
            DataFrame with signals (1: buy, -1: sell, 0: hold)
        """
        signals = pd.DataFrame(index=data.index)
        signals['Price'] = data['Close']
        
        # Calculate moving averages
        signals['SMA_Short'] = data['Close'].rolling(window=self.short_window).mean()
        signals['SMA_Long'] = data['Close'].rolling(window=self.long_window).mean()
        
        # Generate signals (start after long window to ensure both MAs are valid)
        signals['Signal'] = 0.0
        mask = signals.index >= signals.index[self.long_window]
        signals.loc[mask, 'Signal'] = np.where(
            signals.loc[mask, 'SMA_Short'] > signals.loc[mask, 'SMA_Long'],
            1, -1
        )
        
        # Generate trading orders
        signals['Position'] = signals['Signal'].diff()
        
        return signals


class MeanReversion:
    """
    Mean Reversion strategy.
    
    Assumes that prices revert to their mean. Buys when price is below
    the mean by a threshold and sells when above.
    """
    
    def __init__(self, lookback_period: int = 20, z_threshold: float = 2.0):
        """
        Initialize the strategy.
        
        Args:
            lookback_period: Period for calculating mean and std
            z_threshold: Z-score threshold for signals
        """
        self.lookback_period = lookback_period
        self.z_threshold = z_threshold
    
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Generate trading signals based on mean reversion.
        
        Args:
            data: Stock data with Close prices
        
        Returns:
            DataFrame with signals
        """
        signals = pd.DataFrame(index=data.index)
        signals['Price'] = data['Close']
        
        # Calculate rolling mean and standard deviation
        signals['Rolling_Mean'] = data['Close'].rolling(window=self.lookback_period).mean()
        signals['Rolling_Std'] = data['Close'].rolling(window=self.lookback_period).std()
        
        # Calculate z-score
        signals['Z_Score'] = (signals['Price'] - signals['Rolling_Mean']) / signals['Rolling_Std']
        
        # Generate signals
        signals['Signal'] = 0
        signals.loc[signals['Z_Score'] < -self.z_threshold, 'Signal'] = 1  # Buy
        signals.loc[signals['Z_Score'] > self.z_threshold, 'Signal'] = -1  # Sell
        
        # Generate trading orders
        signals['Position'] = signals['Signal'].diff()
        
        return signals


class BollingerBands:
    """
    Bollinger Bands strategy.
    
    Buys when price touches lower band and sells when it touches upper band.
    """
    
    def __init__(self, window: int = 20, num_std: float = 2.0):
        """
        Initialize the strategy.
        
        Args:
            window: Period for moving average
            num_std: Number of standard deviations for bands
        """
        self.window = window
        self.num_std = num_std
    
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Generate trading signals based on Bollinger Bands.
        
        Args:
            data: Stock data with Close prices
        
        Returns:
            DataFrame with signals
        """
        signals = pd.DataFrame(index=data.index)
        signals['Price'] = data['Close']
        
        # Calculate Bollinger Bands
        signals['SMA'] = data['Close'].rolling(window=self.window).mean()
        rolling_std = data['Close'].rolling(window=self.window).std()
        signals['Upper_Band'] = signals['SMA'] + (self.num_std * rolling_std)
        signals['Lower_Band'] = signals['SMA'] - (self.num_std * rolling_std)
        
        # Generate signals
        signals['Signal'] = 0
        signals.loc[signals['Price'] <= signals['Lower_Band'], 'Signal'] = 1  # Buy
        signals.loc[signals['Price'] >= signals['Upper_Band'], 'Signal'] = -1  # Sell
        
        # Generate trading orders
        signals['Position'] = signals['Signal'].diff()
        
        return signals


class RSIStrategy:
    """
    RSI (Relative Strength Index) strategy.
    
    Buys when RSI indicates oversold condition and sells when overbought.
    """
    
    def __init__(self, period: int = 14, oversold: float = 30, overbought: float = 70):
        """
        Initialize the strategy.
        
        Args:
            period: Period for RSI calculation
            oversold: RSI threshold for oversold condition
            overbought: RSI threshold for overbought condition
        """
        self.period = period
        self.oversold = oversold
        self.overbought = overbought
    
    def calculate_rsi(self, data: pd.Series) -> pd.Series:
        """Calculate RSI indicator."""
        delta = data.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=self.period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=self.period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Generate trading signals based on RSI.
        
        Args:
            data: Stock data with Close prices
        
        Returns:
            DataFrame with signals
        """
        signals = pd.DataFrame(index=data.index)
        signals['Price'] = data['Close']
        
        # Calculate RSI
        signals['RSI'] = self.calculate_rsi(data['Close'])
        
        # Generate signals
        signals['Signal'] = 0
        signals.loc[signals['RSI'] < self.oversold, 'Signal'] = 1  # Buy (oversold)
        signals.loc[signals['RSI'] > self.overbought, 'Signal'] = -1  # Sell (overbought)
        
        # Generate trading orders
        signals['Position'] = signals['Signal'].diff()
        
        return signals


class MACDStrategy:
    """
    MACD (Moving Average Convergence Divergence) strategy.
    """
    
    def __init__(self, fast_period: int = 12, slow_period: int = 26, signal_period: int = 9):
        """
        Initialize the strategy.
        
        Args:
            fast_period: Fast EMA period
            slow_period: Slow EMA period
            signal_period: Signal line period
        """
        self.fast_period = fast_period
        self.slow_period = slow_period
        self.signal_period = signal_period
    
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Generate trading signals based on MACD.
        
        Args:
            data: Stock data with Close prices
        
        Returns:
            DataFrame with signals
        """
        signals = pd.DataFrame(index=data.index)
        signals['Price'] = data['Close']
        
        # Calculate MACD
        ema_fast = data['Close'].ewm(span=self.fast_period, adjust=False).mean()
        ema_slow = data['Close'].ewm(span=self.slow_period, adjust=False).mean()
        signals['MACD'] = ema_fast - ema_slow
        signals['Signal_Line'] = signals['MACD'].ewm(span=self.signal_period, adjust=False).mean()
        signals['MACD_Histogram'] = signals['MACD'] - signals['Signal_Line']
        
        # Generate signals (ensure signal line has sufficient data)
        signals['Signal'] = 0.0
        min_periods = self.slow_period + self.signal_period - 1
        mask = signals.index >= signals.index[min_periods]
        signals.loc[mask, 'Signal'] = np.where(
            signals.loc[mask, 'MACD'] > signals.loc[mask, 'Signal_Line'],
            1, -1
        )
        
        # Generate trading orders
        signals['Position'] = signals['Signal'].diff()
        
        return signals
