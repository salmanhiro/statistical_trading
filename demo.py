"""
Demo script using synthetic stock data to demonstrate the trading algorithm.
This script can be run without internet access.
"""

import sys
import os
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from data.preprocessor import DataPreprocessor
from strategies.strategies import (
    MovingAverageCrossover, 
    MeanReversion, 
    BollingerBands,
    RSIStrategy,
    MACDStrategy
)
from backtest.backtester import Backtester
from models.predictor import StockPricePredictor, TimeSeriesAnalyzer


def generate_synthetic_stock_data(days=500, start_price=150, symbol='DEMO'):
    """
    Generate synthetic stock data for demonstration.
    
    Args:
        days: Number of days to generate
        start_price: Starting price
        symbol: Stock symbol
    
    Returns:
        DataFrame with synthetic OHLCV data
    """
    np.random.seed(42)
    
    # Generate dates
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)
    dates = pd.date_range(start=start_date, end=end_date, freq='D')[:days]
    
    # Generate more realistic price data with trend and cycles
    trend = np.linspace(0, 0.0005, days)  # Gradual upward trend
    cycle = 0.001 * np.sin(np.linspace(0, 8*np.pi, days))  # Market cycles
    noise = np.random.normal(0, 0.015, days)  # 1.5% daily volatility
    daily_returns = trend + cycle + noise
    
    # Calculate prices from returns
    prices = start_price * np.cumprod(1 + daily_returns)
    
    # Generate OHLC data
    data = pd.DataFrame(index=dates)
    data['Close'] = prices
    data['Open'] = prices * (1 + np.random.uniform(-0.005, 0.005, days))
    data['High'] = np.maximum(data['Open'], data['Close']) * (1 + np.abs(np.random.uniform(0, 0.01, days)))
    data['Low'] = np.minimum(data['Open'], data['Close']) * (1 - np.abs(np.random.uniform(0, 0.01, days)))
    data['Volume'] = np.random.uniform(50000000, 150000000, days).astype(int)
    
    return data


def main():
    """Main execution function."""
    print("="*60)
    print("STATISTICAL TRADING ALGORITHM - DEMO")
    print("="*60)
    print()
    
    # Configuration
    symbol = 'DEMO'
    initial_capital = 100000
    commission = 0.001
    
    print(f"1. GENERATING SYNTHETIC DATA FOR {symbol}")
    print("-" * 60)
    
    # Generate synthetic stock data
    data = generate_synthetic_stock_data(days=500, start_price=150, symbol=symbol)
    
    print(f"   ✓ Generated {len(data)} days of historical data")
    print(f"   ✓ Date range: {data.index[0].date()} to {data.index[-1].date()}")
    print(f"   ✓ Starting price: ${data['Close'].iloc[0]:.2f}")
    print(f"   ✓ Ending price: ${data['Close'].iloc[-1]:.2f}")
    print()
    
    # Save raw data
    os.makedirs('data/raw', exist_ok=True)
    data.to_csv(f'data/raw/{symbol}_raw.csv')
    print(f"   ✓ Saved raw data to data/raw/{symbol}_raw.csv")
    print()
    
    print("2. PREPROCESSING DATA")
    print("-" * 60)
    
    # Initialize preprocessor
    preprocessor = DataPreprocessor()
    
    # Clean data
    clean_data = preprocessor.clean_data(data)
    print(f"   ✓ Data cleaned ({len(clean_data)} rows)")
    
    # Add technical indicators
    data_with_indicators = preprocessor.add_technical_indicators(clean_data)
    print(f"   ✓ Added technical indicators")
    print(f"      - Moving Averages (SMA 20, 50, 200)")
    print(f"      - MACD and Signal Line")
    print(f"      - Bollinger Bands")
    print(f"      - RSI")
    print(f"      - ATR")
    
    # Create features for ML
    data_with_features = preprocessor.create_features(data_with_indicators, lookback=10)
    print(f"   ✓ Created feature columns for ML ({data_with_features.shape[1]} features)")
    print()
    
    # Save processed data
    os.makedirs('data/processed', exist_ok=True)
    data_with_features.to_csv(f'data/processed/{symbol}_processed.csv')
    print(f"   ✓ Saved processed data to data/processed/{symbol}_processed.csv")
    print()
    
    print("3. ANALYZING STATISTICAL PROPERTIES")
    print("-" * 60)
    
    # Check stationarity
    analyzer = TimeSeriesAnalyzer()
    stationarity = analyzer.check_stationarity(clean_data['Close'])
    print(f"   ADF Statistic: {stationarity['adf_statistic']:.4f}")
    print(f"   P-value: {stationarity['p_value']:.4f}")
    print(f"   Is Stationary: {'Yes' if stationarity['is_stationary'] else 'No'}")
    
    # Basic statistics
    print(f"\n   Price Statistics:")
    print(f"      Mean: ${clean_data['Close'].mean():.2f}")
    print(f"      Std Dev: ${clean_data['Close'].std():.2f}")
    print(f"      Min: ${clean_data['Close'].min():.2f}")
    print(f"      Max: ${clean_data['Close'].max():.2f}")
    print()
    
    print("4. TESTING TRADING STRATEGIES")
    print("-" * 60)
    
    strategies = {
        'Moving Average Crossover': MovingAverageCrossover(short_window=20, long_window=50),
        'Mean Reversion': MeanReversion(lookback_period=20, z_threshold=2.0),
        'Bollinger Bands': BollingerBands(window=20, num_std=2.0),
        'RSI Strategy': RSIStrategy(period=14, oversold=30, overbought=70),
        'MACD Strategy': MACDStrategy(fast_period=12, slow_period=26, signal_period=9)
    }
    
    best_strategy = None
    best_return = -float('inf')
    results_summary = []
    
    for strategy_name, strategy in strategies.items():
        print(f"\n   Testing: {strategy_name}")
        
        # Generate signals
        signals = strategy.generate_signals(clean_data)
        
        # Run backtest
        backtester = Backtester(initial_capital=initial_capital, commission=commission)
        portfolio = backtester.run_backtest(signals)
        
        # Calculate metrics
        metrics = backtester.calculate_metrics()
        
        print(f"      Total Return: {metrics['Total Return']*100:.2f}%")
        print(f"      Sharpe Ratio: {metrics['Sharpe Ratio']:.4f}")
        print(f"      Max Drawdown: {metrics['Maximum Drawdown']*100:.2f}%")
        print(f"      Win Rate: {metrics['Win Rate']*100:.2f}%")
        print(f"      Total Trades: {metrics['Total Trades']:.0f}")
        
        # Store results
        results_summary.append({
            'Strategy': strategy_name,
            'Total Return': metrics['Total Return'],
            'Sharpe Ratio': metrics['Sharpe Ratio'],
            'Max Drawdown': metrics['Maximum Drawdown'],
            'Win Rate': metrics['Win Rate']
        })
        
        # Track best strategy
        if metrics['Total Return'] > best_return:
            best_return = metrics['Total Return']
            best_strategy = (strategy_name, strategy, backtester)
    
    print()
    print("="*60)
    print(f"BEST STRATEGY: {best_strategy[0]}")
    print("="*60)
    
    # Print detailed metrics for best strategy
    best_strategy[2].print_metrics()
    
    # Create results directory
    os.makedirs('results', exist_ok=True)
    
    # Save strategy comparison
    results_df = pd.DataFrame(results_summary)
    results_df.to_csv('results/strategy_comparison.csv', index=False)
    print("Strategy comparison saved to results/strategy_comparison.csv\n")
    
    # Plot results for best strategy
    print("5. VISUALIZING RESULTS")
    print("-" * 60)
    try:
        # Use Agg backend to avoid display issues
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        
        fig = best_strategy[2].plot_results(f'results/{symbol}_backtest.png')
        plt.close(fig)
        print(f"   ✓ Backtest visualization saved to results/{symbol}_backtest.png")
    except Exception as e:
        print(f"   ! Could not save plot: {e}")
    print()
    
    print("6. TRAINING PRICE PREDICTION MODEL")
    print("-" * 60)
    
    # Prepare data for ML model
    features, target = preprocessor.prepare_for_prediction(
        data_with_features,
        target_col='Close',
        prediction_days=1
    )
    
    print(f"   ✓ Prepared {len(features)} samples with {features.shape[1]} features")
    
    # Train multiple models
    model_types = ['linear', 'ridge', 'random_forest']
    best_model = None
    best_r2 = -float('inf')
    model_results = []
    
    for model_type in model_types:
        print(f"\n   Training {model_type} model...")
        
        predictor = StockPricePredictor(model_type=model_type)
        X_train, X_test, y_train, y_test = predictor.prepare_data(features, target, test_size=0.2)
        
        predictor.train(X_train, y_train)
        metrics = predictor.evaluate(X_test, y_test)
        
        print(f"      RMSE: ${metrics['rmse']:.2f}")
        print(f"      MAE: ${metrics['mae']:.2f}")
        print(f"      R² Score: {metrics['r2']:.4f}")
        
        model_results.append({
            'Model': model_type,
            'RMSE': metrics['rmse'],
            'MAE': metrics['mae'],
            'R2': metrics['r2']
        })
        
        if metrics['r2'] > best_r2:
            best_r2 = metrics['r2']
            best_model = (model_type, predictor)
    
    print()
    print(f"   Best Model: {best_model[0]} (R² = {best_r2:.4f})")
    
    # Save best model
    best_model[1].save_model(f'results/{symbol}_prediction_model.pkl')
    print(f"   ✓ Model saved to results/{symbol}_prediction_model.pkl")
    
    # Save model comparison
    model_df = pd.DataFrame(model_results)
    model_df.to_csv('results/model_comparison.csv', index=False)
    print(f"   ✓ Model comparison saved to results/model_comparison.csv")
    print()
    
    print("="*60)
    print("ANALYSIS COMPLETE!")
    print("="*60)
    print()
    print(f"Results saved in 'results/' directory:")
    print(f"  - Backtest visualization: {symbol}_backtest.png")
    print(f"  - Prediction model: {symbol}_prediction_model.pkl")
    print(f"  - Strategy comparison: strategy_comparison.csv")
    print(f"  - Model comparison: model_comparison.csv")
    print()
    print(f"Data saved in 'data/' directory:")
    print(f"  - Raw data: data/raw/{symbol}_raw.csv")
    print(f"  - Processed data: data/processed/{symbol}_processed.csv")
    print()
    print("="*60)
    print("SUMMARY")
    print("="*60)
    print(f"\nBest Trading Strategy: {best_strategy[0]}")
    print(f"  Return: {best_return*100:.2f}%")
    print(f"\nBest Prediction Model: {best_model[0]}")
    print(f"  R² Score: {best_r2:.4f}")
    print()
    

if __name__ == '__main__':
    main()
