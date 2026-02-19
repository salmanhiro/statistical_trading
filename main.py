"""
Main example script demonstrating the statistical trading algorithm.

This script:
1. Fetches historical stock data using yfinance API
2. Preprocesses data and adds technical indicators
3. Applies multiple trading strategies
4. Backtests the strategies
5. Trains a machine learning model for price prediction
6. Displays results and performance metrics
"""

import sys
import os
import yaml
import warnings
warnings.filterwarnings('ignore')

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from data.collector import StockDataCollector
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


def load_config(config_path='config.yaml'):
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def main():
    """Main execution function."""
    print("="*60)
    print("STATISTICAL TRADING ALGORITHM")
    print("="*60)
    print()
    
    # Load configuration
    config = load_config()
    
    # Configuration parameters
    symbols = config['trading']['symbols']
    data_period = config['trading']['data_period']
    short_window = config['trading']['strategy']['short_window']
    long_window = config['trading']['strategy']['long_window']
    initial_capital = config['trading']['backtest']['initial_capital']
    commission = config['trading']['backtest']['commission']
    
    # Select first symbol for demonstration
    symbol = symbols[0]
    
    print(f"1. FETCHING DATA FOR {symbol}")
    print("-" * 60)
    
    # Initialize data collector
    collector = StockDataCollector()
    
    # Fetch stock data
    data = collector.fetch_stock_data(symbol, period=data_period)
    
    if data.empty:
        print(f"Failed to fetch data for {symbol}")
        return
    
    print(f"   ✓ Fetched {len(data)} days of historical data")
    print(f"   ✓ Date range: {data.index[0].date()} to {data.index[-1].date()}")
    print()
    
    # Get stock info
    info = collector.get_stock_info(symbol)
    if info:
        print(f"   Stock: {info.get('longName', symbol)}")
        print(f"   Sector: {info.get('sector', 'N/A')}")
        print(f"   Industry: {info.get('industry', 'N/A')}")
    print()
    
    # Save raw data
    os.makedirs('data/raw', exist_ok=True)
    collector.save_data(data, symbol, f'data/raw/{symbol}_raw.csv')
    
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
    
    # Create features for ML
    data_with_features = preprocessor.create_features(data_with_indicators, lookback=10)
    print(f"   ✓ Created feature columns for ML")
    print()
    
    # Save processed data
    os.makedirs('data/processed', exist_ok=True)
    data_with_features.to_csv(f'data/processed/{symbol}_processed.csv')
    
    print("3. ANALYZING STATISTICAL PROPERTIES")
    print("-" * 60)
    
    # Check stationarity
    analyzer = TimeSeriesAnalyzer()
    stationarity = analyzer.check_stationarity(clean_data['Close'])
    print(f"   ADF Statistic: {stationarity['adf_statistic']:.4f}")
    print(f"   P-value: {stationarity['p_value']:.4f}")
    print(f"   Is Stationary: {'Yes' if stationarity['is_stationary'] else 'No'}")
    print()
    
    print("4. TESTING TRADING STRATEGIES")
    print("-" * 60)
    
    strategies = {
        'Moving Average Crossover': MovingAverageCrossover(short_window, long_window),
        'Mean Reversion': MeanReversion(lookback_period=20, z_threshold=2.0),
        'Bollinger Bands': BollingerBands(window=20, num_std=2.0),
        'RSI Strategy': RSIStrategy(period=14, oversold=30, overbought=70),
        'MACD Strategy': MACDStrategy(fast_period=12, slow_period=26, signal_period=9)
    }
    
    best_strategy = None
    best_return = -float('inf')
    
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
        print(f"      Total Trades: {metrics['Total Trades']:.0f}")
        
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
    
    # Plot results for best strategy
    print("5. VISUALIZING RESULTS")
    print("-" * 60)
    os.makedirs('results', exist_ok=True)
    fig = best_strategy[2].plot_results(f'results/{symbol}_backtest.png')
    print(f"   ✓ Backtest visualization saved")
    print()
    
    print("6. TRAINING PRICE PREDICTION MODEL")
    print("-" * 60)
    
    # Prepare data for ML model
    features, target = preprocessor.prepare_for_prediction(
        data_with_features,
        target_col='Close',
        prediction_days=1
    )
    
    # Train multiple models
    model_types = ['linear', 'ridge', 'random_forest']
    best_model = None
    best_r2 = -float('inf')
    
    for model_type in model_types:
        print(f"\n   Training {model_type} model...")
        
        predictor = StockPricePredictor(model_type=model_type)
        X_train, X_test, y_train, y_test = predictor.prepare_data(features, target, test_size=0.2)
        
        predictor.train(X_train, y_train)
        metrics = predictor.evaluate(X_test, y_test)
        
        print(f"      RMSE: ${metrics['rmse']:.2f}")
        print(f"      MAE: ${metrics['mae']:.2f}")
        print(f"      R² Score: {metrics['r2']:.4f}")
        
        if metrics['r2'] > best_r2:
            best_r2 = metrics['r2']
            best_model = (model_type, predictor)
    
    print()
    print(f"   Best Model: {best_model[0]} (R² = {best_r2:.4f})")
    
    # Save best model
    best_model[1].save_model(f'results/{symbol}_prediction_model.pkl')
    print(f"   ✓ Model saved")
    print()
    
    print("="*60)
    print("ANALYSIS COMPLETE!")
    print("="*60)
    print()
    print(f"Results saved in 'results/' directory:")
    print(f"  - Backtest visualization: {symbol}_backtest.png")
    print(f"  - Prediction model: {symbol}_prediction_model.pkl")
    print()
    print(f"Data saved in 'data/' directory:")
    print(f"  - Raw data: data/raw/{symbol}_raw.csv")
    print(f"  - Processed data: data/processed/{symbol}_processed.csv")
    print()
    

if __name__ == '__main__':
    main()
