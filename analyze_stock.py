"""
Quick start script to analyze a specific stock.

Usage:
    python analyze_stock.py AAPL
    python analyze_stock.py MSFT --period 1y
    python analyze_stock.py GOOGL --strategy macd
"""

import sys
import os
import argparse
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
from models.predictor import StockPricePredictor


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(description='Analyze a stock with algorithmic trading strategies')
    parser.add_argument('symbol', type=str, help='Stock symbol (e.g., AAPL, MSFT, GOOGL)')
    parser.add_argument('--period', type=str, default='2y', 
                       help='Data period (e.g., 1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y)')
    parser.add_argument('--strategy', type=str, default='all',
                       choices=['all', 'ma', 'mean_reversion', 'bollinger', 'rsi', 'macd'],
                       help='Trading strategy to use')
    parser.add_argument('--capital', type=float, default=100000,
                       help='Initial capital for backtesting')
    parser.add_argument('--commission', type=float, default=0.001,
                       help='Commission rate (e.g., 0.001 = 0.1%%)')
    
    args = parser.parse_args()
    
    print("="*60)
    print(f"ANALYZING {args.symbol}")
    print("="*60)
    print()
    
    # Fetch data
    print(f"Fetching {args.period} of data...")
    collector = StockDataCollector()
    data = collector.fetch_stock_data(args.symbol, period=args.period)
    
    if data.empty:
        print(f"Error: Could not fetch data for {args.symbol}")
        print("Please check your internet connection and verify the symbol is correct.")
        return 1
    
    print(f"✓ Fetched {len(data)} days of data")
    print(f"  Date range: {data.index[0].date()} to {data.index[-1].date()}")
    print(f"  Latest price: ${data['Close'].iloc[-1]:.2f}")
    print()
    
    # Preprocess data
    print("Processing data...")
    preprocessor = DataPreprocessor()
    clean_data = preprocessor.clean_data(data)
    data_with_indicators = preprocessor.add_technical_indicators(clean_data)
    print(f"✓ Added technical indicators")
    print()
    
    # Create results directory
    os.makedirs('results', exist_ok=True)
    
    # Define strategies
    strategies = {}
    if args.strategy == 'all' or args.strategy == 'ma':
        strategies['Moving Average Crossover'] = MovingAverageCrossover(20, 50)
    if args.strategy == 'all' or args.strategy == 'mean_reversion':
        strategies['Mean Reversion'] = MeanReversion(20, 2.0)
    if args.strategy == 'all' or args.strategy == 'bollinger':
        strategies['Bollinger Bands'] = BollingerBands(20, 2.0)
    if args.strategy == 'all' or args.strategy == 'rsi':
        strategies['RSI Strategy'] = RSIStrategy(14, 30, 70)
    if args.strategy == 'all' or args.strategy == 'macd':
        strategies['MACD Strategy'] = MACDStrategy(12, 26, 9)
    
    # Test strategies
    print("Testing strategies...")
    print("-" * 60)
    
    best_strategy = None
    best_return = -float('inf')
    
    for strategy_name, strategy in strategies.items():
        signals = strategy.generate_signals(clean_data)
        backtester = Backtester(args.capital, args.commission)
        portfolio = backtester.run_backtest(signals)
        metrics = backtester.calculate_metrics()
        
        print(f"\n{strategy_name}:")
        print(f"  Total Return: {metrics['Total Return']*100:>8.2f}%")
        print(f"  Sharpe Ratio: {metrics['Sharpe Ratio']:>8.4f}")
        print(f"  Max Drawdown: {metrics['Maximum Drawdown']*100:>8.2f}%")
        print(f"  Win Rate:     {metrics['Win Rate']*100:>8.2f}%")
        print(f"  Total Trades: {metrics['Total Trades']:>8.0f}")
        
        if metrics['Total Return'] > best_return:
            best_return = metrics['Total Return']
            best_strategy = (strategy_name, backtester)
    
    print()
    print("="*60)
    print(f"BEST STRATEGY: {best_strategy[0]}")
    print("="*60)
    best_strategy[1].print_metrics()
    
    # Save visualization
    print("Saving results...")
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        
        fig = best_strategy[1].plot_results(f'results/{args.symbol}_backtest.png')
        plt.close(fig)
        print(f"✓ Visualization saved to results/{args.symbol}_backtest.png")
    except Exception as e:
        print(f"! Could not save plot: {e}")
    
    # Train prediction model
    print("\nTraining prediction model...")
    data_with_features = preprocessor.create_features(data_with_indicators, lookback=10)
    features, target = preprocessor.prepare_for_prediction(data_with_features)
    
    predictor = StockPricePredictor(model_type='ridge')
    X_train, X_test, y_train, y_test = predictor.prepare_data(features, target, test_size=0.2)
    predictor.train(X_train, y_train)
    metrics = predictor.evaluate(X_test, y_test)
    
    print(f"  RMSE: ${metrics['rmse']:.2f}")
    print(f"  MAE:  ${metrics['mae']:.2f}")
    print(f"  R²:   {metrics['r2']:.4f}")
    
    predictor.save_model(f'results/{args.symbol}_model.pkl')
    print(f"✓ Model saved to results/{args.symbol}_model.pkl")
    
    print()
    print("="*60)
    print("ANALYSIS COMPLETE!")
    print("="*60)
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
