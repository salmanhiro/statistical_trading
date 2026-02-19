# Statistical Trading Algorithm

A comprehensive algorithmic trading project using statistical inference methods to predict US stock prices. This project fetches historical data via API, implements multiple trading strategies, and uses machine learning for price prediction. This program mainly built with copilot so need to check again.

## Features

- **Data Collection**: Fetches historical stock data using yfinance API
- **Technical Indicators**: Implements 20+ technical indicators (SMA, EMA, RSI, MACD, Bollinger Bands, etc.)
- **Trading Strategies**: Multiple statistical trading strategies:
  - Moving Average Crossover
  - Mean Reversion
  - Bollinger Bands
  - RSI Strategy
  - MACD Strategy
- **Machine Learning**: Price prediction using:
  - Linear Regression
  - Ridge Regression
  - Random Forest
- **Backtesting**: Comprehensive backtesting framework with performance metrics
- **Statistical Analysis**: Time series analysis and stationarity tests

## Project Structure

```
statistical_trading/
├── src/
│   ├── data/
│   │   ├── collector.py         # Stock data collection via yfinance
│   │   └── preprocessor.py      # Data preprocessing and feature engineering
│   ├── models/
│   │   └── predictor.py         # ML models for price prediction
│   ├── strategies/
│   │   └── strategies.py        # Trading strategies implementation
│   └── backtest/
│       └── backtester.py        # Backtesting framework
├── data/
│   ├── raw/                     # Raw stock data (CSV files)
│   └── processed/               # Processed data with features
├── results/                     # Backtest results and visualizations
├── config.yaml                  # Configuration file
├── requirements.txt             # Python dependencies
└── main.py                      # Main execution script
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/salmanhiro/statistical_trading.git
cd statistical_trading
```

2. Install the package (with dependencies):
```bash
pip install -e .
```

## Usage

### Quick Start (With Internet Access)

Run the main example script:
```bash
python main.py
```

This will:
1. Fetch 2 years of historical data for AAPL (configurable)
2. Preprocess data and add technical indicators
3. Test 5 different trading strategies
4. Backtest and compare strategies
5. Train ML models for price prediction
6. Save results and visualizations

### Demo with Synthetic Data (No Internet Required)

Run the demo script with synthetic data:
```bash
python demo.py
```

This generates realistic synthetic stock data and demonstrates the full workflow without requiring internet access. Perfect for testing and learning how the system works.

### Analyze a Specific Stock

Use the analyze_stock.py script to quickly analyze any stock:
```bash
# Analyze Apple stock with default settings (2 years)
python analyze_stock.py AAPL

# Analyze Microsoft with 1 year of data
python analyze_stock.py MSFT --period 1y

# Test only the MACD strategy on Google
python analyze_stock.py GOOGL --strategy macd

# Custom initial capital and commission
python analyze_stock.py TSLA --capital 50000 --commission 0.002
```

### Configuration

Edit `config.yaml` to customize:
- Stock symbols to analyze
- Data period
- Strategy parameters
- Risk management settings
- Initial capital and commission rates

### Using Individual Components

#### Fetch Stock Data
```python
from src.data.collector import StockDataCollector

collector = StockDataCollector()
data = collector.fetch_stock_data('AAPL', period='2y')
```

#### Apply Trading Strategy
```python
from src.strategies.strategies import MovingAverageCrossover

strategy = MovingAverageCrossover(short_window=20, long_window=50)
signals = strategy.generate_signals(data)
```

#### Backtest Strategy
```python
from src.backtest.backtester import Backtester

backtester = Backtester(initial_capital=100000, commission=0.001)
portfolio = backtester.run_backtest(signals)
metrics = backtester.calculate_metrics()
backtester.print_metrics()
```

#### Train Prediction Model
```python
from src.models.predictor import StockPricePredictor
from src.data.preprocessor import DataPreprocessor

# Prepare data
preprocessor = DataPreprocessor()
data_with_features = preprocessor.create_features(data)
features, target = preprocessor.prepare_for_prediction(data_with_features)

# Train model
predictor = StockPricePredictor(model_type='random_forest')
X_train, X_test, y_train, y_test = predictor.prepare_data(features, target)
predictor.train(X_train, y_train)

# Evaluate
metrics = predictor.evaluate(X_test, y_test)
print(f"R² Score: {metrics['r2']:.4f}")
```

## Performance Metrics

The backtester calculates the following metrics:
- **Total Return**: Overall profit/loss percentage
- **Annualized Return**: Return adjusted for time period
- **Sharpe Ratio**: Risk-adjusted return
- **Maximum Drawdown**: Largest peak-to-trough decline
- **Win Rate**: Percentage of profitable trading days
- **Total Trades**: Number of buy/sell transactions

## Technical Indicators

The preprocessor adds these indicators:
- Simple Moving Averages (SMA): 20, 50, 200 days
- Exponential Moving Averages (EMA): 12, 26 days
- MACD and Signal Line
- Bollinger Bands
- RSI (Relative Strength Index)
- ATR (Average True Range)
- Rate of Change (ROC)
- Volume indicators

## Dependencies

- yfinance: Stock data API
- pandas: Data manipulation
- numpy: Numerical computing
- scikit-learn: Machine learning
- matplotlib: Visualization
- seaborn: Statistical visualization
- scipy: Scientific computing
- statsmodels: Statistical analysis

## Example Output

When running `main.py`, you'll see:
```
============================================================
STATISTICAL TRADING ALGORITHM
============================================================

1. FETCHING DATA FOR AAPL
------------------------------------------------------------
   ✓ Fetched 504 days of historical data
   ✓ Date range: 2024-02-10 to 2026-02-09
   Stock: Apple Inc.
   Sector: Technology

2. PREPROCESSING DATA
------------------------------------------------------------
   ✓ Data cleaned (504 rows)
   ✓ Added technical indicators
   ✓ Created feature columns for ML

3. ANALYZING STATISTICAL PROPERTIES
------------------------------------------------------------
   ADF Statistic: -2.1234
   P-value: 0.2341
   Is Stationary: No

4. TESTING TRADING STRATEGIES
------------------------------------------------------------
   Testing: Moving Average Crossover
      Total Return: 15.23%
      Sharpe Ratio: 1.2345
      Max Drawdown: -8.45%
      Total Trades: 12
...

==================================================
BACKTEST PERFORMANCE METRICS
==================================================
Total Return............................ 15.23%
Annualized Return....................... 18.45%
Sharpe Ratio............................  1.2345
Maximum Drawdown........................ -8.45%
Final Portfolio Value................. 115,230.00
==================================================
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is open source and available for educational purposes.

## Disclaimer

This software is for educational purposes only. Do not use it for actual trading without proper testing and risk management. Past performance does not guarantee future results.
