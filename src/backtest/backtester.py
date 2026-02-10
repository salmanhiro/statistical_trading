"""
Backtesting framework for evaluating trading strategies.
"""

import pandas as pd
import numpy as np
from typing import Optional, Dict
import matplotlib.pyplot as plt


class Backtester:
    """
    Backtest trading strategies on historical data.
    """
    
    def __init__(
        self, 
        initial_capital: float = 100000,
        commission: float = 0.001
    ):
        """
        Initialize the backtester.
        
        Args:
            initial_capital: Starting capital for backtesting
            commission: Commission rate per trade (e.g., 0.001 = 0.1%)
        """
        self.initial_capital = initial_capital
        self.commission = commission
        self.portfolio = None
    
    def run_backtest(self, signals: pd.DataFrame) -> pd.DataFrame:
        """
        Run backtest on trading signals.
        
        Args:
            signals: DataFrame with 'Price' and 'Position' columns
        
        Returns:
            DataFrame with portfolio performance
        """
        # Initialize portfolio
        portfolio = pd.DataFrame(index=signals.index)
        portfolio['Price'] = signals['Price']
        portfolio['Signal'] = signals['Signal']
        portfolio['Position'] = signals['Position']
        
        # Calculate shares to hold
        portfolio['Holdings'] = 0.0
        portfolio['Cash'] = self.initial_capital
        portfolio['Total'] = self.initial_capital
        
        # Track position
        position = 0
        cash = self.initial_capital
        shares = 0
        
        for i, date in enumerate(portfolio.index):
            price = portfolio.loc[date, 'Price']
            signal = portfolio.loc[date, 'Signal']
            
            if pd.isna(price) or pd.isna(signal):
                portfolio.loc[date, 'Holdings'] = shares * price if not pd.isna(price) else 0
                portfolio.loc[date, 'Cash'] = cash
                portfolio.loc[date, 'Total'] = portfolio.loc[date, 'Holdings'] + cash
                continue
            
            # Execute trades based on signals
            if signal == 1 and position <= 0:  # Buy signal
                # Calculate shares to buy (use 95% of cash to maintain liquidity)
                # The 95% threshold keeps 5% cash reserve for handling fees and small price variations
                shares_to_buy = int(cash * 0.95 / price)
                if shares_to_buy > 0:
                    cost = shares_to_buy * price * (1 + self.commission)
                    if cost <= cash:
                        shares += shares_to_buy
                        cash -= cost
                        position = 1
            
            elif signal == -1 and position >= 0:  # Sell signal
                # Sell all shares
                if shares > 0:
                    revenue = shares * price * (1 - self.commission)
                    cash += revenue
                    shares = 0
                    position = -1
            
            # Update portfolio values
            portfolio.loc[date, 'Holdings'] = shares * price
            portfolio.loc[date, 'Cash'] = cash
            portfolio.loc[date, 'Total'] = portfolio.loc[date, 'Holdings'] + cash
        
        # Calculate returns
        portfolio['Returns'] = portfolio['Total'].pct_change()
        portfolio['Cumulative_Returns'] = (1 + portfolio['Returns']).cumprod()
        
        self.portfolio = portfolio
        return portfolio
    
    def calculate_metrics(self) -> Dict[str, float]:
        """
        Calculate performance metrics.
        
        Returns:
            Dictionary of performance metrics
        """
        if self.portfolio is None:
            raise ValueError("Must run backtest first")
        
        total_return = (self.portfolio['Total'].iloc[-1] - self.initial_capital) / self.initial_capital
        
        # Calculate daily returns
        returns = self.portfolio['Returns'].dropna()
        
        # Annualized metrics (assuming 252 trading days)
        trading_days = 252
        num_days = len(returns)
        
        if num_days > 0:
            annualized_return = (1 + total_return) ** (trading_days / num_days) - 1
            annualized_volatility = returns.std() * np.sqrt(trading_days)
            sharpe_ratio = annualized_return / annualized_volatility if annualized_volatility > 0 else 0
        else:
            annualized_return = 0
            annualized_volatility = 0
            sharpe_ratio = 0
        
        # Maximum drawdown
        cumulative = self.portfolio['Total']
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = drawdown.min()
        
        # Win rate
        winning_days = (returns > 0).sum()
        total_trading_days = len(returns)
        win_rate = winning_days / total_trading_days if total_trading_days > 0 else 0
        
        metrics = {
            'Total Return': total_return,
            'Annualized Return': annualized_return,
            'Annualized Volatility': annualized_volatility,
            'Sharpe Ratio': sharpe_ratio,
            'Maximum Drawdown': max_drawdown,
            'Win Rate': win_rate,
            'Final Portfolio Value': self.portfolio['Total'].iloc[-1],
            'Total Trades': (self.portfolio['Position'] != 0).sum()
        }
        
        return metrics
    
    def plot_results(self, save_path: Optional[str] = None):
        """
        Plot backtest results.
        
        Args:
            save_path: Path to save the plot (optional)
        """
        if self.portfolio is None:
            raise ValueError("Must run backtest first")
        
        fig, axes = plt.subplots(3, 1, figsize=(14, 10))
        
        # Plot 1: Portfolio value over time
        axes[0].plot(self.portfolio.index, self.portfolio['Total'], label='Portfolio Value', linewidth=2)
        axes[0].axhline(y=self.initial_capital, color='r', linestyle='--', label='Initial Capital')
        axes[0].set_title('Portfolio Value Over Time', fontsize=14, fontweight='bold')
        axes[0].set_ylabel('Value ($)', fontsize=12)
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Plot 2: Price and buy/sell signals
        axes[1].plot(self.portfolio.index, self.portfolio['Price'], label='Price', linewidth=2, alpha=0.7)
        
        # Mark buy signals
        buy_signals = self.portfolio[self.portfolio['Position'] > 0]
        if len(buy_signals) > 0:
            axes[1].scatter(buy_signals.index, buy_signals['Price'], 
                          color='green', marker='^', s=100, label='Buy', zorder=5)
        
        # Mark sell signals
        sell_signals = self.portfolio[self.portfolio['Position'] < 0]
        if len(sell_signals) > 0:
            axes[1].scatter(sell_signals.index, sell_signals['Price'], 
                          color='red', marker='v', s=100, label='Sell', zorder=5)
        
        axes[1].set_title('Price and Trading Signals', fontsize=14, fontweight='bold')
        axes[1].set_ylabel('Price ($)', fontsize=12)
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        # Plot 3: Returns distribution
        returns = self.portfolio['Returns'].dropna()
        axes[2].hist(returns, bins=50, alpha=0.7, edgecolor='black')
        axes[2].axvline(x=0, color='r', linestyle='--', linewidth=2)
        axes[2].set_title('Returns Distribution', fontsize=14, fontweight='bold')
        axes[2].set_xlabel('Daily Returns', fontsize=12)
        axes[2].set_ylabel('Frequency', fontsize=12)
        axes[2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Plot saved to {save_path}")
        
        return fig
    
    def print_metrics(self):
        """Print performance metrics in a formatted way."""
        metrics = self.calculate_metrics()
        
        print("\n" + "="*50)
        print("BACKTEST PERFORMANCE METRICS")
        print("="*50)
        
        for key, value in metrics.items():
            if 'Return' in key or 'Volatility' in key or 'Drawdown' in key or 'Win Rate' in key:
                print(f"{key:.<40} {value*100:>8.2f}%")
            elif 'Value' in key or 'Trades' in key:
                print(f"{key:.<40} {value:>8,.2f}")
            else:
                print(f"{key:.<40} {value:>8.4f}")
        
        print("="*50 + "\n")
