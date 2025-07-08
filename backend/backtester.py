import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, Any
import warnings
from backend.utils import DataValidator, MathUtils
from backend.strategies import BaseStrategy

warnings.filterwarnings('ignore')

class BacktestEngine:
    """
    Comprehensive backtesting engine for trading strategies.
    """
    
    def __init__(self, initial_capital: float = 100000, 
                 commission_rate: float = 0.001,
                 slippage_rate: float = 0.0005, # 0.05% difference is acceptable
                 ):
        """
        Initialize backtesting engine.
        
        Args:
            initial_capital: Starting portfolio value
            commission_rate: Commission rate (0.001 = 0.1%)
            slippage_rate: Slippage rate (0.0005 = 0.05%)
        """
        self.initial_capital = initial_capital
        self.commission_rate = commission_rate
        self.slippage_rate = slippage_rate
        
        # Reset portfolio state
        self.reset_portfolio()
            
    def reset_portfolio(self):
        """Reset portfolio to initial state."""
        self.cash = self.initial_capital
        self.positions = {}  # {symbol: shares}
        self.portfolio_value = self.initial_capital
        self.equity_curve = []
        self.trades = []
        self.daily_returns = []
        self.drawdown_series = []
        self.max_portfolio_value = self.initial_capital
        
        # Performance tracking
        self.total_trades = 0
        self.winning_trades = 0
        self.losing_trades = 0
        self.total_commission = 0
        self.total_slippage = 0
    
    def run_backtest(self, strategy: BaseStrategy, data: pd.DataFrame, 
                    symbol: str) -> Dict[str, Any]:
        """
        Run complete backtest for a strategy.
        
        Args:
            strategy: Trading strategy to test
            data: Historical market data with features
            symbol: Stock symbol being tested
            
        Returns:
            Dict: Comprehensive backtest results
        """        
        # Validate data
        if not DataValidator.validate_ohlcv_data(data):
            raise ValueError("Invalid market data provided")
        
        # Reset portfolio
        self.reset_portfolio()
        
        # Generate signals
        signals = strategy.generate_signals(data)
        
        # Execute trades
        self._execute_backtest(strategy, data, signals, symbol)
        
        # Calculate performance metrics
        results = self._calculate_performance_metrics(data, symbol)
        
        # Add strategy-specific information
        results['strategy_name'] = strategy.name
        results['strategy_params'] = strategy.params
        results['symbol'] = symbol
        results['start_date'] = data.index[0]
        results['end_date'] = data.index[-1]
        results['total_days'] = len(data)
        
        return results
    
    def _execute_backtest(self, strategy: BaseStrategy, data: pd.DataFrame, 
                         signals: pd.Series, symbol: str):
        """Execute the backtest day by day."""

        executed_trades = 0
        
        for i, (date, row) in enumerate(data.iterrows()):
            # Get signal for this date
            if i < len(signals):
                signal = signals.iloc[i]
            else:
                signal = 0
            
            # Skip if no signal
            if signal == 0:
                continue
                            
            # Execute trade
            if signal != 0:
                self._execute_trade(signal, row, date, symbol, strategy)
                executed_trades += 1
            
            # Update portfolio value
            self._update_portfolio_value(row, date, symbol)
        
    def _execute_trade(self, signal: int, market_data: pd.Series, 
                      date: datetime, symbol: str, strategy: BaseStrategy):
        """Execute a single trade"""
        
        # Get price
        price = market_data['Close']
        
        # Get current position
        current_position = self.positions.get(symbol, 0)
        
        if signal > 0:  # Buy signal
            # Only buy if we don't have a position i.e. we are holding
            if current_position == 0:
                # Use fixed position size
                max_investment = self.cash * 0.60  # Use 60% of cash
                shares_to_buy = int(max_investment / price)
                
                # If we can afford to buy the shares
                if shares_to_buy > 0:
                    trade_value = shares_to_buy * price
            
                    self.cash -= trade_value
                    self.positions[symbol] = shares_to_buy
                    
                    # Record trade
                    self._record_trade(date, symbol, 'BUY', shares_to_buy, price, 0, 0)
                
        elif signal < 0:  # Sell signal
            # Only sell if we have a position
            if current_position > 0:
                shares_to_sell = current_position
                trade_value = shares_to_sell * price
                
                # Execute sell
                self.cash += trade_value
                self.positions[symbol] = 0
                
                # Record trade
                self._record_trade(date, symbol, 'SELL', shares_to_sell, price, 0, 0)

    def _record_trade(self, date: datetime, symbol: str, action: str, 
                     shares: int, price: float, commission: float, slippage: float):
        """Record a trade for analysis."""
        
        trade = {
            'date': date,
            'symbol': symbol,
            'action': action,
            'shares': shares,
            'price': price,
            'value': shares * price,
            'commission': commission,
            'slippage': slippage,
            'net_value': shares * price - commission
        }
        
        self.trades.append(trade)
        self.total_trades += 1
        
        # Calculate P&L for sell trades
        if action == 'SELL' and len(self.trades) > 1:
            # Find corresponding buy trade
            for prev_trade in reversed(self.trades[:-1]):
                if prev_trade['symbol'] == symbol and prev_trade['action'] == 'BUY':
                    pnl = ((price - prev_trade['price']) * shares) - commission - prev_trade['commission']
                    trade['pnl'] = pnl
                    trade['pnl_pct'] = pnl / (prev_trade['price'] * shares)
                    trade['holding_days'] = (date - prev_trade['date']).days
                    
                    if pnl > 0:
                        self.winning_trades += 1
                    else:
                        self.losing_trades += 1
                    break
            
    def _update_portfolio_value(self, market_data: pd.Series, date: datetime, symbol: str):
        """Update portfolio value and equity curve."""
        
        # Calculate position value
        position_value = 0
        if symbol in self.positions and self.positions[symbol] > 0:
            position_value = self.positions[symbol] * market_data['Close']
        
        self.portfolio_value = self.cash + position_value
        
        # Update equity curve
        self.equity_curve.append({
            'date': date,
            'portfolio_value': self.portfolio_value,
            'cash': self.cash,
            'position_value': position_value,
            'daily_return': (self.portfolio_value / self.initial_capital) - 1 # Percentage return
        })
        
        # Calculate daily returns
        if len(self.equity_curve) > 1:
            prev_value = self.equity_curve[-2]['portfolio_value']
            daily_return = (self.portfolio_value - prev_value) / prev_value
            self.daily_returns.append(daily_return)
        else:
            # First day there is no return
            self.daily_returns.append(0.0)
        
        # Update drawdown
        self.max_portfolio_value = max(self.max_portfolio_value, self.portfolio_value)
        drawdown = (self.portfolio_value - self.max_portfolio_value) / self.max_portfolio_value
        self.drawdown_series.append(drawdown)
    
    def _calculate_performance_metrics(self, data: pd.DataFrame, symbol: str) -> Dict[str, Any]:
        """Calculate comprehensive performance metrics."""
        
        # Convert to DataFrames for easier analysis
        equity_df = pd.DataFrame(self.equity_curve)
        trades_df = pd.DataFrame(self.trades) if self.trades else pd.DataFrame()
        
        total_return = (self.portfolio_value - self.initial_capital) / self.initial_capital
        annualized_return = (self.portfolio_value / self.initial_capital) ** (252 / len(data)) - 1

        # Risk metrics
        returns_series = pd.Series(self.daily_returns)
        volatility = MathUtils.calculate_volatility(returns_series, True) if len(returns_series) > 1 else 0
        sharpe_ratio = MathUtils.calculate_sharpe_ratio(returns_series) if len(returns_series) > 1 else 0
        
        # Drawdown metrics
        max_drawdown = min(self.drawdown_series) if self.drawdown_series else 0
        drawdown_df = pd.Series(self.drawdown_series)
        
        # Calculate drawdown duration
        drawdown_periods = []
        in_drawdown = False
        drawdown_start = None
        
        for i, dd in enumerate(drawdown_df):
            if dd < 0 and not in_drawdown:
                in_drawdown = True
                drawdown_start = i
            elif dd >= 0 and in_drawdown:
                in_drawdown = False
                if drawdown_start is not None:
                    drawdown_periods.append(i - drawdown_start)
        
        avg_drawdown_duration = np.mean(drawdown_periods) if drawdown_periods else 0
        max_drawdown_duration = max(drawdown_periods) if drawdown_periods else 0
        
        # Trade analysis
        win_rate = self.winning_trades / self.total_trades if self.total_trades > 0 else 0
        
        # Calculate average win/loss
        winning_trades_pnl = []
        losing_trades_pnl = []
        
        if not trades_df.empty and 'pnl' in trades_df.columns:
            winning_trades_pnl = trades_df[trades_df['pnl'] > 0]['pnl'].tolist()
            losing_trades_pnl = trades_df[trades_df['pnl'] < 0]['pnl'].tolist()
        
        avg_win = np.mean(winning_trades_pnl) if winning_trades_pnl else 0
        avg_loss = np.mean(losing_trades_pnl) if losing_trades_pnl else 0
        profit_factor = abs(avg_win * self.winning_trades / (avg_loss * self.losing_trades)) if avg_loss != 0 and self.losing_trades > 0 else 0
        
        # Market comparison
        market_return = (data['Close'].iloc[-1] - data['Close'].iloc[0]) / data['Close'].iloc[0]
        excess_return = total_return - market_return
        
        # Information ratio
        tracking_error = 0
        information_ratio = 0
        
        if len(returns_series) > 1:
            market_returns = data['Close'].pct_change().dropna()
            if len(market_returns) == len(returns_series):
                excess_returns = returns_series - market_returns
                tracking_error = excess_returns.std() * np.sqrt(252)
                information_ratio = excess_returns.mean() / excess_returns.std() * np.sqrt(252) if excess_returns.std() > 0 else 0
        
        # Performance metrics dictionary
        metrics = {
            # Return metrics
            'total_return': total_return,
            'annualized_return': annualized_return,
            'market_return': market_return,
            'excess_return': excess_return,
            'final_portfolio_value': self.portfolio_value,
            
            # Risk metrics
            'volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
            'information_ratio': information_ratio,
            'tracking_error': tracking_error,
            
            # Drawdown metrics
            'max_drawdown': max_drawdown,
            'avg_drawdown_duration': avg_drawdown_duration,
            'max_drawdown_duration': max_drawdown_duration,
            
            # Trade metrics
            'total_trades': self.total_trades,
            'winning_trades': self.winning_trades,
            'losing_trades': self.losing_trades,
            'win_rate': win_rate,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'profit_factor': profit_factor,
            
            # Cost metrics
            'total_commission': self.total_commission,
            'total_slippage': self.total_slippage,
            'commission_pct': self.total_commission / self.initial_capital,
            'slippage_pct': self.total_slippage / self.initial_capital,
            
            # Detailed data
            'equity_curve': equity_df,
            'trades': trades_df,
            'daily_returns': returns_series,
            'drawdown_series': drawdown_df
        };
        
        return metrics
    
    def generate_performance_report(self, results: Dict[str, Any]) -> str:
        """Generate a formatted performance report."""
        
        report = f"""
=== BACKTEST PERFORMANCE REPORT ===
Strategy: {results['strategy_name']}
Symbol: {results['symbol']}
Period: {results['start_date'].strftime('%Y-%m-%d')} to {results['end_date'].strftime('%Y-%m-%d')}
Total Days: {results['total_days']}

=== RETURN METRICS ===
Total Return:           {results['total_return']:>10.2%}
Annualized Return:      {results['annualized_return']:>10.2%}
Market Return:          {results['market_return']:>10.2%}
Excess Return:          {results['excess_return']:>10.2%}
Final Portfolio Value:  ${results['final_portfolio_value']:>10,.2f}

=== RISK METRICS ===
Volatility (Annual):    {results['volatility']:>10.2%}
Sharpe Ratio:          {results['sharpe_ratio']:>10.2f}
Information Ratio:     {results['information_ratio']:>10.2f}
Tracking Error:        {results['tracking_error']:>10.2%}

=== DRAWDOWN METRICS ===
Max Drawdown:          {results['max_drawdown']:>10.2%}
Avg Drawdown Duration: {results['avg_drawdown_duration']:>10.1f} days
Max Drawdown Duration: {results['max_drawdown_duration']:>10.0f} days

=== TRADE METRICS ===
Total Trades:          {results['total_trades']:>10}
Winning Trades:        {results['winning_trades']:>10}
Losing Trades:         {results['losing_trades']:>10}
Win Rate:              {results['win_rate']:>10.2%}
Average Win:           ${results['avg_win']:>10.2f}
Average Loss:          ${results['avg_loss']:>10.2f}
Profit Factor:         {results['profit_factor']:>10.2f}

=== COST ANALYSIS ===
Total Commission:      ${results['total_commission']:>10.2f}
Total Slippage:        ${results['total_slippage']:>10.2f}
Commission %:          {results['commission_pct']:>10.2%}
Slippage %:            {results['slippage_pct']:>10.2%}

=== STRATEGY PARAMETERS ===
"""
        
        for key, value in results['strategy_params'].items():
            report += f"{key}: {value}\n"
        
        return report