import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
import warnings
warnings.filterwarnings('ignore')

# Import utilities and strategies
from utils import DateUtils, DataValidator, FileUtils, LoggingUtils, MathUtils
from strategies import BaseStrategy

class BacktestEngine:
    """
    Comprehensive backtesting engine for trading strategies.
    
    Logic: Simulates trading strategy execution on historical data with
    realistic transaction costs, slippage, and risk management. Provides
    detailed performance analytics and trade-by-trade analysis.
    
    Why chosen: Realistic backtesting is crucial for strategy validation.
    This engine provides institutional-grade backtesting capabilities
    with proper position management and performance attribution.
    """
    
    def __init__(self, initial_capital: float = 100000, 
                 commission_rate: float = 0.001,
                 slippage_rate: float = 0.0005,
                 log_level: str = "INFO"):
        """
        Initialize backtesting engine.
        
        Args:
            initial_capital: Starting portfolio value
            commission_rate: Commission rate (0.001 = 0.1%)
            slippage_rate: Slippage rate (0.0005 = 0.05%)
            log_level: Logging level
        """
        self.initial_capital = initial_capital
        self.commission_rate = commission_rate
        self.slippage_rate = slippage_rate
        
        # Setup logging
        self.logger = LoggingUtils.setup_logger(
            "BacktestEngine",
            level=log_level
        )
        
        # Portfolio state
        self.reset_portfolio()
        
        self.logger.info(f"BacktestEngine initialized with ${initial_capital:,.2f} capital")
    
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
        
        Logic: Executes strategy on historical data, managing positions,
        calculating performance metrics, and providing detailed analysis.
        
        Args:
            strategy: Trading strategy to test
            data: Historical market data with features
            symbol: Stock symbol being tested
            
        Returns:
            Dict: Comprehensive backtest results
        """
        self.logger.info(f"Starting backtest for {strategy.name} on {symbol}")
        
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
        
        self.logger.info(f"Backtest completed for {strategy.name} on {symbol}")
        self.logger.info(f"Total return: {results['total_return']:.2%}")
        self.logger.info(f"Sharpe ratio: {results['sharpe_ratio']:.2f}")
        self.logger.info(f"Max drawdown: {results['max_drawdown']:.2%}")
        
        return results
    
    def _execute_backtest(self, strategy: BaseStrategy, data: pd.DataFrame, 
                         signals: pd.Series, symbol: str):
        """Execute the backtest day by day."""
        
        for i, (date, row) in enumerate(data.iterrows()):
            # Get current signal
            signal = signals.iloc[i] if i < len(signals) else 0
            
            # Apply risk management
            adjusted_signal = strategy.apply_risk_management(data.iloc[:i+1], signal)
            
            # Execute trade if signal changed
            if adjusted_signal != 0:
                self._execute_trade(adjusted_signal, row, date, symbol, strategy)
            
            # Update portfolio value
            self._update_portfolio_value(row, date, symbol)
            
            # Update strategy position tracking
            if hasattr(strategy, 'current_position'):
                strategy.current_position = self.positions.get(symbol, 0)
                if strategy.current_position != 0 and hasattr(strategy, 'entry_price'):
                    strategy.entry_price = row['Close']
    
    def _execute_trade(self, signal: int, market_data: pd.Series, 
                      date: datetime, symbol: str, strategy: BaseStrategy):
        """Execute a single trade."""
        
        # Get execution price (including slippage)
        base_price = market_data['Close']
        slippage = base_price * self.slippage_rate * (1 if signal > 0 else -1)
        execution_price = base_price + slippage
        
        # Calculate position size
        position_size = strategy.calculate_position_size(
            pd.DataFrame([market_data]), signal
        )
        
        # Calculate shares to trade
        current_position = self.positions.get(symbol, 0)
        portfolio_value = self.portfolio_value
        
        if signal > 0:  # Buy signal
            max_investment = portfolio_value * position_size
            shares_to_buy = int(max_investment / execution_price)
            
            if shares_to_buy > 0:
                trade_value = shares_to_buy * execution_price
                commission = trade_value * self.commission_rate
                
                # Check if we have enough cash
                if trade_value + commission <= self.cash:
                    self._record_trade(date, symbol, 'BUY', shares_to_buy, 
                                     execution_price, commission, slippage)
                    
                    self.cash -= (trade_value + commission)
                    self.positions[symbol] = current_position + shares_to_buy
                    self.total_commission += commission
                    self.total_slippage += abs(slippage * shares_to_buy)
                    
        elif signal < 0:  # Sell signal
            shares_to_sell = current_position
            
            if shares_to_sell > 0:
                trade_value = shares_to_sell * execution_price
                commission = trade_value * self.commission_rate
                
                self._record_trade(date, symbol, 'SELL', shares_to_sell, 
                                 execution_price, commission, slippage)
                
                self.cash += (trade_value - commission)
                self.positions[symbol] = 0
                self.total_commission += commission
                self.total_slippage += abs(slippage * shares_to_sell)
    
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
                    pnl = (price - prev_trade['price']) * shares - commission - prev_trade['commission']
                    trade['pnl'] = pnl
                    trade['pnl_pct'] = pnl / (prev_trade['price'] * shares)
                    trade['holding_days'] = (date - prev_trade['date']).days
                    
                    if pnl > 0:
                        self.winning_trades += 1
                    else:
                        self.losing_trades += 1
                    break
        
        self.logger.info(f"Trade executed: {action} {shares} shares of {symbol} at ${price:.2f}")
    
    def _update_portfolio_value(self, market_data: pd.Series, date: datetime, symbol: str):
        """Update portfolio value and equity curve."""
        
        # Calculate position value
        position_value = 0
        if symbol in self.positions and self.positions[symbol] > 0:
            position_value = self.positions[symbol] * market_data['Close']
        
        # Update portfolio value
        self.portfolio_value = self.cash + position_value
        
        # Update equity curve
        self.equity_curve.append({
            'date': date,
            'portfolio_value': self.portfolio_value,
            'cash': self.cash,
            'position_value': position_value,
            'daily_return': (self.portfolio_value / self.initial_capital) - 1
        })
        
        # Calculate daily returns
        if len(self.equity_curve) > 1:
            prev_value = self.equity_curve[-2]['portfolio_value']
            daily_return = (self.portfolio_value - prev_value) / prev_value
            self.daily_returns.append(daily_return)
        
        # Update drawdown
        self.max_portfolio_value = max(self.max_portfolio_value, self.portfolio_value)
        drawdown = (self.portfolio_value - self.max_portfolio_value) / self.max_portfolio_value
        self.drawdown_series.append(drawdown)
    
    def _calculate_performance_metrics(self, data: pd.DataFrame, symbol: str) -> Dict[str, Any]:
        """Calculate comprehensive performance metrics."""
        
        # Convert to DataFrames for easier analysis
        equity_df = pd.DataFrame(self.equity_curve)
        trades_df = pd.DataFrame(self.trades) if self.trades else pd.DataFrame()
        
        # Basic returns
        total_return = (self.portfolio_value - self.initial_capital) / self.initial_capital
        annualized_return = (self.portfolio_value / self.initial_capital) ** (252 / len(data)) - 1
        
        # Risk metrics
        returns_series = pd.Series(self.daily_returns)
        volatility = returns_series.std() * np.sqrt(252) if len(returns_series) > 1 else 0
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
        }
        
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

# Example usage and testing
if __name__ == "__main__":
    from strategies import StrategyFactory
    
    # Create sample data
    print("Testing BacktestEngine...")
    
    dates = pd.date_range(start='2024-01-01', periods=100, freq='D')
    sample_data = pd.DataFrame({
        'Open': np.random.randn(100).cumsum() + 100,
        'High': np.random.randn(100).cumsum() + 102,
        'Low': np.random.randn(100).cumsum() + 98,
        'Close': np.random.randn(100).cumsum() + 100,
        'Volume': np.random.randint(1000000, 10000000, 100),
        'RSI': np.random.uniform(20, 80, 100),
        'BB_High': np.random.randn(100).cumsum() + 105,
        'BB_Low': np.random.randn(100).cumsum() + 95,
        'BB_Position': np.random.uniform(0, 1, 100),
        'MACD': np.random.randn(100),
        'MACD_Signal': np.random.randn(100),
        'Volume_Ratio': np.random.uniform(0.5, 2.0, 100),
        'SMA_10': np.random.randn(100).cumsum() + 99,
        'SMA_50': np.random.randn(100).cumsum() + 98,
        'Price_Change': np.random.randn(100) * 0.02,
        'Price_Change_5d': np.random.randn(100) * 0.05,
        'sentiment_mean': np.random.uniform(-0.5, 0.5, 100)
    }, index=dates)
    
    # Ensure proper OHLCV relationships
    sample_data['High'] = sample_data[['Open', 'High', 'Low', 'Close']].max(axis=1)
    sample_data['Low'] = sample_data[['Open', 'High', 'Low', 'Close']].min(axis=1)
    
    try:
        # Initialize backtester
        backtester = BacktestEngine(initial_capital=100000)
        
        # Test with mean reversion strategy
        strategy = StrategyFactory.create_strategy('mean_reversion')
        
        # Run backtest
        results = backtester.run_backtest(strategy, sample_data, 'TEST')
        
        # Generate report
        report = backtester.generate_performance_report(results)
        print(report)
        
        print("\nBacktest completed successfully!")
        
    except Exception as e:
        print(f"Error during backtesting: {str(e)}")
        import traceback
        traceback.print_exc()
