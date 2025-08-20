import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, Any
import warnings

warnings.filterwarnings('ignore')

class BacktestEngine:
    """
    Comprehensive backtesting engine for trading strategies.
    """
    
    def __init__(self, initial_capital: float = 100000, 
                 commission_rate: float = 0.001,
                 slippage_rate: float = 0.0005):
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
        self.reset_portfolio()
    
    def _validate_data(self, data: pd.DataFrame) -> bool:
        """Validate OHLCV data."""
        required_columns = ['Open', 'High', 'Low', 'Close']
        if not all(col in data.columns for col in required_columns):
            return False
        
        if data.empty or len(data) < 2:
            return False
        
        # Check for valid price data
        price_data = data[required_columns]
        if (price_data <= 0).any().any():
            return False
        
        # Check OHLC consistency
        if ((data['High'] < data['Low']) | 
            (data['High'] < data['Open']) | 
            (data['High'] < data['Close']) |
            (data['Low'] > data['Open']) | 
            (data['Low'] > data['Close'])).any():
            return False
        
        return True
    
    def _calculate_annualized_return(self, total_return: float, days: int) -> float:
        """Calculate annualized return."""
        if days <= 0:
            return 0.0
        years = days / 252  # Assuming 252 trading days per year
        if years <= 0 or total_return <= -1:
            return 0.0
        return (1 + total_return) ** (1 / years) - 1
    
    def _calculate_volatility(self, returns: pd.Series) -> float:
        """Calculate annualized volatility."""
        if len(returns) < 2:
            return 0.0
        return returns.std() * np.sqrt(252)
    
    def _calculate_sharpe_ratio(self, returns: pd.Series, risk_free_rate: float = 0.02) -> float:
        """Calculate Sharpe ratio."""
        if len(returns) < 2:
            return 0.0
        
        excess_returns = returns - risk_free_rate / 252
        volatility = self._calculate_volatility(returns)
        
        if volatility == 0:
            return 0.0
        
        return excess_returns.mean() * 252 / volatility

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
    
    def run_backtest(self, strategy, data: pd.DataFrame, symbol: str) -> Dict[str, Any]:
        """
        Run complete backtest for a strategy.
        
        Args:
            strategy: Trading strategy to test
            data: Historical market data with features
            symbol: Stock symbol being tested
            
        Returns:
            Dict: Comprehensive backtest results
        """        
        if not self._validate_data(data):
            raise ValueError("Invalid market data provided")
        
        self.reset_portfolio()
        
        try:
            signals = strategy.generate_signals(data)
            self._execute_backtest(strategy, data, signals, symbol)
            results = self._calculate_performance_metrics(data, symbol)
            
            results['strategy_name'] = getattr(strategy, 'name', 'Unknown')
            results['strategy_params'] = getattr(strategy, 'params', {})
            results['symbol'] = symbol
            results['start_date'] = data.index[0]
            results['end_date'] = data.index[-1]
            results['total_days'] = len(data)
            
            return results
            
        except Exception as e:
            return {
                'error': f"Backtest failed: {str(e)}",
                'total_return': 0.0,
                'final_portfolio_value': self.initial_capital
            }

    def _execute_backtest(self, strategy, data: pd.DataFrame, 
                         signals: pd.Series, symbol: str):
        """Execute the backtest day by day."""
        for i, (date, row) in enumerate(data.iterrows()):
            # Get signal for current date
            signal = signals.get(date, 0) if date in signals.index else 0
            
            # Skip if no signal
            if signal == 0:
                self._update_portfolio_value(row, date, symbol)
                continue
            
            # Execute trade based on signal
            self._execute_trade(signal, row, date, symbol, strategy)
            self._update_portfolio_value(row, date, symbol)
        
        # Close any remaining open position at the end
        if self.positions.get(symbol, 0) != 0:
            final_row = data.iloc[-1]
            self._close_position(symbol, final_row['Close'], data.index[-1], strategy)

    def _execute_trade(self, signal: int, market_data: pd.Series, 
                      date: datetime, symbol: str, strategy):
        """Execute a single trade with improved position sizing."""
        price = market_data['Close']
        current_position = self.positions.get(symbol, 0)
        
        # Apply slippage
        if signal > 0:
            trade_price = price * (1 + self.slippage_rate)
        else:
            trade_price = price * (1 - self.slippage_rate)

        if signal > 0 and current_position <= 0:  # Buy signal
            if current_position < 0:  # Cover short first
                self._close_position(symbol, trade_price, date, strategy)
            
            # Enter new long position
            position_size = strategy.calculate_position_size(market_data)
            investment = self.portfolio_value * position_size
            shares_to_buy = max(1, int(investment / trade_price))
            
            trade_value = shares_to_buy * trade_price
            commission = trade_value * self.commission_rate
            
            if self.cash >= trade_value + commission:
                self.cash -= (trade_value + commission)
                self.positions[symbol] = shares_to_buy
                self.total_commission += commission
                slippage_cost = shares_to_buy * price * self.slippage_rate
                self.total_slippage += slippage_cost
                
                # Update strategy position
                strategy.update_position(1, trade_price, date)
                
                # Record trade
                self._record_trade(date, symbol, 'BUY', shares_to_buy, trade_price, 
                                 commission, slippage_cost)
                
        elif signal < 0 and current_position > 0:  # Sell signal
            self._close_position(symbol, trade_price, date, strategy)

    def _close_position(self, symbol: str, trade_price: float, date: datetime, strategy):
        """Close existing position."""
        current_position = self.positions.get(symbol, 0)
        if current_position == 0:
            return
            
        shares = abs(current_position)
        trade_value = shares * trade_price
        commission = trade_value * self.commission_rate
        
        if current_position > 0:  # Closing long position
            self.cash += (trade_value - commission)
            action = 'SELL'
            pnl_gross = trade_value - (shares * self._get_avg_entry_price(symbol))
        else:  # Covering short position
            self.cash -= (trade_value + commission)
            action = 'COVER'
            pnl_gross = (shares * self._get_avg_entry_price(symbol)) - trade_value
        
        self.positions[symbol] = 0
        self.total_commission += commission
        slippage_cost = shares * trade_price * self.slippage_rate
        self.total_slippage += slippage_cost
        
        # Calculate net P&L
        pnl_net = pnl_gross - commission - slippage_cost
        
        # Update strategy position
        strategy.update_position(0, trade_price, date)
        
        # Record trade with P&L
        self._record_trade(date, symbol, action, shares, trade_price, 
                         commission, slippage_cost, pnl_net)
        
        # Update win/loss counters
        if pnl_net > 0:
            self.winning_trades += 1
        elif pnl_net < 0:
            self.losing_trades += 1

    def _get_avg_entry_price(self, symbol: str) -> float:
        """Get average entry price for current position."""
        if symbol not in self.positions or self.positions[symbol] == 0:
            return 0.0
        
        # Find the most recent BUY trade for this symbol
        for trade in reversed(self.trades):
            if trade['symbol'] == symbol and trade['action'] == 'BUY':
                return trade['price']
        
        return 0.0

    def _record_trade(self, date: datetime, symbol: str, action: str, 
                     shares: int, price: float, commission: float, 
                     slippage: float, pnl: float = None):
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
            'pnl': pnl
        }
        
        self.trades.append(trade)
        self.total_trades += 1

    def _update_portfolio_value(self, market_data: pd.Series, date: datetime, symbol: str):
        """Update portfolio value and equity curve."""
        position_value = self.positions.get(symbol, 0) * market_data['Close']
        self.portfolio_value = self.cash + position_value
        
        if len(self.equity_curve) > 0:
            prev_value = self.equity_curve[-1]['portfolio_value']
            daily_return = (self.portfolio_value - prev_value) / prev_value if prev_value != 0 else 0
            self.daily_returns.append(daily_return)
        else:
            self.daily_returns.append(0.0)

        self.equity_curve.append({
            'date': date,
            'portfolio_value': self.portfolio_value,
        })
        
        self.max_portfolio_value = max(self.max_portfolio_value, self.portfolio_value)
        drawdown = (self.portfolio_value - self.max_portfolio_value) / self.max_portfolio_value if self.max_portfolio_value != 0 else 0
        self.drawdown_series.append(drawdown)
    
    def _calculate_performance_metrics(self, data: pd.DataFrame, symbol: str) -> Dict[str, Any]:
        """Calculate comprehensive performance metrics."""
        if not self.equity_curve:
            return {
                'total_return': 0.0,
                'final_portfolio_value': self.initial_capital,
                'total_trades': 0
            }

        equity_df = pd.DataFrame(self.equity_curve)
        if not equity_df.empty:
            equity_df = equity_df.set_index('date')['portfolio_value']
        
        trades_df = pd.DataFrame(self.trades) if self.trades else pd.DataFrame()
        
        total_return = (self.portfolio_value - self.initial_capital) / self.initial_capital
        annualized_return = self._calculate_annualized_return(total_return, len(data))
        
        returns_series = pd.Series(self.daily_returns)
        volatility = self._calculate_volatility(returns_series)
        sharpe_ratio = self._calculate_sharpe_ratio(returns_series)
        
        max_drawdown = min(self.drawdown_series) if self.drawdown_series else 0
        
        # Calculate trade statistics
        completed_trades = 0
        avg_win = avg_loss = profit_factor = 0
        
        if not trades_df.empty:
            # Count only trades with P&L (completed trades)
            pnl_trades = trades_df[trades_df['pnl'].notna()]
            completed_trades = len(pnl_trades)
            
            if not pnl_trades.empty:
                winning_pnl = pnl_trades[pnl_trades['pnl'] > 0]['pnl']
                losing_pnl = pnl_trades[pnl_trades['pnl'] < 0]['pnl']
                
                avg_win = winning_pnl.mean() if not winning_pnl.empty else 0
                avg_loss = losing_pnl.mean() if not losing_pnl.empty else 0
                
                total_wins = winning_pnl.sum() if not winning_pnl.empty else 0
                total_losses = abs(losing_pnl.sum()) if not losing_pnl.empty else 0
                
                profit_factor = total_wins / total_losses if total_losses > 0 else float('inf')

        win_rate = self.winning_trades / max(completed_trades, 1)

        metrics = {
            'total_return': total_return,
            'annualized_return': annualized_return,
            'final_portfolio_value': self.portfolio_value,
            'volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'total_trades': completed_trades,
            'winning_trades': self.winning_trades,
            'losing_trades': self.losing_trades,
            'win_rate': win_rate,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'profit_factor': profit_factor,
            'total_commission': self.total_commission,
            'total_slippage': self.total_slippage,
            'commission_pct': self.total_commission / self.initial_capital,
            'slippage_pct': self.total_slippage / self.initial_capital,
            'equity_curve': equity_df if not equity_df.empty else pd.Series(),
            'trades': trades_df,
            'daily_returns': returns_series,
            'drawdown_series': pd.Series(self.drawdown_series, 
                                       index=equity_df.index if not equity_df.empty else [])
        }
        
        return metrics
