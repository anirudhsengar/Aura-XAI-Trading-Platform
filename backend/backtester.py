import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, Any
import warnings

warnings.filterwarnings('ignore')

class DataValidator:
    """Data validation utilities."""
    
    @staticmethod
    def validate_ohlcv_data(data: pd.DataFrame) -> bool:
        """Validate OHLCV data format."""
        required_cols = ['Close']
        return all(col in data.columns for col in required_cols) and len(data) > 0

class MathUtils:
    """Mathematical utilities for performance calculations."""
    
    @staticmethod
    def calculate_annualized_return(total_return: float, num_periods: int, periods_per_year: int = 252) -> float:
        """Calculate annualized return."""
        if num_periods <= 0:
            return 0.0
        return (1 + total_return) ** (periods_per_year / num_periods) - 1
    
    @staticmethod
    def calculate_volatility(returns: pd.Series, periods_per_year: int = 252) -> float:
        """Calculate annualized volatility."""
        if len(returns) == 0:
            return 0.0
        return returns.std() * np.sqrt(periods_per_year)
    
    @staticmethod
    def calculate_sharpe_ratio(returns: pd.Series, risk_free_rate: float = 0.02) -> float:
        """Calculate Sharpe ratio."""
        if len(returns) == 0 or returns.std() == 0:
            return 0.0
        excess_returns = returns.mean() * 252 - risk_free_rate
        return excess_returns / (returns.std() * np.sqrt(252))

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
        if not DataValidator.validate_ohlcv_data(data):
            raise ValueError("Invalid market data provided")
        
        self.reset_portfolio()
        signals = strategy.generate_signals(data)
        self._execute_backtest(strategy, data, signals, symbol)
        results = self._calculate_performance_metrics(data, symbol)
        
        results['strategy_name'] = strategy.name
        results['strategy_params'] = strategy.params
        results['symbol'] = symbol
        results['start_date'] = data.index[0]
        results['end_date'] = data.index[-1]
        results['total_days'] = len(data)
        
        return results
    
    def _execute_backtest(self, strategy, data: pd.DataFrame, 
                         signals: pd.Series, symbol: str):
        """Execute the backtest day by day."""
        current_trade_entry = None  # Track current trade for proper exit handling
        
        for i, (date, row) in enumerate(data.iterrows()):
            # Get signal for current date, handle missing signals
            if date in signals.index:
                signal = signals.loc[date]
            else:
                signal = 0
            
            # Apply risk management
            signal = strategy.apply_risk_management(row, signal)

            if signal != 0:
                trade_result = self._execute_trade(signal, row, date, symbol, strategy)
                if trade_result:
                    if signal > 0:  # Entry
                        current_trade_entry = trade_result
                    elif signal < 0 and current_trade_entry:  # Exit
                        # Calculate P&L for completed trade
                        self._finalize_trade(current_trade_entry, trade_result, date)
                        current_trade_entry = None
            
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

        trade_result = None

        if signal > 0 and current_position <= 0:  # Buy signal (enter long or cover short)
            if current_position < 0:  # Cover short first
                self._close_position(symbol, trade_price, date, strategy)
            
            # Enter new long position
            position_size = strategy.calculate_position_size(market_data)
            investment = self.portfolio_value * position_size
            shares_to_buy = int(investment / trade_price)
            
            if shares_to_buy > 0:
                trade_value = shares_to_buy * trade_price
                commission = trade_value * self.commission_rate
                
                if self.cash >= trade_value + commission:
                    self.cash -= (trade_value + commission)
                    self.positions[symbol] = shares_to_buy
                    self.total_commission += commission
                    slippage_cost = shares_to_buy * (trade_price - price)
                    self.total_slippage += slippage_cost
                    strategy.update_position(1, trade_price, date)
                    
                    trade_result = {
                        'entry_date': date,
                        'entry_price': trade_price,
                        'shares': shares_to_buy,
                        'entry_commission': commission,
                        'entry_slippage': slippage_cost
                    }
                    
                    self._record_trade(date, symbol, 'BUY', shares_to_buy, trade_price, commission, slippage_cost)
                
        elif signal < 0 and current_position > 0:  # Sell signal (exit long)
            trade_result = self._close_position(symbol, trade_price, date, strategy)

        return trade_result
    
    def _close_position(self, symbol: str, trade_price: float, date: datetime, strategy):
        """Close existing position."""
        current_position = self.positions.get(symbol, 0)
        if current_position == 0:
            return None
            
        shares_to_sell = abs(current_position)
        trade_value = shares_to_sell * trade_price
        commission = trade_value * self.commission_rate
        
        if current_position > 0:  # Closing long position
            self.cash += (trade_value - commission)
            action = 'SELL'
        else:  # Covering short position
            self.cash -= (trade_value + commission)
            action = 'COVER'
        
        self.positions[symbol] = 0
        self.total_commission += commission
        slippage_cost = shares_to_sell * abs(trade_price * self.slippage_rate)
        self.total_slippage += slippage_cost
        strategy.update_position(0, trade_price, date)
        
        self._record_trade(date, symbol, action, shares_to_sell, trade_price, commission, slippage_cost)
        
        return {
            'exit_date': date,
            'exit_price': trade_price,
            'shares': shares_to_sell,
            'exit_commission': commission,
            'exit_slippage': slippage_cost
        }

    def _finalize_trade(self, entry_trade: Dict, exit_trade: Dict, exit_date: datetime):
        """Finalize a completed trade with P&L calculation."""
        if not entry_trade or not exit_trade:
            return
            
        pnl_gross = (exit_trade['exit_price'] - entry_trade['entry_price']) * entry_trade['shares']
        total_commission = entry_trade['entry_commission'] + exit_trade['exit_commission']
        total_slippage = entry_trade['entry_slippage'] + exit_trade['exit_slippage']
        pnl_net = pnl_gross - total_commission - total_slippage
        
        holding_days = (exit_date - entry_trade['entry_date']).days
        
        # Update trade records with P&L
        if self.trades:
            # Find the corresponding buy trade and update
            for trade in reversed(self.trades):
                if (trade['action'] == 'BUY' and 
                    trade['date'] == entry_trade['entry_date'] and
                    'pnl' not in trade):
                    trade['pnl'] = pnl_net
                    trade['pnl_pct'] = pnl_net / (entry_trade['entry_price'] * entry_trade['shares'])
                    trade['holding_days'] = holding_days
                    break
            
            # Update the sell trade as well
            for trade in reversed(self.trades):
                if (trade['action'] == 'SELL' and 
                    trade['date'] == exit_date and
                    'pnl' not in trade):
                    trade['pnl'] = pnl_net
                    trade['pnl_pct'] = pnl_net / (entry_trade['entry_price'] * entry_trade['shares'])
                    trade['holding_days'] = holding_days
                    break
        
        if pnl_net > 0:
            self.winning_trades += 1
        else:
            self.losing_trades += 1

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
            'net_value': shares * price - commission if action == 'BUY' else shares * price + commission
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
            return {}

        equity_df = pd.DataFrame(self.equity_curve).set_index('date')['portfolio_value']
        trades_df = pd.DataFrame(self.trades) if self.trades else pd.DataFrame()
        
        total_return = (self.portfolio_value - self.initial_capital) / self.initial_capital
        annualized_return = MathUtils.calculate_annualized_return(total_return, len(data))
        
        returns_series = pd.Series(self.daily_returns)
        volatility = MathUtils.calculate_volatility(returns_series)
        sharpe_ratio = MathUtils.calculate_sharpe_ratio(returns_series)
        
        max_drawdown = min(self.drawdown_series) if self.drawdown_series else 0
        
        win_rate = self.winning_trades / max(self.total_trades, 1)
        
        # Enhanced trade analysis
        if not trades_df.empty and 'pnl' in trades_df.columns:
            pnl_series = trades_df[trades_df['pnl'].notna()]['pnl']
            if not pnl_series.empty:
                avg_win = pnl_series[pnl_series > 0].mean() if not pnl_series[pnl_series > 0].empty else 0
                avg_loss = pnl_series[pnl_series < 0].mean() if not pnl_series[pnl_series < 0].empty else 0
                profit_factor = abs(pnl_series[pnl_series > 0].sum() / pnl_series[pnl_series < 0].sum()) if pnl_series[pnl_series < 0].sum() != 0 else float('inf')
                
                # Add entry/exit date columns for trade analysis
                if 'action' in trades_df.columns:
                    buy_trades = trades_df[trades_df['action'] == 'BUY'].copy()
                    sell_trades = trades_df[trades_df['action'] == 'SELL'].copy()
                    
                    if not buy_trades.empty and not sell_trades.empty:
                        # Match buy and sell trades
                        matched_trades = []
                        for _, buy_trade in buy_trades.iterrows():
                            # Find corresponding sell trade
                            future_sells = sell_trades[sell_trades['date'] > buy_trade['date']]
                            if not future_sells.empty:
                                sell_trade = future_sells.iloc[0]
                                matched_trades.append({
                                    'entry_date': buy_trade['date'],
                                    'exit_date': sell_trade['date'],
                                    'entry_price': buy_trade['price'],
                                    'exit_price': sell_trade['price'],
                                    'shares': buy_trade['shares'],
                                    'pnl': sell_trade.get('pnl', 0),
                                    'holding_days': (sell_trade['date'] - buy_trade['date']).days
                                })
                        
                        if matched_trades:
                            trades_df = pd.DataFrame(matched_trades)
            else:
                avg_win = avg_loss = profit_factor = 0
        else:
            avg_win = avg_loss = profit_factor = 0

        metrics = {
            'total_return': total_return,
            'annualized_return': annualized_return,
            'final_portfolio_value': self.portfolio_value,
            'volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'total_trades': len(trades_df) if not trades_df.empty else 0,
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
            'equity_curve': equity_df,
            'trades': trades_df,
            'daily_returns': returns_series,
            'drawdown_series': pd.Series(self.drawdown_series, index=equity_df.index)
        }
        
        return metrics
