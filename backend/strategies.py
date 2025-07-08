import pandas as pd
import numpy as np
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Dict, List, Any
import warnings

warnings.filterwarnings('ignore')

class BaseStrategy(ABC):
    """
    Base class for all trading strategies.
    """
    
    def __init__(self, name: str, params: Dict[str, Any] = None):
        """
        Initialize base strategy.
        
        Args:
            name: Strategy name
            params: Strategy parameters
        """
        self.name = name
        self.params = params or {}
        
        # Risk management parameters
        self.max_position_size = self.params.get('max_position_size', 0.1)  # 10% of portfolio
        self.stop_loss_pct = self.params.get('stop_loss_pct', 0.05)  # 5% stop loss
        self.take_profit_pct = self.params.get('take_profit_pct', 0.15)  # 15% take profit
        self.max_drawdown_limit = self.params.get('max_drawdown_limit', 0.20)  # 20% max drawdown
        
        # Position tracking
        self.current_position = 0
        self.entry_price = None
        self.trades = []
    
    @abstractmethod
    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """
        Generate trading signals based on the strategy logic.
        
        Args:
            data: DataFrame with market data and features
            
        Returns:
            pd.Series: Trading signals (1 for buy, -1 for sell, 0 for hold)
        """
        pass
    
    def calculate_position_size(self, data: pd.DataFrame, signal: int) -> float:
        """
        Calculate position size based on risk management rules.
        
        Args:
            data: Market data
            signal: Trading signal
            
        Returns:
            float: Position size (fraction of portfolio)
        """
        if signal == 0:
            return 0
        
        # Calculate monthly volatility based position size
        if 'Volatility_20' in data.columns:
            volatility = data['Volatility_20'].iloc[-1]
            if volatility > 0:
                # Inverse volatility scaling
                vol_adjusted_size = self.max_position_size * (0.02 / volatility) # 2% daily target volatility
                return min(vol_adjusted_size, self.max_position_size)
        
        return self.max_position_size
    
    def apply_risk_management(self, data: pd.DataFrame, signal: int) -> int:
        """
        Apply risk management rules to trading signals.
        
        Args:
            data: Market data
            signal: Original signal
            
        Returns:
            int: Risk-adjusted signal
        """
        
        current_price = data['Close'].iloc[-1]
        
        # If we have a position, check for exit conditions
        if self.current_position != 0 and self.entry_price is not None:
            pnl_pct = (current_price - self.entry_price) / self.entry_price
            
            # Long position risk management
            if self.current_position > 0:
                # Stop loss
                if pnl_pct <= -self.stop_loss_pct:
                    return -1  # Exit long position
                
                # Take profit
                if pnl_pct >= self.take_profit_pct:
                    return -1  # Exit long position
            
            # Short position risk management
            elif self.current_position < 0:
                # Stop loss for short
                if pnl_pct >= self.stop_loss_pct:
                    return 1  # Exit short position
                
                # Take profit for short
                if pnl_pct <= -self.take_profit_pct:
                    return 1  # Exit short position
        
        return signal

    def update_position(self, signal: int, price: float, timestamp: datetime):
        """
        Update position based on signal.
        
        Args:
            signal: Trading signal
            price: Current price
            timestamp: Current timestamp
        """
        if signal != 0:
            trade_info = {
                'timestamp': timestamp,
                'signal': signal,
                'price': price,
                'position_before': self.current_position,
                'position_after': signal
            }
            
            self.trades.append(trade_info)
            
            # Update position
            self.current_position = signal
            self.entry_price = price

class SimpleMAStrategy(BaseStrategy):
    """
    Simple Moving Average Crossover Strategy.
    """
    
    def __init__(self, params: Dict[str, Any] = None):
        default_params = {
            'fast_ma': 10,
            'slow_ma': 20
        }
        
        if params:
            default_params.update(params)
        
        super().__init__("SimpleMA", default_params)
    
    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """
        Generate simple MA crossover signals.
        
        Logic: Buy when fast MA > slow MA, sell when fast MA < slow MA.
        """
        signals = pd.Series(0, index=data.index)
        
        # Check for required columns
        if 'SMA_10' not in data.columns or 'SMA_20' not in data.columns:
            return signals
        
        # Simple crossover logic
        fast_ma = data['SMA_10']
        slow_ma = data['SMA_20']
        
        # Buy when fast MA is above slow MA
        buy_condition = fast_ma > slow_ma
        
        # Sell when fast MA is below slow MA
        sell_condition = fast_ma < slow_ma
        
        # Generate signals
        signals[buy_condition] = 1
        signals[sell_condition] = -1
        
        # Only trade on crossovers (when signal changes)
        signals = signals.diff().fillna(0)
        signals = signals.replace({2: 1, -2: -1})  # Clean up diff artifacts
        
        return signals

class MeanReversionStrategy(BaseStrategy):
    """
    Mean Reversion Strategy using Bollinger Bands and RSI.
    """
    
    def __init__(self, params: Dict[str, Any] = None):
        default_params = {
            'bb_period': 20,
            'bb_std': 2,
            'rsi_period': 14,
            'rsi_oversold': 30,
            'rsi_overbought': 70,
            'volume_threshold': 1.5
        }
        
        if params:
            default_params.update(params)
        
        super().__init__("MeanReversion", default_params)
    
    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """
        Generate mean reversion signals.
        """
        signals = pd.Series(0, index=data.index)
        
        # Ensure we have required columns
        required_cols = ['Close', 'BB_High', 'BB_Low', 'RSI', 'Volume_Ratio']
        if not all(col in data.columns for col in required_cols):
            return signals
        
        # More relaxed mean reversion conditions for better signal generation
        oversold_condition = (
            (data['Close'] < data['BB_Low']) &
            (data['RSI'] < self.params['rsi_oversold'])
            # Removed volume threshold to be less restrictive
        )
        
        overbought_condition = (
            (data['Close'] > data['BB_High']) &
            (data['RSI'] > self.params['rsi_overbought'])
            # Removed volume threshold to be less restrictive
        )
        
        # Generate signals
        signals[oversold_condition] = 1   # Buy signal
        signals[overbought_condition] = -1  # Sell signal
        
        # Less restrictive momentum filter
        if 'Price_Change_5d' in data.columns:
            # Only avoid extreme momentum
            strong_downtrend = data['Price_Change_5d'] < -0.15  # More extreme threshold
            signals[oversold_condition & strong_downtrend] = 0
            
            strong_uptrend = data['Price_Change_5d'] > 0.15  # More extreme threshold
            signals[overbought_condition & strong_uptrend] = 0
        
        return signals

class MomentumStrategy(BaseStrategy):
    """
    Momentum Strategy using multiple timeframes and trend confirmation.
    """
    
    def __init__(self, params: Dict[str, Any] = None):
        default_params = {
            'fast_ma': 10,
            'slow_ma': 50,
            'rsi_momentum_threshold': 50,
            'macd_confirmation': True,
            'volume_confirmation': True,
            'min_momentum_strength': 0.02
        }
        
        if params:
            default_params.update(params)
        
        super().__init__("Momentum", default_params)
    
    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """
        Generate momentum signals with multiple confirmations.
        """
        signals = pd.Series(0, index=data.index)
        
        # Ensure we have required columns
        required_cols = ['Close', 'SMA_10', 'SMA_50', 'RSI', 'MACD', 'MACD_Signal']
        if not all(col in data.columns for col in required_cols):
            return signals
        
        # Simplified momentum conditions - less restrictive
        price_above_fast_ma = data['Close'] > data['SMA_10']
        fast_ma_above_slow = data['SMA_10'] > data['SMA_50']
        
        price_below_fast_ma = data['Close'] < data['SMA_10']
        fast_ma_below_slow = data['SMA_10'] < data['SMA_50']
        
        # RSI momentum
        rsi_bullish = data['RSI'] > self.params['rsi_momentum_threshold']
        rsi_bearish = data['RSI'] < self.params['rsi_momentum_threshold']
        
        # MACD confirmation
        macd_bullish = data['MACD'] > data['MACD_Signal']
        macd_bearish = data['MACD'] < data['MACD_Signal']
        
        # Simplified bullish momentum conditions (removed some requirements)
        bullish_momentum = (
            price_above_fast_ma &
            fast_ma_above_slow &
            rsi_bullish &
            macd_bullish
        )
        
        # Simplified bearish momentum conditions
        bearish_momentum = (
            price_below_fast_ma &
            fast_ma_below_slow &
            rsi_bearish &
            macd_bearish
        )
        
        # Generate signals
        signals[bullish_momentum] = 1   # Buy signal
        signals[bearish_momentum] = -1  # Sell signal
        
        return signals

class StrategyFactory:
    """
    Factory class for creating trading strategies.
    """
    
    @staticmethod
    def create_strategy(strategy_name: str, params: Dict[str, Any] = None) -> BaseStrategy:
        """Create a strategy instance."""
        strategy_map = {
            'simple_ma': SimpleMAStrategy,
            'mean_reversion': MeanReversionStrategy,
            'momentum': MomentumStrategy
        }
        
        if strategy_name not in strategy_map:
            raise ValueError(f"Unknown strategy: {strategy_name}")
        
        return strategy_map[strategy_name](params)
    
    @staticmethod
    def get_available_strategies() -> List[str]:
        """Get list of available strategies."""
        return ['simple_ma', 'mean_reversion', 'momentum']
    
    @staticmethod
    def get_strategy_description(strategy_name: str) -> str:
        """Get description of a strategy."""
        descriptions = {
            'simple_ma': "Simple MA: Basic moving average crossover strategy",
            'mean_reversion': "Mean Reversion: Trades based on price returning to statistical mean",
            'momentum': "Momentum: Follows trending price movements with multiple confirmations"
        }
        
        return descriptions.get(strategy_name, "Unknown strategy")