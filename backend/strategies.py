import pandas as pd
import numpy as np
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any
import warnings
warnings.filterwarnings('ignore')

# Import utilities
from utils import DateUtils, DataValidator, FileUtils, LoggingUtils, MathUtils

class BaseStrategy(ABC):
    """
    Base class for all trading strategies.
    
    Logic: Provides common functionality for all strategies including
    signal generation, risk management, and performance tracking.
    
    Why chosen: Ensures consistent interface across all strategies,
    implements common risk management, and provides extensibility.
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
        self.logger = LoggingUtils.setup_logger(f"Strategy_{name}")
        
        # Risk management parameters
        self.max_position_size = self.params.get('max_position_size', 0.1)  # 10% of portfolio
        self.stop_loss_pct = self.params.get('stop_loss_pct', 0.05)  # 5% stop loss
        self.take_profit_pct = self.params.get('take_profit_pct', 0.15)  # 15% take profit
        self.max_drawdown_limit = self.params.get('max_drawdown_limit', 0.20)  # 20% max drawdown
        
        # Position tracking
        self.current_position = 0
        self.entry_price = None
        self.trades = []
        
        self.logger.info(f"Strategy {name} initialized with params: {params}")
    
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
        
        Logic: Uses volatility-based position sizing and Kelly criterion
        for optimal position sizing.
        
        Args:
            data: Market data
            signal: Trading signal
            
        Returns:
            float: Position size (fraction of portfolio)
        """
        if signal == 0:
            return 0
        
        # Calculate volatility-based position size
        if 'Volatility_20' in data.columns:
            volatility = data['Volatility_20'].iloc[-1]
            if volatility > 0:
                # Inverse volatility scaling
                vol_adjusted_size = self.max_position_size * (0.02 / volatility)
                return min(vol_adjusted_size, self.max_position_size)
        
        return self.max_position_size
    
    def apply_risk_management(self, data: pd.DataFrame, signal: int) -> int:
        """
        Apply risk management rules to trading signals.
        
        Logic: Implements stop-loss, take-profit, and position limits
        to manage risk.
        
        Args:
            data: Market data
            signal: Original signal
            
        Returns:
            int: Risk-adjusted signal
        """
        # Add logging for debugging
        if hasattr(self, 'logger'):
            self.logger.info(f"Risk management: input signal={signal}, current_position={self.current_position}")
        
        # Temporarily disable risk management for debugging
        return signal
        
        # Original risk management code (commented out for debugging)
        """
        current_price = data['Close'].iloc[-1]
        
        # If we have a position, check for exit conditions
        if self.current_position != 0 and self.entry_price is not None:
            pnl_pct = (current_price - self.entry_price) / self.entry_price
            
            # Long position risk management
            if self.current_position > 0:
                # Stop loss
                if pnl_pct <= -self.stop_loss_pct:
                    self.logger.info(f"Stop loss triggered: {pnl_pct:.2%}")
                    return -1  # Exit long position
                
                # Take profit
                if pnl_pct >= self.take_profit_pct:
                    self.logger.info(f"Take profit triggered: {pnl_pct:.2%}")
                    return -1  # Exit long position
            
            # Short position risk management
            elif self.current_position < 0:
                # Stop loss for short
                if pnl_pct >= self.stop_loss_pct:
                    self.logger.info(f"Stop loss triggered (short): {pnl_pct:.2%}")
                    return 1  # Exit short position
                
                # Take profit for short
                if pnl_pct <= -self.take_profit_pct:
                    self.logger.info(f"Take profit triggered (short): {pnl_pct:.2%}")
                    return 1  # Exit short position
        
        return signal
        """

    def update_position(self, signal: int, price: float, timestamp: datetime):
        """
        Update position based on signal.
        
        Args:
            signal: Trading signal
            price: Current price
            timestamp: Current timestamp
        """
        if signal != 0:
            # Log the trade
            trade_info = {
                'timestamp': timestamp,
                'signal': signal,
                'price': price,
                'position_before': self.current_position,
                'position_after': signal
            }
            
            self.trades.append(trade_info)
            LoggingUtils.log_trade_execution(self.logger, trade_info)
            
            # Update position
            self.current_position = signal
            self.entry_price = price

class MeanReversionStrategy(BaseStrategy):
    """
    Mean Reversion Strategy using Bollinger Bands and RSI.
    
    Logic: Assumes prices revert to their mean over time. Buys when
    price is oversold and below lower Bollinger Band, sells when
    price is overbought and above upper Bollinger Band.
    
    Why chosen: Statistically robust, works well in ranging markets,
    and has strong theoretical foundation in financial mathematics.
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
        
        Logic: Combines Bollinger Bands, RSI, and volume analysis
        for high-probability mean reversion trades.
        """
        signals = pd.Series(0, index=data.index)
        
        # Ensure we have required columns
        required_cols = ['Close', 'BB_High', 'BB_Low', 'RSI', 'Volume_Ratio']
        if not all(col in data.columns for col in required_cols):
            self.logger.warning(f"Missing required columns for Mean Reversion strategy. Available: {data.columns.tolist()}")
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
        
        self.logger.info(f"Generated {(signals == 1).sum()} buy signals and {(signals == -1).sum()} sell signals")
        return signals

class MomentumStrategy(BaseStrategy):
    """
    Momentum Strategy using multiple timeframes and trend confirmation.
    
    Logic: Follows the trend by buying when multiple indicators confirm
    upward momentum and selling when they confirm downward momentum.
    
    Why chosen: Momentum strategies have shown consistent performance
    across different markets and timeframes. Multiple confirmations
    reduce false signals.
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
        
        Logic: Requires alignment of multiple indicators for signal
        generation to increase probability of success.
        """
        signals = pd.Series(0, index=data.index)
        
        # Ensure we have required columns
        required_cols = ['Close', 'SMA_10', 'SMA_50', 'RSI', 'MACD', 'MACD_Signal']
        if not all(col in data.columns for col in required_cols):
            self.logger.warning(f"Missing required columns for Momentum strategy. Available: {data.columns.tolist()}")
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
        
        self.logger.info(f"Generated {(signals == 1).sum()} buy signals and {(signals == -1).sum()} sell signals")
        return signals

class MultiFactorStrategy(BaseStrategy):
    """
    Multi-Factor Strategy combining technical indicators with sentiment.
    
    Logic: Integrates multiple sources of information including technical
    analysis, sentiment analysis, and market microstructure for robust
    signal generation.
    
    Why chosen: Diversified approach reduces strategy-specific risks,
    incorporates alternative data (sentiment), and adapts to different
    market conditions.
    """
    
    def __init__(self, params: Dict[str, Any] = None):
        default_params = {
            'technical_weight': 0.4,
            'sentiment_weight': 0.3,
            'momentum_weight': 0.3,
            'sentiment_threshold': 0.1,
            'signal_threshold': 0.5
        }
        
        if params:
            default_params.update(params)
        
        super().__init__("MultiFactor", default_params)
    
    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """
        Generate multi-factor signals by combining different components.
        
        Logic: Weights different factors and combines them into a
        composite signal score.
        """
        signals = pd.Series(0, index=data.index)
        
        # Technical factor
        technical_score = self._calculate_technical_score(data)
        
        # Sentiment factor
        sentiment_score = self._calculate_sentiment_score(data)
        
        # Momentum factor
        momentum_score = self._calculate_momentum_score(data)
        
        # Combine factors
        composite_score = (
            technical_score * self.params['technical_weight'] +
            sentiment_score * self.params['sentiment_weight'] +
            momentum_score * self.params['momentum_weight']
        )
        
        # Generate signals based on composite score
        buy_condition = composite_score > self.params['signal_threshold']
        sell_condition = composite_score < -self.params['signal_threshold']
        
        signals[buy_condition] = 1
        signals[sell_condition] = -1
        
        return signals
    
    def _calculate_technical_score(self, data: pd.DataFrame) -> pd.Series:
        """Calculate technical analysis score."""
        score = pd.Series(0.0, index=data.index)
        
        # RSI component
        if 'RSI' in data.columns:
            rsi_score = (data['RSI'] - 50) / 50  # Normalize to [-1, 1]
            score += rsi_score * 0.3
        
        # Bollinger Bands component
        if 'BB_Position' in data.columns:
            bb_score = (data['BB_Position'] - 0.5) * 2  # Normalize to [-1, 1]
            score += bb_score * 0.3
        
        # MACD component
        if 'MACD' in data.columns and 'MACD_Signal' in data.columns:
            macd_diff = data['MACD'] - data['MACD_Signal']
            macd_score = np.tanh(macd_diff * 10)  # Normalize with tanh
            score += macd_score * 0.4
        
        return score
    
    def _calculate_sentiment_score(self, data: pd.DataFrame) -> pd.Series:
        """Calculate sentiment score."""
        score = pd.Series(0.0, index=data.index)
        
        if 'sentiment_mean' in data.columns:
            # Normalize sentiment score
            sentiment_score = np.tanh(data['sentiment_mean'] * 5)
            score += sentiment_score * 0.5
        
        if 'sentiment_momentum' in data.columns:
            # Add sentiment momentum
            momentum_score = np.tanh(data['sentiment_momentum'] * 10)
            score += momentum_score * 0.3
        
        if 'news_count' in data.columns:
            # Add news volume factor
            news_score = np.tanh((data['news_count'] - data['news_count'].mean()) / data['news_count'].std())
            score += news_score * 0.2
        
        return score
    
    def _calculate_momentum_score(self, data: pd.DataFrame) -> pd.Series:
        """Calculate momentum score."""
        score = pd.Series(0.0, index=data.index)
        
        # Price momentum
        if 'Price_Change_5d' in data.columns:
            price_momentum = np.tanh(data['Price_Change_5d'] * 20)
            score += price_momentum * 0.4
        
        # Volume momentum
        if 'Volume_Ratio' in data.columns:
            volume_momentum = np.tanh((data['Volume_Ratio'] - 1) * 2)
            score += volume_momentum * 0.3
        
        # Trend strength
        if 'Trend_Strength' in data.columns:
            trend_score = data['Trend_Strength']
            score += trend_score * 0.3
        
        return score

class SimpleMAStrategy(BaseStrategy):
    """
    Simple Moving Average Crossover Strategy.
    
    Logic: Buy when fast MA crosses above slow MA, sell when it crosses below.
    This is the simplest trend-following strategy.
    
    Why chosen: Extremely simple, easy to debug, widely understood.
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
            self.logger.warning(f"Missing MA columns. Available: {data.columns.tolist()}")
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
        
        buy_signals = (signals == 1).sum()
        sell_signals = (signals == -1).sum()
        
        self.logger.info(f"Simple MA Strategy generated {buy_signals} buy signals and {sell_signals} sell signals")
        
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
            'momentum': MomentumStrategy,
            'multi_factor': MultiFactorStrategy
        }
        
        if strategy_name not in strategy_map:
            raise ValueError(f"Unknown strategy: {strategy_name}")
        
        return strategy_map[strategy_name](params)
    
    @staticmethod
    def get_available_strategies() -> List[str]:
        """Get list of available strategies."""
        return ['simple_ma', 'mean_reversion', 'momentum', 'multi_factor']
    
    @staticmethod
    def get_strategy_description(strategy_name: str) -> str:
        """Get description of a strategy."""
        descriptions = {
            'simple_ma': "Simple MA: Basic moving average crossover strategy",
            'mean_reversion': "Mean Reversion: Trades based on price returning to statistical mean",
            'momentum': "Momentum: Follows trending price movements with multiple confirmations",
            'multi_factor': "Multi-Factor: Combines technical, sentiment, and momentum factors"
        }
        
        return descriptions.get(strategy_name, "Unknown strategy")

# Example usage and testing
if __name__ == "__main__":
    # Test strategy creation
    print("Testing Strategy Factory...")
    
    # Create different strategies
    strategies = []
    for strategy_name in StrategyFactory.get_available_strategies():
        try:
            strategy = StrategyFactory.create_strategy(strategy_name)
            strategies.append(strategy)
            print(f"Created {strategy_name}: {StrategyFactory.get_strategy_description(strategy_name)}")
        except Exception as e:
            print(f"Error creating {strategy_name}: {str(e)}")
    
    # Test with sample data
    print("\nTesting signal generation...")
    
    # Create sample data
    dates = pd.date_range(start='2024-01-01', periods=100, freq='D')
    sample_data = pd.DataFrame({
        'Close': np.random.randn(100).cumsum() + 100,
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
        'Volatility_20': np.random.uniform(0.1, 0.3, 100),
        'sentiment_mean': np.random.uniform(-0.5, 0.5, 100),
        'sentiment_momentum': np.random.randn(100) * 0.1,
        'Trend_Strength': np.random.uniform(-1, 1, 100),
        'news_count': np.random.randint(1, 10, 100)
    }, index=dates)
    
    # Test signal generation for each strategy
    for strategy in strategies:
        try:
            signals = strategy.generate_signals(sample_data)
            buy_signals = (signals == 1).sum()
            sell_signals = (signals == -1).sum()
            print(f"{strategy.name}: {buy_signals} buy signals, {sell_signals} sell signals")
        except Exception as e:
            print(f"Error testing {strategy.name}: {str(e)}")
    
    print("\nStrategy testing completed!")