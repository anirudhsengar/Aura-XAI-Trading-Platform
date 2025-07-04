import pandas as pd
import numpy as np
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any
import warnings
warnings.filterwarnings('ignore')

# Machine Learning
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

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
            self.logger.warning("Missing required columns for Mean Reversion strategy")
            return signals
        
        # Mean reversion conditions
        oversold_condition = (
            (data['Close'] < data['BB_Low']) &
            (data['RSI'] < self.params['rsi_oversold']) &
            (data['Volume_Ratio'] > self.params['volume_threshold'])
        )
        
        overbought_condition = (
            (data['Close'] > data['BB_High']) &
            (data['RSI'] > self.params['rsi_overbought']) &
            (data['Volume_Ratio'] > self.params['volume_threshold'])
        )
        
        # Generate signals
        signals[oversold_condition] = 1   # Buy signal
        signals[overbought_condition] = -1  # Sell signal
        
        # Add momentum filter to avoid catching falling knives
        if 'Price_Change_5d' in data.columns:
            # Don't buy if there's strong downward momentum
            strong_downtrend = data['Price_Change_5d'] < -0.10
            signals[oversold_condition & strong_downtrend] = 0
            
            # Don't sell if there's strong upward momentum
            strong_uptrend = data['Price_Change_5d'] > 0.10
            signals[overbought_condition & strong_uptrend] = 0
        
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
            self.logger.warning("Missing required columns for Momentum strategy")
            return signals
        
        # Momentum conditions
        price_above_fast_ma = data['Close'] > data['SMA_10']
        price_above_slow_ma = data['Close'] > data['SMA_50']
        fast_ma_above_slow = data['SMA_10'] > data['SMA_50']
        
        price_below_fast_ma = data['Close'] < data['SMA_10']
        price_below_slow_ma = data['Close'] < data['SMA_50']
        fast_ma_below_slow = data['SMA_10'] < data['SMA_50']
        
        # RSI momentum
        rsi_bullish = data['RSI'] > self.params['rsi_momentum_threshold']
        rsi_bearish = data['RSI'] < self.params['rsi_momentum_threshold']
        
        # MACD confirmation
        macd_bullish = data['MACD'] > data['MACD_Signal']
        macd_bearish = data['MACD'] < data['MACD_Signal']
        
        # Volume confirmation
        volume_confirmation = True
        if self.params['volume_confirmation'] and 'Volume_Ratio' in data.columns:
            volume_above_average = data['Volume_Ratio'] > 1.2
            volume_confirmation = volume_above_average
        
        # Momentum strength
        momentum_strength = True
        if 'Price_Change_5d' in data.columns:
            strong_positive_momentum = data['Price_Change_5d'] > self.params['min_momentum_strength']
            strong_negative_momentum = data['Price_Change_5d'] < -self.params['min_momentum_strength']
            
            # Bullish momentum conditions
            bullish_momentum = (
                price_above_fast_ma &
                price_above_slow_ma &
                fast_ma_above_slow &
                rsi_bullish &
                macd_bullish &
                volume_confirmation &
                strong_positive_momentum
            )
            
            # Bearish momentum conditions
            bearish_momentum = (
                price_below_fast_ma &
                price_below_slow_ma &
                fast_ma_below_slow &
                rsi_bearish &
                macd_bearish &
                volume_confirmation &
                strong_negative_momentum
            )
        else:
            # Without momentum strength filter
            bullish_momentum = (
                price_above_fast_ma &
                price_above_slow_ma &
                fast_ma_above_slow &
                rsi_bullish &
                macd_bullish &
                volume_confirmation
            )
            
            bearish_momentum = (
                price_below_fast_ma &
                price_below_slow_ma &
                fast_ma_below_slow &
                rsi_bearish &
                macd_bearish &
                volume_confirmation
            )
        
        # Generate signals
        signals[bullish_momentum] = 1   # Buy signal
        signals[bearish_momentum] = -1  # Sell signal
        
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

class MLStrategy(BaseStrategy):
    """
    Machine Learning Strategy using ensemble methods.
    
    Logic: Uses machine learning models to predict price direction
    based on historical patterns in technical and sentiment features.
    
    Why chosen: Can capture complex non-linear relationships in data,
    adapts to changing market conditions, and provides probabilistic
    predictions for better risk management.
    """
    
    def __init__(self, params: Dict[str, Any] = None):
        default_params = {
            'lookback_period': 252,  # 1 year
            'prediction_horizon': 5,  # 5 days ahead
            'retrain_frequency': 50,  # Retrain every 50 days
            'min_accuracy': 0.55,  # Minimum accuracy to use model
            'feature_importance_threshold': 0.01
        }
        
        if params:
            default_params.update(params)
        
        super().__init__("MLStrategy", default_params)
        
        # Initialize models
        self.models = {
            'rf': RandomForestClassifier(n_estimators=100, random_state=42),
            'gb': GradientBoostingClassifier(n_estimators=100, random_state=42),
            'lr': LogisticRegression(random_state=42)
        }
        
        self.scaler = StandardScaler()
        self.is_trained = False
        self.last_train_date = None
        self.feature_importance = {}
    
    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """
        Generate ML-based signals.
        
        Logic: Uses ensemble of ML models to predict price direction
        and generates signals based on prediction consensus.
        """
        signals = pd.Series(0, index=data.index)
        
        if len(data) < self.params['lookback_period']:
            self.logger.warning("Insufficient data for ML strategy")
            return signals
        
        # Prepare features
        features = self._prepare_features(data)
        
        if features.empty:
            self.logger.warning("No valid features for ML strategy")
            return signals
        
        # Check if we need to retrain
        if self._should_retrain(data):
            self._train_models(data, features)
        
        if not self.is_trained:
            self.logger.warning("ML models not trained yet")
            return signals
        
        # Generate predictions for recent data
        recent_features = features.tail(self.params['prediction_horizon'])
        predictions = self._predict(recent_features)
        
        # Convert predictions to signals
        for i, pred in enumerate(predictions):
            if i < len(signals):
                signals.iloc[-(len(predictions)-i)] = pred
        
        return signals
    
    def _prepare_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Prepare features for ML model."""
        feature_cols = [
            'RSI', 'MACD', 'BB_Position', 'Volume_Ratio',
            'Price_Change', 'Price_Change_5d', 'Volatility_20',
            'sentiment_mean', 'sentiment_momentum', 'Trend_Strength'
        ]
        
        # Select available features
        available_features = [col for col in feature_cols if col in data.columns]
        
        if not available_features:
            return pd.DataFrame()
        
        features = data[available_features].copy()
        
        # Add lag features
        for col in available_features[:5]:  # Only for main technical indicators
            if col in features.columns:
                features[f'{col}_lag1'] = features[col].shift(1)
                features[f'{col}_lag2'] = features[col].shift(2)
        
        # Add rolling statistics
        for col in available_features[:3]:
            if col in features.columns:
                features[f'{col}_ma3'] = features[col].rolling(3).mean()
                features[f'{col}_std3'] = features[col].rolling(3).std()
        
        # Drop NaN values
        features = features.dropna()
        
        return features
    
    def _create_labels(self, data: pd.DataFrame) -> pd.Series:
        """Create target labels for training."""
        # Predict if price will go up in the next N days
        future_returns = data['Close'].shift(-self.params['prediction_horizon']) / data['Close'] - 1
        
        # Create labels: 1 for up, -1 for down, 0 for neutral
        labels = pd.Series(0, index=data.index)
        labels[future_returns > 0.02] = 1   # Up if > 2% return
        labels[future_returns < -0.02] = -1  # Down if < -2% return
        
        return labels
    
    def _should_retrain(self, data: pd.DataFrame) -> bool:
        """Check if models should be retrained."""
        if not self.is_trained:
            return True
        
        if self.last_train_date is None:
            return True
        
        # Check if enough time has passed
        current_date = data.index[-1]
        days_since_train = (current_date - self.last_train_date).days
        
        return days_since_train >= self.params['retrain_frequency']
    
    def _train_models(self, data: pd.DataFrame, features: pd.DataFrame):
        """Train ML models."""
        self.logger.info("Training ML models...")
        
        # Create labels
        labels = self._create_labels(data)
        
        # Align features and labels
        common_index = features.index.intersection(labels.index)
        if len(common_index) < 100:  # Minimum training samples
            self.logger.warning("Insufficient training data")
            return
        
        X = features.loc[common_index]
        y = labels.loc[common_index]
        
        # Remove neutral labels for training
        non_neutral = y != 0
        X_train = X[non_neutral]
        y_train = y[non_neutral]
        
        if len(X_train) < 50:
            self.logger.warning("Insufficient non-neutral training samples")
            return
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        
        # Train models
        model_accuracies = {}
        
        for name, model in self.models.items():
            try:
                # Train model
                model.fit(X_train_scaled, y_train)
                
                # Evaluate on training data (could add validation split)
                y_pred = model.predict(X_train_scaled)
                accuracy = accuracy_score(y_train, y_pred)
                model_accuracies[name] = accuracy
                
                self.logger.info(f"Model {name} accuracy: {accuracy:.3f}")
                
            except Exception as e:
                self.logger.error(f"Error training model {name}: {str(e)}")
                model_accuracies[name] = 0
        
        # Check if models meet minimum accuracy
        best_accuracy = max(model_accuracies.values())
        if best_accuracy >= self.params['min_accuracy']:
            self.is_trained = True
            self.last_train_date = data.index[-1]
            self.logger.info(f"ML models trained successfully. Best accuracy: {best_accuracy:.3f}")
        else:
            self.logger.warning(f"Model accuracy {best_accuracy:.3f} below threshold {self.params['min_accuracy']}")
    
    def _predict(self, features: pd.DataFrame) -> List[int]:
        """Make predictions using ensemble of models."""
        if features.empty:
            return []
        
        # Scale features
        X_scaled = self.scaler.transform(features)
        
        # Get predictions from all models
        predictions = []
        
        for name, model in self.models.items():
            try:
                pred = model.predict(X_scaled)
                predictions.append(pred)
            except Exception as e:
                self.logger.error(f"Error predicting with model {name}: {str(e)}")
                predictions.append(np.zeros(len(features)))
        
        # Ensemble predictions (majority vote)
        ensemble_pred = []
        for i in range(len(features)):
            votes = [pred[i] for pred in predictions]
            # Take majority vote
            ensemble_pred.append(max(set(votes), key=votes.count))
        
        return ensemble_pred

class StrategyFactory:
    """
    Factory class for creating trading strategies.
    
    Logic: Centralized strategy creation with parameter validation
    and strategy selection based on market conditions.
    
    Why chosen: Provides clean interface for strategy creation,
    enables easy addition of new strategies, and ensures proper
    parameter validation.
    """
    
    @staticmethod
    def create_strategy(strategy_name: str, params: Dict[str, Any] = None) -> BaseStrategy:
        """
        Create a strategy instance.
        
        Args:
            strategy_name: Name of the strategy
            params: Strategy parameters
            
        Returns:
            BaseStrategy: Strategy instance
        """
        strategy_map = {
            'mean_reversion': MeanReversionStrategy,
            'momentum': MomentumStrategy,
            'multi_factor': MultiFactorStrategy,
            'ml_strategy': MLStrategy
        }
        
        if strategy_name not in strategy_map:
            raise ValueError(f"Unknown strategy: {strategy_name}")
        
        return strategy_map[strategy_name](params)
    
    @staticmethod
    def get_available_strategies() -> List[str]:
        """Get list of available strategies."""
        return ['mean_reversion', 'momentum', 'multi_factor', 'ml_strategy']
    
    @staticmethod
    def get_strategy_description(strategy_name: str) -> str:
        """Get description of a strategy."""
        descriptions = {
            'mean_reversion': "Mean Reversion: Trades based on price returning to statistical mean",
            'momentum': "Momentum: Follows trending price movements with multiple confirmations",
            'multi_factor': "Multi-Factor: Combines technical, sentiment, and momentum factors",
            'ml_strategy': "ML Strategy: Uses machine learning models to predict price direction"
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
