import pandas as pd
import numpy as np
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Dict, Any
import warnings
import ta
from sklearn.preprocessing import StandardScaler

# Remove eager SHAP import to avoid TF gradient conflicts and use lazy import instead
SHAP_AVAILABLE = None  # Unknown until first use

try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False
    print("Warning: TensorFlow not available. LSTM strategy will not work.")

warnings.filterwarnings('ignore')

class BaseStrategy(ABC):
    """
    Base class for all trading strategies.
    """
    
    def __init__(self, name: str, params: Dict[str, Any] = None):
        """
        Initialize base strategy.
        """
        self.name = name
        self.params = params or {}
        self.max_position_size = self.params.get('max_position_size', 0.1)
        self.stop_loss_pct = self.params.get('stop_loss_pct', 0.05)
        self.take_profit_pct = self.params.get('take_profit_pct', 0.15)
        self.current_position = 0
        self.entry_price = None
    
    @abstractmethod
    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """
        Generate trading signals.
        """
        pass
    
    @staticmethod
    def create_strategy(strategy_name: str, params: Dict[str, Any] = None):
        """
        Factory method to create strategy instances.
        """
        strategy_map = {
            'simple_ma': SimpleMAStrategy,
            'mean_reversion': MeanReversionStrategy,
            'momentum': MomentumStrategy,
            'lstm_strategy': LSTMStrategy
        }
        
        if strategy_name not in strategy_map:
            raise ValueError(f"Unknown strategy: {strategy_name}")
        
        strategy_class = strategy_map[strategy_name]
        return strategy_class(params)

    def calculate_position_size(self, market_data: pd.Series) -> float:
        """
        Calculate position size based on volatility.
        """
        volatility = market_data.get('Volatility_20', 0.02)
        if volatility <= 0 or pd.isna(volatility):
            volatility = 0.02  # Default volatility
        
        # Risk-adjusted position sizing
        volatility_adjusted_size = min(self.max_position_size, 0.02 / volatility)
        return max(0.01, volatility_adjusted_size)  # Minimum 1% position

    def apply_risk_management(self, market_data: pd.Series, signal: int) -> int:
        """
        Apply stop loss and take profit.
        """
        if self.current_position != 0 and self.entry_price is not None:
            current_price = market_data['Close']
            pnl_pct = (current_price - self.entry_price) / self.entry_price
            
            if self.current_position > 0:  # Long position
                if pnl_pct <= -self.stop_loss_pct or pnl_pct >= self.take_profit_pct:
                    return -1  # Exit long
            elif self.current_position < 0:  # Short position
                if pnl_pct >= self.stop_loss_pct or pnl_pct <= -self.take_profit_pct:
                    return 1  # Exit short
        return signal

    def update_position(self, signal: int, price: float, timestamp: datetime):
        """
        Update position based on signal.
        - signal > 0: enter long if flat, or exit short if short
        - signal < 0: enter short if flat, or exit long if long
        """
        # Entry long or exit short
        if signal > 0:
            if self.current_position < 0:
                # Exit short
                self.current_position = 0
                self.entry_price = None
            elif self.current_position == 0:
                # Enter long
                self.current_position = 1
                self.entry_price = price

        # Entry short or exit long
        elif signal < 0:
            if self.current_position > 0:
                # Exit long
                self.current_position = 0
                self.entry_price = None
            elif self.current_position == 0:
                # Enter short
                self.current_position = -1
                self.entry_price = price

class SimpleMAStrategy(BaseStrategy):
    """
    Simple Moving Average Crossover Strategy.
    """
    def __init__(self, params: Dict[str, Any] = None):
        default_params = {'fast_ma': 10, 'slow_ma': 50}
        if params:
            default_params.update(params)
        super().__init__("SimpleMA", default_params)
    
    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """Generate signals based on MA crossover with proper state management."""
        fast_ma_col = f'SMA_{self.params["fast_ma"]}'
        slow_ma_col = f'SMA_{self.params["slow_ma"]}'
        
        if fast_ma_col not in data.columns or slow_ma_col not in data.columns:
            raise ValueError(f"Required MA columns not found in data")
        
        fast_ma = data[fast_ma_col]
        slow_ma = data[slow_ma_col]
        
        # Initialize signals series
        signals = pd.Series(0, index=data.index)
        
        # Create boolean series for crossover detection
        fast_above_slow = (fast_ma > slow_ma).fillna(False)
        fast_above_slow_prev = fast_above_slow.shift(1).fillna(False)
        
        # Generate signals only on crossovers
        bullish_crossover = fast_above_slow & (~fast_above_slow_prev)  # Fast MA crosses above slow MA
        bearish_crossover = (~fast_above_slow) & fast_above_slow_prev  # Fast MA crosses below slow MA
        
        # Set signals: 1 for buy, -1 for sell
        signals[bullish_crossover] = 1
        signals[bearish_crossover] = -1
        
        # Apply risk management for each signal
        for i, (timestamp, market_data) in enumerate(data.iterrows()):
            if signals.loc[timestamp] != 0:
                # Apply risk management
                original_signal = signals.loc[timestamp]
                adjusted_signal = self.apply_risk_management(market_data, original_signal)
                signals.loc[timestamp] = adjusted_signal
                
                # Update position if signal is valid
                if adjusted_signal != 0:
                    self.update_position(adjusted_signal, market_data['Close'], timestamp)
        
        return signals

class MeanReversionStrategy(BaseStrategy):
    """
    Mean Reversion Strategy using Bollinger Bands and RSI.
    """
    def __init__(self, params: Dict[str, Any] = None):
        default_params = {
            'bb_period': 20,
            'rsi_period': 14,
            'rsi_oversold': 30,
            'rsi_overbought': 70,
            'rsi_neutral_upper': 55,  # Upper bound for neutral RSI
            'rsi_neutral_lower': 45   # Lower bound for neutral RSI
        }
        if params:
            default_params.update(params)
        super().__init__("MeanReversion", default_params)

    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """Generate mean reversion signals with proper position management."""
        required_cols = ['Close', 'BB_Low', 'BB_High', 'RSI']
        for col in required_cols:
            if col not in data.columns:
                raise ValueError(f"Required column {col} not found in data")
        
        signals = pd.Series(0, index=data.index)
        current_position = 0
        
        for i, (timestamp, market_data) in enumerate(data.iterrows()):
            signal = 0
            
            # Entry conditions for mean reversion
            if current_position == 0:  # No current position
                # Long entry: Oversold conditions
                if (market_data['Close'] < market_data['BB_Low'] and 
                    market_data['RSI'] < self.params['rsi_oversold']):
                    signal = 1
                    current_position = 1
                
                # Short entry: Overbought conditions  
                elif (market_data['Close'] > market_data['BB_High'] and 
                      market_data['RSI'] > self.params['rsi_overbought']):
                    signal = -1
                    current_position = -1
            
            # Exit conditions
            elif current_position == 1:  # Long position
                # Exit long when price returns to middle or RSI normalizes
                bb_middle = (market_data['BB_High'] + market_data['BB_Low']) / 2
                if (market_data['Close'] >= bb_middle or 
                    market_data['RSI'] > self.params['rsi_neutral_upper']):
                    signal = -1  # Exit long
                    current_position = 0
            
            elif current_position == -1:  # Short position
                # Exit short when price returns to middle or RSI normalizes
                bb_middle = (market_data['BB_High'] + market_data['BB_Low']) / 2
                if (market_data['Close'] <= bb_middle or 
                    market_data['RSI'] < self.params['rsi_neutral_lower']):
                    signal = 1  # Exit short
                    current_position = 0
            
            # Apply risk management if we have a signal
            if signal != 0:
                adjusted_signal = self.apply_risk_management(market_data, signal)
                signals.loc[timestamp] = adjusted_signal
                
                # Update position tracking
                if adjusted_signal != 0:
                    self.update_position(adjusted_signal, market_data['Close'], timestamp)
                    # Update local position tracking based on the actual signal
                    if adjusted_signal == 1 and current_position <= 0:
                        current_position = 1
                    elif adjusted_signal == -1 and current_position >= 0:
                        current_position = -1 if current_position == 0 else 0
                    elif adjusted_signal == -1 and current_position == 1:
                        current_position = 0
                    elif adjusted_signal == 1 and current_position == -1:
                        current_position = 0
        
        return signals

class MomentumStrategy(BaseStrategy):
    """
    Momentum Strategy using multiple indicators.
    """
    def __init__(self, params: Dict[str, Any] = None):
        default_params = {
            'fast_ma': 12,
            'slow_ma': 26,
            'rsi_momentum_threshold': 50,
            'rsi_strong_momentum': 60,  # Strong momentum threshold
            'rsi_weak_momentum': 40,   # Weak momentum threshold
            'macd_strength_threshold': 0.001  # Minimum MACD strength
        }
        if params:
            default_params.update(params)
        super().__init__("Momentum", default_params)

    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """Generate momentum signals with proper position management."""
        fast_ma_col = f'SMA_{self.params["fast_ma"]}'
        slow_ma_col = f'SMA_{self.params["slow_ma"]}'
        
        required_cols = [fast_ma_col, slow_ma_col, 'RSI', 'MACD', 'MACD_Signal']
        for col in required_cols:
            if col not in data.columns:
                raise ValueError(f"Required column {col} not found in data")
        
        signals = pd.Series(0, index=data.index)
        current_position = 0
        
        for i, (timestamp, market_data) in enumerate(data.iterrows()):
            signal = 0
            
            # Calculate momentum strength indicators
            ma_trend = market_data[fast_ma_col] > market_data[slow_ma_col]
            rsi_momentum = market_data['RSI']
            macd_momentum = market_data['MACD'] - market_data['MACD_Signal']
            macd_bullish = market_data['MACD'] > market_data['MACD_Signal']
            
            # Entry conditions for momentum
            if current_position == 0:  # No current position
                # Long entry: Strong bullish momentum
                if (ma_trend and 
                    rsi_momentum > self.params['rsi_strong_momentum'] and 
                    macd_bullish and 
                    abs(macd_momentum) > self.params['macd_strength_threshold']):
                    signal = 1
                    current_position = 1
                
                # Short entry: Strong bearish momentum  
                elif (not ma_trend and 
                      rsi_momentum < self.params['rsi_weak_momentum'] and 
                      not macd_bullish and 
                      abs(macd_momentum) > self.params['macd_strength_threshold']):
                    signal = -1
                    current_position = -1
            
            # Exit conditions - momentum weakening
            elif current_position == 1:  # Long position
                # Exit long when momentum weakens
                if (not ma_trend or 
                    rsi_momentum < self.params['rsi_momentum_threshold'] or 
                    not macd_bullish):
                    signal = -1  # Exit long
                    current_position = 0
            
            elif current_position == -1:  # Short position
                # Exit short when momentum weakens
                if (ma_trend or 
                    rsi_momentum > self.params['rsi_momentum_threshold'] or 
                    macd_bullish):
                    signal = 1  # Exit short
                    current_position = 0
            
            # Apply risk management if we have a signal
            if signal != 0:
                adjusted_signal = self.apply_risk_management(market_data, signal)
                signals.loc[timestamp] = adjusted_signal
                
                # Update position tracking
                if adjusted_signal != 0:
                    self.update_position(adjusted_signal, market_data['Close'], timestamp)
                    # Update local position tracking based on the actual signal
                    if adjusted_signal == 1 and current_position <= 0:
                        current_position = 1
                    elif adjusted_signal == -1 and current_position >= 0:
                        current_position = -1 if current_position == 0 else 0
                    elif adjusted_signal == -1 and current_position == 1:
                        current_position = 0
                    elif adjusted_signal == 1 and current_position == -1:
                        current_position = 0
        
        return signals

class LSTMStrategy(BaseStrategy):
    """
    LSTM-based Deep Learning Trading Strategy with SHAP explanations.
    """
    
    def __init__(self, params: Dict[str, Any] = None):
        if not TENSORFLOW_AVAILABLE:
            raise ImportError("TensorFlow is required for LSTM strategy. Please install with: pip install tensorflow")
        
        default_params = {
            'lookback_window': 60,
            'prediction_horizon': 5,
            'lstm_units': [64, 32],
            'dropout_rate': 0.2,
            'learning_rate': 0.001,
            'batch_size': 32,
            'epochs': 100,
            'validation_split': 0.2,
            'early_stopping_patience': 15,
            'min_data_length': 300,
            'retrain_frequency': 30,
            'signal_threshold': 0.45,   # lower default for 3-class softmax
            'min_prob_margin': 0.10,    # margin between up and down probs
            # labeling controls
            'labeling_mode': 'quantile',      # 'quantile' or 'fixed'
            'labeling_pos_q': 0.6,            # 60th percentile for up
            'labeling_neg_q': 0.4,            # 40th percentile for down
            'return_threshold': 0.02,         # used when labeling_mode='fixed'
            'use_shap': True,
            'max_features': 20,
            'max_cache_size': 100,
        }
        if params:
            default_params.update(params)
        
        super().__init__("LSTM_Strategy", default_params)
        
        # Model components
        self.model = None
        self.scaler = None
        self.feature_scaler = None
        self.last_training_date = None
        self.feature_importance = {}
        self.shap_explainer = None
        self.shap_values = None
        # SHAP runtime attributes
        self._shap_mode = None            # 'deep' or 'kernel'
        self._kernel_input_shape = None   # (lookback, n_features) when kernel is used
        self._shap_module = None          # keep module ref if needed
        # convenience flag
        self.shap_enabled = bool(self.params.get('use_shap', False))
        # cache background for SHAP re-init
        self._background_data = None
        
        # SHAP explanation storage
        self.explanation_cache = {}
        
    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """
        Generate trading signals using LSTM predictions with risk management.
        """
        print(f"LSTM Strategy: Processing {len(data)} data points")
        
        try:
            # Check minimum data requirement
            if len(data) < self.params['min_data_length']:
                print(f"LSTM Strategy: Insufficient data ({len(data)} < {self.params['min_data_length']})")
                return pd.Series(0, index=data.index)
            
            # Prepare features for LSTM
            features_df = self._prepare_lstm_features(data)
            if features_df is None or len(features_df) < self.params['lookback_window']:
                print("LSTM Strategy: Feature preparation failed")
                return pd.Series(0, index=data.index)
            
            # Check if model needs training/retraining
            if self._should_retrain(data.index[-1]):
                print("LSTM Strategy: Training/retraining model...")
                self._train_model(features_df, data)
            
            # Generate predictions and signals with risk management
            signals = self._generate_lstm_signals_with_risk_management(features_df, data)
            
            # Generate SHAP explanations for recent predictions
            if self.params.get('use_shap') and self.shap_explainer is not None:
                self._generate_shap_explanations(features_df.tail(30))
            
            # Clean up old cache entries
            self._cleanup_explanation_cache()
            
            signal_summary = signals[signals != 0]
            print(f"LSTM Strategy: Generated {len(signal_summary)} signals")
            
            return signals
            
        except Exception as e:
            print(f"LSTM Strategy: Error in signal generation: {str(e)}")
            import traceback
            traceback.print_exc()
            return pd.Series(0, index=data.index)
    
    def _generate_lstm_signals_with_risk_management(self, features_df: pd.DataFrame, price_data: pd.DataFrame) -> pd.Series:
        """
        Generate trading signals from LSTM predictions with risk management integration.
        Uses self.current_position (persistent state) instead of a local shadow variable.
        """
        signals = pd.Series(0, index=price_data.index)
        
        if self.model is None or self.feature_scaler is None:
            print("LSTM Strategy: Model not trained")
            return signals
        
        try:
            lookback = self.params['lookback_window']
            prediction_cache = {}
            
            for i in range(lookback, len(features_df)):
                current_date = features_df.index[i]
                if current_date not in price_data.index:
                    continue

                market_data = price_data.loc[current_date]
                signal = 0

                # If flat, consider new entry based on prediction
                if self.current_position == 0:
                    seq = features_df.iloc[i - lookback:i].values
                    seq_scaled = self.feature_scaler.transform(
                        seq.reshape(-1, seq.shape[1])
                    ).reshape(1, lookback, -1)

                    pred = self.model.predict(seq_scaled, verbose=0)[0]
                    # Map probs: index 0->down(-1), 1->neutral(0), 2->up(1)
                    p_down, p_neutral, p_up = float(pred[0]), float(pred[1]), float(pred[2])
                    max_prob = float(np.max(pred))
                    predicted_class = int(np.argmax(pred) - 1)  # -1, 0, 1
                    # Directional dominance between up and down
                    directional_margin = abs(p_up - p_down)
                    directional_class = 1 if p_up > p_down else -1

                    # Cache for explanation
                    prediction_cache[current_date] = {
                        'sequence': seq_scaled[0],
                        'prediction': pred,
                        'features': features_df.columns.tolist()
                    }

                    # Entry logic:
                    # - standard: confident top class and directional (not neutral)
                    # - margin: strong up-vs-down dominance even if neutral is top
                    if (
                        (max_prob >= self.params.get('signal_threshold', 0.45) and predicted_class != 0)
                        or (directional_margin >= self.params.get('min_prob_margin', 0.10))
                    ):
                        signal = predicted_class if predicted_class != 0 else directional_class
                        # Cache explanation for entry
                        self._generate_signal_explanation(
                            current_date,
                            seq_scaled[0],
                            pred,
                            features_df.columns.tolist(),
                            signal_type="trade"
                        )

                # If in a position, apply risk management for exits
                else:
                    rm_signal = self.apply_risk_management(market_data, 0)
                    if rm_signal != 0:
                        signal = rm_signal
                        # Explanation for risk exit using last prediction if available
                        cached = prediction_cache.get(current_date)
                        if cached:
                            self._generate_signal_explanation(
                                current_date,
                                cached['sequence'],
                                cached['prediction'],
                                cached['features'],
                                signal_type="risk_exit"
                            )

                # Record signal and update position state
                if signal != 0:
                    signals.loc[current_date] = signal
                    self.update_position(signal, market_data['Close'], current_date)

            print(f"LSTM Strategy: Generated explanations for {len(self.explanation_cache)} signals")
            return signals

        except Exception as e:
            print(f"LSTM Strategy: Signal generation error: {str(e)}")
            return signals

    def _shap_has_values(self, shap_values) -> bool:
        """
        Safely check if SHAP values exist (handles list/ndarray/None).
        """
        try:
            if shap_values is None:
                return False
            # List or tuple of arrays
            if isinstance(shap_values, (list, tuple)):
                return len(shap_values) > 0 and all(
                    (sv is not None) and (not isinstance(sv, np.ndarray) or sv.size > 0)
                    for sv in shap_values
                )
            # Single ndarray
            if isinstance(shap_values, np.ndarray):
                return shap_values.size > 0
            return bool(shap_values)
        except Exception:
            return False

    def _generate_signal_explanation(self, date, sequence, prediction, features, signal_type="trade"):
        """
        Generate explanation for a specific trading signal.
        Always stores minimal context; adds SHAP values if available.
        """
        try:
            shap_values = []
            if self.params.get('use_shap', False) and self.shap_explainer is not None:
                try:
                    if self._shap_mode == 'deep':
                        shap_values = self.shap_explainer.shap_values(sequence.reshape(1, *sequence.shape))
                    elif self._shap_mode == 'kernel':
                        x_flat = sequence.reshape(1, -1)
                        shap_values = self.shap_explainer.shap_values(x_flat, nsamples=min(100, x_flat.shape[1] * 2))
                except Exception as e:
                    print(f"LSTM Strategy: Error generating SHAP values for {signal_type} on {date}: {str(e)}")
                    shap_values = []
                    # Runtime fallback: if Deep fails at inference time, switch to Kernel once and retry
                    if self._shap_mode == 'deep' and self._background_data is not None:
                        try:
                            self._initialize_shap_explainer(self._background_data, prefer_kernel=True)
                            if self.shap_explainer is not None and self._shap_mode == 'kernel':
                                x_flat = sequence.reshape(1, -1)
                                shap_values = self.shap_explainer.shap_values(x_flat, nsamples=min(100, x_flat.shape[1] * 2))
                        except Exception as e2:
                            print(f"LSTM Strategy: Kernel fallback failed for {signal_type} on {date}: {str(e2)}")
                            shap_values = []
            # Store explanation with or without SHAP
            self.explanation_cache[date] = {
                'shap_values': shap_values,
                'features': features,
                'sequence': sequence,  # keep for fallback feature importance
                'prediction': prediction.tolist(),
                'prediction_date': date,
                'signal_type': signal_type,
                'max_probability': float(np.max(prediction)),
                'predicted_class': int(np.argmax(prediction) - 1),
                'timestamp': datetime.now()
            }
            # Avoid ambiguous truth-value on numpy arrays
            had_shap = self._shap_has_values(shap_values)
            print(f"LSTM Strategy: Cached {'SHAP' if had_shap else 'minimal'} explanation for {signal_type} on {date}")
        except Exception as e:
            print(f"LSTM Strategy: Error caching signal explanation: {str(e)}")

    def _generate_shap_explanations(self, recent_features: pd.DataFrame):
        """
        Generate SHAP explanations for recent predictions (backup method).
        Uses trained model + scaler to compute predictions and cache explanations.
        """
        if self.model is None or self.feature_scaler is None:
            return
        try:
            lookback = self.params['lookback_window']
            for i in range(lookback, len(recent_features)):
                current_date = recent_features.index[i]
                if current_date in self.explanation_cache:
                    continue
                try:
                    sequence = recent_features.iloc[i - lookback:i].values
                    sequence_scaled = self.feature_scaler.transform(
                        sequence.reshape(-1, sequence.shape[1])
                    ).reshape(1, lookback, -1)
                    prediction = self.model.predict(sequence_scaled, verbose=0)[0]
                    self._generate_signal_explanation(
                        current_date,
                        sequence_scaled[0],
                        prediction,
                        recent_features.columns.tolist(),
                        signal_type="analysis"
                    )
                except Exception as e:
                    print(f"Error processing sequence at index {i}: {str(e)}")
                    continue
        except Exception as e:
            print(f"LSTM Strategy: SHAP/analysis explanation error: {str(e)}")

    def get_feature_importance(self, date=None) -> Dict[str, float]:
        """
        Get feature importance.
        Uses SHAP when available; falls back to sequence magnitude-based proxy otherwise.
        """
        if not self.explanation_cache:
            return {}
        try:
            # Select the explanation: specific date or latest
            if date and date in self.explanation_cache:
                explanation = self.explanation_cache[date]
            else:
                latest_date = max(self.explanation_cache.keys())
                explanation = self.explanation_cache[latest_date]
            
            features = explanation.get('features', [])
            shap_values = explanation.get('shap_values', [])
            
            # Preferred: SHAP-based importance
            if self._shap_has_values(shap_values) and features:
                importance = {}
                for feat_idx, feature in enumerate(features):
                    total_importance = 0
                    valid_classes = 0
                    # Normalize shap_values into iterable of arrays
                    iterable = shap_values if isinstance(shap_values, (list, tuple)) else [shap_values]
                    for class_shap in iterable:
                        if class_shap is None:
                            continue
                        arr = class_shap
                        # KernelExplainer may produce flat SHAP (samples, features_flat). Reshape back.
                        if isinstance(arr, np.ndarray) and arr.ndim == 2 and self._shap_mode == 'kernel' and self._kernel_input_shape:
                            lookback, n_feats = self._kernel_input_shape
                            try:
                                arr = arr.reshape(1, lookback, n_feats)
                            except Exception:
                                pass
                        if isinstance(arr, np.ndarray) and arr.ndim >= 3 and feat_idx < arr.shape[-1]:
                            feature_importance = np.mean(np.abs(arr[0, :, feat_idx]))
                            if not np.isnan(feature_importance):
                                total_importance += feature_importance
                                valid_classes += 1
                    if valid_classes > 0:
                        importance[feature] = total_importance / valid_classes
                total_importance = sum(importance.values())
                if total_importance > 0:
                    importance = {k: v/total_importance for k, v in importance.items()}
                return dict(sorted(importance.items(), key=lambda x: x[1], reverse=True))
            
            # Fallback: use input sequence magnitude as a rough proxy
            sequence = explanation.get('sequence', None)
            if sequence is not None and len(features) == sequence.shape[-1]:
                per_feature = np.mean(np.abs(sequence), axis=0)
                total = np.sum(per_feature)
                if total > 0:
                    per_feature = per_feature / total
                importance = {feat: float(per_feature[idx]) for idx, feat in enumerate(features)}
                return dict(sorted(importance.items(), key=lambda x: x[1], reverse=True))
            
            return {}
        except Exception as e:
            print(f"LSTM Strategy: Feature importance error: {str(e)}")
            return {}

    def get_prediction_explanation(self, date=None) -> Dict[str, Any]:
        """
        Get detailed explanation for a specific prediction.
        """
        if not self.explanation_cache:
            return {'error': 'No explanations available'}
        
        try:
            if date and date in self.explanation_cache:
                explanation = self.explanation_cache[date]
            else:
                # Return most recent explanation
                latest_date = max(self.explanation_cache.keys())
                explanation = self.explanation_cache[latest_date]
            
            # Add summary information
            summary = {
                'explanation_date': explanation.get('prediction_date'),
                'signal_type': explanation.get('signal_type', 'unknown'),
                'predicted_class': explanation.get('predicted_class', 0),
                'max_probability': explanation.get('max_probability', 0.0),
                'confidence_level': 'High' if explanation.get('max_probability', 0) > 0.8 else 'Medium' if explanation.get('max_probability', 0) > 0.6 else 'Low',
                'features_count': len(explanation.get('features', [])),
                'prediction_classes': {
                    -1: 'Sell Signal',
                    0: 'Hold/Neutral',
                    1: 'Buy Signal'
                }
            }
            
            # Combine with original explanation
            result = {**explanation, 'summary': summary}
            return result
            
        except Exception as e:
            return {'error': f'Failed to get explanation: {str(e)}'}
    
    def _cleanup_explanation_cache(self):
        """
        Clean up old explanation cache entries to prevent memory issues.
        """
        if len(self.explanation_cache) > self.params['max_cache_size']:
            # Keep only the most recent entries
            sorted_dates = sorted(self.explanation_cache.keys())
            excess_count = len(sorted_dates) - self.params['max_cache_size']
            
            for date in sorted_dates[:excess_count]:
                del self.explanation_cache[date]
            
            print(f"LSTM Strategy: Cleaned up {excess_count} old explanation cache entries")

    def _prepare_lstm_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Prepare comprehensive features optimized for LSTM training.
        """
        try:
            df = data.copy()
            features = pd.DataFrame(index=df.index)
            
            # Price-based features
            features['close'] = df['Close']
            features['high'] = df['High'] if 'High' in df.columns else df['Close']
            features['low'] = df['Low'] if 'Low' in df.columns else df['Close']
            features['open'] = df['Open'] if 'Open' in df.columns else df['Close']
            
            # Returns and momentum
            features['return_1d'] = df['Close'].pct_change()
            features['return_5d'] = df['Close'].pct_change(5)
            features['return_20d'] = df['Close'].pct_change(20)
            
            # Volatility measures
            features['volatility_5d'] = features['return_1d'].rolling(5).std()
            features['volatility_20d'] = features['return_1d'].rolling(20).std()
            
            # Technical indicators
            if 'RSI' in df.columns:
                features['rsi'] = df['RSI']
            else:
                features['rsi'] = ta.momentum.rsi(df['Close'], window=14)
            
            # MACD
            if 'MACD' in df.columns:
                features['macd'] = df['MACD']
                features['macd_signal'] = df.get('MACD_Signal', features['macd'].ewm(span=9).mean())
            else:
                features['macd'] = ta.trend.macd(df['Close'])
                features['macd_signal'] = ta.trend.macd_signal(df['Close'])
            
            features['macd_histogram'] = features['macd'] - features['macd_signal']
            
            # Bollinger Bands
            if all(col in df.columns for col in ['BB_Low', 'BB_High']):
                bb_mid = (df['BB_Low'] + df['BB_High']) / 2
                features['bb_position'] = (df['Close'] - df['BB_Low']) / (df['BB_High'] - df['BB_Low'])
                features['bb_squeeze'] = (df['BB_High'] - df['BB_Low']) / bb_mid
            else:
                sma_20 = df['Close'].rolling(20).mean()
                bb_std = df['Close'].rolling(20).std()
                bb_upper = sma_20 + 2 * bb_std
                bb_lower = sma_20 - 2 * bb_std
                features['bb_position'] = (df['Close'] - bb_lower) / (bb_upper - bb_lower)
                features['bb_squeeze'] = (bb_upper - bb_lower) / sma_20
            
            # Moving averages and crossovers
            for period in [5, 10, 20, 50]:
                ma_col = f'SMA_{period}'
                if ma_col in df.columns:
                    features[f'ma_{period}'] = df[ma_col]
                else:
                    features[f'ma_{period}'] = df['Close'].rolling(period).mean()
                
                features[f'ma_ratio_{period}'] = df['Close'] / features[f'ma_{period}']
            
            # MA crossover signals
            features['ma_cross_5_20'] = np.where(features['ma_5'] > features['ma_20'], 1, -1)
            features['ma_cross_10_50'] = np.where(features['ma_10'] > features['ma_50'], 1, -1)
            
            # Volume features
            if 'Volume' in df.columns:
                features['volume'] = df['Volume']
                features['volume_ma'] = df['Volume'].rolling(20).mean()
                features['volume_ratio'] = features['volume'] / features['volume_ma']
                
                # Volume-price trend
                features['vpt'] = (features['return_1d'] * features['volume']).cumsum()
                features['vpt_ma'] = features['vpt'].rolling(20).mean()
            else:
                features['volume_ratio'] = 1.0
            
            # Advanced momentum indicators
            # Stochastic Oscillator
            try:
                features['stoch_k'] = ta.momentum.stoch(features['high'], features['low'], 
                                                      features['close'], window=14)
                features['stoch_d'] = features['stoch_k'].rolling(3).mean()
            except:
                # Fallback calculation
                lowest_low = features['low'].rolling(14).min()
                highest_high = features['high'].rolling(14).max()
                features['stoch_k'] = ((features['close'] - lowest_low) / 
                                     (highest_high - lowest_low)) * 100
                features['stoch_d'] = features['stoch_k'].rolling(3).mean()
            
            # Williams %R
            high_14 = features['high'].rolling(14).max()
            low_14 = features['low'].rolling(14).min()
            features['williams_r'] = ((high_14 - features['close']) / 
                                    (high_14 - low_14)) * -100
            
            # Average True Range
            try:
                features['atr'] = ta.volatility.average_true_range(features['high'], 
                                                                  features['low'], 
                                                                  features['close'])
            except:
                # Fallback ATR
                hl = features['high'] - features['low']
                hc = abs(features['high'] - features['close'].shift(1))
                lc = abs(features['low'] - features['close'].shift(1))
                features['atr'] = pd.concat([hl, hc, lc], axis=1).max(axis=1).rolling(14).mean()
            
            # Market regime features
            features['trend_strength'] = abs(features['ma_ratio_20'] - 1)
            features['volatility_regime'] = features['volatility_20d'] / features['volatility_20d'].rolling(60).mean()
            
            # Price patterns
            features['higher_high'] = ((features['high'] > features['high'].shift(1)) & 
                                     (features['high'].shift(1) > features['high'].shift(2))).astype(int)
            features['lower_low'] = ((features['low'] < features['low'].shift(1)) & 
                                   (features['low'].shift(1) < features['low'].shift(2))).astype(int)
            
            # Clean features
            features = features.replace([np.inf, -np.inf], np.nan)
            features = features.fillna(method='ffill').fillna(0)
            
            # Limit features for performance
            if len(features.columns) > self.params['max_features']:
                # Select most relevant features based on correlation with returns
                returns = features['return_1d'].shift(-1)  # Next day return
                correlations = abs(features.corrwith(returns)).sort_values(ascending=False)
                top_features = correlations.head(self.params['max_features']).index.tolist()
                features = features[top_features]
            
            print(f"LSTM Strategy: Prepared {features.shape[1]} features")
            return features
            
        except Exception as e:
            print(f"LSTM Strategy: Feature preparation error: {str(e)}")
            return None
    
    def _should_retrain(self, current_date) -> bool:
        """
        Determine if the model should be retrained.
        """
        if self.model is None or self.last_training_date is None:
            return True
        
        days_since_training = (current_date - self.last_training_date).days
        return days_since_training >= self.params['retrain_frequency']
    
    def _train_model(self, features_df: pd.DataFrame, price_data: pd.DataFrame):
        """
        Train the LSTM model with proper sequence preparation.
        """
        try:
            # Prepare training data
            X, y = self._prepare_sequences(features_df, price_data)
            
            if X is None or len(X) < 50:
                print("LSTM Strategy: Insufficient data for training")
                return
            
            # Split data
            split_idx = int(len(X) * (1 - self.params['validation_split']))
            X_train, X_val = X[:split_idx], X[split_idx:]
            y_train, y_val = y[:split_idx], y[split_idx:]
            
            # Scale features
            self.feature_scaler = StandardScaler()
            n_samples, n_timesteps, n_features = X_train.shape
            X_train_scaled = self.feature_scaler.fit_transform(
                X_train.reshape(-1, n_features)
            ).reshape(n_samples, n_timesteps, n_features)
            
            X_val_scaled = self.feature_scaler.transform(
                X_val.reshape(-1, n_features)
            ).reshape(X_val.shape[0], n_timesteps, n_features)
            
            # Build LSTM model
            self.model = self._build_lstm_model(n_features)
            
            # Training callbacks
            callbacks = [
                EarlyStopping(monitor='val_loss', patience=self.params['early_stopping_patience'], 
                            restore_best_weights=True),
                ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6)
            ]
            
            # Train model
            history = self.model.fit(
                X_train_scaled, y_train,
                validation_data=(X_val_scaled, y_val),
                epochs=self.params['epochs'],
                batch_size=self.params['batch_size'],
                callbacks=callbacks,
                verbose=0
            )
            
            # Initialize SHAP explainer
            if self.params['use_shap']:
                bg = X_train_scaled[:min(100, len(X_train_scaled))]  # Use subset for efficiency
                self._background_data = bg
                self._initialize_shap_explainer(bg)  # may choose Deep or Kernel
            
            self.last_training_date = features_df.index[-1]
            print(f"LSTM Strategy: Model trained successfully. Final val_loss: {min(history.history['val_loss']):.4f}")
        except Exception as e:
            print(f"LSTM Strategy: Training error: {str(e)}")
            import traceback
            traceback.print_exc()
    
    def _prepare_sequences(self, features_df: pd.DataFrame, price_data: pd.DataFrame):
        """
        Prepare sequences for LSTM training.
        """
        try:
            # Create target variable (future price movement)
            horizon = self.params['prediction_horizon']
            future_returns = price_data['Close'].pct_change(horizon).shift(-horizon)
            # Align features and targets
            common_index = features_df.index.intersection(future_returns.index)
            fr = future_returns.loc[common_index]
            features_aligned = features_df.loc[common_index]
            # Labeling
            mode = self.params.get('labeling_mode', 'quantile')
            y = pd.Series(0, index=fr.index)
            if mode == 'quantile':
                pos_q = float(self.params.get('labeling_pos_q', 0.6))
                neg_q = float(self.params.get('labeling_neg_q', 0.4))
                pos_thr = fr.quantile(pos_q)
                neg_thr = fr.quantile(neg_q)
                y[fr > pos_thr] = 1
                y[fr < neg_thr] = -1
            else:
                # fixed absolute threshold
                thr = float(self.params.get('return_threshold', 0.02))
                y[fr > thr] = 1
                y[fr < -thr] = -1

            # Create sequences
            lookback = self.params['lookback_window']
            X_list, y_list = [], []
            for i in range(lookback, len(features_aligned)):
                if not pd.isna(y.iloc[i]):
                    X_list.append(features_aligned.iloc[i - lookback:i].values)
                    y_list.append(y.iloc[i])

            if len(X_list) == 0:
                return None, None

            X = np.array(X_list)
            y_arr = np.array(y_list)

            # Convert to categorical for multi-class classification
            y_categorical = tf.keras.utils.to_categorical(y_arr + 1, num_classes=3)  # Shift to 0,1,2

            print(f"LSTM Strategy: Prepared {len(X)} sequences with shape {X.shape}")
            return X, y_categorical

        except Exception as e:
            print(f"LSTM Strategy: Sequence preparation error: {str(e)}")
            return None, None
    
    def _build_lstm_model(self, n_features: int):
        """
        Build and compile LSTM model architecture.
        """
        model = Sequential()
        
        # First LSTM layer
        model.add(LSTM(
            self.params['lstm_units'][0],
            return_sequences=True,
            input_shape=(self.params['lookback_window'], n_features)
        ))
        model.add(BatchNormalization())
        model.add(Dropout(self.params['dropout_rate']))
        
        # Second LSTM layer
        if len(self.params['lstm_units']) > 1:
            model.add(LSTM(self.params['lstm_units'][1], return_sequences=False))
            model.add(BatchNormalization())
            model.add(Dropout(self.params['dropout_rate']))
        
        # Dense layers
        model.add(Dense(32, activation='relu'))
        model.add(Dropout(self.params['dropout_rate']))
        model.add(Dense(16, activation='relu'))
        
        # Output layer (3 classes: down, neutral, up)
        model.add(Dense(3, activation='softmax'))
        
        # Compile model
        model.compile(
            optimizer=Adam(learning_rate=self.params['learning_rate']),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def _initialize_shap_explainer(self, background_data: np.ndarray, prefer_kernel: bool = False):
        """
        Initialize SHAP explainer for model interpretability.
        Lazy import SHAP; prefer DeepExplainer, fallback to KernelExplainer to avoid TF gradient issues.
        If prefer_kernel is True, skip Deep and go straight to Kernel.
        """
        if not self.params.get('use_shap', False):
            print("LSTM Strategy: SHAP disabled by params")
            return
        try:
            import importlib
            shap = importlib.import_module('shap')
            self._shap_module = shap
        except Exception as e:
            print(f"LSTM Strategy: SHAP import failed ({e}), disabling explanations")
            self.params['use_shap'] = False
            self.shap_explainer = None
            self._shap_mode = None
            return

        self._background_data = background_data  # persist for runtime fallback

        if not prefer_kernel:
            # Try DeepExplainer first
            try:
                self.shap_explainer = shap.DeepExplainer(self.model, background_data)
                self._shap_mode = 'deep'
                self._kernel_input_shape = None
                print("LSTM Strategy: SHAP DeepExplainer initialized")
                return
            except Exception as e:
                print(f"LSTM Strategy: SHAP DeepExplainer init failed, falling back to KernelExplainer: {e}")

        # Fallback to KernelExplainer with flattened sequences
        try:
            lookback = background_data.shape[1]
            n_features = background_data.shape[2]
            self._kernel_input_shape = (lookback, n_features)
            bg_flat = background_data.reshape(background_data.shape[0], -1)
            def predict_fn(x_flat):
                x = x_flat.reshape((-1, lookback, n_features))
                return self.model.predict(x, verbose=0)
            self.shap_explainer = self._shap_module.KernelExplainer(predict_fn, bg_flat)
            self._shap_mode = 'kernel'
            print("LSTM Strategy: SHAP KernelExplainer initialized")
        except Exception as e2:
            print(f"LSTM Strategy: SHAP initialization error: {str(e2)}")
            self.shap_explainer = None
            self.params['use_shap'] = False
            self._shap_mode = None