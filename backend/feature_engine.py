import pandas as pd
import numpy as np
import ta
from typing import Dict, Any
import warnings

warnings.filterwarnings('ignore')

class FeatureEngine:
    """
    Feature engineering class for technical indicators and market features.
    """
    
    def __init__(self):
        """Initialize FeatureEngine."""
        pass
    
    def calculate_technical_indicators(self, data: pd.DataFrame, 
                                     strategy_params: Dict[str, Any] = None) -> pd.DataFrame:
        """
        Calculate comprehensive technical indicators.
        
        Args:
            data: Market data with OHLCV columns
            strategy_params: Strategy-specific parameters
            
        Returns:
            pd.DataFrame: Data with technical indicators
        """
        if strategy_params is None:
            strategy_params = {}
        
        df = data.copy()
        
        # Validate minimum data requirements
        if len(df) < 2:
            raise ValueError("At least 2 data points are required")
        
        # Ensure we have required columns
        if 'Close' not in df.columns:
            raise ValueError("Close column is required")
        
        # Fill missing OHLC columns if needed
        for col in ['Open', 'High', 'Low']:
            if col not in df.columns:
                df[col] = df['Close']
        
        if 'Volume' not in df.columns:
            df['Volume'] = 1000000  # Default volume
        
        # Ensure all price columns are positive
        price_cols = ['Open', 'High', 'Low', 'Close']
        for col in price_cols:
            df[col] = df[col].abs()
            df[col] = df[col].replace(0, df[col].mean())  # Replace zeros with mean
        
        try:
            # Moving Averages
            self._add_moving_averages(df, strategy_params)
            
            # RSI
            self._add_rsi(df)
            
            # MACD
            self._add_macd(df)
            
            # Bollinger Bands
            self._add_bollinger_bands(df)
            
            # ATR
            self._add_atr(df)
            
            # Volume indicators
            self._add_volume_indicators(df)
            
            # Price-based features
            self._add_price_features(df)
            
            # Volatility indicators
            self._add_volatility_indicators(df)
            
            # Additional momentum indicators
            self._add_momentum_indicators(df)
            
            # Clean the data
            df = self._clean_features(df)
            
        except Exception as e:
            print(f"Warning: Error calculating some indicators: {str(e)}")
            # Continue with basic indicators if advanced ones fail
            df = self._add_basic_indicators(df)
            df = self._clean_features(df)
        
        return df

    def _add_basic_indicators(self, df: pd.DataFrame):
        """Add basic indicators as fallback."""
        # Basic moving averages
        for period in [10, 20, 50]:
            if period <= len(df):
                df[f'SMA_{period}'] = df['Close'].rolling(window=period, min_periods=1).mean()
        
        # Basic RSI
        delta = df['Close'].diff()
        gain = delta.where(delta > 0, 0).rolling(window=14, min_periods=1).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14, min_periods=1).mean()
        rs = gain / (loss + 1e-10)  # Avoid division by zero
        df['RSI'] = 100 - (100 / (1 + rs))

    def _add_moving_averages(self, df: pd.DataFrame, strategy_params: Dict[str, Any]):
        """Add moving averages based on strategy parameters."""
        # Default periods
        periods = [5, 10, 20, 50, 100, 200]
        
        # Add strategy-specific periods
        if 'fast_ma' in strategy_params:
            periods.append(strategy_params['fast_ma'])
        if 'slow_ma' in strategy_params:
            periods.append(strategy_params['slow_ma'])
        
        # Remove duplicates and sort
        periods = sorted(list(set(periods)))
        
        for period in periods:
            if period <= len(df):
                df[f'SMA_{period}'] = df['Close'].rolling(window=period).mean()
                df[f'EMA_{period}'] = df['Close'].ewm(span=period).mean()
    
    def _add_rsi(self, df: pd.DataFrame):
        """Add RSI indicator."""
        try:
            df['RSI'] = ta.momentum.rsi(df['Close'], window=14)
        except:
            # Fallback RSI calculation
            delta = df['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            df['RSI'] = 100 - (100 / (1 + rs))
    
    def _add_macd(self, df: pd.DataFrame):
        """Add MACD indicator."""
        try:
            df['MACD'] = ta.trend.macd(df['Close'])
            df['MACD_Signal'] = ta.trend.macd_signal(df['Close'])
            df['MACD_Histogram'] = ta.trend.macd_diff(df['Close'])
        except:
            # Fallback MACD calculation
            ema_12 = df['Close'].ewm(span=12).mean()
            ema_26 = df['Close'].ewm(span=26).mean()
            df['MACD'] = ema_12 - ema_26
            df['MACD_Signal'] = df['MACD'].ewm(span=9).mean()
            df['MACD_Histogram'] = df['MACD'] - df['MACD_Signal']
    
    def _add_bollinger_bands(self, df: pd.DataFrame):
        """Add Bollinger Bands."""
        try:
            df['BB_High'] = ta.volatility.bollinger_hband(df['Close'])
            df['BB_Low'] = ta.volatility.bollinger_lband(df['Close'])
            df['BB_Mid'] = ta.volatility.bollinger_mavg(df['Close'])
        except:
            # Fallback calculation
            sma_20 = df['Close'].rolling(20, min_periods=1).mean()
            std_20 = df['Close'].rolling(20, min_periods=1).std()
            df['BB_High'] = sma_20 + 2 * std_20
            df['BB_Low'] = sma_20 - 2 * std_20
            df['BB_Mid'] = sma_20
        
        # BB position with division by zero protection
        bb_range = df['BB_High'] - df['BB_Low']
        df['BB_Position'] = np.where(
            bb_range > 1e-10, 
            (df['Close'] - df['BB_Low']) / bb_range, 
            0.5
        )
        df['BB_Width'] = np.where(
            df['BB_Mid'] > 1e-10,
            bb_range / df['BB_Mid'],
            0
        )

    def _add_atr(self, df: pd.DataFrame):
        """Add Average True Range."""
        try:
            df['ATR'] = ta.volatility.average_true_range(df['High'], df['Low'], df['Close'])
        except:
            # Fallback ATR calculation
            high_low = df['High'] - df['Low']
            high_close = abs(df['High'] - df['Close'].shift(1))
            low_close = abs(df['Low'] - df['Close'].shift(1))
            true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
            df['ATR'] = true_range.rolling(14).mean()
        
        # ATR as percentage of price
        df['ATR_Pct'] = df['ATR'] / df['Close'] * 100
    
    def _add_volume_indicators(self, df: pd.DataFrame):
        """Add volume-based indicators."""
        # Volume moving averages
        df['Volume_MA_20'] = df['Volume'].rolling(20, min_periods=1).mean()
        df['Volume_Ratio'] = np.where(
            df['Volume_MA_20'] > 0,
            df['Volume'] / df['Volume_MA_20'],
            1.0
        )
        
        # On Balance Volume
        price_change = np.sign(df['Close'].diff().fillna(0))
        obv = (price_change * df['Volume']).cumsum()
        df['OBV'] = obv
        df['OBV_MA'] = obv.rolling(20, min_periods=1).mean()
        
        # Volume Price Trend with protection
        close_change = df['Close'].diff() / (df['Close'].shift(1) + 1e-10)
        vpt = (close_change * df['Volume']).fillna(0).cumsum()
        df['VPT'] = vpt
        
        # Volume Weighted Average Price (approximation)
        typical_price = (df['High'] + df['Low'] + df['Close']) / 3
        volume_sum = df['Volume'].rolling(20, min_periods=1).sum()
        vwap = (typical_price * df['Volume']).rolling(20, min_periods=1).sum() / (volume_sum + 1e-10)
        df['VWAP'] = vwap
        df['VWAP_Ratio'] = df['Close'] / (vwap + 1e-10)
    
    def _add_price_features(self, df: pd.DataFrame):
        """Add price-based features."""
        # Returns
        df['Return_1d'] = df['Close'].pct_change().fillna(0)
        df['Return_5d'] = df['Close'].pct_change(5).fillna(0)
        df['Return_20d'] = df['Close'].pct_change(20).fillna(0)
        
        # Price position in recent range
        for period in [10, 20, 50]:
            if period <= len(df):
                high_period = df['High'].rolling(period, min_periods=1).max()
                low_period = df['Low'].rolling(period, min_periods=1).min()
                price_range = high_period - low_period
                df[f'Price_Position_{period}d'] = np.where(
                    price_range > 1e-10,
                    (df['Close'] - low_period) / price_range,
                    0.5
                )
        
        # Gap analysis with protection
        prev_close = df['Close'].shift(1)
        df['Gap'] = np.where(
            prev_close > 1e-10,
            (df['Open'] - prev_close) / prev_close,
            0
        )
        df['Gap_Filled'] = (
            ((df['Gap'] > 0) & (df['Low'] <= prev_close)) |
            ((df['Gap'] < 0) & (df['High'] >= prev_close))
        ).astype(int)
        
        # Intraday features with protection
        df['High_Low_Ratio'] = df['High'] / (df['Low'] + 1e-10)
        df['Open_Close_Ratio'] = df['Open'] / (df['Close'] + 1e-10)
        df['Intraday_Range'] = (df['High'] - df['Low']) / (df['Close'] + 1e-10)

    def _add_volatility_indicators(self, df: pd.DataFrame):
        """Add volatility indicators."""
        # Historical volatility
        for period in [5, 10, 20, 50]:
            returns = df['Close'].pct_change()
            df[f'Volatility_{period}'] = returns.rolling(period).std() * np.sqrt(252)
        
        # Garman-Klass volatility
        df['GK_Volatility'] = np.sqrt(
            ((np.log(df['High']) - np.log(df['Low']))**2)/2 - 
            (2*np.log(2) - 1)*((np.log(df['Close']) - np.log(df['Open']))**2)
        )
        
        # True Range as percentage
        high_low = df['High'] - df['Low']
        high_close = abs(df['High'] - df['Close'].shift(1))
        low_close = abs(df['Low'] - df['Close'].shift(1))
        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        df['True_Range_Pct'] = true_range / df['Close'] * 100
    
    def _add_momentum_indicators(self, df: pd.DataFrame):
        """Add additional momentum indicators."""
        # Stochastic Oscillator
        try:
            df['Stochastic_K'] = ta.momentum.stoch(df['High'], df['Low'], df['Close'])
            df['Stochastic_D'] = df['Stochastic_K'].rolling(3, min_periods=1).mean()
        except:
            # Fallback calculation
            lowest_low = df['Low'].rolling(14, min_periods=1).min()
            highest_high = df['High'].rolling(14, min_periods=1).max()
            stoch_range = highest_high - lowest_low
            df['Stochastic_K'] = np.where(
                stoch_range > 1e-10,
                ((df['Close'] - lowest_low) / stoch_range) * 100,
                50
            )
            df['Stochastic_D'] = df['Stochastic_K'].rolling(3, min_periods=1).mean()
        
        # Williams %R
        high_14 = df['High'].rolling(14, min_periods=1).max()
        low_14 = df['Low'].rolling(14, min_periods=1).min()
        williams_range = high_14 - low_14
        df['Williams_R'] = np.where(
            williams_range > 1e-10,
            ((high_14 - df['Close']) / williams_range) * -100,
            -50
        )
        
        # Rate of Change
        for period in [5, 10, 20]:
            if period <= len(df):
                df[f'ROC_{period}'] = df['Close'].pct_change(period).fillna(0) * 100
        
        # Commodity Channel Index (approximation)
        typical_price = (df['High'] + df['Low'] + df['Close']) / 3
        sma_tp = typical_price.rolling(20, min_periods=1).mean()
        mad = typical_price.rolling(20, min_periods=1).apply(
            lambda x: np.mean(np.abs(x - x.mean())) if len(x) > 0 else 0
        )
        df['CCI'] = np.where(
            mad > 1e-10,
            (typical_price - sma_tp) / (0.015 * mad),
            0
        )
        
        # Money Flow Index (approximation)
        typical_price_mfi = (df['High'] + df['Low'] + df['Close']) / 3
        money_flow = typical_price_mfi * df['Volume']
        
        price_change_mfi = typical_price_mfi.diff()
        positive_flow = money_flow.where(price_change_mfi > 0, 0)
        negative_flow = money_flow.where(price_change_mfi < 0, 0)
        
        positive_mf = positive_flow.rolling(14, min_periods=1).sum()
        negative_mf = negative_flow.rolling(14, min_periods=1).sum()
        
        mfr = positive_mf / (negative_mf + 1e-10)
        df['MFI'] = 100 - (100 / (1 + mfr))

    def _clean_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and finalize features."""
        # Replace infinite values with NaN
        df = df.replace([np.inf, -np.inf], np.nan)
        
        # Use modern pandas fillna methods
        df = df.ffill()  # Forward fill
        df = df.bfill()  # Backward fill
        
        # Fill any remaining NaN values with reasonable defaults
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        for col in numeric_columns:
            if df[col].isna().any():
                if 'ratio' in col.lower() or 'position' in col.lower():
                    df[col] = df[col].fillna(1.0)
                elif 'rsi' in col.lower() or 'stochastic' in col.lower():
                    df[col] = df[col].fillna(50.0)
                else:
                    df[col] = df[col].fillna(0.0)
        
        # Ensure all numeric columns are float64
        df[numeric_columns] = df[numeric_columns].astype(np.float64)
        
        # Clip extreme values to reasonable ranges
        df['RSI'] = df['RSI'].clip(0, 100)
        df['Stochastic_K'] = df['Stochastic_K'].clip(0, 100)
        df['Stochastic_D'] = df['Stochastic_D'].clip(0, 100)
        df['Williams_R'] = df['Williams_R'].clip(-100, 0)
        df['MFI'] = df['MFI'].clip(0, 100)
        df['BB_Position'] = df['BB_Position'].clip(0, 1)
        
        return df

    def get_feature_names(self, category: str = 'all') -> list:
        """
        Get list of feature names by category.
        
        Args:
            category: Feature category ('all', 'price', 'volume', 'momentum', 'volatility')
            
        Returns:
            list: List of feature names
        """
        features = {
            'price': [
                'Close', 'Open', 'High', 'Low', 'Return_1d', 'Return_5d', 'Return_20d',
                'Price_Position_10d', 'Price_Position_20d', 'Price_Position_50d',
                'Gap', 'Gap_Filled', 'High_Low_Ratio', 'Open_Close_Ratio', 'Intraday_Range'
            ],
            'volume': [
                'Volume', 'Volume_MA_20', 'Volume_Ratio', 'OBV', 'OBV_MA', 'VPT', 
                'VWAP', 'VWAP_Ratio', 'MFI'
            ],
            'momentum': [
                'RSI', 'MACD', 'MACD_Signal', 'MACD_Histogram', 'Stochastic_K', 'Stochastic_D', 
                'Williams_R', 'ROC_5', 'ROC_10', 'ROC_20', 'CCI'
            ],
            'volatility': [
                'ATR', 'ATR_Pct', 'Volatility_5', 'Volatility_10', 'Volatility_20', 'Volatility_50',
                'GK_Volatility', 'True_Range_Pct', 'BB_Width'
            ],
            'moving_averages': [
                'SMA_5', 'SMA_10', 'SMA_20', 'SMA_50', 'SMA_100', 'SMA_200',
                'EMA_5', 'EMA_10', 'EMA_20', 'EMA_50', 'EMA_100', 'EMA_200'
            ],
            'bollinger': ['BB_High', 'BB_Low', 'BB_Mid', 'BB_Position', 'BB_Width']
        }
        
        if category == 'all':
            all_features = []
            for feat_list in features.values():
                all_features.extend(feat_list)
            return sorted(list(set(all_features)))  # Remove duplicates and sort
        
        return features.get(category, [])
