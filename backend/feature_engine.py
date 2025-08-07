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
        
        # Ensure we have required columns
        if 'Close' not in df.columns:
            raise ValueError("Close column is required")
        
        # Fill missing OHLC columns if needed
        for col in ['Open', 'High', 'Low']:
            if col not in df.columns:
                df[col] = df['Close']
        
        if 'Volume' not in df.columns:
            df['Volume'] = 1000000
        
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
        
        return df
    
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
            sma_20 = df['Close'].rolling(20).mean()
            std_20 = df['Close'].rolling(20).std()
            df['BB_High'] = sma_20 + 2 * std_20
            df['BB_Low'] = sma_20 - 2 * std_20
            df['BB_Mid'] = sma_20
        
        # BB position
        bb_range = df['BB_High'] - df['BB_Low']
        df['BB_Position'] = np.where(bb_range > 0, 
                                   (df['Close'] - df['BB_Low']) / bb_range, 
                                   0.5)
        df['BB_Width'] = bb_range / df['BB_Mid']
    
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
        df['Volume_MA_20'] = df['Volume'].rolling(20).mean()
        df['Volume_Ratio'] = df['Volume'] / df['Volume_MA_20']
        
        # On Balance Volume
        obv = (np.sign(df['Close'].diff()) * df['Volume']).fillna(0).cumsum()
        df['OBV'] = obv
        df['OBV_MA'] = obv.rolling(20).mean()
        
        # Volume Price Trend
        vpt = ((df['Close'].diff() / df['Close'].shift(1)) * df['Volume']).fillna(0).cumsum()
        df['VPT'] = vpt
        
        # Volume Weighted Average Price (approximation)
        typical_price = (df['High'] + df['Low'] + df['Close']) / 3
        vwap = (typical_price * df['Volume']).rolling(20).sum() / df['Volume'].rolling(20).sum()
        df['VWAP'] = vwap
        df['VWAP_Ratio'] = df['Close'] / vwap
    
    def _add_price_features(self, df: pd.DataFrame):
        """Add price-based features."""
        # Returns
        df['Return_1d'] = df['Close'].pct_change()
        df['Return_5d'] = df['Close'].pct_change(5)
        df['Return_20d'] = df['Close'].pct_change(20)
        
        # Price position in recent range
        for period in [10, 20, 50]:
            high_period = df['High'].rolling(period).max()
            low_period = df['Low'].rolling(period).min()
            df[f'Price_Position_{period}d'] = (df['Close'] - low_period) / (high_period - low_period)
        
        # Gap analysis
        df['Gap'] = (df['Open'] - df['Close'].shift(1)) / df['Close'].shift(1)
        df['Gap_Filled'] = (
            ((df['Gap'] > 0) & (df['Low'] <= df['Close'].shift(1))) |
            ((df['Gap'] < 0) & (df['High'] >= df['Close'].shift(1)))
        ).astype(int)
        
        # Intraday features
        df['High_Low_Ratio'] = df['High'] / df['Low']
        df['Open_Close_Ratio'] = df['Open'] / df['Close']
        df['Intraday_Range'] = (df['High'] - df['Low']) / df['Close']
    
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
            df['Stochastic_D'] = df['Stochastic_K'].rolling(3).mean()
        except:
            # Fallback calculation
            lowest_low = df['Low'].rolling(14).min()
            highest_high = df['High'].rolling(14).max()
            df['Stochastic_K'] = ((df['Close'] - lowest_low) / (highest_high - lowest_low)) * 100
            df['Stochastic_D'] = df['Stochastic_K'].rolling(3).mean()
        
        # Williams %R
        high_14 = df['High'].rolling(14).max()
        low_14 = df['Low'].rolling(14).min()
        df['Williams_R'] = ((high_14 - df['Close']) / (high_14 - low_14)) * -100
        
        # Rate of Change
        for period in [5, 10, 20]:
            df[f'ROC_{period}'] = df['Close'].pct_change(period) * 100
        
        # Commodity Channel Index (approximation)
        typical_price = (df['High'] + df['Low'] + df['Close']) / 3
        sma_tp = typical_price.rolling(20).mean()
        mad = typical_price.rolling(20).apply(lambda x: np.mean(np.abs(x - x.mean())))
        df['CCI'] = (typical_price - sma_tp) / (0.015 * mad)
        
        # Money Flow Index (approximation)
        typical_price_mfi = (df['High'] + df['Low'] + df['Close']) / 3
        money_flow = typical_price_mfi * df['Volume']
        
        positive_flow = money_flow.where(typical_price_mfi > typical_price_mfi.shift(1), 0)
        negative_flow = money_flow.where(typical_price_mfi < typical_price_mfi.shift(1), 0)
        
        positive_mf = positive_flow.rolling(14).sum()
        negative_mf = negative_flow.rolling(14).sum()
        
        mfr = positive_mf / negative_mf
        df['MFI'] = 100 - (100 / (1 + mfr))
    
    def _clean_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and finalize features."""
        # Replace infinite values with NaN
        df = df.replace([np.inf, -np.inf], np.nan)
        
        # Forward fill NaN values
        df = df.fillna(method='ffill')
        
        # Backward fill any remaining NaN values at the beginning
        df = df.fillna(method='bfill')
        
        # Fill any remaining NaN values with 0
        df = df.fillna(0)
        
        # Ensure all numeric columns are float64
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        df[numeric_columns] = df[numeric_columns].astype(np.float64)
        
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
            'price': ['Close', 'Open', 'High', 'Low', 'Return_1d', 'Return_5d', 'Return_20d'],
            'volume': ['Volume', 'Volume_Ratio', 'OBV', 'VPT', 'VWAP', 'MFI'],
            'momentum': ['RSI', 'MACD', 'MACD_Signal', 'Stochastic_K', 'Stochastic_D', 'Williams_R', 'ROC_5', 'ROC_10', 'ROC_20', 'CCI'],
            'volatility': ['ATR', 'ATR_Pct', 'Volatility_5', 'Volatility_10', 'Volatility_20', 'GK_Volatility', 'BB_Width'],
            'moving_averages': ['SMA_5', 'SMA_10', 'SMA_20', 'SMA_50', 'EMA_5', 'EMA_10', 'EMA_20', 'EMA_50'],
            'bollinger': ['BB_High', 'BB_Low', 'BB_Mid', 'BB_Position']
        }
        
        if category == 'all':
            all_features = []
            for feat_list in features.values():
                all_features.extend(feat_list)
            return all_features
        
        return features.get(category, [])
