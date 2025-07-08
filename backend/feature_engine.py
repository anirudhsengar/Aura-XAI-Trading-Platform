import pandas as pd
import numpy as np
import ta
import warnings
import re
from backend.utils import DataValidator
from textblob import TextBlob
from pathlib import Path

warnings.filterwarnings('ignore')

class FeatureEngine:
    """
    Feature Engine for calculating technical indicators and sentiment analysis.
    """
    
    def __init__(self):
        """
        Initialize FeatureEngine.
        """        
        # Initialize sentiment analysis models
        self._initialize_sentiment_models()
    
    def calculate_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate comprehensive technical indicators for market data.
        
        Args:
            df: DataFrame with OHLCV data            
        Returns:
            pd.DataFrame: DataFrame with technical indicators added
        """
        # Validate input data
        if not DataValidator.validate_ohlcv_data(df):
            raise ValueError("Invalid OHLCV data provided")
        
        # Create a copy to avoid modifying original data
        result_df = df.copy()
        
        # 1. TREND INDICATORS
        # Moving Averages
        result_df['SMA_10'] = ta.trend.sma_indicator(df['Close'], window=10)
        result_df['SMA_20'] = ta.trend.sma_indicator(df['Close'], window=20)
        result_df['SMA_50'] = ta.trend.sma_indicator(df['Close'], window=50)
        result_df['SMA_200'] = ta.trend.sma_indicator(df['Close'], window=200)
        
        result_df['EMA_10'] = ta.trend.ema_indicator(df['Close'], window=10)
        result_df['EMA_20'] = ta.trend.ema_indicator(df['Close'], window=20)
        result_df['EMA_50'] = ta.trend.ema_indicator(df['Close'], window=50)
        
        # MACD
        result_df['MACD'] = ta.trend.macd(df['Close'])
        result_df['MACD_Signal'] = ta.trend.macd_signal(df['Close'])
        result_df['MACD_Histogram'] = ta.trend.macd_diff(df['Close'])
        
        # Bollinger Bands
        result_df['BB_High'] = ta.volatility.bollinger_hband(df['Close'])
        result_df['BB_Low'] = ta.volatility.bollinger_lband(df['Close'])
        result_df['BB_Middle'] = ta.volatility.bollinger_mavg(df['Close'])
        result_df['BB_Width'] = result_df['BB_High'] - result_df['BB_Low']
        result_df['BB_Position'] = (df['Close'] - result_df['BB_Low']) / result_df['BB_Width']
        
        # 2. MOMENTUM INDICATORS
        # RSI
        result_df['RSI'] = ta.momentum.rsi(df['Close'], window=14)
        result_df['RSI_Oversold'] = (result_df['RSI'] < 30).astype(int)
        result_df['RSI_Overbought'] = (result_df['RSI'] > 70).astype(int)
        
        # Stochastic Oscillator
        result_df['Stoch_K'] = ta.momentum.stoch(df['High'], df['Low'], df['Close'])
        result_df['Stoch_D'] = ta.momentum.stoch_signal(df['High'], df['Low'], df['Close'])
        
        # Williams %R
        result_df['Williams_R'] = ta.momentum.williams_r(df['High'], df['Low'], df['Close'])
        
        # 3. VOLATILITY INDICATORS
        # Average True Range
        result_df['ATR'] = ta.volatility.average_true_range(df['High'], df['Low'], df['Close'])
        
        # Volatility (rolling standard deviation)
        result_df['Volatility_10'] = df['Close'].rolling(window=10).std()
        result_df['Volatility_20'] = df['Close'].rolling(window=20).std()
        
        # 4. VOLUME INDICATORS
        # On-Balance Volume
        result_df['OBV'] = ta.volume.on_balance_volume(df['Close'], df['Volume'])
        
        # Volume SMA
        result_df['Volume_SMA_10'] = ta.trend.sma_indicator(df['Volume'], window=10)
        result_df['Volume_SMA_20'] = ta.trend.sma_indicator(df['Volume'], window=20)
        
        # Volume Ratio
        result_df['Volume_Ratio'] = df['Volume'] / result_df['Volume_SMA_20']
        
        # 5. PRICE-BASED FEATURES
        # Price changes
        result_df['Price_Change'] = df['Close'].pct_change()
        result_df['Price_Change_5d'] = df['Close'].pct_change(periods=5)
        result_df['Price_Change_10d'] = df['Close'].pct_change(periods=10)
        
        # High-Low spread
        result_df['HL_Spread'] = (df['High'] - df['Low']) / df['Close']
        
        # Gap analysis
        result_df['Gap'] = (df['Open'] - df['Close'].shift(1)) / df['Close'].shift(1)
        
        # 6. CUSTOM INDICATORS
        # Trend strength
        result_df['Trend_Strength'] = self._calculate_trend_strength(df)
        
        # Support/Resistance levels
        result_df['Support_Level'] = self._calculate_support_resistance(df, 'support')
        result_df['Resistance_Level'] = self._calculate_support_resistance(df, 'resistance')
        
        # 7. PATTERN RECOGNITION FLAGS
        # Golden Cross (50-day MA crosses above 200-day MA)
        result_df['Golden_Cross'] = (
            (result_df['SMA_50'] > result_df['SMA_200']) & 
            (result_df['SMA_50'].shift(1) <= result_df['SMA_200'].shift(1))
        ).astype(int)
        
        # Death Cross (50-day MA crosses below 200-day MA)
        result_df['Death_Cross'] = (
            (result_df['SMA_50'] < result_df['SMA_200']) & 
            (result_df['SMA_50'].shift(1) >= result_df['SMA_200'].shift(1))
        ).astype(int)
        
        # MACD Bullish/Bearish crossovers
        result_df['MACD_Bullish_Cross'] = (
            (result_df['MACD'] > result_df['MACD_Signal']) & 
            (result_df['MACD'].shift(1) <= result_df['MACD_Signal'].shift(1))
        ).astype(int)
        
        result_df['MACD_Bearish_Cross'] = (
            (result_df['MACD'] < result_df['MACD_Signal']) & 
            (result_df['MACD'].shift(1) >= result_df['MACD_Signal'].shift(1))
        ).astype(int)
        
        return result_df
    
    def _initialize_sentiment_models(self):
        """Initialize sentiment analysis models."""
        # Placeholder for sentiment model initialization
        pass
    
    def _calculate_trend_strength(self, df: pd.DataFrame) -> pd.Series:
        """
        Calculate trend strength indicator.
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            pd.Series: Trend strength values
        """
        window = 20
        
        # Calculate price changes
        price_changes = df['Close'].diff()
        
        # Count positive and negative changes in rolling window
        positive_changes = (price_changes > 0).rolling(window=window).sum()
        negative_changes = (price_changes < 0).rolling(window=window).sum()
        
        # Calculate trend strength
        trend_strength = (positive_changes - negative_changes) / window
        
        return trend_strength
    
    def _calculate_support_resistance(self, df: pd.DataFrame, level_type: str) -> pd.Series:
        """
        Calculate support and resistance levels.
        
        Args:
            df: DataFrame with OHLCV data
            level_type: 'support' or 'resistance'
            
        Returns:
            pd.Series: Support/resistance levels
        """
        window = 20
        
        if level_type == 'support':
            # Find local minima
            levels = df['Low'].rolling(window=window, center=True).min()
        else:  # resistance
            # Find local maxima
            levels = df['High'].rolling(window=window, center=True).max()
        
        return levels