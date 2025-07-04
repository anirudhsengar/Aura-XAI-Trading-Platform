import pandas as pd
import numpy as np
import ta
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
import warnings
from pathlib import Path
warnings.filterwarnings('ignore')

# NLP and sentiment analysis
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
import torch
from textblob import TextBlob
import re

# Import utilities
from utils import DateUtils, DataValidator, FileUtils, LoggingUtils, MathUtils

class FeatureEngine:
    """
    Feature Engine for calculating technical indicators and sentiment analysis.
    """
    
    def __init__(self, cache_dir: str = "../data", log_level: str = "INFO"):
        """
        Initialize FeatureEngine with caching and logging.
        
        Args:
            cache_dir: Directory for caching computed features
            log_level: Logging level
        """
        self.cache_dir = Path(cache_dir) if isinstance(cache_dir, str) else cache_dir
        self.features_cache_dir = self.cache_dir / "features"
        
        # Ensure directories exist
        FileUtils.ensure_directory_exists(str(self.features_cache_dir))
        
        # Setup logging
        self.logger = LoggingUtils.setup_logger(
            "FeatureEngine", 
            log_file=str(self.cache_dir.parent / "logs" / "feature_engine.log"),
            level=log_level
        )
        
        # Initialize sentiment analysis models
        self._initialize_sentiment_models()
        
        # Configuration
        self.cache_duration_hours = 24
        
        self.logger.info("FeatureEngine initialized successfully")
    
    def _initialize_sentiment_models(self):
        """
        Initialize sentiment analysis models.
        """
        try:
            # Initialize FinBERT for financial sentiment analysis
            self.finbert_tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
            self.finbert_model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")
            
            # Create sentiment pipeline
            self.sentiment_pipeline = pipeline(
                "sentiment-analysis",
                model=self.finbert_model,
                tokenizer=self.finbert_tokenizer,
                device=0 if torch.cuda.is_available() else -1
            )
            
            self.logger.info("FinBERT sentiment model loaded successfully")
            
        except Exception as e:
            self.logger.warning(f"Failed to load FinBERT model: {str(e)}")
            self.logger.info("Falling back to TextBlob for sentiment analysis")
            self.sentiment_pipeline = None
    
    def calculate_technical_indicators(self, df: pd.DataFrame, symbol: str = None) -> pd.DataFrame:
        """
        Calculate comprehensive technical indicators for market data.
        
        Args:
            df: DataFrame with OHLCV data
            symbol: Stock symbol for caching (optional)
            
        Returns:
            pd.DataFrame: DataFrame with technical indicators added
        """
        # Validate input data
        if not DataValidator.validate_ohlcv_data(df):
            raise ValueError("Invalid OHLCV data provided")
        
        # Check cache if symbol is provided
        if symbol:
            cache_file = self.features_cache_dir / f"{symbol}_technical_indicators.parquet"
            if cache_file.exists():
                file_age = datetime.now() - datetime.fromtimestamp(cache_file.stat().st_mtime)
                if file_age.total_seconds() < self.cache_duration_hours * 3600:
                    self.logger.info(f"Loading cached technical indicators for {symbol}")
                    return FileUtils.load_dataframe(str(cache_file), format='parquet')
        
        self.logger.info(f"Calculating technical indicators for {symbol or 'data'}")
        
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
        
        # Cache the results if symbol is provided
        if symbol:
            FileUtils.save_dataframe(result_df, str(cache_file), format='parquet')
            self.logger.info(f"Cached technical indicators for {symbol}")
        
        self.logger.info(f"Calculated {len(result_df.columns) - len(df.columns)} technical indicators")
        
        return result_df
    
    def calculate_sentiment_features(self, news_df: pd.DataFrame, symbol: str = None) -> pd.DataFrame:
        """
        Calculate sentiment features from news data.
        
        Args:
            news_df: DataFrame with news data
            symbol: Stock symbol for caching (optional)
            
        Returns:
            pd.DataFrame: DataFrame with sentiment features by date
        """
        if news_df.empty:
            self.logger.warning("No news data provided for sentiment analysis")
            return pd.DataFrame()
        
        # Check cache if symbol is provided
        if symbol:
            cache_file = self.features_cache_dir / f"{symbol}_sentiment_features.parquet"
            if cache_file.exists():
                file_age = datetime.now() - datetime.fromtimestamp(cache_file.stat().st_mtime)
                if file_age.total_seconds() < self.cache_duration_hours * 3600:
                    self.logger.info(f"Loading cached sentiment features for {symbol}")
                    return FileUtils.load_dataframe(str(cache_file), format='parquet')
        
        self.logger.info(f"Calculating sentiment features for {symbol or 'data'}")
        
        # Create a copy to work with
        sentiment_df = news_df.copy()
        
        # Calculate sentiment scores for headlines
        sentiment_df['headline_sentiment'] = sentiment_df['headline'].apply(
            self._analyze_sentiment
        )
        
        # Calculate sentiment scores for summaries if available
        if 'summary' in sentiment_df.columns:
            sentiment_df['summary_sentiment'] = sentiment_df['summary'].apply(
                self._analyze_sentiment
            )
        else:
            sentiment_df['summary_sentiment'] = sentiment_df['headline_sentiment']
        
        # Combine headline and summary sentiment
        sentiment_df['combined_sentiment'] = (
            sentiment_df['headline_sentiment'] * 0.6 + 
            sentiment_df['summary_sentiment'] * 0.4
        )
        
        # Add sentiment intensity (absolute value)
        sentiment_df['sentiment_intensity'] = abs(sentiment_df['combined_sentiment'])
        
        # Add sentiment category
        sentiment_df['sentiment_category'] = sentiment_df['combined_sentiment'].apply(
            lambda x: 'positive' if x > 0.1 else 'negative' if x < -0.1 else 'neutral'
        )
        
        # Group by date and calculate daily sentiment metrics
        sentiment_df['date'] = pd.to_datetime(sentiment_df['date']).dt.date
        
        daily_sentiment = sentiment_df.groupby('date').agg({
            'combined_sentiment': ['mean', 'std', 'count'],
            'sentiment_intensity': ['mean', 'max'],
            'headline_sentiment': ['mean', 'min', 'max'],
            'summary_sentiment': ['mean', 'min', 'max']
        }).reset_index()
        
        # Flatten column names
        daily_sentiment.columns = [
            'date', 'sentiment_mean', 'sentiment_std', 'news_count',
            'sentiment_intensity_mean', 'sentiment_intensity_max',
            'headline_sentiment_mean', 'headline_sentiment_min', 'headline_sentiment_max',
            'summary_sentiment_mean', 'summary_sentiment_min', 'summary_sentiment_max'
        ]
        
        # Add rolling sentiment features
        daily_sentiment['sentiment_mean_3d'] = daily_sentiment['sentiment_mean'].rolling(window=3).mean()
        daily_sentiment['sentiment_mean_7d'] = daily_sentiment['sentiment_mean'].rolling(window=7).mean()
        
        # Sentiment momentum (change in sentiment)
        daily_sentiment['sentiment_momentum'] = daily_sentiment['sentiment_mean'].diff()
        
        # Sentiment volatility
        daily_sentiment['sentiment_volatility'] = daily_sentiment['sentiment_mean'].rolling(window=5).std()
        
        # News volume features
        daily_sentiment['news_volume_change'] = daily_sentiment['news_count'].pct_change()
        
        # Set date as index
        daily_sentiment['date'] = pd.to_datetime(daily_sentiment['date'])
        daily_sentiment.set_index('date', inplace=True)
        
        # Cache the results if symbol is provided
        if symbol:
            FileUtils.save_dataframe(daily_sentiment, str(cache_file), format='parquet')
            self.logger.info(f"Cached sentiment features for {symbol}")
        
        self.logger.info(f"Calculated sentiment features for {len(daily_sentiment)} days")
        
        return daily_sentiment
    
    def combine_features(self, market_data: pd.DataFrame, sentiment_data: pd.DataFrame) -> pd.DataFrame:
        """
        Combine market data with sentiment features.
        
        Logic: Merges technical indicators with sentiment features,
        handling missing data and creating interaction features.
        
        Args:
            market_data: DataFrame with market data and technical indicators
            sentiment_data: DataFrame with sentiment features
            
        Returns:
            pd.DataFrame: Combined feature set
        """
        self.logger.info("Combining market data with sentiment features")
        
        # Ensure both dataframes have datetime index
        if not isinstance(market_data.index, pd.DatetimeIndex):
            market_data.index = pd.to_datetime(market_data.index)
        
        if not isinstance(sentiment_data.index, pd.DatetimeIndex):
            sentiment_data.index = pd.to_datetime(sentiment_data.index)
        
        # Remove timezone information to avoid tz-aware/tz-naive conflicts
        if market_data.index.tz is not None:
            market_data.index = market_data.index.tz_localize(None)
        
        if sentiment_data.index.tz is not None:
            sentiment_data.index = sentiment_data.index.tz_localize(None)
        
        # Merge on date index
        combined_df = market_data.join(sentiment_data, how='left')
        
        # Forward fill sentiment data for missing dates
        sentiment_columns = sentiment_data.columns
        combined_df[sentiment_columns] = combined_df[sentiment_columns].fillna(method='ffill')
        
        # Fill any remaining NaN values with neutral sentiment
        combined_df[sentiment_columns] = combined_df[sentiment_columns].fillna(0)
        
        # Create interaction features
        if 'sentiment_mean' in combined_df.columns:
            # Sentiment-adjusted RSI
            combined_df['RSI_Sentiment_Adjusted'] = (
                combined_df['RSI'] + combined_df['sentiment_mean'] * 10
            )
            
            # Sentiment-Volume interaction
            if 'Volume_Ratio' in combined_df.columns:
                combined_df['Sentiment_Volume_Interaction'] = (
                    combined_df['sentiment_mean'] * combined_df['Volume_Ratio']
                )
            
            # Sentiment-Momentum interaction
            if 'Price_Change' in combined_df.columns:
                combined_df['Sentiment_Momentum_Interaction'] = (
                    combined_df['sentiment_mean'] * combined_df['Price_Change']
                )
        
        self.logger.info(f"Combined features dataset has {len(combined_df.columns)} features")
        
        return combined_df
    
    def _analyze_sentiment(self, text: str) -> float:
        """
        Analyze sentiment of a text string.
        
        Args:
            text: Text to analyze
            
        Returns:
            float: Sentiment score
        """
        if not text or pd.isna(text):
            return 0.0
        
        # Clean text
        text = self._clean_text(text)
        
        if not text:
            return 0.0
        
        try:
            if self.sentiment_pipeline:
                # Use FinBERT
                result = self.sentiment_pipeline(text[:512])  # Limit text length
                
                # Convert to numeric score
                if result[0]['label'] == 'positive':
                    return result[0]['score']
                elif result[0]['label'] == 'negative':
                    return -result[0]['score']
                else:  # neutral
                    return 0.0
            else:
                # Fall back to TextBlob
                blob = TextBlob(text)
                return blob.sentiment.polarity
                
        except Exception as e:
            self.logger.warning(f"Error analyzing sentiment: {str(e)}")
            return 0.0
    
    def _clean_text(self, text: str) -> str:
        """
        Clean text for sentiment analysis.
        
        Args:
            text: Raw text
            
        Returns:
            str: Cleaned text
        """
        if not text:
            return ""
        
        # Remove URLs
        text = re.sub(r'http\S+', '', text)
        
        # Remove special characters but keep basic punctuation
        text = re.sub(r'[^a-zA-Z0-9\s.,!?-]', '', text)
        
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        return text.strip()
    
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

# Example usage and testing
if __name__ == "__main__":
    from pathlib import Path
    
    # Initialize FeatureEngine
    fe = FeatureEngine()
    
    # Test with sample data
    print("Testing FeatureEngine...")
    
    # Create sample market data
    dates = pd.date_range(start='2024-01-01', periods=100, freq='D')
    sample_market_data = pd.DataFrame({
        'Open': np.random.randn(100).cumsum() + 100,
        'High': np.random.randn(100).cumsum() + 102,
        'Low': np.random.randn(100).cumsum() + 98,
        'Close': np.random.randn(100).cumsum() + 100,
        'Volume': np.random.randint(1000000, 10000000, 100)
    }, index=dates)
    
    # Ensure High >= Low >= Close relationships
    sample_market_data['High'] = sample_market_data[['Open', 'High', 'Low', 'Close']].max(axis=1)
    sample_market_data['Low'] = sample_market_data[['Open', 'High', 'Low', 'Close']].min(axis=1)
    
    try:
        # Test technical indicators
        print("Testing technical indicators...")
        technical_features = fe.calculate_technical_indicators(sample_market_data, "TEST")
        print(f"Technical features shape: {technical_features.shape}")
        print(f"Added {technical_features.shape[1] - sample_market_data.shape[1]} indicators")
        
        # Create sample news data
        sample_news_data = pd.DataFrame({
            'date': pd.date_range(start='2024-01-01', periods=50, freq='D'),
            'headline': [f"Stock news headline {i}" for i in range(50)],
            'summary': [f"Stock news summary {i}" for i in range(50)]
        })
        
        # Test sentiment features
        print("\nTesting sentiment analysis...")
        sentiment_features = fe.calculate_sentiment_features(sample_news_data, "TEST")
        print(f"Sentiment features shape: {sentiment_features.shape}")
        
        # Test combined features
        print("\nTesting feature combination...")
        combined_features = fe.combine_features(technical_features, sentiment_features)
        print(f"Combined features shape: {combined_features.shape}")
        
        print("\nFeatureEngine testing completed successfully!")
        
    except Exception as e:
        print(f"Error during testing: {str(e)}")
        import traceback
        traceback.print_exc()
