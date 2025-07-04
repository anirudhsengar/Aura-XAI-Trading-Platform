import pandas as pd
import numpy as np
import yfinance as yf
import requests
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import time
import os
from pathlib import Path
from dotenv import load_dotenv

# Import utilities
from utils import DateUtils, DataValidator, FileUtils, LoggingUtils

class DataManager:
    """
    Data Manager for fetching and managing financial and news data.
    """
    
    def __init__(self, cache_dir: str = "../data", log_level: str = "INFO"):
        """
        Initialize DataManager with caching and logging.
        
        Args:
            cache_dir: Directory for caching data
            log_level: Logging level
        """
        # Load environment variables from .env file
        load_dotenv()
        
        self.cache_dir = Path(cache_dir)
        self.market_data_dir = self.cache_dir / "market_data"
        self.news_data_dir = self.cache_dir / "news_data"
        self.cache_data_dir = self.cache_dir / "cache"
        
        # Ensure directories exist
        FileUtils.ensure_directory_exists(str(self.market_data_dir))
        FileUtils.ensure_directory_exists(str(self.news_data_dir))
        FileUtils.ensure_directory_exists(str(self.cache_data_dir))
        
        # Setup logging
        self.logger = LoggingUtils.setup_logger(
            "DataManager", 
            log_file=str(self.cache_dir.parent / "logs" / "data_manager.log"),
            level=log_level
        )
        
        # Configuration
        self.cache_duration_hours = 24  # Cache data for 24 hours
        self.max_retries = 3
        self.retry_delay = 1  # seconds
        
        # News API configuration - now loaded from .env file
        self.news_api_key = os.getenv("NEWS_API_KEY", "")
        self.finnhub_api_key = os.getenv("FINNHUB_API_KEY", "")
        
        # Log API key status (without revealing the keys)
        self.logger.info(f"NewsAPI key loaded: {'Yes' if self.news_api_key else 'No'}")
        self.logger.info(f"Finnhub key loaded: {'Yes' if self.finnhub_api_key else 'No'}")
        
        self.logger.info("DataManager initialized successfully")
    
    def fetch_market_data(self, symbol: str, start_date: datetime, end_date: datetime, 
                         force_refresh: bool = False) -> pd.DataFrame:
        """
        Fetch OHLCV market data for a given symbol and date range.
        
        Args:
            symbol: Stock ticker symbol
            start_date: Start date for data
            end_date: End date for data
            force_refresh: Force refresh data from API
            
        Returns:
            pd.DataFrame: OHLCV data with DatetimeIndex
        """
        # Validate inputs
        if not DataValidator.validate_stock_symbol(symbol):
            raise ValueError(f"Invalid stock symbol: {symbol}")
        
        if not DataValidator.validate_date_range(start_date, end_date):
            raise ValueError(f"Invalid date range: {start_date} to {end_date}")
        
        LoggingUtils.log_data_fetch(self.logger, symbol, str(start_date), str(end_date))
        
        # Check cache first
        cache_file = self.market_data_dir / f"{symbol}_{start_date.strftime('%Y%m%d')}_{end_date.strftime('%Y%m%d')}.parquet"
        
        if not force_refresh and cache_file.exists():
            file_age = datetime.now() - datetime.fromtimestamp(cache_file.stat().st_mtime)
            if file_age.total_seconds() < self.cache_duration_hours * 3600:
                self.logger.info(f"Loading cached data for {symbol}")
                df = FileUtils.load_dataframe(str(cache_file), format='parquet')
                if not df.empty and DataValidator.validate_ohlcv_data(df):
                    return df
        
        # Fetch from API with retry logic
        for attempt in range(self.max_retries):
            try:
                self.logger.info(f"Fetching market data for {symbol} (attempt {attempt + 1})")
                
                # Use yfinance to fetch data
                ticker = yf.Ticker(symbol)
                df = ticker.history(
                    start=start_date.strftime('%Y-%m-%d'),
                    end=end_date.strftime('%Y-%m-%d'),
                    interval='1d'
                )
                
                if df.empty:
                    raise ValueError(f"No data returned for {symbol}")
                
                # Clean and validate data
                df = self._clean_market_data(df)
                
                if not DataValidator.validate_ohlcv_data(df):
                    raise ValueError(f"Invalid OHLCV data for {symbol}")
                
                # Cache the data
                FileUtils.save_dataframe(df, str(cache_file), format='parquet')
                self.logger.info(f"Successfully fetched and cached data for {symbol}")
                
                return df
                
            except Exception as e:
                self.logger.warning(f"Attempt {attempt + 1} failed for {symbol}: {str(e)}")
                if attempt == self.max_retries - 1:
                    self.logger.error(f"Failed to fetch data for {symbol} after {self.max_retries} attempts")
                    raise
                time.sleep(self.retry_delay)
        
        return pd.DataFrame()
    
    def fetch_news_data(self, symbol: str, start_date: datetime, end_date: datetime,
                       force_refresh: bool = False) -> pd.DataFrame:
        """
        Fetch news data for a given symbol and date range.
        
        Args:
            symbol: Stock ticker symbol
            start_date: Start date for news
            end_date: End date for news
            force_refresh: Force refresh data from API
            
        Returns:
            pd.DataFrame: News data with headlines and metadata
        """
        # Validate inputs
        if not DataValidator.validate_stock_symbol(symbol):
            raise ValueError(f"Invalid stock symbol: {symbol}")
        
        self.logger.info(f"Fetching news data for {symbol} from {start_date} to {end_date}")
        
        # Check cache first
        cache_file = self.news_data_dir / f"{symbol}_news_{start_date.strftime('%Y%m%d')}_{end_date.strftime('%Y%m%d')}.parquet"
        
        if not force_refresh and cache_file.exists():
            file_age = datetime.now() - datetime.fromtimestamp(cache_file.stat().st_mtime)
            if file_age.total_seconds() < self.cache_duration_hours * 3600:
                self.logger.info(f"Loading cached news data for {symbol}")
                df = FileUtils.load_dataframe(str(cache_file), format='parquet')
                if not df.empty:
                    return df
        
        # Fetch news from multiple sources
        news_data = []
        
        # Try Yahoo Finance news first (free)
        try:
            ticker = yf.Ticker(symbol)
            news = ticker.news
            
            for article in news:
                if 'providerPublishTime' in article:
                    publish_time = datetime.fromtimestamp(article['providerPublishTime'])
                    if start_date <= publish_time <= end_date:
                        news_data.append({
                            'date': publish_time,
                            'headline': article.get('title', ''),
                            'summary': article.get('summary', ''),
                            'source': 'Yahoo Finance',
                            'url': article.get('link', '')
                        })
        except Exception as e:
            self.logger.warning(f"Failed to fetch Yahoo Finance news for {symbol}: {str(e)}")
        
        # Try Finnhub API if available
        if self.finnhub_api_key:
            try:
                news_data.extend(self._fetch_finnhub_news(symbol, start_date, end_date))
            except Exception as e:
                self.logger.warning(f"Failed to fetch Finnhub news for {symbol}: {str(e)}")
        
        # Try NewsAPI if available
        if self.news_api_key:
            try:
                news_data.extend(self._fetch_newsapi_data(symbol, start_date, end_date))
            except Exception as e:
                self.logger.warning(f"Failed to fetch NewsAPI data for {symbol}: {str(e)}")
        
        # Create DataFrame from news data
        if news_data:
            df = pd.DataFrame(news_data)
            df['date'] = pd.to_datetime(df['date'])
            df = df.sort_values('date').reset_index(drop=True)
            
            # Remove duplicates based on headline
            df = df.drop_duplicates(subset=['headline'], keep='first')
            
            # Cache the data
            FileUtils.save_dataframe(df, str(cache_file), format='parquet')
            self.logger.info(f"Successfully fetched and cached {len(df)} news articles for {symbol}")
            
            return df
        else:
            self.logger.warning(f"No news data found for {symbol}")
            return pd.DataFrame()
    
    def get_company_info(self, symbol: str) -> Dict:
        """
        Get company information for a given symbol. Basic information like 
        sector, industry, market cap, etc. for context in analysis.
        
        Args:
            symbol: Stock ticker symbol
            
        Returns:
            Dict: Company information
        """
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info
            
            # Extract relevant information
            company_info = {
                'symbol': symbol,
                'name': info.get('longName', ''),
                'sector': info.get('sector', ''),
                'industry': info.get('industry', ''),
                'market_cap': info.get('marketCap', 0),
                'currency': info.get('currency', 'USD'),
                'exchange': info.get('exchange', ''),
                'country': info.get('country', ''),
                'website': info.get('website', ''),
                'description': info.get('longBusinessSummary', '')
            }
            
            self.logger.info(f"Retrieved company info for {symbol}")
            return company_info
            
        except Exception as e:
            self.logger.error(f"Failed to get company info for {symbol}: {str(e)}")
            return {}
    
    def _clean_market_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean and standardize market data.
        
        Args:
            df: Raw market data DataFrame
            
        Returns:
            pd.DataFrame: Cleaned market data
        """
        # Ensure we have the required columns
        required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        for col in required_columns:
            if col not in df.columns:
                raise ValueError(f"Missing required column: {col}")
        
        # Remove rows with NaN values
        df = df.dropna()
        
        # Remove rows with zero or negative prices
        price_columns = ['Open', 'High', 'Low', 'Close']
        df = df[(df[price_columns] > 0).all(axis=1)]
        
        # Remove rows with negative volume
        df = df[df['Volume'] >= 0]
        
        # Ensure High >= Low
        df = df[df['High'] >= df['Low']]
        
        # Round prices to 2 decimal places
        for col in price_columns:
            df[col] = df[col].round(2)
        
        # Convert volume to int
        df['Volume'] = df['Volume'].astype(int)
        
        return df
    
    def _fetch_finnhub_news(self, symbol: str, start_date: datetime, end_date: datetime) -> List[Dict]:
        """
        Fetch news from Finnhub API.
        
        Args:
            symbol: Stock ticker symbol
            start_date: Start date
            end_date: End date
            
        Returns:
            List[Dict]: News articles
        """
        news_data = []
        
        try:
            url = "https://finnhub.io/api/v1/company-news"
            params = {
                'symbol': symbol,
                'from': start_date.strftime('%Y-%m-%d'),
                'to': end_date.strftime('%Y-%m-%d'),
                'token': self.finnhub_api_key
            }
            
            response = requests.get(url, params=params)
            response.raise_for_status()
            
            articles = response.json()
            
            for article in articles:
                news_data.append({
                    'date': datetime.fromtimestamp(article.get('datetime', 0)),
                    'headline': article.get('headline', ''),
                    'summary': article.get('summary', ''),
                    'source': 'Finnhub',
                    'url': article.get('url', '')
                })
                
        except Exception as e:
            self.logger.error(f"Error fetching Finnhub news: {str(e)}")
            
        return news_data
    
    def _fetch_newsapi_data(self, symbol: str, start_date: datetime, end_date: datetime) -> List[Dict]:
        """
        Fetch news from NewsAPI.
        
        Args:
            symbol: Stock ticker symbol
            start_date: Start date
            end_date: End date
            
        Returns:
            List[Dict]: News articles
        """
        news_data = []
        
        try:
            url = "https://newsapi.org/v2/everything"
            params = {
                'q': symbol,
                'from': start_date.strftime('%Y-%m-%d'),
                'to': end_date.strftime('%Y-%m-%d'),
                'sortBy': 'publishedAt',
                'apiKey': self.news_api_key,
                'language': 'en'
            }
            
            response = requests.get(url, params=params)
            response.raise_for_status()
            
            data = response.json()
            articles = data.get('articles', [])
            
            for article in articles:
                news_data.append({
                    'date': pd.to_datetime(article.get('publishedAt')),
                    'headline': article.get('title', ''),
                    'summary': article.get('description', ''),
                    'source': 'NewsAPI',
                    'url': article.get('url', '')
                })
                
        except Exception as e:
            self.logger.error(f"Error fetching NewsAPI data: {str(e)}")
            
        return news_data
    
    def get_available_symbols(self) -> List[str]:
        """
        Get list of available symbols from cached data.
        
        Logic: Scans cached market data to provide a list of available
        symbols for user selection.
        
        Returns:
            List[str]: Available stock symbols
        """
        symbols = []
        
        try:
            for file in self.market_data_dir.glob("*.parquet"):
                # Extract symbol from filename (format: SYMBOL_YYYYMMDD_YYYYMMDD.parquet)
                symbol = file.stem.split('_')[0]
                if symbol not in symbols:
                    symbols.append(symbol)
                    
        except Exception as e:
            self.logger.error(f"Error getting available symbols: {str(e)}")
            
        return sorted(symbols)
    
    def clear_cache(self, symbol: str = None, older_than_days: int = 30) -> None:
        """
        Clear cached data.
        
        Logic: Removes old cached data to free up space and ensure
        fresh data retrieval.
        
        Args:
            symbol: Specific symbol to clear (None for all)
            older_than_days: Remove files older than this many days
        """
        cutoff_date = datetime.now() - timedelta(days=older_than_days)
        
        directories = [self.market_data_dir, self.news_data_dir, self.cache_data_dir]
        
        for directory in directories:
            for file in directory.glob("*.parquet"):
                if symbol and not file.name.startswith(symbol):
                    continue
                    
                file_date = datetime.fromtimestamp(file.stat().st_mtime)
                if file_date < cutoff_date:
                    file.unlink()
                    self.logger.info(f"Removed old cache file: {file.name}")

# Example usage and testing
if __name__ == "__main__":
    # Initialize DataManager
    dm = DataManager()
    
    # Test market data fetching
    try:
        print("Testing market data fetching...")
        end_date = datetime.now()
        start_date = end_date - timedelta(days=30)
        
        market_data = dm.fetch_market_data("AAPL", start_date, end_date)
        print(f"Fetched {len(market_data)} days of market data for AAPL")
        print(market_data.head())
        
        # Test news data fetching
        print("\nTesting news data fetching...")
        news_data = dm.fetch_news_data("AAPL", start_date, end_date)
        print(f"Fetched {len(news_data)} news articles for AAPL")
        if not news_data.empty:
            print(news_data.head())
        
        # Test company info
        print("\nTesting company info...")
        company_info = dm.get_company_info("AAPL")
        print(f"Company name: {company_info.get('name', 'N/A')}")
        print(f"Sector: {company_info.get('sector', 'N/A')}")
        
    except Exception as e:
        print(f"Error during testing: {str(e)}")
