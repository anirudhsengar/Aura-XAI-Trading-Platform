import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
from typing import Optional, Dict, Any
import warnings
import time
import hashlib

warnings.filterwarnings('ignore')

class DataManager:
    """
    Data management class for fetching and processing market data.
    """
    
    def __init__(self, cache_size_limit: int = 100, cache_expiry_hours: int = 24):
        """Initialize DataManager with cache management."""
        self.cache = {}
        self.cache_timestamps = {}
        self.cache_size_limit = cache_size_limit
        self.cache_expiry_hours = cache_expiry_hours
        self.last_api_call = 0
        self.min_api_interval = 0.1  # Minimum seconds between API calls
        
    def fetch_market_data(self, symbol: str, start_date: datetime, 
                         end_date: datetime) -> pd.DataFrame:
        """
        Fetch market data from Yahoo Finance.
        
        Args:
            symbol: Stock symbol (e.g., 'AAPL')
            start_date: Start date for data
            end_date: End date for data
            
        Returns:
            pd.DataFrame: Market data with OHLCV columns
        """
        try:
            # Validate inputs
            symbol = symbol.upper().strip()
            if not symbol:
                raise ValueError("Symbol cannot be empty")
            
            if start_date >= end_date:
                raise ValueError("Start date must be before end date")
            
            # Create cache key
            cache_key = self._generate_cache_key(symbol, start_date, end_date)
            
            # Check cache first
            if self._is_cache_valid(cache_key):
                return self.cache[cache_key].copy()
            
            # Rate limiting
            self._rate_limit()
            
            # Fetch data from yfinance
            ticker = yf.Ticker(symbol)
            data = ticker.history(start=start_date, end=end_date, auto_adjust=True, prepost=True)
            
            if data.empty:
                raise ValueError(f"No data found for symbol {symbol} between {start_date} and {end_date}")
            
            # Process and clean the data
            data = self._process_market_data(data)
            
            # Cache the result
            self._update_cache(cache_key, data)
            
            return data
            
        except Exception as e:
            raise Exception(f"Error fetching data for {symbol}: {str(e)}")
    
    def _generate_cache_key(self, symbol: str, start_date: datetime, end_date: datetime) -> str:
        """Generate a unique cache key."""
        key_string = f"{symbol}_{start_date.isoformat()}_{end_date.isoformat()}"
        return hashlib.md5(key_string.encode()).hexdigest()
    
    def _is_cache_valid(self, cache_key: str) -> bool:
        """Check if cache entry is valid and not expired."""
        if cache_key not in self.cache:
            return False
        
        # Check expiry
        cache_time = self.cache_timestamps.get(cache_key)
        if cache_time:
            expiry_time = cache_time + timedelta(hours=self.cache_expiry_hours)
            if datetime.now() > expiry_time:
                self._remove_from_cache(cache_key)
                return False
        
        return True
    
    def _rate_limit(self):
        """Implement rate limiting for API calls."""
        current_time = time.time()
        time_since_last_call = current_time - self.last_api_call
        
        if time_since_last_call < self.min_api_interval:
            time.sleep(self.min_api_interval - time_since_last_call)
        
        self.last_api_call = time.time()
    
    def _process_market_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Process and clean market data."""
        # Clean column names - handle both old and new yfinance formats
        column_mapping = {
            'Adj Close': 'Close',
            'Adj_Close': 'Close'
        }
        data = data.rename(columns=column_mapping)
        
        # Ensure column names are properly formatted
        data.columns = [col.replace(' ', '_').title() for col in data.columns]
        
        # Ensure we have required columns
        required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        for col in required_columns:
            if col not in data.columns:
                if col == 'Volume':
                    data[col] = 1000000  # Default volume
                else:
                    data[col] = data['Close']  # Use close as fallback
        
        # Data quality checks
        data = self._clean_market_data(data)
        
        return data
    
    def _clean_market_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Clean and validate market data."""
        # Remove any rows with all NaN values
        data = data.dropna(how='all')
        
        if data.empty:
            raise ValueError("No valid data after cleaning")
        
        # Ensure positive prices
        price_columns = ['Open', 'High', 'Low', 'Close']
        for col in price_columns:
            if col in data.columns:
                # Replace non-positive prices with forward-filled values
                data[col] = data[col].where(data[col] > 0)
                data[col] = data[col].ffill()  # Use modern pandas method
                
                # If still NaN at the beginning, use the first valid value
                first_valid = data[col].first_valid_index()
                if first_valid is not None:
                    data[col] = data[col].fillna(data[col].loc[first_valid])
        
        # Ensure volume is non-negative
        if 'Volume' in data.columns:
            data['Volume'] = data['Volume'].where(data['Volume'] >= 0, 0)
            data['Volume'] = data['Volume'].fillna(0)
        
        # OHLC validation - ensure High >= Low and High >= Open, Close
        if all(col in data.columns for col in ['Open', 'High', 'Low', 'Close']):
            # Fix any OHLC inconsistencies
            data['High'] = data[['Open', 'High', 'Low', 'Close']].max(axis=1)
            data['Low'] = data[['Open', 'High', 'Low', 'Close']].min(axis=1)
        
        # Remove any remaining rows with NaN in critical columns
        critical_columns = ['Close']
        data = data.dropna(subset=critical_columns)
        
        # Forward fill any remaining NaN values
        data = data.ffill()
        
        # Ensure all numeric columns are proper types
        numeric_columns = data.select_dtypes(include=['number']).columns
        data[numeric_columns] = data[numeric_columns].astype('float64')
        
        return data
    
    def _update_cache(self, cache_key: str, data: pd.DataFrame):
        """Update cache with size management."""
        # Clean expired entries first
        self._clean_expired_cache()
        
        # If cache is full, remove oldest entry
        if len(self.cache) >= self.cache_size_limit:
            oldest_key = min(self.cache_timestamps.keys(), 
                           key=lambda k: self.cache_timestamps[k])
            self._remove_from_cache(oldest_key)
        
        # Add new entry
        self.cache[cache_key] = data.copy()
        self.cache_timestamps[cache_key] = datetime.now()
    
    def _remove_from_cache(self, cache_key: str):
        """Remove entry from cache."""
        if cache_key in self.cache:
            del self.cache[cache_key]
        if cache_key in self.cache_timestamps:
            del self.cache_timestamps[cache_key]
    
    def _clean_expired_cache(self):
        """Remove expired cache entries."""
        current_time = datetime.now()
        expired_keys = []
        
        for key, timestamp in self.cache_timestamps.items():
            if current_time - timestamp > timedelta(hours=self.cache_expiry_hours):
                expired_keys.append(key)
        
        for key in expired_keys:
            self._remove_from_cache(key)
    
    def get_latest_price(self, symbol: str) -> Optional[float]:
        """
        Get the latest price for a symbol.
        
        Args:
            symbol: Stock symbol
            
        Returns:
            float: Latest closing price
        """
        try:
            symbol = symbol.upper().strip()
            self._rate_limit()
            
            ticker = yf.Ticker(symbol)
            data = ticker.history(period="1d")
            
            if not data.empty:
                # Handle both old and new column formats
                if 'Close' in data.columns:
                    return float(data['Close'].iloc[-1])
                elif 'Adj Close' in data.columns:
                    return float(data['Adj Close'].iloc[-1])
            return None
            
        except Exception as e:
            print(f"Error fetching latest price for {symbol}: {str(e)}")
            return None
    
    def get_company_info(self, symbol: str) -> Dict[str, Any]:
        """
        Get company information.
        
        Args:
            symbol: Stock symbol
            
        Returns:
            Dict: Company information
        """
        try:
            symbol = symbol.upper().strip()
            self._rate_limit()
            
            ticker = yf.Ticker(symbol)
            info = ticker.info
            
            # Handle cases where info might be None or empty
            if not info:
                return {'name': symbol, 'sector': 'Unknown', 'industry': 'Unknown'}
            
            return {
                'name': info.get('longName', info.get('shortName', symbol)),
                'sector': info.get('sector', 'Unknown'),
                'industry': info.get('industry', 'Unknown'),
                'market_cap': info.get('marketCap', 0),
                'pe_ratio': info.get('trailingPE', info.get('forwardPE', None)),
                'dividend_yield': info.get('dividendYield', None),
                'beta': info.get('beta', None),
                'currency': info.get('currency', 'USD')
            }
            
        except Exception as e:
            print(f"Error fetching company info for {symbol}: {str(e)}")
            return {'name': symbol, 'sector': 'Unknown', 'industry': 'Unknown'}
    
    def validate_symbol(self, symbol: str) -> bool:
        """
        Validate if a symbol exists and has data.
        
        Args:
            symbol: Stock symbol to validate
            
        Returns:
            bool: True if symbol is valid
        """
        try:
            symbol = symbol.upper().strip()
            if not symbol:
                return False
            
            self._rate_limit()
            
            ticker = yf.Ticker(symbol)
            # Try to get recent data
            data = ticker.history(period="5d")
            
            # Check if we have valid data
            return not data.empty and len(data) > 0
            
        except Exception:
            return False
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        return {
            'cache_size': len(self.cache),
            'cache_limit': self.cache_size_limit,
            'cache_expiry_hours': self.cache_expiry_hours,
            'oldest_entry': min(self.cache_timestamps.values()) if self.cache_timestamps else None,
            'newest_entry': max(self.cache_timestamps.values()) if self.cache_timestamps else None
        }
    
    def clear_cache(self):
        """Clear the data cache."""
        self.cache.clear()
        self.cache_timestamps.clear()
