import pandas as pd
import yfinance as yf
from datetime import datetime
from typing import Optional, Dict, Any
import warnings

warnings.filterwarnings('ignore')

class DataManager:
    """
    Data management class for fetching and processing market data.
    """
    
    def __init__(self):
        """Initialize DataManager."""
        self.cache = {}
        
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
            # Create cache key
            cache_key = f"{symbol}_{start_date}_{end_date}"
            
            # Check cache first
            if cache_key in self.cache:
                return self.cache[cache_key].copy()
            
            # Fetch data from yfinance
            ticker = yf.Ticker(symbol)
            data = ticker.history(start=start_date, end=end_date)
            
            if data.empty:
                raise ValueError(f"No data found for symbol {symbol}")
            
            # Clean column names
            data.columns = [col.title() for col in data.columns]
            
            # Ensure we have required columns
            required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
            for col in required_columns:
                if col not in data.columns:
                    if col == 'Volume':
                        data[col] = 1000000  # Default volume
                    else:
                        data[col] = data['Close']  # Use close as fallback
            
            # Remove any rows with all NaN values
            data = data.dropna(how='all')
            
            # Forward fill any remaining NaN values
            data = data.fillna(method='ffill')
            
            # Cache the result
            self.cache[cache_key] = data.copy()
            
            return data
            
        except Exception as e:
            raise Exception(f"Error fetching data for {symbol}: {str(e)}")
    
    def fetch_multiple_symbols(self, symbols: list, start_date: datetime, 
                              end_date: datetime) -> Dict[str, pd.DataFrame]:
        """
        Fetch data for multiple symbols.
        
        Args:
            symbols: List of stock symbols
            start_date: Start date
            end_date: End date
            
        Returns:
            Dict: Dictionary with symbol as key and DataFrame as value
        """
        results = {}
        
        for symbol in symbols:
            try:
                results[symbol] = self.fetch_market_data(symbol, start_date, end_date)
            except Exception as e:
                print(f"Error fetching data for {symbol}: {str(e)}")
                continue
                
        return results
    
    def get_latest_price(self, symbol: str) -> Optional[float]:
        """
        Get the latest price for a symbol.
        
        Args:
            symbol: Stock symbol
            
        Returns:
            float: Latest closing price
        """
        try:
            ticker = yf.Ticker(symbol)
            data = ticker.history(period="1d")
            
            if not data.empty:
                return data['Close'].iloc[-1]
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
            ticker = yf.Ticker(symbol)
            info = ticker.info
            
            return {
                'name': info.get('longName', symbol),
                'sector': info.get('sector', 'Unknown'),
                'industry': info.get('industry', 'Unknown'),
                'market_cap': info.get('marketCap', 0),
                'pe_ratio': info.get('trailingPE', None),
                'dividend_yield': info.get('dividendYield', None)
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
            ticker = yf.Ticker(symbol)
            data = ticker.history(period="5d")
            return not data.empty
            
        except Exception:
            return False
    
    def clear_cache(self):
        """Clear the data cache."""
        self.cache.clear()
