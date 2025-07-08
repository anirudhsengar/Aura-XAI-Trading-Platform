import pandas as pd
import yfinance as yf
from datetime import datetime
import os

class DataManager:
    """
    Data Manager for fetching and managing financial.
    """
    
    def __init__(self):
        """
        Initialize DataManager.
        """
        # API configurations
        self.finnhub_api_key = os.getenv("FINNHUB_API_KEY", "")
    
    def fetch_market_data(self, symbol: str, start_date: datetime, end_date: datetime) -> pd.DataFrame:
        """
        Fetch OHLCV market data for a given symbol and date range.
        
        Args:
            symbol: Stock ticker symbol
            start_date: Start date for data
            end_date: End date for data
            
        Returns:
            pd.DataFrame: OHLCV data with DatetimeIndex
        """
        try:
            # Fetch data from Yahoo Finance
            ticker = yf.Ticker(symbol)
            data = ticker.history(
                start=start_date,
                end=end_date,
                interval='1d'
            )
            
            if data.empty:
                raise ValueError(f"No data found for symbol {symbol}")
            
            # Ensure all numeric columns are properly typed
            numeric_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
            for col in numeric_columns:
                if col in data.columns:
                    data[col] = pd.to_numeric(data[col], errors='coerce')
            
            # Remove any rows with NaN values in critical columns
            data = data.dropna(subset=['Open', 'High', 'Low', 'Close', 'Volume'])
            
            # Ensure index is datetime
            if not isinstance(data.index, pd.DatetimeIndex):
                data.index = pd.to_datetime(data.index)
            
            # Remove timezone info to avoid conflicts
            if data.index.tz is not None:
                data.index = data.index.tz_localize(None)
            
            return data
            
        except Exception as e:
            raise
    
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
    
    def _calculate_basic_metrics(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate basic financial metrics.
        
        Args:
            data: Raw market data
            
        Returns:
            pd.DataFrame: Data with basic metrics
        """
        try:
            # Ensure we have numeric data
            data = data.copy()
            
            # Convert to numeric if needed
            numeric_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
            for col in numeric_columns:
                if col in data.columns:
                    data[col] = pd.to_numeric(data[col], errors='coerce')
            
            # Calculate returns
            data['Returns'] = data['Close'].pct_change()
            
            # Calculate daily range
            data['Daily_Range'] = (data['High'] - data['Low']) / data['Close']
            
            # Calculate volume ratios
            data['Volume_MA'] = data['Volume'].rolling(window=20).mean()
            data['Volume_Ratio'] = data['Volume'] / data['Volume_MA']
            
            # Fill NaN values
            data['Returns'] = data['Returns'].fillna(0)
            data['Daily_Range'] = data['Daily_Range'].fillna(0)
            data['Volume_Ratio'] = data['Volume_Ratio'].fillna(1)
            
            return data
            
        except Exception as e:
            raise