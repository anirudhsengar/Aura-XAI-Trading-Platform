import pandas as pd
import numpy as np
import os
import json
import logging
from datetime import datetime, timedelta
from typing import Union, List, Dict, Any, Optional
import warnings
warnings.filterwarnings('ignore')

class DateUtils:
    """
    Date handling utilities for financial data processing.
    """
    
    @staticmethod
    def is_trading_day(date: datetime) -> bool:
        """
        Check if a given date is a trading day (excludes weekends).
        
        Args:
            date: datetime object to check
            
        Returns:
            bool: True if trading day, False otherwise
        """
        return date.weekday() < 5  # Monday=0, Sunday=6
    
    @staticmethod
    def get_trading_days(start_date: datetime, end_date: datetime) -> List[datetime]:
        """
        Get all trading days between two dates.
        
        Args:
            start_date: Start date
            end_date: End date
            
        Returns:
            List of trading days
        """
        dates = pd.date_range(start=start_date, end=end_date, freq='D')
        return [date for date in dates if DateUtils.is_trading_day(date)]
    
    @staticmethod
    def format_date_for_api(date: datetime) -> str:
        """
        Format date for API calls (YYYY-MM-DD format).
        
        Args:
            date: datetime object
            
        Returns:
            str: Formatted date string
        """
        return date.strftime('%Y-%m-%d')
    
    @staticmethod
    def parse_date_string(date_str: str) -> datetime:
        """
        Parse date string to datetime object.
        
        Args:
            date_str: Date string in various formats
            
        Returns:
            datetime: Parsed datetime object
        """
        try:
            return pd.to_datetime(date_str)
        except:
            raise ValueError(f"Unable to parse date: {date_str}")

class DataValidator:
    """
    Data validation utilities for user inputs and API responses.
    """
    
    @staticmethod
    def validate_stock_symbol(symbol: str) -> bool:
        """
        Validate stock symbol format.
        
        Args:
            symbol: Stock ticker symbol
            
        Returns:
            bool: True if valid format
        """
        if not symbol or not isinstance(symbol, str):
            return False
        
        # Basic validation: 1-5 characters, letters only
        return len(symbol) <= 5 and symbol.isalpha() and symbol.isupper()
    
    @staticmethod
    def validate_date_range(start_date: datetime, end_date: datetime) -> bool:
        """
        Validate date range for backtesting.
        
        Args:
            start_date: Start date
            end_date: End date
            
        Returns:
            bool: True if valid range
        """
        if start_date >= end_date:
            return False
        
        # Check if range is reasonable (not too far in future)
        if end_date > datetime.now():
            return False
        
        # Check minimum range (at least 30 days for meaningful backtest)
        if (end_date - start_date).days < 30:
            return False
        
        return True
    
    @staticmethod
    def validate_ohlcv_data(df: pd.DataFrame) -> bool:
        """
        Validate OHLCV data structure and content.
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            bool: True if valid data
        """
        required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        
        # Check if all required columns exist
        if not all(col in df.columns for col in required_columns):
            return False
        
        # Check for negative prices or volumes
        if (df[['Open', 'High', 'Low', 'Close']] < 0).any().any():
            return False
        
        if (df['Volume'] < 0).any():
            return False
        
        # Check High >= Low logic
        if (df['High'] < df['Low']).any():
            return False
        
        return True
    
    @staticmethod
    def validate_strategy_params(params: Dict[str, Any]) -> bool:
        """
        Validate strategy parameters.
        
        Args:
            params: Dictionary of strategy parameters
            
        Returns:
            bool: True if valid parameters
        """
        # Check for required parameters
        if 'lookback_period' in params:
            if not isinstance(params['lookback_period'], int) or params['lookback_period'] < 1:
                return False
        
        if 'rsi_threshold' in params:
            if not isinstance(params['rsi_threshold'], (int, float)) or not 0 <= params['rsi_threshold'] <= 100:
                return False
        
        return True

class MathUtils:
    """
    Mathematical and statistical utilities for trading calculations.
    """
    
    @staticmethod
    def calculate_returns(prices: pd.Series) -> pd.Series:
        """
        Calculate simple returns from price series.
        
        Args:
            prices: Series of prices
            
        Returns:
            pd.Series: Simple returns
        """
        return prices.pct_change().dropna()
    
    @staticmethod
    def calculate_log_returns(prices: pd.Series) -> pd.Series:
        """
        Calculate logarithmic returns from price series.
        
        Args:
            prices: Series of prices
            
        Returns:
            pd.Series: Log returns
        """
        return np.log(prices / prices.shift(1)).dropna()
    
    @staticmethod
    def calculate_volatility(returns: pd.Series, annualize: bool = True) -> float:
        """
        Calculate volatility (standard deviation of returns).
        
        Args:
            returns: Series of returns
            annualize: Whether to annualize the volatility
            
        Returns:
            float: Volatility
        """
        vol = returns.std()
        if annualize:
            vol *= np.sqrt(252)  # 252 trading days per year
        return vol
    
    @staticmethod
    def calculate_sharpe_ratio(returns: pd.Series, risk_free_rate: float = 0.02) -> float:
        """
        Calculate Sharpe ratio.
        
        Args:
            returns: Series of returns
            risk_free_rate: Risk-free rate (default 2%)
            
        Returns:
            float: Sharpe ratio
        """
        excess_returns = returns - risk_free_rate / 252  # Daily risk-free rate
        return excess_returns.mean() / excess_returns.std() * np.sqrt(252)
    
    @staticmethod
    def calculate_max_drawdown(equity_curve: pd.Series) -> float:
        """
        Calculate maximum drawdown from equity curve.
        
        Args:
            equity_curve: Series of portfolio values
            
        Returns:
            float: Maximum drawdown as percentage
        """
        peak = equity_curve.expanding().max()
        drawdown = (equity_curve - peak) / peak
        return drawdown.min()
    
    @staticmethod
    def calculate_win_rate(trades: pd.Series) -> float:
        """
        Calculate win rate from trade P&L series.
        
        Args:
            trades: Series of trade P&L values
            
        Returns:
            float: Win rate as percentage
        """
        winning_trades = (trades > 0).sum()
        total_trades = len(trades)
        return winning_trades / total_trades if total_trades > 0 else 0

class FileUtils:
    """
    File I/O utilities for data storage and retrieval.
    """
    
    @staticmethod
    def ensure_directory_exists(path: str) -> None:
        """
        Ensure directory exists, create if not.
        
        Args:
            path: Directory path
        """
        os.makedirs(path, exist_ok=True)
    
    @staticmethod
    def save_json(data: Dict[str, Any], filepath: str) -> None:
        """
        Save dictionary to JSON file.
        
        Args:
            data: Dictionary to save
            filepath: Path to save file
        """
        FileUtils.ensure_directory_exists(os.path.dirname(filepath))
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2, default=str)
    
    @staticmethod
    def load_json(filepath: str) -> Dict[str, Any]:
        """
        Load dictionary from JSON file.
        
        Args:
            filepath: Path to JSON file
            
        Returns:
            Dict: Loaded data
        """
        try:
            with open(filepath, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            return {}
    
    @staticmethod
    def save_dataframe(df: pd.DataFrame, filepath: str, format: str = 'parquet') -> None:
        """
        Save DataFrame to file.
        
        Args:
            df: DataFrame to save
            filepath: Path to save file
            format: File format ('parquet', 'csv')
        """
        FileUtils.ensure_directory_exists(os.path.dirname(filepath))
        
        if format == 'parquet':
            df.to_parquet(filepath)
        elif format == 'csv':
            df.to_csv(filepath, index=False)
        else:
            raise ValueError(f"Unsupported format: {format}")
    
    @staticmethod
    def load_dataframe(filepath: str, format: str = 'parquet') -> pd.DataFrame:
        """
        Load DataFrame from file.
        
        Args:
            filepath: Path to file
            format: File format ('parquet', 'csv')
            
        Returns:
            pd.DataFrame: Loaded data
        """
        try:
            if format == 'parquet':
                return pd.read_parquet(filepath)
            elif format == 'csv':
                return pd.read_csv(filepath, index_col=0, parse_dates=True)
            else:
                raise ValueError(f"Unsupported format: {format}")
        except FileNotFoundError:
            return pd.DataFrame()

class LoggingUtils:
    """
    Logging utilities for debugging and monitoring.
    """
    
    @staticmethod
    def setup_logger(name: str, log_file: str = None, level: str = 'INFO') -> logging.Logger:
        """
        Set up logger with consistent formatting.
        
        Args:
            name: Logger name
            log_file: Optional log file path
            level: Logging level
            
        Returns:
            logging.Logger: Configured logger
        """
        logger = logging.getLogger(name)
        logger.setLevel(getattr(logging, level.upper()))
        
        # Create formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
        
        # File handler if specified
        if log_file:
            FileUtils.ensure_directory_exists(os.path.dirname(log_file))
            file_handler = logging.FileHandler(log_file)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
        
        return logger
    
    @staticmethod
    def log_trade_execution(logger: logging.Logger, trade_info: Dict[str, Any]) -> None:
        """
        Log trade execution details.
        
        Args:
            logger: Logger instance
            trade_info: Trade information dictionary
        """
        logger.info(f"Trade executed: {trade_info}")
    
    @staticmethod
    def log_data_fetch(logger: logging.Logger, symbol: str, start_date: str, end_date: str) -> None:
        """
        Log data fetch operation.
        
        Args:
            logger: Logger instance
            symbol: Stock symbol
            start_date: Start date
            end_date: End date
        """
        logger.info(f"Fetching data for {symbol} from {start_date} to {end_date}")

# Example usage and testing functions
if __name__ == "__main__":
    # Test date utilities
    print("Testing DateUtils...")
    today = datetime.now()
    print(f"Is today a trading day? {DateUtils.is_trading_day(today)}")
    
    # Test data validation
    print("\nTesting DataValidator...")
    print(f"Is 'AAPL' a valid symbol? {DataValidator.validate_stock_symbol('AAPL')}")
    print(f"Is 'apple' a valid symbol? {DataValidator.validate_stock_symbol('apple')}")
    
    # Test math utilities
    print("\nTesting MathUtils...")
    sample_prices = pd.Series([100, 101, 99, 102, 98])
    returns = MathUtils.calculate_returns(sample_prices)
    print(f"Sample returns: {returns.values}")
    
    # Test logging
    print("\nTesting LoggingUtils...")
    logger = LoggingUtils.setup_logger("test_logger")
    logger.info("Test log message")
    
    print("All utility functions working correctly!")
