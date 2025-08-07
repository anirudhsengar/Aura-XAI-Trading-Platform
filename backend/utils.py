import pandas as pd
import numpy as np
import os
from datetime import datetime
from typing import List, Dict, Any
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
            pd.Series: Simple returns of len n - 1 as initial capital is taken as the benchmark
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
    def calculate_sharpe_ratio(returns: pd.Series, risk_free_rate: float = 0.044) -> float:
        """
        Calculate Sharpe ratio.
        
        Args:
            returns: Series of returns
            risk_free_rate: Risk-free rate i.e. 4.40% in the UAE
            
        Returns:
            float: Sharpe ratio
        """
        if len(returns) == 0 or returns.std() == 0:
            return 0.0
        
        excess_returns = returns - risk_free_rate / 252  # Daily risk-free rate
        volatility = MathUtils.calculate_volatility(excess_returns, False)
        if volatility == 0:
            return 0.0
        return (excess_returns.mean() * 252) / (volatility * np.sqrt(252))
    
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
        return drawdown.min() # Since drawdown is negative
    
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

    @staticmethod
    def calculate_annualized_return(total_return: float, num_days: int) -> float:
        """
        Calculates the annualized return.

        Args:
            total_return: The total return over the period.
            num_days: The number of days in the period.

        Returns:
            The annualized return.
        """
        if num_days == 0:
            return 0
        return (1 + total_return) ** (252 / num_days) - 1

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

class DataValidator:
    """
    Data validation utilities for trading data.
    """
    
    @staticmethod
    def validate_ohlcv_data(data: pd.DataFrame) -> bool:
        """
        Validate OHLCV data format.
        
        Args:
            data: DataFrame with market data
            
        Returns:
            bool: True if data is valid
        """
        if data is None or data.empty:
            return False
            
        # Check for required columns
        required_cols = ['Close']
        if not all(col in data.columns for col in required_cols):
            return False
            
        # Check for valid index (should be datetime)
        if not isinstance(data.index, pd.DatetimeIndex):
            return False
            
        # Check for NaN values in critical columns
        if data['Close'].isna().all():
            return False
            
        return True
    
    @staticmethod
    def validate_strategy_params(strategy_name: str, params: Dict[str, Any]) -> bool:
        """
        Validate strategy parameters.
        
        Args:
            strategy_name: Name of the strategy
            params: Strategy parameters
            
        Returns:
            bool: True if parameters are valid
        """
        if not params:
            return True
            
        try:
            if strategy_name == 'simple_ma':
                fast_ma = params.get('fast_ma', 10)
                slow_ma = params.get('slow_ma', 50)
                return fast_ma < slow_ma and fast_ma > 0 and slow_ma > 0
                
            elif strategy_name == 'mean_reversion':
                rsi_oversold = params.get('rsi_oversold', 30)
                rsi_overbought = params.get('rsi_overbought', 70)
                return rsi_oversold < rsi_overbought and 0 < rsi_oversold < 100 and 0 < rsi_overbought < 100
                
            elif strategy_name == 'momentum':
                fast_ma = params.get('fast_ma', 12)
                slow_ma = params.get('slow_ma', 26)
                return fast_ma < slow_ma and fast_ma > 0 and slow_ma > 0
                
            elif strategy_name in ['ml_strategy', 'lstm_strategy']:
                return True  # More complex validation can be added
                
            return True
            
        except Exception:
            return False
    
    @staticmethod
    def validate_date_range(start_date, end_date) -> bool:
        """
        Validate date range.
        
        Args:
            start_date: Start date
            end_date: End date
            
        Returns:
            bool: True if date range is valid
        """
        try:
            return start_date < end_date
        except Exception:
            return False

class MathUtils:
    """
    Mathematical utilities for performance calculations.
    """
    
    @staticmethod
    def calculate_annualized_return(total_return: float, num_periods: int, 
                                  periods_per_year: int = 252) -> float:
        """
        Calculate annualized return.
        
        Args:
            total_return: Total return over the period
            num_periods: Number of periods
            periods_per_year: Number of periods per year (252 for daily)
            
        Returns:
            float: Annualized return
        """
        if num_periods <= 0:
            return 0.0
        try:
            return (1 + total_return) ** (periods_per_year / num_periods) - 1
        except (OverflowError, ZeroDivisionError):
            return 0.0
    
    @staticmethod
    def calculate_volatility(returns: pd.Series, periods_per_year: int = 252) -> float:
        """
        Calculate annualized volatility.
        
        Args:
            returns: Series of returns
            periods_per_year: Number of periods per year
            
        Returns:
            float: Annualized volatility
        """
        if len(returns) == 0 or returns.std() == 0:
            return 0.0
        return returns.std() * np.sqrt(periods_per_year)
    
    @staticmethod
    def calculate_sharpe_ratio(returns: pd.Series, risk_free_rate: float = 0.02) -> float:
        """
        Calculate Sharpe ratio.
        
        Args:
            returns: Series of returns
            risk_free_rate: Risk-free rate (annualized)
            
        Returns:
            float: Sharpe ratio
        """
        if len(returns) == 0 or returns.std() == 0:
            return 0.0
        
        try:
            excess_returns = returns.mean() * 252 - risk_free_rate
            return excess_returns / (returns.std() * np.sqrt(252))
        except (ZeroDivisionError, OverflowError):
            return 0.0
    
    @staticmethod
    def calculate_max_drawdown(equity_curve: pd.Series) -> float:
        """
        Calculate maximum drawdown.
        
        Args:
            equity_curve: Series of portfolio values
            
        Returns:
            float: Maximum drawdown (negative value)
        """
        if equity_curve.empty:
            return 0.0
            
        running_max = equity_curve.expanding().max()
        drawdowns = (equity_curve - running_max) / running_max
        return drawdowns.min()
    
    @staticmethod
    def calculate_calmar_ratio(annualized_return: float, max_drawdown: float) -> float:
        """
        Calculate Calmar ratio.
        
        Args:
            annualized_return: Annualized return
            max_drawdown: Maximum drawdown
            
        Returns:
            float: Calmar ratio
        """
        if max_drawdown == 0:
            return float('inf') if annualized_return > 0 else 0.0
        return annualized_return / abs(max_drawdown)

class TechnicalUtils:
    """
    Technical analysis utilities.
    """
    
    @staticmethod
    def calculate_sma(data: pd.Series, window: int) -> pd.Series:
        """Calculate Simple Moving Average."""
        return data.rolling(window=window).mean()
    
    @staticmethod
    def calculate_ema(data: pd.Series, window: int) -> pd.Series:
        """Calculate Exponential Moving Average."""
        return data.ewm(span=window).mean()
    
    @staticmethod
    def calculate_rsi(data: pd.Series, window: int = 14) -> pd.Series:
        """Calculate Relative Strength Index."""
        delta = data.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
    
    @staticmethod
    def calculate_bollinger_bands(data: pd.Series, window: int = 20, 
                                num_std: float = 2.0) -> Dict[str, pd.Series]:
        """Calculate Bollinger Bands."""
        sma = data.rolling(window=window).mean()
        std = data.rolling(window=window).std()
        
        return {
            'middle': sma,
            'upper': sma + (std * num_std),
            'lower': sma - (std * num_std)
        }
    
    @staticmethod
    def calculate_macd(data: pd.Series, fast: int = 12, slow: int = 26, 
                      signal: int = 9) -> Dict[str, pd.Series]:
        """Calculate MACD indicator."""
        ema_fast = data.ewm(span=fast).mean()
        ema_slow = data.ewm(span=slow).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal).mean()
        histogram = macd_line - signal_line
        
        return {
            'macd': macd_line,
            'signal': signal_line,
            'histogram': histogram
        }

class DataCleaner:
    """
    Data cleaning utilities.
    """
    
    @staticmethod
    def clean_market_data(data: pd.DataFrame) -> pd.DataFrame:
        """
        Clean market data by handling missing values and outliers.
        
        Args:
            data: Raw market data
            
        Returns:
            pd.DataFrame: Cleaned data
        """
        cleaned_data = data.copy()
        
        # Forward fill missing values
        cleaned_data = cleaned_data.fillna(method='ffill')
        
        # Remove rows where all OHLC values are the same (likely data errors)
        if all(col in cleaned_data.columns for col in ['Open', 'High', 'Low', 'Close']):
            same_values = (cleaned_data['Open'] == cleaned_data['High']) & \
                         (cleaned_data['High'] == cleaned_data['Low']) & \
                         (cleaned_data['Low'] == cleaned_data['Close'])
            cleaned_data = cleaned_data[~same_values]
        
        # Handle volume outliers
        if 'Volume' in cleaned_data.columns:
            volume_median = cleaned_data['Volume'].median()
            volume_outlier_threshold = volume_median * 10
            cleaned_data.loc[cleaned_data['Volume'] > volume_outlier_threshold, 'Volume'] = volume_median
        
        return cleaned_data
    
    @staticmethod
    def remove_outliers(data: pd.Series, method: str = 'iqr', factor: float = 1.5) -> pd.Series:
        """
        Remove outliers from a data series.
        
        Args:
            data: Input data series
            method: Method to use ('iqr' or 'zscore')
            factor: Factor for outlier detection
            
        Returns:
            pd.Series: Data with outliers removed
        """
        cleaned_data = data.copy()
        
        if method == 'iqr':
            Q1 = cleaned_data.quantile(0.25)
            Q3 = cleaned_data.quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - factor * IQR
            upper_bound = Q3 + factor * IQR
            cleaned_data = cleaned_data[(cleaned_data >= lower_bound) & (cleaned_data <= upper_bound)]
            
        elif method == 'zscore':
            z_scores = np.abs((cleaned_data - cleaned_data.mean()) / cleaned_data.std())
            cleaned_data = cleaned_data[z_scores < factor]
        
        return cleaned_data

class PerformanceAnalyzer:
    """
    Performance analysis utilities.
    """
    
    @staticmethod
    def analyze_trades(trades_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Analyze trading performance.
        
        Args:
            trades_df: DataFrame with trade information
            
        Returns:
            Dict: Trade analysis results
        """
        if trades_df.empty:
            return {}
        
        analysis = {}
        
        # Basic trade statistics
        analysis['total_trades'] = len(trades_df)
        
        if 'pnl' in trades_df.columns:
            pnl_series = trades_df['pnl'].dropna()
            
            if not pnl_series.empty:
                analysis['winning_trades'] = len(pnl_series[pnl_series > 0])
                analysis['losing_trades'] = len(pnl_series[pnl_series < 0])
                analysis['win_rate'] = analysis['winning_trades'] / len(pnl_series) if len(pnl_series) > 0 else 0
                
                analysis['avg_win'] = pnl_series[pnl_series > 0].mean() if analysis['winning_trades'] > 0 else 0
                analysis['avg_loss'] = pnl_series[pnl_series < 0].mean() if analysis['losing_trades'] > 0 else 0
                
                analysis['largest_win'] = pnl_series.max()
                analysis['largest_loss'] = pnl_series.min()
                
                # Profit factor
                gross_profit = pnl_series[pnl_series > 0].sum()
                gross_loss = abs(pnl_series[pnl_series < 0].sum())
                analysis['profit_factor'] = gross_profit / gross_loss if gross_loss > 0 else float('inf')
        
        # Holding period analysis
        if 'holding_days' in trades_df.columns:
            holding_days = trades_df['holding_days'].dropna()
            if not holding_days.empty:
                analysis['avg_holding_days'] = holding_days.mean()
                analysis['max_holding_days'] = holding_days.max()
                analysis['min_holding_days'] = holding_days.min()
        
        return analysis
    
    @staticmethod
    def calculate_risk_metrics(returns: pd.Series) -> Dict[str, float]:
        """
        Calculate risk metrics.
        
        Args:
            returns: Series of returns
            
        Returns:
            Dict: Risk metrics
        """
        if returns.empty:
            return {}
        
        metrics = {}
        
        # Basic statistics
        metrics['mean_return'] = returns.mean()
        metrics['volatility'] = returns.std()
        metrics['skewness'] = returns.skew()
        metrics['kurtosis'] = returns.kurtosis()
        
        # Downside risk
        negative_returns = returns[returns < 0]
        metrics['downside_volatility'] = negative_returns.std() if not negative_returns.empty else 0
        
        # Value at Risk (95% confidence)
        metrics['var_95'] = returns.quantile(0.05)
        
        # Conditional Value at Risk
        var_threshold = metrics['var_95']
        conditional_returns = returns[returns <= var_threshold]
        metrics['cvar_95'] = conditional_returns.mean() if not conditional_returns.empty else 0
        
        return metrics