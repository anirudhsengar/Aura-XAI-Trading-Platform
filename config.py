import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class Config:
    """
    Centralized configuration for the Aura platform.
    
    Logic: Stores all configuration parameters in one place for easy
    management and modification across the application.
    
    Why chosen: Centralized configuration makes it easy to modify
    settings without changing code throughout the application.
    """
    
    # Application Settings
    APP_NAME = "Aura - Explainable AI Trading Platform"
    VERSION = "1.0.0"
    DEBUG = os.getenv("DEBUG", "False").lower() == "true"
    
    # Paths
    BASE_DIR = Path(__file__).parent
    DATA_DIR = BASE_DIR / "data"
    BACKEND_DIR = BASE_DIR / "backend"
    LOGS_DIR = BASE_DIR / "logs"
    MODELS_DIR = BASE_DIR / "models"
    
    # Data directories
    MARKET_DATA_DIR = DATA_DIR / "market_data"
    NEWS_DATA_DIR = DATA_DIR / "news_data"
    FEATURES_DIR = DATA_DIR / "features"
    CACHE_DIR = DATA_DIR / "cache"
    RESULTS_DIR = DATA_DIR / "results"
    
    # API Keys
    NEWS_API_KEY = os.getenv("NEWS_API_KEY", "")
    FINNHUB_API_KEY = os.getenv("FINNHUB_API_KEY", "")
    ALPHA_VANTAGE_API_KEY = os.getenv("ALPHA_VANTAGE_API_KEY", "")
    
    # Data Settings
    CACHE_DURATION_HOURS = int(os.getenv("CACHE_DURATION_HOURS", "24"))
    MAX_RETRIES = int(os.getenv("MAX_RETRIES", "3"))
    RETRY_DELAY = float(os.getenv("RETRY_DELAY", "1.0"))
    
    # Trading Settings
    DEFAULT_INITIAL_CAPITAL = float(os.getenv("DEFAULT_INITIAL_CAPITAL", "100000"))
    DEFAULT_COMMISSION_RATE = float(os.getenv("DEFAULT_COMMISSION_RATE", "0.001"))
    DEFAULT_SLIPPAGE_RATE = float(os.getenv("DEFAULT_SLIPPAGE_RATE", "0.0005"))
    
    # Risk Management
    MAX_POSITION_SIZE = float(os.getenv("MAX_POSITION_SIZE", "0.1"))
    DEFAULT_STOP_LOSS = float(os.getenv("DEFAULT_STOP_LOSS", "0.05"))
    DEFAULT_TAKE_PROFIT = float(os.getenv("DEFAULT_TAKE_PROFIT", "0.15"))
    MAX_DRAWDOWN_LIMIT = float(os.getenv("MAX_DRAWDOWN_LIMIT", "0.20"))
    
    # Feature Engineering
    DEFAULT_TECHNICAL_INDICATORS = [
        "RSI", "MACD", "BB_Position", "Volume_Ratio", "Price_Change",
        "SMA_10", "SMA_50", "SMA_200", "Volatility_20", "ATR"
    ]
    
    # ML Settings
    ML_MODEL_RETRAIN_FREQUENCY = int(os.getenv("ML_MODEL_RETRAIN_FREQUENCY", "50"))
    ML_MIN_ACCURACY_THRESHOLD = float(os.getenv("ML_MIN_ACCURACY_THRESHOLD", "0.55"))
    ML_LOOKBACK_PERIOD = int(os.getenv("ML_LOOKBACK_PERIOD", "252"))
    
    # Sentiment Analysis
    SENTIMENT_MODEL_NAME = os.getenv("SENTIMENT_MODEL_NAME", "ProsusAI/finbert")
    SENTIMENT_BATCH_SIZE = int(os.getenv("SENTIMENT_BATCH_SIZE", "16"))
    
    # Logging
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
    LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    LOG_FILE_MAX_SIZE = int(os.getenv("LOG_FILE_MAX_SIZE", "10485760"))  # 10MB
    LOG_FILE_BACKUP_COUNT = int(os.getenv("LOG_FILE_BACKUP_COUNT", "5"))
    
    # Streamlit Settings
    STREAMLIT_THEME = "light"
    STREAMLIT_SIDEBAR_STATE = "expanded"
    STREAMLIT_WIDE_MODE = True
    
    # Performance Settings
    ENABLE_CACHING = os.getenv("ENABLE_CACHING", "True").lower() == "true"
    CACHE_TTL = int(os.getenv("CACHE_TTL", "3600"))  # 1 hour
    
    # Strategy Default Parameters
    STRATEGY_DEFAULTS = {
        "mean_reversion": {
            "bb_period": 20,
            "bb_std": 2,
            "rsi_period": 14,
            "rsi_oversold": 30,
            "rsi_overbought": 70,
            "volume_threshold": 1.5
        },
        "momentum": {
            "fast_ma": 10,
            "slow_ma": 50,
            "rsi_momentum_threshold": 50,
            "macd_confirmation": True,
            "volume_confirmation": True,
            "min_momentum_strength": 0.02
        },
        "multi_factor": {
            "technical_weight": 0.4,
            "sentiment_weight": 0.3,
            "momentum_weight": 0.3,
            "sentiment_threshold": 0.1,
            "signal_threshold": 0.5
        },
        "ml_strategy": {
            "lookback_period": 252,
            "prediction_horizon": 5,
            "retrain_frequency": 50,
            "min_accuracy": 0.55,
            "feature_importance_threshold": 0.01
        }
    }
    
    @classmethod
    def ensure_directories(cls):
        """Ensure all necessary directories exist."""
        directories = [
            cls.DATA_DIR,
            cls.LOGS_DIR,
            cls.MODELS_DIR,
            cls.MARKET_DATA_DIR,
            cls.NEWS_DATA_DIR,
            cls.FEATURES_DIR,
            cls.CACHE_DIR,
            cls.RESULTS_DIR
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
    
    @classmethod
    def validate_api_keys(cls):
        """Validate that required API keys are present."""
        missing_keys = []
        
        if not cls.NEWS_API_KEY:
            missing_keys.append("NEWS_API_KEY")
        
        if not cls.FINNHUB_API_KEY:
            missing_keys.append("FINNHUB_API_KEY")
        
        return missing_keys
    
    @classmethod
    def get_strategy_defaults(cls, strategy_name: str):
        """Get default parameters for a strategy."""
        return cls.STRATEGY_DEFAULTS.get(strategy_name, {})
    
    @classmethod
    def get_data_path(cls, data_type: str, filename: str = ""):
        """Get path for different data types."""
        path_map = {
            "market": cls.MARKET_DATA_DIR,
            "news": cls.NEWS_DATA_DIR,
            "features": cls.FEATURES_DIR,
            "cache": cls.CACHE_DIR,
            "results": cls.RESULTS_DIR,
            "models": cls.MODELS_DIR,
            "logs": cls.LOGS_DIR
        }
        
        base_path = path_map.get(data_type, cls.DATA_DIR)
        return base_path / filename if filename else base_path

# Initialize directories when module is imported
Config.ensure_directories()
