# Trading Bot Configuration
import os

# Load environment variables from .env file manually if dotenv is not available
def load_env_file():
    """Load .env file manually if python-dotenv is not available"""
    env_path = os.path.join(os.path.dirname(__file__), '..', '.env')
    if os.path.exists(env_path):
        with open(env_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, value = line.split('=', 1)
                    os.environ[key.strip()] = value.strip()

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    # Fallback to manual loading if dotenv is not available
    load_env_file()

# Database Configuration
DATABASE_CONFIG = {
    'host': os.getenv('DB_HOST', 'localhost'),
    'port': int(os.getenv('DB_PORT', 3306)),
    'user': os.getenv('DB_USER', 'root'),
    'password': os.getenv('DB_PASSWORD', ''),
    'database': os.getenv('DB_NAME', 'TB'),
    'charset': 'utf8mb4'
}

# Trading Configuration
TRADING_CONFIG = {
    'timeframe': '4h',  # 4-hour timeframe as specified
    'demo_balance': 100.0,  # Starting demo balance in USD
    'confidence_threshold': 0.7,  # 70% confidence minimum
    'symbols': ['BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'DOGEUSDT'],
    'max_positions': 4,  # Maximum concurrent positions
    'risk_per_trade': 0.02,  # 2% risk per trade
}

# Take Profit / Stop Loss Configuration
TP_SL_CONFIG = {
    'tp1_percent': 3.0,  # First take profit at 3%
    'tp2_percent': 5.0,  # Second take profit at 5%
    'tp3_percent': 8.0,  # Third take profit at 8%
    'initial_sl_percent': 2.0,  # Initial stop loss at 2%
    'trailing_enabled': True,
}

# CoinEx API Configuration
COINEX_CONFIG = {
    'api_key': os.getenv('COINEX_API_KEY', ''),
    'secret_key': os.getenv('COINEX_SECRET_KEY', ''),
    'sandbox_mode': os.getenv('COINEX_SANDBOX', 'false').lower() == 'true',  # Default to spot trading API
    'base_url': 'https://api.coinex.com/v1/',  # Spot trading API
    'sandbox_url': 'https://api.coinex.com/v1/',  # Use same spot API for better compatibility
}

# Machine Learning Configuration
ML_CONFIG = {
    'training_data_size': 58000,  # Use 58k historical records as specified
    'rfe_sample_size': 1000,  # Use last 1000 samples for RFE selection
    'selected_features': 50,  # Select 50 best indicators via RFE
    'test_size': 0.2,  # 20% for testing
    'random_state': 42,
    'model_retrain_interval': 24,  # Retrain every 24 hours
}

# Web Dashboard Configuration
WEB_CONFIG = {
    'host': '0.0.0.0',
    'port': 5000,
    'debug': os.getenv('DEBUG', 'false').lower() == 'true',
    'secret_key': os.getenv('SECRET_KEY', 'your-secret-key-change-this'),
}

# Data Update Configuration
DATA_CONFIG = {
    'update_interval': 1,  # Update every 1 second as specified
    'batch_size': 100,
    'max_retries': 3,
    'timeout': 30,
}

# Logging Configuration
LOGGING_CONFIG = {
    'level': 'INFO',
    'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    'file': 'logs/trading_bot.log',
    'max_size': 10 * 1024 * 1024,  # 10MB
    'backup_count': 5,
}