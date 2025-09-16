# Trading Bot Configuration
import os
from config.config_loader import load_config

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

# Load configuration from YAML
_config = load_config()

# Database Configuration
DATABASE_CONFIG = {
    'host': os.getenv('DB_HOST', _config.get('database', {}).get('host', 'localhost')),
    'port': int(os.getenv('DB_PORT', _config.get('database', {}).get('port', 3306))),
    'user': os.getenv('DB_USER', _config.get('database', {}).get('user', 'root')),
    'password': os.getenv('DB_PASSWORD', _config.get('database', {}).get('password', '')),
    'database': os.getenv('DB_NAME', _config.get('database', {}).get('database', 'TB')),
    'charset': 'utf8mb4'
}

# Trading Configuration - Load from config.yaml
TRADING_CONFIG = _config.get('trading', {})

# Take Profit / Stop Loss Configuration
TP_SL_CONFIG = _config.get('tp_sl', {})

# CoinEx API Configuration
coinex_config = _config.get('data_sources', {}) if _config.get('data_sources', {}).get('primary') == 'coinex' else {}
COINEX_CONFIG = {
    'api_key': os.getenv('COINEX_API_KEY', ''),
    'secret_key': os.getenv('COINEX_SECRET_KEY', ''),
    'sandbox_mode': os.getenv('COINEX_SANDBOX', 'false').lower() == 'true',
    'base_url': 'https://api.coinex.com/v1/',
    'sandbox_url': 'https://api.coinex.com/v1/',
}

# Machine Learning Configuration
ML_CONFIG = _config.get('ml', {})

# Web Dashboard Configuration
WEB_CONFIG = _config.get('web', {})
WEB_CONFIG['secret_key'] = os.getenv('SECRET_KEY', 'your-secret-key-change-this')
WEB_CONFIG['debug'] = os.getenv('DEBUG', str(WEB_CONFIG.get('debug', False))).lower() == 'true'

# Data Update Configuration
DATA_CONFIG = _config.get('data', {})

# Logging Configuration
LOGGING_CONFIG = _config.get('logging', {})

# Feature Selection Configuration
FEATURE_SELECTION_CONFIG = _config.get('feature_selection', {})

# Legacy configurations for backward compatibility
XGB_PRO_CONFIG = _config.get('xgboost', {})
LABELING_CONFIG = _config.get('ml', {}).get('labeling', {})
