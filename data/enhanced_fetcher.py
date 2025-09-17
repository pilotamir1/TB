"""
Enhanced Data Fetcher with multi-source support and indicator integration
"""

import pandas as pd
import logging
import threading
import time
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta

from utils.api_client import MultiSourceAPIClient
from indicators.enhanced_calculator import EnhancedIndicatorCalculator
from database.connection import db_connection
from database.models import Candle

class EnhancedDataFetcher:
    """Enhanced data fetcher with failover and indicator integration"""
    
    def __init__(self, api_client: MultiSourceAPIClient, config: Dict[str, Any] = None):
        self.api_client = api_client
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        
        # Enhanced calculator
        self.calculator = EnhancedIndicatorCalculator()
        
        # Configuration
        trading_config = self.config.get('trading', {})
        data_config = self.config.get('data_sources', {})
        
        self.symbols = trading_config.get('symbols', ['BTCUSDT', 'ETHUSDT'])
        self.timeframe = trading_config.get('timeframe', '4h')
        self.update_interval = data_config.get('timeout_seconds', 30)
        
        # Threading
        self.update_thread = None
        self.stop_updates = False
        
        # Enhanced data cache with health tracking
        self.latest_prices = {}
        self.latest_data_cache = {}
        self.last_update_times = {}
        self.data_health_status = {}
        
        self.logger.info("Enhanced data fetcher initialized")
    
    def start_real_time_updates(self):
        """Start real-time data updates with enhanced error handling"""
        if self.update_thread and self.update_thread.is_alive():
            self.logger.warning("Real-time updates already running")
            return
        
        self.stop_updates = False
        self.update_thread = threading.Thread(target=self._enhanced_update_loop, daemon=True)
        self.update_thread.start()
        
        self.logger.info("Enhanced real-time data updates started")
    
    def stop_real_time_updates(self):
        """Stop real-time data updates"""
        self.stop_updates = True
        if self.update_thread:
            self.update_thread.join(timeout=10)
        
        self.logger.info("Real-time data updates stopped")
    
    def _enhanced_update_loop(self):
        """Enhanced update loop with health monitoring"""
        while not self.stop_updates:
            try:
                update_start = time.time()
                
                # Update data for all symbols
                for symbol in self.symbols:
                    try:
                        self._update_symbol_data(symbol)
                    except Exception as e:
                        self.logger.error(f"Failed to update {symbol}: {e}")
                        self._mark_symbol_unhealthy(symbol, str(e))
                
                # Calculate update time
                update_duration = time.time() - update_start
                self.logger.debug(f"Data update completed in {update_duration:.2f}s")
                
                # Sleep until next update
                sleep_time = max(1, self.update_interval - update_duration)
                time.sleep(sleep_time)
                
            except Exception as e:
                self.logger.error(f"Error in update loop: {e}")
                time.sleep(30)  # Wait before retrying
    
    def _update_symbol_data(self, symbol: str):
        """Update data for a specific symbol with health tracking"""
        try:
            # Get latest market data (reduced to 5 to avoid API pressure)
            market_data = self.api_client.get_market_data(symbol, self.timeframe, limit=5)
            
            if not market_data:
                raise ValueError(f"No market data received for {symbol}")
            
            # Convert to DataFrame
            df = self._convert_market_data_to_df(market_data, symbol)
            
            if df is None or len(df) == 0:
                raise ValueError(f"Failed to convert market data for {symbol}")
            
            # Update cache
            self.latest_data_cache[symbol] = df
            self.last_update_times[symbol] = datetime.now()
            
            # Update latest price
            self.latest_prices[symbol] = df['close'].iloc[-1]
            
            # Mark symbol as healthy
            self._mark_symbol_healthy(symbol)
            
            self.logger.debug(f"Updated {symbol}: {len(df)} records, latest price: {self.latest_prices[symbol]:.4f}")
            
        except Exception as e:
            self._mark_symbol_unhealthy(symbol, str(e))
            raise
    
    def _convert_market_data_to_df(self, market_data: List, symbol: str) -> pd.DataFrame:
        """Convert market data to standardized DataFrame"""
        try:
            if not market_data:
                return None
            
            # Handle different API response formats
            if isinstance(market_data[0], dict):
                # CoinEx format
                df = pd.DataFrame(market_data)
                if 'timestamp' in df.columns:
                    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
            elif isinstance(market_data[0], (list, tuple)):
                # Binance format - [timestamp, open, high, low, close, volume, ...]
                columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
                df = pd.DataFrame(market_data, columns=columns[:len(market_data[0])])
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            else:
                raise ValueError(f"Unknown market data format for {symbol}")
            
            # Ensure required columns
            required_columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
            for col in required_columns:
                if col not in df.columns:
                    if col == 'timestamp':
                        df['timestamp'] = pd.to_datetime('now')
                    else:
                        df[col] = 0.0
            
            # Convert to numeric
            numeric_columns = ['open', 'high', 'low', 'close', 'volume']
            for col in numeric_columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # Sort by timestamp
            df = df.sort_values('timestamp').reset_index(drop=True)
            
            # Remove duplicates
            df = df.drop_duplicates(subset=['timestamp']).reset_index(drop=True)
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error converting market data for {symbol}: {e}")
            return None
    
    def _mark_symbol_healthy(self, symbol: str):
        """Mark symbol as healthy"""
        self.data_health_status[symbol] = {
            'status': 'healthy',
            'last_update': datetime.now(),
            'error_message': None,
            'consecutive_failures': 0
        }
    
    def _mark_symbol_unhealthy(self, symbol: str, error_message: str):
        """Mark symbol as unhealthy"""
        current_status = self.data_health_status.get(symbol, {})
        failures = current_status.get('consecutive_failures', 0) + 1
        
        self.data_health_status[symbol] = {
            'status': 'unhealthy',
            'last_update': datetime.now(),
            'error_message': error_message,
            'consecutive_failures': failures
        }
        
        if failures >= 3:
            self.logger.warning(f"Symbol {symbol} has {failures} consecutive failures")
    
    def get_latest_data_with_indicators(self, symbol: str, lookback_periods: int = 500) -> pd.DataFrame:
        """Get latest data with enhanced indicators calculated"""
        try:
            # First try to get from cache
            if symbol in self.latest_data_cache:
                df = self.latest_data_cache[symbol].copy()
                
                # Extend with historical data if not enough
                if len(df) < lookback_periods:
                    historical_df = self.get_historical_data(symbol, limit=lookback_periods)
                    if historical_df is not None:
                        # Combine historical and latest data
                        combined = pd.concat([historical_df, df]).drop_duplicates(
                            subset=['timestamp'], keep='last'
                        ).sort_values('timestamp').reset_index(drop=True)
                        df = combined
                
            else:
                # Get fresh data
                df = self.get_historical_data(symbol, limit=lookback_periods)
            
            if df is None or len(df) < 50:
                self.logger.warning(f"Insufficient data for {symbol}: {len(df) if df is not None else 0} records")
                return None
            
            # Calculate enhanced indicators
            df_with_indicators = self.calculator.calculate_all_enhanced_indicators(df)
            
            self.logger.debug(f"Prepared {symbol} data: {len(df_with_indicators)} records with {len(df_with_indicators.columns)} features")
            
            return df_with_indicators
            
        except Exception as e:
            self.logger.error(f"Error getting data with indicators for {symbol}: {e}")
            return None
    
    def get_historical_data(self, symbol: str, limit: int = 1000) -> pd.DataFrame:
        """Get historical data for symbol"""
        try:
            # Try to get from database first
            df_db = self._get_data_from_database(symbol, limit)
            
            # Get fresh data from API
            market_data = self.api_client.get_market_data(symbol, self.timeframe, limit=limit)
            df_api = self._convert_market_data_to_df(market_data, symbol)
            
            if df_api is None or len(df_api) == 0:
                self.logger.warning(f"No API data for {symbol}, using database data")
                return df_db
            
            # Combine database and API data if both available
            if df_db is not None and len(df_db) > 0:
                combined = pd.concat([df_db, df_api]).drop_duplicates(
                    subset=['timestamp'], keep='last'
                ).sort_values('timestamp').reset_index(drop=True)
                
                # Keep only the requested limit
                combined = combined.tail(limit)
                
                self.logger.debug(f"Combined data for {symbol}: {len(combined)} records")
                return combined
            else:
                self.logger.debug(f"API data for {symbol}: {len(df_api)} records")
                return df_api
                
        except Exception as e:
            self.logger.error(f"Error getting historical data for {symbol}: {e}")
            return None
    
    def _get_data_from_database(self, symbol: str, limit: int) -> pd.DataFrame:
        """Get data from database"""
        try:
            session = db_connection.get_session()
            
            candles = session.query(Candle).filter(
                Candle.symbol == symbol,
                Candle.timeframe == self.timeframe
            ).order_by(Candle.timestamp.desc()).limit(limit).all()
            
            session.close()
            
            if not candles:
                return None
            
            # Convert to DataFrame
            data = []
            for candle in candles:
                data.append({
                    'timestamp': candle.timestamp,
                    'open': candle.open,
                    'high': candle.high,
                    'low': candle.low,
                    'close': candle.close,
                    'volume': candle.volume
                })
            
            df = pd.DataFrame(data)
            df = df.sort_values('timestamp').reset_index(drop=True)
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error getting data from database for {symbol}: {e}")
            return None
    
    def get_market_overview(self) -> Dict[str, Any]:
        """Get market overview for all symbols"""
        try:
            overview = {}
            
            for symbol in self.symbols:
                try:
                    # Get ticker data
                    ticker = self.api_client.get_ticker(symbol)
                    
                    # Get latest price from cache or ticker
                    price = self.latest_prices.get(symbol, ticker.get('last', 0))
                    
                    # Calculate 24h change (simplified)
                    change_24h = ticker.get('change_rate', 0)
                    if isinstance(change_24h, str) and change_24h.endswith('%'):
                        change_24h = float(change_24h[:-1])
                    
                    overview[symbol] = {
                        'price': float(price),
                        'change_24h': float(change_24h),
                        'volume': ticker.get('volume', 0),
                        'high_24h': ticker.get('high', price),
                        'low_24h': ticker.get('low', price),
                        'last_update': self.last_update_times.get(symbol, datetime.now()).isoformat(),
                        'health_status': self.data_health_status.get(symbol, {}).get('status', 'unknown')
                    }
                    
                except Exception as e:
                    self.logger.error(f"Error getting overview for {symbol}: {e}")
                    overview[symbol] = {
                        'price': 0,
                        'change_24h': 0,
                        'error': str(e),
                        'health_status': 'error'
                    }
            
            return overview
            
        except Exception as e:
            self.logger.error(f"Error getting market overview: {e}")
            return {}
    
    def get_data_health_status(self) -> Dict[str, Any]:
        """Get data health status for all symbols"""
        overall_status = {
            'healthy_symbols': 0,
            'unhealthy_symbols': 0,
            'total_symbols': len(self.symbols),
            'last_update': datetime.now().isoformat(),
            'symbols': {}
        }
        
        for symbol in self.symbols:
            symbol_health = self.data_health_status.get(symbol, {
                'status': 'unknown',
                'last_update': None,
                'error_message': None,
                'consecutive_failures': 0
            })
            
            overall_status['symbols'][symbol] = {
                'status': symbol_health['status'],
                'last_update': symbol_health['last_update'].isoformat() if symbol_health['last_update'] else None,
                'consecutive_failures': symbol_health['consecutive_failures'],
                'error_message': symbol_health['error_message']
            }
            
            if symbol_health['status'] == 'healthy':
                overall_status['healthy_symbols'] += 1
            else:
                overall_status['unhealthy_symbols'] += 1
        
        # Overall health
        if overall_status['unhealthy_symbols'] == 0:
            overall_status['overall_status'] = 'healthy'
        elif overall_status['healthy_symbols'] > 0:
            overall_status['overall_status'] = 'degraded'
        else:
            overall_status['overall_status'] = 'unhealthy'
        
        return overall_status
    
    def get_latest_price(self, symbol: str) -> Optional[float]:
        """Get latest price for symbol"""
        return self.latest_prices.get(symbol)
    
    def is_symbol_healthy(self, symbol: str) -> bool:
        """Check if symbol data is healthy"""
        status = self.data_health_status.get(symbol, {})
        return status.get('status') == 'healthy'