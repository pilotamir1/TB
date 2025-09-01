import pandas as pd
import logging
import threading
import time
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta

from trading.coinex_api import CoinExAPI
from indicators.calculator import IndicatorCalculator
from database.connection import db_connection
from database.models import Candle
from config.settings import TRADING_CONFIG, DATA_CONFIG
from config.config_loader import get_config_value

class DataFetcher:
    """
    Real-time data fetching and management system
    """
    
    def __init__(self, api: CoinExAPI):
        self.api = api
        self.logger = logging.getLogger(__name__)
        self.calculator = IndicatorCalculator()
        
        # Configuration
        self.symbols = TRADING_CONFIG['symbols']
        self.timeframe = TRADING_CONFIG['timeframe']
        self.update_interval = DATA_CONFIG['update_interval']
        
        # Threading
        self.update_thread = None
        self.stop_updates = False
        
        # Data cache
        self.latest_prices = {}
        self.latest_data_cache = {}
        
        # Throttling for data fetch spam prevention
        self.last_fetch_times = {}  # symbol -> timestamp
        self.min_fetch_interval = get_config_value('data.min_fetch_interval_seconds', 30)  # configurable throttling
        
        self.logger.info("Data fetcher initialized")
    
    def start_real_time_updates(self):
        """Start real-time data updates"""
        if self.update_thread and self.update_thread.is_alive():
            self.logger.warning("Real-time updates already running")
            return
        
        self.stop_updates = False
        self.update_thread = threading.Thread(target=self._update_loop, daemon=True)
        self.update_thread.start()
        
        self.logger.info("Real-time data updates started")
    
    def stop_real_time_updates(self):
        """Stop real-time data updates"""
        self.stop_updates = True
        if self.update_thread and self.update_thread.is_alive():
            self.update_thread.join(timeout=10)
        
        self.logger.info("Real-time data updates stopped")
    
    def _should_fetch_data(self, symbol: str) -> bool:
        """Check if we should fetch data for symbol based on throttling rules"""
        current_time = time.time()
        last_fetch = self.last_fetch_times.get(symbol, 0)
        
        time_since_last = current_time - last_fetch
        
        # For 4h timeframe, enforce minimum interval to prevent spam
        if time_since_last < self.min_fetch_interval:
            self.logger.debug(f"Throttling fetch for {symbol}: {time_since_last:.1f}s < {self.min_fetch_interval}s")
            return False
            
        return True
    
    def _record_fetch_time(self, symbol: str):
        """Record the fetch time for a symbol"""
        self.last_fetch_times[symbol] = time.time()

    def _update_loop(self):
        """Main update loop for real-time data with throttling"""
        while not self.stop_updates:
            try:
                # Update latest prices for all symbols
                for symbol in self.symbols:
                    try:
                        # Check throttling before updating
                        if not self._should_fetch_data(symbol):
                            continue
                            
                        self._update_symbol_price(symbol)
                        self._record_fetch_time(symbol)
                        
                        # Update historical data less frequently (every 4 hours for 4h timeframe)
                        if self._should_update_historical(symbol):
                            self._update_historical_data(symbol)
                    
                    except Exception as e:
                        self.logger.error(f"Error updating data for {symbol}: {e}")
                
                time.sleep(self.update_interval)
                
            except Exception as e:
                self.logger.error(f"Error in data update loop: {e}")
                time.sleep(5)
    
    def _update_symbol_price(self, symbol: str):
        """Update current price for a symbol"""
        try:
            ticker = self.api.get_ticker(symbol)
            current_price = float(ticker.get('last', 0))
            
            if current_price > 0:
                self.latest_prices[symbol] = {
                    'price': current_price,
                    'timestamp': datetime.now(),
                    'bid': float(ticker.get('buy', current_price)),
                    'ask': float(ticker.get('sell', current_price)),
                    'volume': float(ticker.get('vol', 0))
                }
        
        except Exception as e:
            self.logger.error(f"Error updating price for {symbol}: {e}")
    
    def _should_update_historical(self, symbol: str) -> bool:
        """Check if historical data should be updated"""
        try:
            # For 4-hour timeframe, update at the beginning of each 4-hour period
            current_time = datetime.now()
            
            # Check if we're at a 4-hour boundary
            if current_time.hour % 4 == 0 and current_time.minute < 5:
                return True
            
            # Also update if we don't have recent data in cache
            if symbol not in self.latest_data_cache:
                return True
            
            last_update = self.latest_data_cache.get(f"{symbol}_last_update")
            if not last_update or (current_time - last_update).total_seconds() > 3600:  # 1 hour
                return True
            
            return False
            
        except Exception as e:
            self.logger.error(f"Error checking update requirement for {symbol}: {e}")
            return False
    
    def _update_historical_data(self, symbol: str):
        """Update historical candlestick data"""
        try:
            self.logger.info(f"Updating historical data for {symbol}")
            
            # Get recent kline data from API (reduced to 5 to avoid API pressure)
            kline_data = self.api.get_kline_data(symbol, self.timeframe, limit=5)
            
            if not kline_data:
                self.logger.warning(f"No kline data received for {symbol}")
                return
            
            # Store in database
            session = db_connection.get_session()
            
            for candle_data in kline_data:
                try:
                    timestamp = candle_data['timestamp']
                    
                    # For 4h timeframe, only process aligned candles
                    if self.timeframe == '4h' and not self._is_aligned_4h(timestamp):
                        continue
                    
                    # Check if candle already exists
                    existing_candle = session.query(Candle).filter(
                        Candle.symbol == symbol,
                        Candle.timestamp == timestamp
                    ).first()
                    
                    if not existing_candle:
                        # Create new candle
                        candle = Candle(
                            symbol=symbol,
                            timestamp=timestamp,
                            open=candle_data['open'],
                            high=candle_data['high'],
                            low=candle_data['low'],
                            close=candle_data['close'],
                            volume=candle_data['volume']
                        )
                        session.add(candle)
                    else:
                        # Update existing candle (in case it's the current incomplete candle)
                        existing_candle.high = max(existing_candle.high, candle_data['high'])
                        existing_candle.low = min(existing_candle.low, candle_data['low'])
                        existing_candle.close = candle_data['close']
                        existing_candle.volume = candle_data['volume']
                
                except Exception as e:
                    self.logger.error(f"Error processing candle data for {symbol}: {e}")
            
            session.commit()
            session.close()
            
            # Update cache timestamp
            self.latest_data_cache[f"{symbol}_last_update"] = datetime.now()
            
            self.logger.info(f"Historical data updated for {symbol}: {len(kline_data)} candles")
            
        except Exception as e:
            self.logger.error(f"Error updating historical data for {symbol}: {e}")
    
    def get_latest_price(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Get latest price for symbol"""
        return self.latest_prices.get(symbol)
    
    def get_historical_data(self, symbol: str, limit: int = 1000) -> pd.DataFrame:
        """Get historical data from database"""
        try:
            session = db_connection.get_session()
            
            # Query candles for symbol, ordered by timestamp descending
            query = session.query(Candle).filter(
                Candle.symbol == symbol
            ).order_by(Candle.timestamp.desc()).limit(limit)
            
            candles = query.all()
            session.close()
            
            if not candles:
                self.logger.warning(f"No historical data found for {symbol}")
                return pd.DataFrame()
            
            # Convert to DataFrame
            data = pd.DataFrame([{
                'timestamp': candle.timestamp,
                'open': candle.open,
                'high': candle.high,
                'low': candle.low,
                'close': candle.close,
                'volume': candle.volume
            } for candle in candles])
            
            # Sort by timestamp (oldest first)
            data = data.sort_values('timestamp').reset_index(drop=True)
            
            return data
            
        except Exception as e:
            self.logger.error(f"Error getting historical data for {symbol}: {e}")
            return pd.DataFrame()
    
    def get_latest_data_with_indicators(self, symbol: str, lookback: int = 200) -> Optional[pd.DataFrame]:
        """
        Get latest data with all technical indicators calculated
        
        Args:
            symbol: Trading symbol
            lookback: Number of periods to include for indicator calculation
            
        Returns:
            DataFrame with OHLCV data and all indicators
        """
        try:
            # Check cache first
            cache_key = f"{symbol}_with_indicators"
            cache_time_key = f"{symbol}_indicators_time"
            
            # Use cache if recent (within 1 minute)
            if (cache_key in self.latest_data_cache and 
                cache_time_key in self.latest_data_cache and
                (datetime.now() - self.latest_data_cache[cache_time_key]).total_seconds() < 60):
                
                return self.latest_data_cache[cache_key]
            
            # Get historical data
            historical_data = self.get_historical_data(symbol, lookback)
            
            if historical_data.empty:
                return None
            
            # For 4h timeframe, filter to only aligned 4h candles
            if self.timeframe == '4h':
                aligned_mask = historical_data['timestamp'].apply(self._is_aligned_4h)
                historical_data = historical_data[aligned_mask].reset_index(drop=True)
                
                if historical_data.empty:
                    self.logger.warning(f"No aligned 4h candles found for {symbol}")
                    return None
            
            # For 4h timeframe, do NOT add synthetic current price row
            # Only add current price for shorter timeframes
            if self.timeframe != '4h':
                current_price_info = self.get_latest_price(symbol)
                if current_price_info:
                    current_time = int(current_price_info['timestamp'].timestamp())
                    current_price = current_price_info['price']
                    
                    # Check if we need to add current data point
                    last_timestamp = historical_data['timestamp'].iloc[-1]
                    
                    # If current time is significantly different from last timestamp, add current point
                    if current_time - last_timestamp > 3600:  # More than 1 hour difference
                        current_row = {
                            'timestamp': current_time,
                            'open': current_price,
                            'high': current_price,
                            'low': current_price,
                            'close': current_price,
                            'volume': current_price_info.get('volume', 0)
                        }
                        historical_data = pd.concat([historical_data, pd.DataFrame([current_row])], ignore_index=True)
            
            # Calculate all technical indicators
            data_with_indicators = self.calculator.calculate_all_indicators(historical_data)
            
            # Cache the result
            self.latest_data_cache[cache_key] = data_with_indicators
            self.latest_data_cache[cache_time_key] = datetime.now()
            
            return data_with_indicators
            
        except Exception as e:
            self.logger.error(f"Error getting data with indicators for {symbol}: {e}")
            return None
    
    def get_recent_performance_data(self, symbol: str, days: int = 30) -> pd.DataFrame:
        """Get recent performance data for analysis"""
        try:
            # Calculate timestamp for N days ago
            days_ago = datetime.now() - timedelta(days=days)
            timestamp_ago = int(days_ago.timestamp())
            
            session = db_connection.get_session()
            
            query = session.query(Candle).filter(
                Candle.symbol == symbol,
                Candle.timestamp >= timestamp_ago
            ).order_by(Candle.timestamp.asc())
            
            candles = query.all()
            session.close()
            
            if not candles:
                return pd.DataFrame()
            
            # Convert to DataFrame
            data = pd.DataFrame([{
                'timestamp': candle.timestamp,
                'open': candle.open,
                'high': candle.high,
                'low': candle.low,
                'close': candle.close,
                'volume': candle.volume,
                'date': datetime.fromtimestamp(candle.timestamp)
            } for candle in candles])
            
            return data
            
        except Exception as e:
            self.logger.error(f"Error getting recent performance data for {symbol}: {e}")
            return pd.DataFrame()
    
    def _is_aligned_4h(self, timestamp: int) -> bool:
        """Check if timestamp is aligned to 4-hour boundary"""
        try:
            dt = datetime.fromtimestamp(timestamp)
            # 4h aligned means hour is divisible by 4 and minute/second are 0
            return dt.hour % 4 == 0 and dt.minute == 0 and dt.second == 0
        except Exception:
            return False
    
    def count_4h_candles(self, symbol: str) -> int:
        """Count number of properly aligned 4h candles for symbol"""
        try:
            session = db_connection.get_session()
            
            candles = session.query(Candle).filter(
                Candle.symbol == symbol
            ).order_by(Candle.timestamp.asc()).all()
            
            session.close()
            
            aligned_count = 0
            for candle in candles:
                if self._is_aligned_4h(candle.timestamp):
                    aligned_count += 1
            
            return aligned_count
            
        except Exception as e:
            self.logger.error(f"Error counting 4h candles for {symbol}: {e}")
            return 0
    
    def backfill_4h(self, symbol: str, min_candles: int = None) -> bool:
        """Backfill 4h aligned candles to ensure minimum count"""
        try:
            if min_candles is None:
                min_candles = DATA_CONFIG.get('min_4h_candles', 800)
            
            current_count = self.count_4h_candles(symbol)
            self.logger.info(f"Current 4h candles for {symbol}: {current_count}")
            
            if current_count >= min_candles:
                self.logger.info(f"Sufficient 4h candles for {symbol} ({current_count} >= {min_candles})")
                return True
            
            # Calculate how many more candles we need
            needed_candles = min_candles - current_count
            
            # For 4h timeframe, we need to go back needed_candles * 4 hours
            hours_back = needed_candles * 4
            
            self.logger.info(f"Backfilling {needed_candles} 4h candles for {symbol} (going back {hours_back} hours)")
            
            # Try to get historical data from API
            # Note: This is a simplified implementation - in production you might need
            # multiple API calls due to limits
            try:
                kline_data = self.api.get_kline_data(symbol, '4h', limit=min(1000, needed_candles))
                
                if not kline_data:
                    self.logger.warning(f"No backfill data received from API for {symbol}")
                    return False
                
                session = db_connection.get_session()
                added_count = 0
                
                for candle_data in kline_data:
                    timestamp = candle_data['timestamp']
                    
                    # Only add if it's 4h aligned
                    if not self._is_aligned_4h(timestamp):
                        continue
                    
                    # Check if candle already exists
                    existing_candle = session.query(Candle).filter(
                        Candle.symbol == symbol,
                        Candle.timestamp == timestamp
                    ).first()
                    
                    if not existing_candle:
                        candle = Candle(
                            symbol=symbol,
                            timestamp=timestamp,
                            open=candle_data['open'],
                            high=candle_data['high'],
                            low=candle_data['low'],
                            close=candle_data['close'],
                            volume=candle_data['volume']
                        )
                        session.add(candle)
                        added_count += 1
                
                session.commit()
                session.close()
                
                self.logger.info(f"Backfilled {added_count} 4h candles for {symbol}")
                
                # Check if we now have enough
                final_count = self.count_4h_candles(symbol)
                if final_count >= min_candles:
                    return True
                else:
                    self.logger.warning(f"Still insufficient 4h candles for {symbol} after backfill: {final_count} < {min_candles}")
                    return False
                    
            except Exception as api_error:
                self.logger.error(f"API error during backfill for {symbol}: {api_error}")
                return False
            
        except Exception as e:
            self.logger.error(f"Error backfilling 4h data for {symbol}: {e}")
            return False
    
    def validate_data_quality(self, symbol: str) -> Dict[str, Any]:
        """Validate data quality for a symbol"""
        try:
            data = self.get_historical_data(symbol, limit=1000)
            
            if data.empty:
                return {
                    'valid': False,
                    'reason': 'No data available'
                }
            
            # Check for gaps in data
            timestamps = data['timestamp'].sort_values()
            time_diffs = timestamps.diff().dropna()
            
            # For 4-hour timeframe, expect ~14400 seconds (4 hours) between candles
            expected_interval = 14400
            large_gaps = time_diffs[time_diffs > expected_interval * 2]
            
            # Check for price anomalies
            price_changes = data['close'].pct_change().abs()
            extreme_changes = price_changes[price_changes > 0.5]  # More than 50% change
            
            # Check for volume anomalies
            volume_zeros = (data['volume'] == 0).sum()
            
            quality_report = {
                'valid': True,
                'total_records': len(data),
                'date_range': {
                    'start': datetime.fromtimestamp(data['timestamp'].min()),
                    'end': datetime.fromtimestamp(data['timestamp'].max())
                },
                'large_gaps': len(large_gaps),
                'extreme_price_changes': len(extreme_changes),
                'zero_volume_candles': volume_zeros,
                'data_completeness': (len(data) - len(large_gaps)) / len(data) * 100
            }
            
            # Mark as invalid if data quality is poor
            if (quality_report['large_gaps'] > len(data) * 0.1 or  # More than 10% gaps
                quality_report['extreme_price_changes'] > len(data) * 0.05):  # More than 5% extreme changes
                quality_report['valid'] = False
                quality_report['reason'] = 'Poor data quality detected'
            
            return quality_report
            
        except Exception as e:
            self.logger.error(f"Error validating data quality for {symbol}: {e}")
            return {
                'valid': False,
                'reason': f'Validation error: {str(e)}'
            }
    
    def get_market_overview(self) -> Dict[str, Any]:
        """Get market overview for all symbols"""
        try:
            overview = {}
            
            for symbol in self.symbols:
                price_info = self.get_latest_price(symbol)
                if price_info:
                    # Get 24h change
                    recent_data = self.get_recent_performance_data(symbol, days=1)
                    price_change_24h = 0.0
                    
                    if not recent_data.empty and len(recent_data) > 1:
                        price_24h_ago = recent_data['close'].iloc[0]
                        current_price = price_info['price']
                        price_change_24h = ((current_price - price_24h_ago) / price_24h_ago) * 100
                    
                    overview[symbol] = {
                        'price': price_info['price'],
                        'change_24h': price_change_24h,
                        'volume': price_info.get('volume', 0),
                        'last_update': price_info['timestamp']
                    }
            
            return overview
            
        except Exception as e:
            self.logger.error(f"Error getting market overview: {e}")
            return {}
    
    def force_data_refresh(self, symbol: str = None):
        """Force refresh of data for symbol or all symbols"""
        try:
            symbols_to_refresh = [symbol] if symbol else self.symbols
            
            for sym in symbols_to_refresh:
                # Clear cache
                cache_keys_to_clear = [key for key in self.latest_data_cache.keys() if sym in key]
                for key in cache_keys_to_clear:
                    del self.latest_data_cache[key]
                
                # Force update
                self._update_symbol_price(sym)
                self._update_historical_data(sym)
            
            self.logger.info(f"Data refreshed for: {symbols_to_refresh}")
            
        except Exception as e:
            self.logger.error(f"Error forcing data refresh: {e}")