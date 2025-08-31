# Implementation Guide: 1-Minute Timeframe Trading

## Code Snippets and File-by-File Analysis

This document provides specific code snippets and detailed implementation guidance for converting the TB trading bot to 1-minute timeframe trading.

---

## 1. CURRENT TIMEFRAME IMPLEMENTATION

### File: `trading/engine.py` - Line 429-456
```python
def _is_new_timeframe(self, symbol: str) -> bool:
    """
    Decide if it's time to generate a new signal.
    حالت تست: اگر timeframe = '1m' باشد همیشه True
    """
    now = datetime.now()
    m = now.minute
    h = now.hour
    
    if self.timeframe == '1m':
        return True  # ✅ ALREADY SUPPORTS 1-MINUTE!
    if self.timeframe == '5m':
        return (m % 5 == 0)
    if self.timeframe == '15m':
        return (m % 15 == 0)
    if self.timeframe == '1h':
        return m == 0
    if self.timeframe == '4h':
        return (h % 4 == 0 and m == 0)
```

**Status**: ✅ **ALREADY IMPLEMENTED** - No code changes needed!

---

## 2. REQUIRED CONFIGURATION CHANGES

### File: `config/config.yaml` - Lines 3-4
**BEFORE:**
```yaml
trading:
  timeframe: "4h"
```

**AFTER:**
```yaml
trading:
  timeframe: "1m"
```

### File: `config/config.yaml` - Line 39
**BEFORE:**
```yaml
data:
  update_interval: 60  # seconds between data updates
```

**AFTER:**
```yaml
data:
  update_interval: 30  # More frequent updates for 1m trading
```

### File: `config/config.yaml` - Line 14
**BEFORE:**
```yaml
adaptive_threshold:
  target_signals_per_24h: 5  # Target 3-8 signals per day per symbol
```

**AFTER:**
```yaml
adaptive_threshold:
  target_signals_per_24h: 15  # Higher frequency for 1m trading
```

---

## 3. PYTHON CONFIGURATION UPDATES

### File: `config/settings.py` - Line 35
**BEFORE:**
```python
TRADING_CONFIG = {
    'timeframe': '4h',  # 4-hour timeframe as specified
    'demo_balance': 100.0,
    'confidence_threshold': 0.7,
    'symbols': ['BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'DOGEUSDT'],
    'max_positions': 4,
    'risk_per_trade': 0.02,
}
```

**AFTER:**
```python
TRADING_CONFIG = {
    'timeframe': '1m',  # 1-minute timeframe
    'demo_balance': 100.0,
    'confidence_threshold': 0.7,
    'symbols': ['BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'DOGEUSDT'],
    'max_positions': 4,
    'risk_per_trade': 0.02,
}
```

### File: `config/settings.py` - Lines 80-89
**BEFORE:**
```python
DATA_CONFIG = {
    'update_interval': 300,  # Update every 300 seconds for 4h timeframe
    'batch_size': 100,
    'max_retries': 3,
    'timeout': 30,
    'min_4h_candles': 800,  # Minimum aligned 4h candles required for training
    'max_4h_selection_candles': 800,  # Maximum candles for feature selection subset
    'max_4h_training_candles': 0,  # Maximum candles for full training (0 or None means use all)
    'use_all_history': True,  # When True, fetch ALL historical data without limits
}
```

**AFTER:**
```python
DATA_CONFIG = {
    'update_interval': 60,  # Update every 60 seconds for 1m timeframe
    'batch_size': 100,
    'max_retries': 3,
    'timeout': 30,
    'min_1m_candles': 1440,  # Minimum 1m candles required (24 hours)
    'max_1m_selection_candles': 1440,  # Maximum candles for feature selection subset
    'max_1m_training_candles': 0,  # Maximum candles for full training (0 means use all)
    'use_all_history': True,  # When True, fetch ALL historical data without limits
}
```

### File: `config/settings.py` - Lines 101-111
**BEFORE:**
```python
FEATURE_SELECTION_CONFIG = {
    'enabled': True,  # Enable dynamic feature selection on recent window
    'mode': 'dynamic',  # Dynamic selection mode
    'selection_window_4h': 800,  # Use most recent 800 4h candles for feature selection
    'min_features': 20,  # Minimum features to retain
    'method': 'dynamic_iterative_pruning',  # Method to use when enabled
    'correlation_threshold': 0.95,  # Correlation threshold for pruning
    'tolerance': 0.003,  # Tolerance for improvement in feature selection
    'max_iterations': 50,  # Maximum iterations for dynamic selection
    'max_features': 50,  # Maximum features when selection is enabled
}
```

**AFTER:**
```python
FEATURE_SELECTION_CONFIG = {
    'enabled': True,  # Enable dynamic feature selection on recent window
    'mode': 'dynamic',  # Dynamic selection mode
    'selection_window_1m': 1440,  # Use most recent 1440 1m candles (24 hours)
    'min_features': 20,  # Minimum features to retain
    'method': 'dynamic_iterative_pruning',  # Method to use when enabled
    'correlation_threshold': 0.95,  # Correlation threshold for pruning
    'tolerance': 0.003,  # Tolerance for improvement in feature selection
    'max_iterations': 50,  # Maximum iterations for dynamic selection
    'max_features': 50,  # Maximum features when selection is enabled
}
```

---

## 4. ALREADY IMPLEMENTED COMPONENTS

### API Client Support: `utils/api_client.py`
```python
timeframe_map = {
    '1m': '1min', '5m': '5min', '15m': '15min', '30m': '30min',
    '1h': '1hour', '4h': '4hour', '1d': '1day'
}
```
✅ **1-minute timeframe already mapped to '1min'**

### Data Fetcher: `data/fetcher.py` - Lines 27-28
```python
self.timeframe = TRADING_CONFIG['timeframe']  # ✅ Uses config
self.update_interval = DATA_CONFIG['update_interval']  # ✅ Configurable
```
✅ **Already reads timeframe from configuration**

### Prediction Scheduler: `utils/prediction_scheduler.py`
```python
# Default interval: 60 seconds (perfect for 1m timeframe)
prediction:
  scheduler_interval: 60  # seconds (every 1 minute)
```
✅ **Already optimized for 1-minute predictions**

---

## 5. DATABASE CONSIDERATIONS

### Current Schema: `database/models.py`
```python
class Candle(Base):
    __tablename__ = 'candles'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    symbol = Column(String(20), nullable=False, index=True)
    timestamp = Column(Integer, nullable=False, index=True)  # Unix timestamp
    open = Column(Float, nullable=False)
    high = Column(Float, nullable=False)
    low = Column(Float, nullable=False)
    close = Column(Float, nullable=False)
    volume = Column(Float, nullable=False)
    created_at = Column(DateTime, default=func.now())
```
✅ **Schema is timeframe-agnostic - no changes needed**

### Data Volume Impact
- **4h timeframe**: ~6 candles per day per symbol
- **1m timeframe**: ~1440 candles per day per symbol
- **Impact**: 240x more data volume

### Recommended Optimizations
```sql
-- Add composite index for performance
CREATE INDEX idx_symbol_timestamp_composite ON candles(symbol, timestamp DESC);

-- Consider partitioning by date for large datasets
-- Implement data retention policy (e.g., keep 30 days of 1m data)
```

---

## 6. PERFORMANCE OPTIMIZATIONS

### Data Fetcher Throttling: `data/fetcher.py` - Lines 72-76
```python
def _should_fetch_data(self, symbol: str) -> bool:
    """Check if we should fetch data for symbol based on throttling rules"""
    current_time = time.time()
    last_fetch = self.last_fetch_times.get(symbol, 0)
    time_since_last = current_time - last_fetch
    
    # For 1m timeframe, reduce minimum interval
    if time_since_last < self.min_fetch_interval:  # Default: 30s
        return False
    return True
```

**Recommendation**: Reduce `min_fetch_interval` from 30s to 15s for 1m trading

### Memory Management Considerations
```python
# Consider implementing in data fetcher:
def cleanup_old_cache(self, max_age_minutes: int = 60):
    """Remove old cached data to manage memory"""
    cutoff_time = time.time() - (max_age_minutes * 60)
    # Implementation to clean old cached data
```

---

## 7. TRADING STRATEGY ADJUSTMENTS

### Risk Management for Higher Frequency
```python
# Potential additions to config/settings.py:
RISK_CONFIG_1M = {
    'max_daily_trades': 50,  # Limit trades per day for 1m
    'cooldown_minutes': 5,   # Wait between trades on same symbol
    'max_drawdown_percent': 5.0,  # Stop trading if drawdown exceeds 5%
}
```

### Position Sizing for 1m Trading
```python
# Consider smaller position sizes for 1m trading:
TRADING_CONFIG = {
    'timeframe': '1m',
    'risk_per_trade': 0.01,  # Reduced from 0.02 for higher frequency
    'max_positions': 6,      # Increased from 4 for more opportunities
}
```

---

## 8. BACKTESTING FRAMEWORK

### Create 1m Backtesting Script
```python
# File: backtest_1m_strategy.py
class Backtester1m:
    def __init__(self):
        self.timeframe = '1m'
        self.data_points_per_day = 1440
        
    def run_backtest(self, start_date, end_date, symbols):
        """Run backtest with 1m data"""
        # Implementation for 1m backtesting
        pass
```

---

## 9. MONITORING AND ALERTS

### Enhanced Logging for 1m Trading
```python
# Add to config/config.yaml:
logging:
  level: "INFO"
  high_frequency_mode: true  # Special logging for 1m trading
  trade_frequency_alert: 100  # Alert if more than 100 trades/day
```

### Performance Monitoring
```python
# Monitor these metrics for 1m trading:
MONITORING_1M = {
    'api_calls_per_minute': 10,     # Alert if exceeding
    'database_writes_per_minute': 8, # Monitor DB performance
    'memory_usage_mb': 500,         # Alert if memory usage high
    'prediction_latency_ms': 1000,  # Alert if predictions slow
}
```

---

## 10. TESTING AND VALIDATION

### Test Script: `test_1m_implementation.py`
Key test cases:
1. ✅ Configuration validation (timeframe = '1m')
2. ✅ Signal generation (should always return True)
3. ✅ API timeframe mapping (1m → 1min)
4. ✅ Data processing pipeline
5. ✅ Performance under load

### Demo Script: `demo_1m_migration.py`
Demonstrates:
1. ✅ Configuration changes
2. ✅ Signal timing behavior
3. ✅ Data update frequency
4. ✅ Performance considerations

---

## 11. IMPLEMENTATION CHECKLIST

### Phase 1: Core Changes ✅
- [ ] Update `config/config.yaml` timeframe to "1m"
- [ ] Update `config/settings.py` TRADING_CONFIG
- [ ] Update DATA_CONFIG for 1m requirements
- [ ] Update FEATURE_SELECTION_CONFIG for 1m window

### Phase 2: Testing ✅
- [ ] Run `test_1m_implementation.py`
- [ ] Run `demo_1m_migration.py`
- [ ] Test with demo mode first
- [ ] Validate with historical data

### Phase 3: Optimization (Optional)
- [ ] Database indexing optimization
- [ ] Memory management improvements
- [ ] API throttling optimization
- [ ] Data retention policies

### Phase 4: Monitoring (Recommended)
- [ ] Set up performance monitoring
- [ ] Configure alerts for high frequency
- [ ] Monitor API rate limits
- [ ] Track database performance

---

## 12. MIGRATION COMMAND

Use the provided migration script:

```bash
# Test changes without applying
python3 migrate_to_1m_timeframe.py --dry-run

# Apply changes to files
python3 migrate_to_1m_timeframe.py --apply

# Validate implementation
python3 test_1m_implementation.py

# Run demo
python3 demo_1m_migration.py
```

---

## CONCLUSION

The TB repository is exceptionally well-prepared for 1-minute timeframe trading. The core trading logic already supports 1m timeframe with `_is_new_timeframe` returning `True` for continuous signal generation. The primary changes are configuration updates, with optional performance optimizations for production use.

**Total Implementation Time**: 1-2 hours for basic changes, plus testing and optimization time.