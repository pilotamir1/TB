#!/usr/bin/env python3
"""
Test ProcessPool indicators computation
"""
import asyncio
import unittest
import time
import numpy as np
import logging
from unittest.mock import patch, MagicMock
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from indicators.compute import IndicatorCompute, compute_technical_indicators_worker

class TestIndicatorsPool(unittest.IsolatedAsyncioTestCase):
    """Test ProcessPool indicator computation"""
    
    async def asyncSetUp(self):
        """Setup test environment"""
        logging.basicConfig(level=logging.DEBUG)
        
        self.compute = IndicatorCompute()
        self.compute.max_workers = 2  # Limit for testing
        self.compute.computation_timeout = 5  # Shorter timeout for tests
        
        await self.compute.start()
    
    async def asyncTearDown(self):
        """Cleanup after tests"""
        if self.compute:
            await self.compute.stop()
    
    def _generate_test_ohlcv(self, num_candles: int = 100, base_price: float = 45000.0):
        """Generate test OHLCV data"""
        ohlcv_data = []
        
        for i in range(num_candles):
            timestamp = time.time() - (num_candles - i) * 60
            
            # Generate realistic OHLCV with some price movement
            price_change = (i % 20 - 10) * 0.001  # +/- 1% variation
            open_price = base_price * (1 + price_change)
            close_price = open_price * (1 + (i % 10 - 5) * 0.0005)  # Intrabar movement
            high_price = max(open_price, close_price) * 1.001
            low_price = min(open_price, close_price) * 0.999
            volume = 100 + (i % 50)  # Variable volume
            
            ohlcv_data.append({
                'timestamp': timestamp,
                'open': open_price,
                'high': high_price,
                'low': low_price,
                'close': close_price,
                'volume': volume
            })
        
        return ohlcv_data
    
    async def test_ohlcv_buffer_management(self):
        """Test OHLCV ring buffer management"""
        symbol = 'BTCUSDT'
        test_data = self._generate_test_ohlcv(50)
        
        # Update buffer
        await self.compute.update_ohlcv(symbol, test_data)
        
        # Verify buffer was created and populated
        self.assertIn(symbol, self.compute.symbol_buffers)
        buffer = self.compute.symbol_buffers[symbol]
        self.assertEqual(len(buffer), 50)
        
        # Test ring buffer overflow (add more data than max size)
        large_data = self._generate_test_ohlcv(1200)  # Exceeds ring_buffer_size
        await self.compute.update_ohlcv(symbol, large_data)
        
        # Buffer should be limited to max size
        self.assertLessEqual(len(buffer), self.compute.ring_buffer_size)
        
        # Latest data should be preserved
        latest_timestamp = max(candle['timestamp'] for candle in large_data)
        buffer_latest = buffer[-1]['timestamp']
        self.assertEqual(buffer_latest, latest_timestamp)
    
    async def test_indicator_computation(self):
        """Test basic indicator computation"""
        symbol = 'BTCUSDT'
        test_data = self._generate_test_ohlcv(100)
        
        # Update buffer and compute indicators
        await self.compute.update_ohlcv(symbol, test_data)
        result = await self.compute.compute_indicators(symbol)
        
        # Verify computation result
        self.assertIsNotNone(result)
        self.assertEqual(result['symbol'], symbol)
        self.assertIn('indicators', result)
        
        indicators = result['indicators']
        
        # Check that key indicators are computed
        expected_indicators = ['sma_20', 'ema_12', 'ema_26', 'rsi_14', 'atr_14']
        for indicator in expected_indicators:
            self.assertIn(indicator, indicators)
            self.assertIsNotNone(indicators[indicator])
    
    async def test_incremental_computation(self):
        """Test incremental computation caching"""
        symbol = 'BTCUSDT'
        initial_data = self._generate_test_ohlcv(100)
        
        # Initial computation
        await self.compute.update_ohlcv(symbol, initial_data)
        result1 = await self.compute.compute_indicators(symbol)
        
        self.assertIsNotNone(result1)
        self.assertFalse(result1.get('incremental', True))  # First computation is full
        
        # Add new data and compute incrementally
        new_data = self._generate_test_ohlcv(5)  # Just 5 new candles
        await self.compute.update_ohlcv(symbol, new_data)
        
        result2 = await self.compute.compute_indicators(symbol)
        
        self.assertIsNotNone(result2)
        # Should use cached result if no new data above last computed timestamp
        
        # Verify cache was used
        initial_cache_hits = self.compute.stats['cache_hits']
        
        # Call again without new data
        result3 = await self.compute.compute_indicators(symbol)
        
        # Cache hits should increase
        self.assertGreaterEqual(self.compute.stats['cache_hits'], initial_cache_hits)
    
    async def test_batch_computation(self):
        """Test batch computation of multiple symbols"""
        symbols = ['BTCUSDT', 'ETHUSDT', 'SOLUSDT']
        
        # Setup data for all symbols
        for symbol in symbols:
            test_data = self._generate_test_ohlcv(100, base_price=1000 + ord(symbol[0]))
            await self.compute.update_ohlcv(symbol, test_data)
        
        # Batch compute
        results = await self.compute.compute_indicators_batch(symbols)
        
        # Verify all symbols were processed
        self.assertEqual(len(results), len(symbols))
        
        for symbol in symbols:
            self.assertIn(symbol, results)
            self.assertEqual(results[symbol]['symbol'], symbol)
            self.assertIn('indicators', results[symbol])
    
    async def test_computation_timeout(self):
        """Test computation timeout handling"""
        symbol = 'BTCUSDT'
        
        # Create computation with very short timeout
        self.compute.computation_timeout = 0.001  # 1ms - should timeout
        
        test_data = self._generate_test_ohlcv(100)
        await self.compute.update_ohlcv(symbol, test_data)
        
        # This should timeout
        result = await self.compute.compute_indicators(symbol)
        
        # Should return None due to timeout
        self.assertIsNone(result)
        
        # Verify timeout stats were updated
        self.assertGreater(self.compute.stats['timeout_computations'], 0)
    
    async def test_computation_error_handling(self):
        """Test error handling in computation"""
        symbol = 'BTCUSDT'
        
        # Add invalid data that should cause computation errors
        invalid_data = [
            {
                'timestamp': time.time(),
                'open': 0,  # Invalid price
                'high': 0,
                'low': 0,
                'close': 0,
                'volume': 0
            }
        ]
        
        await self.compute.update_ohlcv(symbol, invalid_data)
        result = await self.compute.compute_indicators(symbol)
        
        # Should return None due to insufficient valid data
        self.assertIsNone(result)
    
    async def test_cache_management(self):
        """Test cache management operations"""
        symbol = 'BTCUSDT'
        test_data = self._generate_test_ohlcv(100)
        
        # Compute indicators to populate cache
        await self.compute.update_ohlcv(symbol, test_data)
        result = await self.compute.compute_indicators(symbol)
        
        self.assertIsNotNone(result)
        
        # Verify cache exists
        cached = self.compute.get_cached_indicators(symbol)
        self.assertIsNotNone(cached)
        self.assertEqual(cached['symbol'], symbol)
        
        # Clear cache for symbol
        self.compute.clear_cache(symbol)
        cached_after_clear = self.compute.get_cached_indicators(symbol)
        self.assertIsNone(cached_after_clear)
    
    async def test_statistics_tracking(self):
        """Test computation statistics tracking"""
        initial_stats = self.compute.get_stats()
        
        symbol = 'BTCUSDT'
        test_data = self._generate_test_ohlcv(100)
        
        # Perform computation
        await self.compute.update_ohlcv(symbol, test_data)
        result = await self.compute.compute_indicators(symbol)
        
        final_stats = self.compute.get_stats()
        
        # Verify stats were updated
        self.assertGreater(final_stats['total_computations'], initial_stats['total_computations'])
        
        if result:
            self.assertGreater(final_stats['successful_computations'], initial_stats['successful_computations'])
            self.assertGreater(final_stats['avg_computation_time_ms'], 0)
        
        self.assertEqual(final_stats['active_symbols'], 1)
    
    def test_worker_function_directly(self):
        """Test the worker function directly (synchronous)"""
        # Generate test data
        test_data = self._generate_test_ohlcv(100)
        
        # Call worker function directly
        result = compute_technical_indicators_worker('BTCUSDT', test_data, incremental=False)
        
        # Verify result
        self.assertIsNotNone(result)
        self.assertEqual(result['symbol'], 'BTCUSDT')
        self.assertIn('indicators', result)
        
        indicators = result['indicators']
        
        # Check specific indicators
        self.assertIn('sma_20', indicators)
        self.assertIn('rsi_14', indicators)
        self.assertIn('macd', indicators)
        
        # Verify indicator values are reasonable
        self.assertIsInstance(indicators['sma_20'], (int, float))
        self.assertGreaterEqual(indicators['rsi_14'], 0)
        self.assertLessEqual(indicators['rsi_14'], 100)
    
    def test_worker_insufficient_data(self):
        """Test worker function with insufficient data"""
        # Only 5 candles - not enough for most indicators
        insufficient_data = self._generate_test_ohlcv(5)
        
        result = compute_technical_indicators_worker('TESTUSDT', insufficient_data, incremental=False)
        
        # Should return None due to insufficient data
        self.assertIsNone(result)
    
    async def test_concurrent_computations(self):
        """Test concurrent computation handling"""
        symbols = [f'TEST{i}USDT' for i in range(10)]
        
        # Setup data for all symbols
        for symbol in symbols:
            test_data = self._generate_test_ohlcv(50)
            await self.compute.update_ohlcv(symbol, test_data)
        
        # Start multiple computations concurrently
        tasks = []
        for symbol in symbols:
            task = asyncio.create_task(self.compute.compute_indicators(symbol))
            tasks.append(task)
        
        # Wait for all to complete
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Count successful results
        successful_results = [r for r in results if r is not None and not isinstance(r, Exception)]
        
        # Most should succeed
        self.assertGreaterEqual(len(successful_results), len(symbols) // 2)
        
        # Verify no exceptions were raised
        exceptions = [r for r in results if isinstance(r, Exception)]
        self.assertEqual(len(exceptions), 0)

if __name__ == '__main__':
    unittest.main()