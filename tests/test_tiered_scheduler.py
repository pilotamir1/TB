#!/usr/bin/env python3
"""
Test tiered scheduler functionality
"""
import asyncio
import unittest
import time
import logging
from unittest.mock import AsyncMock, MagicMock, patch
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scanner.scheduler import TieredScheduler

class TestTieredScheduler(unittest.IsolatedAsyncioTestCase):
    """Test tiered scanning scheduler"""
    
    async def asyncSetUp(self):
        """Setup test environment"""
        logging.basicConfig(level=logging.DEBUG)
        
        # Create scheduler with test configuration
        self.scheduler = TieredScheduler()
        
        # Override intervals for faster testing
        self.scheduler.tier1_interval = 0.1  # 100ms
        self.scheduler.tier2_interval = 0.2  # 200ms
        self.scheduler.tier1_size = 5  # Small tier for testing
        
        # Mock external dependencies
        self.scheduler.rest_fetcher = AsyncMock()
        self.scheduler.websocket = MagicMock()
        self.scheduler.websocket.is_connected = True
        
    async def asyncTearDown(self):
        """Cleanup after tests"""
        if self.scheduler and self.scheduler.is_running:
            await self.scheduler.stop()
    
    async def test_symbol_initialization(self):
        """Test symbol loading and initialization"""
        # Mock CoinEx API to return test symbols
        test_symbols = ['BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'DOGEUSDT', 'BNBUSDT']
        
        with patch('scanner.scheduler.CoinExAPI') as mock_api_class:
            mock_api = mock_api_class.return_value
            mock_api.get_coinmarketcap_available_symbols.return_value = test_symbols
            
            await self.scheduler._initialize_symbols()
        
        # Verify symbols were loaded
        self.assertEqual(self.scheduler.all_symbols, test_symbols)
        self.assertGreater(len(self.scheduler.tier1_symbols), 0)
    
    async def test_symbol_ranking(self):
        """Test symbol ranking logic"""
        # Setup test symbols and metrics
        test_symbols = ['BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'DOGEUSDT', 'BNBUSDT']
        self.scheduler.all_symbols = test_symbols
        
        # Mock metrics with different volumes/volatility
        self.scheduler.symbol_metrics = {
            'BTCUSDT': {'volume_24h': 1000000, 'volatility': 5.0},
            'ETHUSDT': {'volume_24h': 500000, 'volatility': 7.0},
            'SOLUSDT': {'volume_24h': 200000, 'volatility': 15.0},
            'DOGEUSDT': {'volume_24h': 800000, 'volatility': 10.0},
            'BNBUSDT': {'volume_24h': 300000, 'volatility': 6.0}
        }
        
        await self.scheduler._rank_symbols()
        
        # Verify ranking (BTC should be first due to highest composite score)
        self.assertEqual(self.scheduler.tier1_symbols[0], 'BTCUSDT')
        
        # Training symbols should be prioritized in tier1
        training_symbols = ['BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'DOGEUSDT']
        tier1_has_training = any(symbol in training_symbols for symbol in self.scheduler.tier1_symbols)
        self.assertTrue(tier1_has_training)
    
    async def test_tier_scheduling_timing(self):
        """Test that tier scheduling happens at correct intervals"""
        # Setup minimal scheduler state
        self.scheduler.tier1_symbols = ['BTCUSDT']
        self.scheduler.tier2_symbols = ['ETHUSDT']
        
        # Mock the scanning methods to track calls
        tier1_calls = []
        tier2_calls = []
        
        async def mock_scan_tier1():
            tier1_calls.append(time.time())
        
        async def mock_scan_tier2():
            tier2_calls.append(time.time())
        
        self.scheduler._scan_tier1_rest_fallback = mock_scan_tier1
        self.scheduler._scan_tier2_batch = mock_scan_tier2
        
        # Mock websocket as disconnected to force REST fallback
        self.scheduler.websocket.is_connected = False
        
        # Start scheduler
        await self.scheduler.start()
        
        # Let it run for a short period
        await asyncio.sleep(0.5)
        
        # Stop scheduler
        await self.scheduler.stop()
        
        # Verify tier1 was called more frequently than tier2
        self.assertGreater(len(tier1_calls), len(tier2_calls))
        
        # Verify timing intervals (approximately)
        if len(tier1_calls) >= 2:
            tier1_interval = tier1_calls[1] - tier1_calls[0]
            self.assertLess(tier1_interval, 0.15)  # Should be ~0.1s
        
        if len(tier2_calls) >= 2:
            tier2_interval = tier2_calls[1] - tier2_calls[0]
            self.assertLess(tier2_interval, 0.3)   # Should be ~0.2s
    
    async def test_price_update_processing(self):
        """Test price update processing and queuing"""
        symbol = 'BTCUSDT'
        price = 45000.0
        timestamp = time.time()
        
        # Process price update
        await self.scheduler._process_price_update(symbol, price, timestamp)
        
        # Verify price was queued
        price_update = await self.scheduler.get_price_update()
        
        self.assertIsNotNone(price_update)
        self.assertEqual(price_update['symbol'], symbol)
        self.assertEqual(price_update['price'], price)
        self.assertEqual(price_update['timestamp'], timestamp)
    
    async def test_signal_generation(self):
        """Test trading signal generation"""
        symbol = 'BTCUSDT'
        test_klines = [
            {'timestamp': time.time(), 'close': 45000.0},
            {'timestamp': time.time() - 60, 'close': 44900.0}
        ]
        
        # Trigger analysis
        await self.scheduler._trigger_analysis(symbol, test_klines)
        
        # Verify signal was generated
        signal = await self.scheduler.get_signal()
        
        self.assertIsNotNone(signal)
        self.assertEqual(signal['symbol'], symbol)
        self.assertEqual(signal['action'], 'analyze')
    
    async def test_websocket_fallback(self):
        """Test fallback to REST when WebSocket is unavailable"""
        # Setup test symbols
        self.scheduler.tier1_symbols = ['BTCUSDT', 'ETHUSDT']
        
        # Mock REST fetcher
        mock_tickers = {
            'BTCUSDT': {'last': '45000'},
            'ETHUSDT': {'last': '2500'}
        }
        self.scheduler.rest_fetcher.fetch_tickers_batch.return_value = mock_tickers
        
        # Disable websocket
        self.scheduler.websocket = None
        
        # Run tier1 fallback scan
        await self.scheduler._scan_tier1_rest_fallback()
        
        # Verify REST fetcher was called
        self.scheduler.rest_fetcher.fetch_tickers_batch.assert_called_once_with(['BTCUSDT', 'ETHUSDT'])
        
        # Verify price updates were generated
        updates_count = 0
        while True:
            update = await self.scheduler.get_price_update()
            if not update:
                break
            updates_count += 1
        
        self.assertEqual(updates_count, 2)  # One for each symbol
    
    async def test_queue_overflow_handling(self):
        """Test queue overflow handling"""
        # Fill price queue to capacity
        for i in range(1100):  # Exceed queue size
            await self.scheduler._process_price_update(f'TEST{i}', 100.0 + i, time.time())
        
        # Queue should not crash and should be at max capacity
        self.assertLessEqual(self.scheduler.price_queue.qsize(), 1000)
        
        # Should still be able to add new updates (old ones dropped)
        await self.scheduler._process_price_update('FINAL', 999.0, time.time())
        
        self.assertLessEqual(self.scheduler.price_queue.qsize(), 1000)
    
    async def test_performance_stats_tracking(self):
        """Test performance statistics tracking"""
        # Initialize stats
        initial_stats = self.scheduler.scan_stats.copy()
        
        # Simulate some activity
        self.scheduler._update_scan_stats('tier1', 0.05)
        self.scheduler._update_scan_stats('tier1', 0.03)
        self.scheduler._update_scan_stats('tier2', 0.15)
        
        # Check stats were updated
        final_stats = self.scheduler.scan_stats
        
        self.assertEqual(final_stats['tier1_scans'], initial_stats['tier1_scans'] + 2)
        self.assertEqual(final_stats['tier2_scans'], initial_stats['tier2_scans'] + 1)
        self.assertGreater(final_stats['tier1_avg_duration'], 0)
        self.assertGreater(final_stats['tier2_avg_duration'], 0)
    
    async def test_candle_close_detection(self):
        """Test candle close detection for analysis triggering"""
        symbol = 'BTCUSDT'
        
        # Setup test klines
        test_klines = [
            {'timestamp': 1000, 'close': 45000.0},
            {'timestamp': 2000, 'close': 45100.0},  # Latest
        ]
        
        # First call should trigger analysis (new candle)
        await self.scheduler._check_candle_close(symbol, 2000, test_klines)
        
        signal1 = await self.scheduler.get_signal()
        self.assertIsNotNone(signal1)
        
        # Second call with same timestamp should not trigger
        await self.scheduler._check_candle_close(symbol, 2000, test_klines)
        
        signal2 = await self.scheduler.get_signal()
        self.assertIsNone(signal2)  # No new signal
        
        # Third call with newer timestamp should trigger
        await self.scheduler._check_candle_close(symbol, 3000, test_klines)
        
        signal3 = await self.scheduler.get_signal()
        self.assertIsNotNone(signal3)

if __name__ == '__main__':
    unittest.main()