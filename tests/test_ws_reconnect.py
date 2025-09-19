#!/usr/bin/env python3
"""
Test WebSocket reconnection and backoff behavior
"""
import asyncio
import unittest
import logging
import time
from unittest.mock import AsyncMock, patch, MagicMock
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from trading.coinex_ws import CoinExWebSocket

class TestWebSocketReconnect(unittest.IsolatedAsyncioTestCase):
    """Test WebSocket reconnection and backoff logic"""
    
    async def asyncSetUp(self):
        """Setup test environment"""
        # Configure logging to see backoff behavior
        logging.basicConfig(level=logging.DEBUG)
        
        self.ws = CoinExWebSocket()
        self.ws.backoff_base = 0.1  # Speed up tests
        self.ws.backoff_max = 1.0   # Limit max backoff for tests
    
    async def asyncTearDown(self):
        """Cleanup after tests"""
        if self.ws:
            await self.ws.stop()
    
    async def test_backoff_calculation(self):
        """Test exponential backoff calculation"""
        # Test backoff progression
        expected_delays = [0.1, 0.2, 0.4, 0.8, 1.0, 1.0]  # Capped at max
        
        for i, expected in enumerate(expected_delays):
            self.ws.reconnect_count = i
            
            start_time = time.time()
            await self.ws._backoff_delay()
            actual_delay = time.time() - start_time
            
            # Allow some tolerance for timing
            self.assertGreaterEqual(actual_delay, expected * 0.8)
            self.assertLessEqual(actual_delay, expected * 1.2)
    
    @patch('websockets.connect')
    async def test_reconnection_attempts(self, mock_connect):
        """Test that reconnection is attempted with proper backoff"""
        # Simulate connection failures
        mock_connect.side_effect = Exception("Connection failed")
        
        self.ws.is_running = True
        
        # Start connection attempt and let it fail a few times
        connection_task = asyncio.create_task(self.ws.connect())
        
        # Let it run for a short time
        await asyncio.sleep(0.5)
        
        # Stop the connection attempts
        self.ws.is_running = False
        
        try:
            await asyncio.wait_for(connection_task, timeout=1.0)
        except asyncio.TimeoutError:
            connection_task.cancel()
        
        # Verify connect was called multiple times
        self.assertGreater(mock_connect.call_count, 1)
        self.assertGreater(self.ws.reconnect_count, 0)
    
    @patch('websockets.connect')
    async def test_successful_reconnection(self, mock_connect):
        """Test successful reconnection after failures"""
        # Setup mock websocket
        mock_websocket = AsyncMock()
        mock_websocket.closed = False
        mock_websocket.ping = AsyncMock(return_value=asyncio.Future())
        mock_websocket.ping.return_value.set_result(None)
        
        # First call fails, second succeeds
        mock_connect.side_effect = [
            Exception("First connection failed"),
            mock_websocket
        ]
        
        # Mock the message handler to return quickly
        async def mock_message_handler():
            await asyncio.sleep(0.1)
        
        async def mock_heartbeat_handler():
            await asyncio.sleep(0.1)
        
        self.ws._message_handler = mock_message_handler
        self.ws._heartbeat_handler = mock_heartbeat_handler
        
        self.ws.is_running = True
        
        # Start connection - should fail once then succeed
        connection_task = asyncio.create_task(self.ws.connect())
        
        # Wait for connection to establish
        await asyncio.sleep(0.3)
        
        # Stop and cleanup
        self.ws.is_running = False
        
        try:
            await asyncio.wait_for(connection_task, timeout=1.0)
        except asyncio.TimeoutError:
            connection_task.cancel()
        
        # Verify we had a failure and recovery
        self.assertEqual(mock_connect.call_count, 2)
        self.assertEqual(self.ws.reconnect_count, 0)  # Reset on successful connection
    
    async def test_subscription_management(self):
        """Test subscription and unsubscription"""
        # Mock websocket
        self.ws.websocket = AsyncMock()
        self.ws.is_connected = True
        
        test_symbols = ['BTCUSDT', 'ETHUSDT']
        
        # Test subscription
        await self.ws.subscribe_ticker(test_symbols)
        
        # Verify subscription calls were made
        self.assertEqual(len(self.ws.subscribed_symbols), len(test_symbols))
        self.assertTrue(all(symbol in self.ws.subscribed_symbols for symbol in test_symbols))
        
        # Test unsubscription
        await self.ws.unsubscribe_all()
        
        # Verify unsubscription
        self.assertEqual(len(self.ws.subscribed_symbols), 0)
        self.assertEqual(len(self.ws.subscriptions), 0)
    
    async def test_message_queue_overflow(self):
        """Test message queue overflow handling"""
        # Fill the queue to capacity
        for i in range(1000):  # Should fill the queue
            message = {
                'timestamp': time.time(),
                'method': 'test',
                'data': {'test_id': i}
            }
            await self.ws._process_message({'method': 'test', 'params': [f'TEST{i}', {'data': 'test'}]})
        
        # Queue should be at capacity but not crashed
        self.assertLessEqual(self.ws.message_queue.qsize(), 1000)
        
        # Adding one more should still work (oldest dropped)
        await self.ws._process_message({'method': 'test', 'params': ['TESTFINAL', {'data': 'final'}]})
        
        self.assertLessEqual(self.ws.message_queue.qsize(), 1000)

if __name__ == '__main__':
    unittest.main()