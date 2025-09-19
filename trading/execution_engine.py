import asyncio
import time
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime
from config.settings import TRADING_CONFIG, TP_SL_CONFIG
from trading.position_manager import PositionManager

class ExecutionEngine:
    """
    Lightweight, event-driven execution engine focused on TP/SL monitoring
    Operates independently from the main scanner to ensure sub-300ms latency
    """
    
    def __init__(self, position_manager: PositionManager):
        self.logger = logging.getLogger(__name__)
        self.position_manager = position_manager
        
        # State management
        self.is_running = False
        self.open_positions = {}  # symbol -> position data
        self.price_updates = asyncio.Queue(maxsize=1000)
        self.execution_tasks = []
        
        # Performance tracking
        self.latency_stats = {
            'total_executions': 0,
            'avg_latency_ms': 0,
            'max_latency_ms': 0,
            'last_update': time.time()
        }
        
        # Configuration
        self.tp_sl_config = TP_SL_CONFIG
        self.max_latency_target_ms = 300
        
    async def start(self):
        """Start the execution engine"""
        if self.is_running:
            return
            
        self.is_running = True
        self.logger.info("Starting execution engine...")
        
        # Start the main execution loop
        self.execution_tasks = [
            asyncio.create_task(self._execution_loop()),
            asyncio.create_task(self._position_monitor()),
            asyncio.create_task(self._performance_monitor())
        ]
        
        self.logger.info("Execution engine started")
    
    async def stop(self):
        """Stop the execution engine"""
        if not self.is_running:
            return
            
        self.is_running = False
        self.logger.info("Stopping execution engine...")
        
        # Cancel all tasks
        for task in self.execution_tasks:
            if not task.done():
                task.cancel()
                
        # Wait for tasks to complete
        if self.execution_tasks:
            await asyncio.gather(*self.execution_tasks, return_exceptions=True)
        
        self.execution_tasks.clear()
        self.logger.info("Execution engine stopped")
    
    async def update_price(self, symbol: str, price: float, timestamp: Optional[float] = None):
        """
        Update price for a symbol (called from WebSocket or REST fallback)
        This is the main entry point for price updates
        """
        if not self.is_running:
            return
            
        if timestamp is None:
            timestamp = time.time()
            
        price_update = {
            'symbol': symbol,
            'price': price,
            'timestamp': timestamp,
            'received_at': time.time()
        }
        
        try:
            self.price_updates.put_nowait(price_update)
        except asyncio.QueueFull:
            # Drop oldest price update to maintain low latency
            try:
                self.price_updates.get_nowait()
                self.price_updates.put_nowait(price_update)
            except asyncio.QueueEmpty:
                pass
    
    async def add_position(self, symbol: str, position_data: Dict[str, Any]):
        """Add a position to monitor"""
        self.open_positions[symbol] = {
            **position_data,
            'added_at': time.time(),
            'last_check': time.time()
        }
        
        self.logger.info(f"Added position for monitoring: {symbol}")
    
    async def remove_position(self, symbol: str):
        """Remove a position from monitoring"""
        if symbol in self.open_positions:
            del self.open_positions[symbol]
            self.logger.info(f"Removed position from monitoring: {symbol}")
    
    async def _execution_loop(self):
        """Main execution loop - processes price updates and checks TP/SL"""
        while self.is_running:
            try:
                # Wait for price update with short timeout to maintain responsiveness
                price_update = await asyncio.wait_for(
                    self.price_updates.get(), 
                    timeout=0.1
                )
                
                start_time = time.time()
                
                # Process the price update
                await self._process_price_update(price_update)
                
                # Track latency
                latency_ms = (time.time() - price_update['received_at']) * 1000
                self._update_latency_stats(latency_ms)
                
                # Warn if latency exceeds target
                if latency_ms > self.max_latency_target_ms:
                    self.logger.warning(
                        f"High execution latency: {latency_ms:.2f}ms for {price_update['symbol']}"
                    )
                
            except asyncio.TimeoutError:
                # No price updates - check if we need to do any maintenance
                await self._maintenance_check()
                continue
            except Exception as e:
                self.logger.error(f"Error in execution loop: {e}")
                await asyncio.sleep(0.01)  # Brief pause to avoid tight error loops
    
    async def _process_price_update(self, price_update: Dict[str, Any]):
        """Process a single price update and check for TP/SL triggers"""
        symbol = price_update['symbol']
        current_price = price_update['price']
        
        # Only check if we have an open position for this symbol
        if symbol not in self.open_positions:
            return
        
        position = self.open_positions[symbol]
        position['last_check'] = time.time()
        
        # Check for TP/SL triggers
        await self._check_tp_sl_triggers(symbol, position, current_price)
    
    async def _check_tp_sl_triggers(self, symbol: str, position: Dict[str, Any], current_price: float):
        """Check if TP or SL should be triggered for a position"""
        try:
            entry_price = position.get('entry_price', 0)
            side = position.get('side', '').lower()
            quantity = position.get('quantity', 0)
            
            if entry_price <= 0 or quantity <= 0:
                return
            
            # Calculate percentage change
            if side == 'buy' or side == 'long':
                pnl_percent = ((current_price - entry_price) / entry_price) * 100
            else:  # sell or short
                pnl_percent = ((entry_price - current_price) / entry_price) * 100
            
            # Check stop loss
            if pnl_percent <= -self.tp_sl_config['initial_sl_percent']:
                await self._execute_stop_loss(symbol, position, current_price)
                return
            
            # Check take profit levels
            if pnl_percent >= self.tp_sl_config['tp1_percent'] and not position.get('tp1_hit', False):
                await self._execute_take_profit(symbol, position, current_price, 1)
            elif pnl_percent >= self.tp_sl_config['tp2_percent'] and not position.get('tp2_hit', False):
                await self._execute_take_profit(symbol, position, current_price, 2)
            elif pnl_percent >= self.tp_sl_config['tp3_percent'] and not position.get('tp3_hit', False):
                await self._execute_take_profit(symbol, position, current_price, 3)
                
        except Exception as e:
            self.logger.error(f"Error checking TP/SL for {symbol}: {e}")
    
    async def _execute_stop_loss(self, symbol: str, position: Dict[str, Any], current_price: float):
        """Execute stop loss order"""
        try:
            self.logger.warning(f"STOP LOSS TRIGGERED for {symbol} at {current_price}")
            
            # Close the entire position
            await self._close_position(symbol, position, current_price, "stop_loss")
            
            # Remove from monitoring
            await self.remove_position(symbol)
            
        except Exception as e:
            self.logger.error(f"Error executing stop loss for {symbol}: {e}")
    
    async def _execute_take_profit(self, symbol: str, position: Dict[str, Any], current_price: float, tp_level: int):
        """Execute take profit order"""
        try:
            self.logger.info(f"TAKE PROFIT {tp_level} TRIGGERED for {symbol} at {current_price}")
            
            # Mark TP level as hit
            position[f'tp{tp_level}_hit'] = True
            position[f'tp{tp_level}_price'] = current_price
            position[f'tp{tp_level}_time'] = time.time()
            
            # For demo purposes, close partial position
            # In real implementation, this would call the position manager
            partial_close_percent = 0.33  # Close 33% at each TP level
            
            if tp_level == 3:  # Close entire position at TP3
                await self._close_position(symbol, position, current_price, f"take_profit_{tp_level}")
                await self.remove_position(symbol)
            else:
                # Partial close
                original_quantity = position.get('quantity', 0)
                close_quantity = original_quantity * partial_close_percent
                position['quantity'] = original_quantity - close_quantity
                
                self.logger.info(f"Partially closed {close_quantity} of {symbol} at TP{tp_level}")
                
        except Exception as e:
            self.logger.error(f"Error executing take profit {tp_level} for {symbol}: {e}")
    
    async def _close_position(self, symbol: str, position: Dict[str, Any], price: float, reason: str):
        """Close a position (demo implementation)"""
        try:
            side = position.get('side', '').lower()
            quantity = position.get('quantity', 0)
            entry_price = position.get('entry_price', 0)
            
            # Calculate PnL
            if side in ['buy', 'long']:
                pnl = (price - entry_price) * quantity
            else:
                pnl = (entry_price - price) * quantity
            
            self.logger.info(
                f"POSITION CLOSED - {symbol}: {side} {quantity} @ {price} "
                f"(entry: {entry_price}, PnL: {pnl:.4f}, reason: {reason})"
            )
            
            # In real implementation, this would call:
            # await self.position_manager.close_position(symbol, quantity, price)
            
        except Exception as e:
            self.logger.error(f"Error closing position for {symbol}: {e}")
    
    async def _position_monitor(self):
        """Monitor open positions and refresh data periodically"""
        while self.is_running:
            try:
                current_time = time.time()
                
                # Check for stale positions (not updated in last 60 seconds)
                stale_positions = []
                for symbol, position in self.open_positions.items():
                    if current_time - position.get('last_check', 0) > 60:
                        stale_positions.append(symbol)
                
                # Log stale positions
                if stale_positions:
                    self.logger.warning(f"Stale positions detected: {stale_positions}")
                
                # Sleep for 30 seconds
                await asyncio.sleep(30)
                
            except Exception as e:
                self.logger.error(f"Error in position monitor: {e}")
                await asyncio.sleep(10)
    
    async def _performance_monitor(self):
        """Monitor and log performance metrics"""
        while self.is_running:
            try:
                await asyncio.sleep(60)  # Report every minute
                
                stats = self.latency_stats.copy()
                positions_count = len(self.open_positions)
                queue_size = self.price_updates.qsize()
                
                self.logger.info(
                    f"Execution Engine Status - Positions: {positions_count}, "
                    f"Queue: {queue_size}, Avg Latency: {stats['avg_latency_ms']:.2f}ms, "
                    f"Max Latency: {stats['max_latency_ms']:.2f}ms, "
                    f"Executions: {stats['total_executions']}"
                )
                
            except Exception as e:
                self.logger.error(f"Error in performance monitor: {e}")
                await asyncio.sleep(30)
    
    async def _maintenance_check(self):
        """Periodic maintenance when no price updates are received"""
        current_time = time.time()
        
        # Check if we haven't received updates for a while
        if hasattr(self, '_last_price_update'):
            if current_time - self._last_price_update > 300:  # 5 minutes
                self.logger.warning("No price updates received for 5 minutes")
        
        self._last_price_update = current_time
    
    def _update_latency_stats(self, latency_ms: float):
        """Update latency statistics"""
        stats = self.latency_stats
        
        stats['total_executions'] += 1
        stats['max_latency_ms'] = max(stats['max_latency_ms'], latency_ms)
        
        # Update rolling average
        if stats['total_executions'] == 1:
            stats['avg_latency_ms'] = latency_ms
        else:
            # Simple moving average with weight on recent values
            alpha = 0.1
            stats['avg_latency_ms'] = (alpha * latency_ms + 
                                     (1 - alpha) * stats['avg_latency_ms'])
        
        stats['last_update'] = time.time()
    
    def get_status(self) -> Dict[str, Any]:
        """Get current execution engine status"""
        return {
            'is_running': self.is_running,
            'open_positions': len(self.open_positions),
            'price_queue_size': self.price_updates.qsize(),
            'latency_stats': self.latency_stats.copy(),
            'positions': list(self.open_positions.keys())
        }