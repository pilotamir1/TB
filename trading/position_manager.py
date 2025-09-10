import logging
from typing import Dict, List, Any, Optional
from datetime import datetime
import threading
import time

from database.connection import db_connection
from database.models import Position
from trading.coinex_api import CoinExAPI
from config.settings import TP_SL_CONFIG, TRADING_CONFIG

class PositionManager:
    """
    Advanced position management with TP/SL trailing system
    """
    
    def __init__(self, api: CoinExAPI):
        self.api = api
        self.logger = logging.getLogger(__name__)
        self.active_positions = {}
        self.monitoring_thread = None
        self.stop_monitoring = False
        
        # TP/SL configuration
        self.tp1_percent = TP_SL_CONFIG['tp1_percent']
        self.tp2_percent = TP_SL_CONFIG['tp2_percent']
        self.tp3_percent = TP_SL_CONFIG['tp3_percent']
        self.initial_sl_percent = TP_SL_CONFIG['initial_sl_percent']
        self.trailing_enabled = TP_SL_CONFIG['trailing_enabled']
    
    def open_position(self, symbol: str, side: str, quantity: float, 
                     entry_price: float, signal_confidence: float) -> Optional[int]:
        """
        Open a new trading position with enhanced validation and logging
        
        Args:
            symbol: Trading symbol
            side: 'LONG' or 'SHORT'
            quantity: Position size
            entry_price: Entry price
            signal_confidence: AI model confidence
            
        Returns:
            Position ID if successful, None otherwise
        """
        try:
            # Validate inputs
            if quantity <= 0:
                self.logger.error(f"Invalid quantity for {symbol}: {quantity}")
                return None
            if entry_price <= 0:
                self.logger.error(f"Invalid entry price for {symbol}: {entry_price}")
                return None
            
            self.logger.info(f"💰 Opening {side} position for {symbol}: "
                           f"Quantity={quantity:.6f}, "
                           f"Entry=${entry_price:.6f}, "
                           f"Confidence={signal_confidence:.1%}")
            
            # Calculate TP/SL levels
            tp_sl_levels = self._calculate_tp_sl_levels(entry_price, side)
            
            # Enhanced logging for TP/SL calculation
            position_value = quantity * entry_price
            risk_amount = abs(entry_price - tp_sl_levels['initial_sl']) * quantity
            risk_percentage = (risk_amount / position_value) * 100
            
            self.logger.info(f"📊 TP/SL levels for {symbol} {side} position: "
                           f"Entry=${entry_price:.6f}, "
                           f"SL=${tp_sl_levels['initial_sl']:.6f} (-3%), "
                           f"TP1=${tp_sl_levels['tp1']:.6f} (+3%), "
                           f"TP2=${tp_sl_levels['tp2']:.6f} (+6%), "
                           f"TP3=${tp_sl_levels['tp3']:.6f} (+10%)")
            
            self.logger.info(f"💼 Position risk: ${risk_amount:.2f} ({risk_percentage:.1f}% of position value)")
            
            # Create position in database
            session = db_connection.get_session()
            
            position = Position(
                symbol=symbol,
                side=side,
                entry_price=entry_price,
                quantity=quantity,
                current_price=entry_price,
                initial_sl=tp_sl_levels['initial_sl'],
                current_sl=tp_sl_levels['initial_sl'],
                tp1_price=tp_sl_levels['tp1'],
                tp2_price=tp_sl_levels['tp2'],
                tp3_price=tp_sl_levels['tp3'],
                status='OPEN'
            )
            
            session.add(position)
            session.commit()
            position_id = position.id
            session.close()
            
            # Add to active positions for monitoring
            self.active_positions[position_id] = {
                'symbol': symbol,
                'side': side,
                'entry_price': entry_price,
                'quantity': quantity,
                'tp_sl_levels': tp_sl_levels,
                'confidence': signal_confidence,
                'opened_at': datetime.now()
            }
            
            # Start monitoring if not already running
            if not self.monitoring_thread or not self.monitoring_thread.is_alive():
                self.start_position_monitoring()
            
            self.logger.info(f"✅ Position opened successfully: ID {position_id}")
            self.logger.info(f"🔍 Started 1-second monitoring for position {position_id}")
            return position_id
            
        except Exception as e:
            self.logger.error(f"❌ Error opening position for {symbol}: {e}")
            return None
    
    def close_position(self, position_id: int, reason: str = "Manual close") -> bool:
        """Close a position"""
        try:
            session = db_connection.get_session()
            position = session.query(Position).get(position_id)
            
            if not position or position.status != 'OPEN':
                self.logger.warning(f"Position {position_id} not found or not open")
                session.close()
                return False
            
            # Get current price
            current_price = self._get_current_price(position.symbol)
            if current_price is None:
                self.logger.error(f"Could not get current price for {position.symbol}")
                session.close()
                return False
            
            # Calculate PnL
            pnl_info = self._calculate_pnl(position, current_price)
            
            # Update position in database
            position.current_price = current_price
            position.pnl = pnl_info['pnl']
            position.pnl_percentage = pnl_info['pnl_percentage']
            position.status = 'CLOSED'
            position.closed_at = datetime.now()
            
            session.commit()
            session.close()
            
            # Remove from active monitoring
            if position_id in self.active_positions:
                del self.active_positions[position_id]
            
            self.logger.info(f"Position {position_id} closed: {reason}, PnL: {pnl_info['pnl']:.4f} ({pnl_info['pnl_percentage']:.2f}%)")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error closing position {position_id}: {e}")
            return False
    
    def emergency_close_position(self, position_id: int, signal_confidence: float) -> bool:
        """
        Emergency close when opposite signal with high confidence is received
        """
        if signal_confidence >= TRADING_CONFIG['confidence_threshold']:
            return self.close_position(position_id, f"Emergency close - opposite signal {signal_confidence:.2f}")
        return False
    
    def start_position_monitoring(self):
        """Start the position monitoring thread"""
        if self.monitoring_thread and self.monitoring_thread.is_alive():
            return
        
        self.stop_monitoring = False
        self.monitoring_thread = threading.Thread(target=self._monitor_positions, daemon=True)
        self.monitoring_thread.start()
        self.logger.info("Position monitoring started")
    
    def stop_position_monitoring(self):
        """Stop position monitoring"""
        self.stop_monitoring = True
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=5)
        self.logger.info("Position monitoring stopped")
    
    def _monitor_positions(self):
        """Monitor all active positions for TP/SL triggers"""
        while not self.stop_monitoring:
            try:
                if not self.active_positions:
                    time.sleep(5)  # Sleep longer if no positions
                    continue
                
                for position_id in list(self.active_positions.keys()):
                    try:
                        self._check_position_triggers(position_id)
                    except Exception as e:
                        self.logger.error(f"Error monitoring position {position_id}: {e}")
                
                time.sleep(1)  # Check every second as specified
                
            except Exception as e:
                self.logger.error(f"Error in position monitoring loop: {e}")
                time.sleep(5)
    
    def _check_position_triggers(self, position_id: int):
        """Check TP/SL triggers for a specific position with enhanced error handling"""
        session = None
        try:
            # Get position from database
            session = db_connection.get_session()
            position = session.query(Position).get(position_id)
            
            if not position or position.status != 'OPEN':
                # Remove from active monitoring
                if position_id in self.active_positions:
                    del self.active_positions[position_id]
                    self.logger.info(f"Removed closed position {position_id} from monitoring")
                if session:
                    session.close()
                return
            
            # Get current price with retry logic
            current_price = self._get_current_price(position.symbol)
            if current_price is None:
                self.logger.warning(f"Failed to get price for {position.symbol}, skipping this check")
                if session:
                    session.close()
                return
            
            # Calculate position age to prevent immediate closures
            position_age = (datetime.now() - position.opened_at).total_seconds()
            if position_age < 10:  # Don't trigger SL/TP in first 10 seconds
                self.logger.debug(f"Position {position_id} too young ({position_age:.1f}s), skipping trigger check")
                if session:
                    session.close()
                return
            
            # Enhanced logging every 30 seconds
            if datetime.now().second % 30 == 0:
                unrealized_pnl = ((current_price - position.entry_price) / position.entry_price) * 100
                self.logger.info(f"📊 Position {position_id} ({position.symbol}): "
                                f"Entry=${position.entry_price:.6f}, "
                                f"Current=${current_price:.6f} ({unrealized_pnl:+.2f}%), "
                                f"SL=${position.current_sl:.6f}, "
                                f"TP1=${position.tp1_price:.6f}, "
                                f"Age={position_age:.0f}s")
            
            # Update current price
            position.current_price = current_price
            position.updated_at = datetime.now()
            
            # Check TP/SL triggers based on position side
            if position.side == 'LONG':
                self._check_long_position_triggers(position, current_price)
            else:
                self._check_short_position_triggers(position, current_price)
            
            session.commit()
            
        except Exception as e:
            self.logger.error(f"Error checking triggers for position {position_id}: {e}")
        finally:
            if session:
                session.close()
    
    def _check_long_position_triggers(self, position: Position, current_price: float):
        """Check triggers for LONG positions with user's specific TP/SL progression"""
        # Validate prices before checking triggers
        if current_price <= 0 or position.current_sl <= 0:
            self.logger.warning(f"Invalid prices for position {position.id}: current={current_price}, sl={position.current_sl}")
            return
        
        # Calculate percentage changes for logging
        sl_distance = ((current_price - position.current_sl) / position.current_sl) * 100
        
        # Check Stop Loss with additional validation
        if current_price <= position.current_sl:
            # Double-check this isn't a false trigger due to bad data
            if current_price < position.entry_price * 0.5:  # More than 50% drop seems like bad data
                self.logger.warning(f"Suspicious SL trigger for position {position.id}: price dropped to {current_price} (from entry {position.entry_price})")
                return
            
            self.logger.info(f"🔻 SL triggered for position {position.id}: {current_price:.6f} <= {position.current_sl:.6f} (SL distance: {sl_distance:.2f}%)")
            self.close_position(position.id, f"Stop Loss triggered at {current_price:.6f}")
            return
        
        # Check Take Profit levels and update trailing SL according to user specifications
        if not position.tp1_hit and current_price >= position.tp1_price:
            # TP1 hit (+3%) - move SL to entry price (breakeven) and set TP2 at +6%
            position.tp1_hit = True
            position.current_sl = position.entry_price  # Move SL to breakeven
            position.tp2_price = position.entry_price * (1 + 6.0 / 100)  # TP2 at +6% from entry
            self.logger.info(f"🎯 TP1 hit for position {position.id} at +3% ({current_price:.6f}). "
                           f"SL moved to breakeven: {position.entry_price:.6f}, "
                           f"TP2 set to +6%: {position.tp2_price:.6f}")
        
        elif position.tp1_hit and not position.tp2_hit and current_price >= position.tp2_price:
            # TP2 hit (+6%) - move SL to TP1 price and set TP3 at +10%
            position.tp2_hit = True
            position.current_sl = position.tp1_price  # Move SL to TP1 (+3%)
            position.tp3_price = position.entry_price * (1 + 10.0 / 100)  # TP3 at +10% from entry
            self.logger.info(f"🎯 TP2 hit for position {position.id} at +6% ({current_price:.6f}). "
                           f"SL moved to TP1: {position.tp1_price:.6f}, "
                           f"TP3 set to +10%: {position.tp3_price:.6f}")
        
        elif position.tp2_hit and not position.tp3_hit and current_price >= position.tp3_price:
            # TP3 hit (+10%) - move SL to TP2 price and continue progression
            position.tp3_hit = True
            position.current_sl = position.tp2_price  # Move SL to TP2 (+6%)
            self.logger.info(f"🎯 TP3 hit for position {position.id} at +10% ({current_price:.6f}). "
                           f"SL moved to TP2: {position.tp2_price:.6f}")
    
    def _check_short_position_triggers(self, position: Position, current_price: float):
        """Check triggers for SHORT positions with user's specific TP/SL progression"""
        # Validate prices before checking triggers
        if current_price <= 0 or position.current_sl <= 0:
            self.logger.warning(f"Invalid prices for position {position.id}: current={current_price}, sl={position.current_sl}")
            return
        
        # Calculate percentage changes for logging
        sl_distance = ((position.current_sl - current_price) / current_price) * 100
        
        # Check Stop Loss with additional validation
        if current_price >= position.current_sl:
            # Double-check this isn't a false trigger due to bad data
            if current_price > position.entry_price * 1.5:  # More than 50% rise seems like bad data
                self.logger.warning(f"Suspicious SL trigger for position {position.id}: price rose to {current_price} (from entry {position.entry_price})")
                return
            
            self.logger.info(f"🔻 SL triggered for position {position.id}: {current_price:.6f} >= {position.current_sl:.6f} (SL distance: {sl_distance:.2f}%)")
            self.close_position(position.id, f"Stop Loss triggered at {current_price:.6f}")
            return
        
        # Check Take Profit levels and update trailing SL according to user specifications
        if not position.tp1_hit and current_price <= position.tp1_price:
            # TP1 hit (-3%) - move SL to entry price (breakeven) and set TP2 at -6%
            position.tp1_hit = True
            position.current_sl = position.entry_price  # Move SL to breakeven
            position.tp2_price = position.entry_price * (1 - 6.0 / 100)  # TP2 at -6% from entry
            self.logger.info(f"🎯 TP1 hit for position {position.id} at -3% ({current_price:.6f}). "
                           f"SL moved to breakeven: {position.entry_price:.6f}, "
                           f"TP2 set to -6%: {position.tp2_price:.6f}")
        
        elif position.tp1_hit and not position.tp2_hit and current_price <= position.tp2_price:
            # TP2 hit (-6%) - move SL to TP1 price and set TP3 at -10%
            position.tp2_hit = True
            position.current_sl = position.tp1_price  # Move SL to TP1 (-3%)
            position.tp3_price = position.entry_price * (1 - 10.0 / 100)  # TP3 at -10% from entry
            self.logger.info(f"🎯 TP2 hit for position {position.id} at -6% ({current_price:.6f}). "
                           f"SL moved to TP1: {position.tp1_price:.6f}, "
                           f"TP3 set to -10%: {position.tp3_price:.6f}")
        
        elif position.tp2_hit and not position.tp3_hit and current_price <= position.tp3_price:
            # TP3 hit (-10%) - move SL to TP2 price and continue progression
            position.tp3_hit = True
            position.current_sl = position.tp2_price  # Move SL to TP2 (-6%)
            self.logger.info(f"🎯 TP3 hit for position {position.id} at -10% ({current_price:.6f}). "
                           f"SL moved to TP2: {position.tp2_price:.6f}")
    
    def _calculate_tp_sl_levels(self, entry_price: float, side: str) -> Dict[str, float]:
        """Calculate TP/SL levels based on entry price and side"""
        if side == 'LONG':
            return {
                'tp1': entry_price * (1 + self.tp1_percent / 100),  # +3% from entry
                'tp2': entry_price * (1 + self.tp2_percent / 100),  # +6% from entry  
                'tp3': entry_price * (1 + self.tp3_percent / 100),  # +10% from entry
                'initial_sl': entry_price * (1 - self.initial_sl_percent / 100)  # -3% from entry
            }
        else:  # SHORT
            return {
                'tp1': entry_price * (1 - self.tp1_percent / 100),  # -3% from entry
                'tp2': entry_price * (1 - self.tp2_percent / 100),  # -6% from entry
                'tp3': entry_price * (1 - self.tp3_percent / 100),  # -10% from entry
                'initial_sl': entry_price * (1 + self.initial_sl_percent / 100)  # +3% from entry
            }
    
    def _get_current_price(self, symbol: str) -> Optional[float]:
        """Get current market price for symbol with robust fallback handling"""
        try:
            ticker = self.api.get_ticker(symbol)
            price = 0.0
            
            # Handle both API response formats (direct and nested)
            if 'ticker' in ticker and isinstance(ticker['ticker'], dict):
                # Nested format from fallback data
                price_str = ticker['ticker'].get('last', '0')
                price = float(price_str) if price_str else 0.0
                self.logger.debug(f"Got nested ticker price for {symbol}: {price}")
            elif isinstance(ticker, dict) and 'last' in ticker:
                # Direct format from live API
                price_str = ticker.get('last', '0')
                price = float(price_str) if price_str else 0.0
                self.logger.debug(f"Got direct ticker price for {symbol}: {price}")
            else:
                # Try to extract price from any available field
                for field in ['last', 'price', 'close']:
                    if field in ticker:
                        price_str = ticker[field]
                        price = float(price_str) if price_str else 0.0
                        if price > 0:
                            self.logger.debug(f"Got price from field '{field}' for {symbol}: {price}")
                            break
            
            if price <= 0:
                self.logger.warning(f"Got invalid price ({price}) for {symbol}, ticker data: {ticker}")
                return None
                
            return price
            
        except (ValueError, TypeError) as e:
            self.logger.error(f"Error parsing price for {symbol}: {e}, ticker: {ticker}")
            return None
        except Exception as e:
            self.logger.error(f"Error getting current price for {symbol}: {e}")
            return None
    
    def _calculate_pnl(self, position: Position, current_price: float) -> Dict[str, float]:
        """Calculate PnL for position"""
        if position.side == 'LONG':
            pnl = (current_price - position.entry_price) * position.quantity
            pnl_percentage = ((current_price - position.entry_price) / position.entry_price) * 100
        else:  # SHORT
            pnl = (position.entry_price - current_price) * position.quantity
            pnl_percentage = ((position.entry_price - current_price) / position.entry_price) * 100
        
        return {
            'pnl': pnl,
            'pnl_percentage': pnl_percentage
        }
    
    def get_active_positions(self) -> List[Dict[str, Any]]:
        """Get all active positions"""
        try:
            session = db_connection.get_session()
            positions = session.query(Position).filter(Position.status == 'OPEN').all()
            
            result = []
            for position in positions:
                current_price = self._get_current_price(position.symbol)
                if current_price:
                    pnl_info = self._calculate_pnl(position, current_price)
                    
                    result.append({
                        'id': position.id,
                        'symbol': position.symbol,
                        'side': position.side,
                        'entry_price': position.entry_price,
                        'current_price': current_price,
                        'quantity': position.quantity,
                        'pnl': pnl_info['pnl'],
                        'pnl_percentage': pnl_info['pnl_percentage'],
                        'current_sl': position.current_sl,
                        'tp1_price': position.tp1_price,
                        'tp2_price': position.tp2_price,
                        'tp3_price': position.tp3_price,
                        'tp1_hit': position.tp1_hit,
                        'tp2_hit': position.tp2_hit,
                        'tp3_hit': position.tp3_hit,
                        'opened_at': position.opened_at
                    })
            
            session.close()
            return result
            
        except Exception as e:
            self.logger.error(f"Error getting active positions: {e}")
            return []
    
    def get_position_summary(self) -> Dict[str, Any]:
        """Get summary of all positions"""
        active_positions = self.get_active_positions()
        
        total_pnl = sum(pos['pnl'] for pos in active_positions)
        total_positions = len(active_positions)
        
        return {
            'total_active_positions': total_positions,
            'total_unrealized_pnl': total_pnl,
            'positions': active_positions
        }