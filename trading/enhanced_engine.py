"""
Enhanced Trading Engine with all integrated components
"""

import logging
import threading
import time
import pandas as pd
from typing import Dict, List, Any, Optional
from datetime import datetime

# Enhanced imports
from config.config_loader import load_config
from utils.logging import get_logger, initialize_logging
from utils.api_client import MultiSourceAPIClient
from utils.prediction_scheduler import PredictionScheduler, AdaptiveThresholdManager
from utils.signal_manager import SignalManager

# ML components
from ml.enhanced_model import EnhancedTradingModel
from ml.enhanced_feature_selection import EnhancedFeatureSelector
from ml.enhanced_labeling import EnhancedTargetLabeler
from indicators.enhanced_calculator import EnhancedIndicatorCalculator

# Legacy components (updated)
from trading.position_manager import PositionManager
from data.fetcher import DataFetcher
from database.connection import db_connection
from database.models import TradingSignal, Position, TradingMetrics

class EnhancedTradingEngine:
    """Enhanced Trading Engine with integrated ML pipeline and adaptive features"""
    
    def __init__(self, demo_mode: bool = True):
        # Load configuration
        self.config = load_config()
        
        # Initialize logging
        self.logger_system = initialize_logging(self.config)
        self.logger = get_logger(__name__)
        
        # Basic setup
        self.demo_mode = demo_mode
        self.is_running = False
        self.stop_trading = False
        
        # Enhanced components
        self.api_client = MultiSourceAPIClient(self.config)
        self.indicator_calculator = EnhancedIndicatorCalculator()
        self.feature_selector = EnhancedFeatureSelector(self.config)
        self.target_labeler = EnhancedTargetLabeler(self.config)
        self.model = EnhancedTradingModel(self.config)
        
        # Adaptive components
        self.threshold_manager = AdaptiveThresholdManager(self.config)
        self.signal_manager = SignalManager(self.config)
        self.prediction_scheduler = None  # Will be initialized after model training
        
        # Legacy components (enhanced)
        self.data_fetcher = DataFetcher(self.api_client)  # Updated to use new API client
        self.position_manager = PositionManager(self.api_client)
        
        # Configuration
        trading_config = self.config.get('trading', {})
        self.symbols = trading_config.get('symbols', ['BTCUSDT', 'ETHUSDT'])
        self.timeframe = trading_config.get('timeframe', '4h')
        self.demo_balance = trading_config.get('demo_balance', 100.0)
        self.used_balance = 0.0
        
        # Threading
        self.trading_thread = None
        
        self.logger.info(f"Enhanced trading engine initialized (Demo: {demo_mode})")
    
    def start_system(self):
        """Start the complete enhanced trading system"""
        try:
            self.logger.info("Starting enhanced trading system...")
            
            # Start signal management services
            self.signal_manager.start()
            
            # Train model if not already trained
            if not self.model.is_trained:
                self.logger.info("Training initial model...")
                self._train_model()
            
            # Initialize prediction scheduler
            if not self.prediction_scheduler:
                self.prediction_scheduler = PredictionScheduler(
                    self.config, 
                    self.model, 
                    self.data_fetcher, 
                    self.threshold_manager
                )
                
                # Add signal handling callback
                self.prediction_scheduler.add_signal_callback(self._handle_new_signal)
            
            # Start prediction scheduler
            self.prediction_scheduler.start()
            
            # Start main trading loop
            self.is_running = True
            self.trading_thread = threading.Thread(target=self._trading_loop, daemon=True)
            self.trading_thread.start()
            
            self.logger.info("Enhanced trading system started successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to start trading system: {e}")
            raise
    
    def stop_system(self):
        """Stop the trading system"""
        try:
            self.logger.info("Stopping enhanced trading system...")
            
            # Stop components in reverse order
            if self.prediction_scheduler:
                self.prediction_scheduler.stop()
            
            self.stop_trading = True
            self.is_running = False
            
            if self.trading_thread:
                self.trading_thread.join(timeout=10)
            
            self.signal_manager.stop()
            
            self.logger.info("Enhanced trading system stopped")
            
        except Exception as e:
            self.logger.error(f"Error stopping trading system: {e}")
    
    def _train_model(self):
        """Train the enhanced ML model"""
        try:
            self.logger.info("Starting enhanced model training...")
            
            # Load and prepare data
            training_data = self._prepare_training_data()
            
            if training_data is None or len(training_data) < 1000:
                raise ValueError("Insufficient training data")
            
            # Calculate enhanced indicators
            self.logger.info("Calculating enhanced indicators...")
            enhanced_data = self.indicator_calculator.calculate_all_enhanced_indicators(training_data)
            
            # Create labels
            self.logger.info("Creating enhanced labels...")
            labeled_data, labels = self.target_labeler.create_labels(enhanced_data)
            
            # Validate labeling
            validation = self.target_labeler.validate_labeling_quality(labeled_data, labels)
            if not validation['is_valid']:
                self.logger.warning(f"Labeling validation issues: {validation['issues']}")
            
            # Balance classes if needed
            labeled_data, labels = self.target_labeler.balance_classes(labeled_data, labels)
            
            # Feature selection
            self.logger.info("Performing enhanced feature selection...")
            
            # Exclude OHLCV columns from feature selection
            feature_columns = [col for col in labeled_data.columns 
                             if col not in ['open', 'high', 'low', 'close', 'volume']]
            
            selected_features, selection_info = self.feature_selector.select_features(
                labeled_data[feature_columns], 
                labels
            )
            
            # Save feature selection artifact
            self.feature_selector.save_selection_artifact("models/selected_features.json")
            
            # Prepare final training data
            X = labeled_data[selected_features]
            y = labels
            
            # Split data into train/val/test for better evaluation
            from sklearn.model_selection import train_test_split
            
            # First split: train+val (80%) / test (20%)
            X_trainval, X_test, y_trainval, y_test = train_test_split(
                X, y, test_size=0.2, stratify=y, random_state=42
            )
            
            # Second split: train (64%) / val (16%)
            X_train, X_val, y_train, y_val = train_test_split(
                X_trainval, y_trainval, test_size=0.2, stratify=y_trainval, random_state=42
            )
            
            self.logger.info(f"Data split - Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
            
            # Train enhanced model
            self.logger.info(f"Training {self.model.model_type} model with {len(selected_features)} features...")
            
            training_metrics = self.model.train(X_train, y_train, X_val, y_val)
            
            # Evaluate on test set for final performance assessment
            if len(X_test) > 0:
                test_metrics = self.model.evaluate_test_set(X_test, y_test)
                training_metrics.update(test_metrics)
                self.logger.info(f"Test accuracy: {test_metrics.get('test_accuracy', 0):.4f}")
            
            # Save model
            model_path = self.model.save_model()
            
            # Log training results with overfitting detection
            train_acc = training_metrics.get('train_accuracy', 0)
            val_acc = training_metrics.get('val_accuracy', 0)
            test_acc = training_metrics.get('test_accuracy', val_acc)  # Use val if no test
            
            self.logger.info(f"Model training completed successfully")
            self.logger.info(f"Training accuracy: {train_acc:.4f}")
            self.logger.info(f"Validation accuracy: {val_acc:.4f}")
            self.logger.info(f"Test accuracy: {test_acc:.4f}")
            
            # Detect potential overfitting
            overfitting_threshold = 0.05  # 5% difference threshold
            if train_acc - val_acc > overfitting_threshold:
                self.logger.warning(f"Potential overfitting detected: train_acc ({train_acc:.4f}) >> val_acc ({val_acc:.4f})")
                self.logger.warning("Consider: reducing model complexity, adding regularization, or collecting more data")
            
            self.logger.info(f"Model saved: {model_path}")
            
            # Log to structured events
            self.logger_system.log_ml_event('training_completed', {
                'model_type': self.model.model_type,
                'training_metrics': training_metrics,
                'feature_count': len(selected_features),
                'training_samples': len(X_train),
                'model_path': model_path
            })
            
            return True
            
        except Exception as e:
            self.logger.error(f"Model training failed: {e}")
            self.logger_system.log_ml_event('training_failed', {
                'error': str(e)
            })
            return False
    
    def _prepare_training_data(self) -> pd.DataFrame:
        """Prepare training data from all symbols"""
        try:
            all_data = []
            
            for symbol in self.symbols:
                self.logger.info(f"Loading data for {symbol}...")
                
                # Get historical data
                symbol_data = self.data_fetcher.get_historical_data(
                    symbol, 
                    limit=10000  # Get more data for better training
                )
                
                if symbol_data is not None and len(symbol_data) > 500:
                    symbol_data['symbol'] = symbol
                    all_data.append(symbol_data)
                    self.logger.info(f"Loaded {len(symbol_data)} records for {symbol}")
                else:
                    self.logger.warning(f"Insufficient data for {symbol}")
            
            if not all_data:
                return None
            
            # Combine all data
            combined_data = pd.concat(all_data, ignore_index=True)
            
            # Sort by timestamp
            if 'timestamp' in combined_data.columns:
                combined_data = combined_data.sort_values('timestamp')
            
            self.logger.info(f"Combined training data: {len(combined_data)} total records")
            
            return combined_data
            
        except Exception as e:
            self.logger.error(f"Error preparing training data: {e}")
            return None
    
    def _trading_loop(self):
        """Main trading loop (simplified, prediction scheduler handles most logic)"""
        self.logger.info("Enhanced trading loop started")
        
        while self.is_running and not self.stop_trading:
            try:
                # Basic system health checks
                self._check_system_health()
                
                # Process any pending position updates
                if self.position_manager:
                    self.position_manager.update_positions()
                
                # Sleep between checks
                time.sleep(60)  # Check every minute
                
            except Exception as e:
                self.logger.error(f"Error in trading loop: {e}")
                time.sleep(30)  # Wait before retrying
    
    def _check_system_health(self):
        """Perform system health checks"""
        try:
            # Check API client health
            health_status = self.api_client.get_health_status()
            
            if health_status['primary_source']['status'] == 'failed' and \
               health_status['secondary_source']['status'] == 'failed':
                self.logger.error("Both data sources failed!")
            
            # Check database connection
            if not db_connection.test_connection():
                self.logger.error("Database connection failed!")
            
            # Log health status occasionally
            if datetime.now().minute % 30 == 0:  # Every 30 minutes
                self.logger.debug("System health check completed")
                
        except Exception as e:
            self.logger.error(f"Health check failed: {e}")
    
    def _handle_new_signal(self, signal):
        """Handle new signal from prediction scheduler"""
        try:
            # Convert signal to dictionary for signal manager
            signal_data = {
                'id': signal.id,
                'timestamp': signal.timestamp,
                'symbol': signal.symbol,
                'direction': signal.direction,
                'confidence': signal.confidence,
                'price': signal.price,
                'threshold_used': signal.threshold_used,
                'features_hash': signal.features_hash,
                'model_version': signal.model_version
            }
            
            # Store signal
            self.signal_manager.handle_new_signal(signal_data)
            
            # Execute trade if in demo mode or real trading
            if self.demo_mode:
                self._execute_demo_trade(signal)
            else:
                self._execute_real_trade(signal)
            
        except Exception as e:
            self.logger.error(f"Error handling signal: {e}")
    
    def _execute_demo_trade(self, signal):
        """Execute demo trade"""
        try:
            trade_size = self.demo_balance * 0.02  # 2% per trade
            
            if signal.direction == 'BUY' and self.used_balance + trade_size <= self.demo_balance:
                self.used_balance += trade_size
                self.logger.info(f"DEMO BUY: {signal.symbol} ${trade_size:.2f} @ {signal.price:.4f}")
                
            elif signal.direction == 'SELL' and self.used_balance >= trade_size:
                self.used_balance -= trade_size
                self.logger.info(f"DEMO SELL: {signal.symbol} ${trade_size:.2f} @ {signal.price:.4f}")
            
            # Log trading event
            self.logger_system.log_trading_event('demo_trade_executed', signal.symbol, {
                'direction': signal.direction,
                'confidence': signal.confidence,
                'price': signal.price,
                'trade_size': trade_size
            })
            
        except Exception as e:
            self.logger.error(f"Demo trade execution failed: {e}")
    
    def _execute_real_trade(self, signal):
        """Execute real trade (placeholder - implement based on risk management)"""
        self.logger.warning("Real trading not implemented yet - this is a demo signal")
        # TODO: Implement real trading logic with proper risk management
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status"""
        try:
            status = {
                'is_running': self.is_running,
                'demo_mode': self.demo_mode,
                'model_trained': self.model.is_trained if self.model else False,
                'symbols': self.symbols,
                'timeframe': self.timeframe,
                'components': {}
            }
            
            # API client status
            if self.api_client:
                status['components']['api_client'] = self.api_client.get_health_status()
            
            # Model status
            if self.model and self.model.is_trained:
                model_info = self.model.get_model_info()
                status['components']['model'] = {
                    'type': model_info.get('model_type'),
                    'version': model_info.get('model_version'),
                    'feature_count': model_info.get('feature_count')
                }
            
            # Prediction scheduler status
            if self.prediction_scheduler:
                status['components']['prediction_scheduler'] = self.prediction_scheduler.get_status()
            
            # Signal manager status
            if self.signal_manager:
                status['components']['signal_manager'] = self.signal_manager.get_statistics()
            
            # Demo trading status
            if self.demo_mode:
                status['demo_trading'] = {
                    'balance': self.demo_balance,
                    'used_balance': self.used_balance,
                    'available_balance': self.demo_balance - self.used_balance
                }
            
            return status
            
        except Exception as e:
            self.logger.error(f"Error getting system status: {e}")
            return {'error': str(e), 'is_running': self.is_running}
    
    def get_positions(self) -> List[Dict[str, Any]]:
        """Get current positions"""
        try:
            if self.position_manager:
                return self.position_manager.get_all_positions()
            return []
        except Exception as e:
            self.logger.error(f"Error getting positions: {e}")
            return []
    
    def force_retrain(self) -> bool:
        """Force model retraining"""
        try:
            self.logger.info("Forcing model retraining...")
            return self._train_model()
        except Exception as e:
            self.logger.error(f"Force retrain failed: {e}")
            return False