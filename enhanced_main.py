#!/usr/bin/env python3
"""
Enhanced AI Trading Bot - Main Application Entry Point

This is the enhanced version of the AI-powered cryptocurrency trading bot with:
- UTF-8 safe logging with Windows compatibility
- Multi-source data feed with automatic failover  
- CatBoost + XGBoost model support with probability calibration
- SHAP-based feature selection with correlation analysis
- Adaptive confidence thresholds for optimal signal rates
- Real-time signal persistence and WebSocket broadcasting
- Comprehensive health monitoring and API endpoints
- Enhanced technical indicators (196+ features)
- Class balancing and improved target labeling

Author: Enhanced by AI Assistant
Repository: amirsofali3/TB
"""

import os
import sys
import signal
import threading
import time
from datetime import datetime

# Add project root to Python path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

# Enhanced imports
from config.config_loader import load_config
from utils.logging import initialize_logging, get_logger
from trading.enhanced_engine import EnhancedTradingEngine
from web.enhanced_app import create_enhanced_app
from database.connection import db_connection
from database.models import Base

class EnhancedTradingBotApplication:
    """Enhanced Trading Bot Application with all integrated components"""
    
    def __init__(self):
        # Load configuration
        self.config = load_config()
        
        # Initialize logging system
        self.logger_system = initialize_logging(self.config)
        self.logger = get_logger(__name__)
        
        # Components
        self.trading_engine = None
        self.web_app = None
        self.flask_thread = None
        
        # Application state
        self.running = False
        self.shutdown_event = threading.Event()
        
        # Setup signal handlers
        signal.signal(signal.SIGINT, self.signal_handler)
        signal.signal(signal.SIGTERM, self.signal_handler)
        
        self.logger.info("Enhanced AI Trading Bot Application initialized")
    
    def signal_handler(self, signum, frame):
        """Handle shutdown signals gracefully"""
        self.logger.info(f"Received signal {signum}, initiating graceful shutdown...")
        self.shutdown()
    
    def initialize_database(self):
        """Initialize database connection with enhanced error handling"""
        try:
            self.logger.info("Initializing database connection...")
            
            # Initialize engine
            if not db_connection.init_engine():
                raise Exception("Database engine initialization failed")
            
            # Test connection with retry
            for attempt in range(3):
                if db_connection.test_connection():
                    break
                else:
                    if attempt < 2:
                        self.logger.warning(f"Database connection attempt {attempt + 1} failed, retrying...")
                        time.sleep(2)
                    else:
                        raise Exception("Database connection test failed after 3 attempts")
            
            # Create tables
            self.logger.info("Creating database tables...")
            Base.metadata.create_all(db_connection.engine)
            
            self.logger.info("Database initialized successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Database initialization failed: {e}")
            self.logger.info("Database initialization failed. Please ensure:")
            self.logger.info("1. MySQL/MariaDB server is running")
            self.logger.info("2. Database 'TB' exists: CREATE DATABASE TB;")
            self.logger.info("3. Credentials in config.yaml or environment variables are correct")
            return False
    
    def initialize_trading_engine(self):
        """Initialize the enhanced trading engine"""
        try:
            self.logger.info("Initializing enhanced trading engine...")
            
            # Create enhanced trading engine
            self.trading_engine = EnhancedTradingEngine(demo_mode=True)
            
            self.logger.info("Enhanced trading engine initialized successfully")
            
            # Log configuration summary
            self.logger.info(f"Model type: {self.config.get('ml', {}).get('model_type', 'xgboost')}")
            self.logger.info(f"Symbols: {self.config.get('trading', {}).get('symbols', [])}")
            self.logger.info(f"Timeframe: {self.config.get('trading', {}).get('timeframe', '4h')}")
            
            adaptive_config = self.config.get('adaptive_threshold', {})
            if adaptive_config.get('enabled', True):
                self.logger.info(f"Adaptive thresholds enabled - target: {adaptive_config.get('target_signals_per_24h', 5)} signals/24h")
            else:
                self.logger.info("Using static confidence threshold")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Enhanced trading engine initialization failed: {e}")
            return False
    
    def initialize_web_interface(self):
        """Initialize the enhanced web dashboard"""
        try:
            self.logger.info("Initializing enhanced web interface...")
            
            # Create enhanced Flask app
            self.web_app = create_enhanced_app(
                trading_engine=self.trading_engine,
                signal_manager=self.trading_engine.signal_manager if self.trading_engine else None,
                prediction_scheduler=self.trading_engine.prediction_scheduler if self.trading_engine else None
            )
            
            self.logger.info("Enhanced web interface initialized successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Enhanced web interface initialization failed: {e}")
            return False
    
    def start_web_interface(self):
        """Start the enhanced web interface"""
        try:
            web_config = self.config.get('web', {})
            
            def run_flask():
                self.web_app.run(
                    host=web_config.get('host', '0.0.0.0'),
                    port=web_config.get('port', 5000),
                    debug=web_config.get('debug', False),
                    use_reloader=False,
                    threaded=True
                )
            
            self.flask_thread = threading.Thread(target=run_flask, daemon=True)
            self.flask_thread.start()
            
            web_host = web_config.get('host', '0.0.0.0')
            web_port = web_config.get('port', 5000)
            self.logger.info(f"Enhanced web interface started on http://{web_host}:{web_port}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to start enhanced web interface: {e}")
            return False
    
    def start(self):
        """Start the complete enhanced trading bot system"""
        try:
            self.logger.info("=" * 80)
            self.logger.info("🚀 STARTING ENHANCED AI TRADING BOT SYSTEM")
            self.logger.info("=" * 80)
            
            # Show configuration summary
            self.logger.info("Configuration Summary:")
            self.logger.info(f"  📊 Model: {self.config.get('ml', {}).get('model_type', 'xgboost').upper()}")
            self.logger.info(f"  💱 Symbols: {', '.join(self.config.get('trading', {}).get('symbols', []))}")
            self.logger.info(f"  ⏰ Timeframe: {self.config.get('trading', {}).get('timeframe', '4h')}")
            self.logger.info(f"  🎯 Adaptive Thresholds: {'✓' if self.config.get('adaptive_threshold', {}).get('enabled') else '✗'}")
            self.logger.info(f"  📡 WebSocket Broadcasts: {'✓' if self.config.get('signals', {}).get('websocket_broadcast') else '✗'}")
            self.logger.info(f"  💾 Signal Persistence: {'✓' if self.config.get('signals', {}).get('persistence_enabled') else '✗'}")
            
            # 1. Initialize database
            self.logger.info("Step 1/4: Database initialization...")
            if not self.initialize_database():
                raise Exception("Database initialization failed")
            self.logger.info("✅ Database initialized")
            
            # 2. Initialize enhanced trading engine
            self.logger.info("Step 2/4: Trading engine initialization...")
            if not self.initialize_trading_engine():
                raise Exception("Enhanced trading engine initialization failed")
            self.logger.info("✅ Enhanced trading engine initialized")
            
            # 3. Initialize enhanced web interface
            self.logger.info("Step 3/4: Web interface initialization...")
            if not self.initialize_web_interface():
                raise Exception("Enhanced web interface initialization failed")
            
            # 4. Start web interface
            if not self.start_web_interface():
                raise Exception("Failed to start enhanced web interface")
            self.logger.info("✅ Enhanced web interface started")
            
            # Give web interface time to start
            time.sleep(2)
            
            # 5. Start enhanced trading system
            self.logger.info("Step 4/4: Starting enhanced trading system...")
            self.trading_engine.start_system()
            self.logger.info("✅ Enhanced trading system started")
            
            self.running = True
            
            # Display startup summary
            self.logger.info("=" * 80)
            self.logger.info("🎉 ENHANCED AI TRADING BOT STARTED SUCCESSFULLY!")
            self.logger.info("=" * 80)
            
            web_config = self.config.get('web', {})
            web_host = web_config.get('host', '0.0.0.0')
            web_port = web_config.get('port', 5000)
            
            self.logger.info("📊 Dashboard URLs:")
            self.logger.info(f"   🌐 Main Dashboard: http://{web_host}:{web_port}")
            self.logger.info(f"   📈 Health Check: http://{web_host}:{web_port}/api/health")
            self.logger.info(f"   🤖 Model Status: http://{web_host}:{web_port}/api/model/status")
            self.logger.info(f"   📡 Recent Signals: http://{web_host}:{web_port}/api/signals/recent")
            
            self.logger.info("💡 Key Features:")
            self.logger.info("   ✓ Multi-algorithm ML models (XGBoost + CatBoost)")
            self.logger.info("   ✓ SHAP-based feature selection")
            self.logger.info("   ✓ Adaptive confidence thresholds")
            self.logger.info("   ✓ Multi-source data feed with failover")
            self.logger.info("   ✓ Real-time signal broadcasting")
            self.logger.info("   ✓ Comprehensive health monitoring")
            self.logger.info("   ✓ Enhanced technical indicators (196+ features)")
            
            self.logger.info("=" * 80)
            
            # Keep the application running
            self.main_loop()
            
        except KeyboardInterrupt:
            self.logger.info("Keyboard interrupt received")
            self.shutdown()
        except Exception as e:
            self.logger.error(f"Failed to start enhanced trading bot: {e}")
            self.shutdown()
            sys.exit(1)
    
    def main_loop(self):
        """Enhanced main application loop with health monitoring"""
        try:
            health_check_interval = self.config.get('health', {}).get('check_interval', 300)  # 5 minutes
            last_health_check = 0
            
            while self.running and not self.shutdown_event.is_set():
                current_time = time.time()
                
                # Periodic health check
                if current_time - last_health_check >= health_check_interval:
                    self._comprehensive_health_check()
                    last_health_check = current_time
                
                # Wait for shutdown signal
                self.shutdown_event.wait(timeout=60)  # Check every minute
                
        except Exception as e:
            self.logger.error(f"Error in main loop: {e}")
            self.shutdown()
    
    def _comprehensive_health_check(self):
        """Perform comprehensive system health check"""
        try:
            self.logger.debug("Performing comprehensive health check...")
            
            health_status = {
                'timestamp': datetime.now().isoformat(),
                'overall_status': 'healthy',
                'issues': []
            }
            
            # Database health
            if not db_connection.test_connection():
                health_status['issues'].append('Database connection failed')
                health_status['overall_status'] = 'degraded'
            
            # Trading engine health
            if self.trading_engine:
                try:
                    engine_status = self.trading_engine.get_system_status()
                    if not engine_status.get('is_running'):
                        health_status['issues'].append('Trading engine not running')
                        health_status['overall_status'] = 'degraded'
                    
                    # Check data source health
                    if 'components' in engine_status and 'api_client' in engine_status['components']:
                        api_status = engine_status['components']['api_client']
                        if api_status.get('primary_source', {}).get('status') == 'failed' and \
                           api_status.get('secondary_source', {}).get('status') == 'failed':
                            health_status['issues'].append('All data sources failed')
                            health_status['overall_status'] = 'critical'
                
                except Exception as e:
                    health_status['issues'].append(f'Trading engine error: {str(e)}')
                    health_status['overall_status'] = 'degraded'
            
            # Log health status
            if health_status['overall_status'] == 'healthy':
                self.logger.debug("System health check: All systems operational")
            else:
                self.logger.warning(f"System health check: {health_status['overall_status']} - Issues: {health_status['issues']}")
            
            # Log structured health event
            self.logger_system.log_ml_event('health_check_completed', health_status)
            
        except Exception as e:
            self.logger.error(f"Health check failed: {e}")
    
    def shutdown(self):
        """Enhanced graceful shutdown"""
        try:
            self.logger.info("🛑 Initiating enhanced graceful shutdown...")
            
            self.running = False
            self.shutdown_event.set()
            
            # Stop trading system
            if self.trading_engine:
                self.logger.info("Stopping enhanced trading system...")
                self.trading_engine.stop_system()
                self.logger.info("✅ Enhanced trading system stopped")
            
            # Web interface will stop when main thread exits
            self.logger.info("🌐 Web interface will stop with main thread")
            
            self.logger.info("✅ Enhanced graceful shutdown completed")
            
        except Exception as e:
            self.logger.error(f"Error during enhanced shutdown: {e}")
        finally:
            self.logger.info("👋 Enhanced AI Trading Bot shutting down")
            sys.exit(0)

def main():
    """Enhanced main entry point"""
    print("""
╔══════════════════════════════════════════════════════════════════════════════════════╗
║                            🤖 ENHANCED AI TRADING BOT 🚀                            ║
║                                                                                      ║
║  🔥 Advanced Features:                                                               ║
║  • 🧠 CatBoost + XGBoost ML models with probability calibration                     ║
║  • 🎯 Adaptive confidence thresholds for optimal signal rates                       ║
║  • 📊 SHAP-based feature selection with 196+ technical indicators                   ║
║  • 🔄 Multi-source data feed with automatic failover (CoinEx → Binance)            ║
║  • 📡 Real-time WebSocket signal broadcasting                                        ║
║  • 💾 Persistent signal storage with status tracking                                ║
║  • 🏥 Comprehensive health monitoring and API endpoints                             ║
║  • 🪟 UTF-8 safe logging with Windows compatibility                                 ║
║  • ⚖️  Smart class balancing and enhanced target labeling                           ║
║  • 📈 Cross-validation and model performance tracking                               ║
║                                                                                      ║
║  Repository: amirsofali3/TB                                                         ║
║  Enhanced by: AI Assistant                                                          ║
╚══════════════════════════════════════════════════════════════════════════════════════╝
    """)
    
    # Create and start the enhanced application
    app = EnhancedTradingBotApplication()
    app.start()

if __name__ == "__main__":
    main()