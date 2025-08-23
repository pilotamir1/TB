#!/usr/bin/env python3
"""
AI Trading Bot - Main Application Entry Point

This is a complete AI-powered cryptocurrency trading bot that:
- Trains ML models on 58k+ historical OHLCV data points
- Uses 196+ technical indicators with RFE feature selection
- Implements advanced TP/SL management with trailing stops
- Provides real-time web dashboard for monitoring
- Supports both demo and live trading on CoinEx
- Operates on 4-hour timeframe with >70% confidence signals

Author: AI Assistant
Created for: amirsofali3/TB repository
"""

import os
import sys
import logging
import signal
import threading
import time
from datetime import datetime

# Add project root to Python path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

# Import components
from config.settings import WEB_CONFIG, TRADING_CONFIG, LOGGING_CONFIG
from database.connection import db_connection
from database.models import SystemLog, Base
from trading.engine import TradingEngine
from web.app import create_app

class TradingBotApplication:
    """Main application class that orchestrates all components"""
    
    def __init__(self):
        # Initialize enhanced logging first
        from config.config_loader import load_config
        from utils.logging import initialize_logging, get_logger
        
        # Load configuration
        config = load_config()
        
        # Initialize enhanced logging system
        self.logger_system = initialize_logging(config)
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
        
        self.logger.info("AI Trading Bot Application initialized with enhanced logging")
    
    def signal_handler(self, signum, frame):
        """Handle shutdown signals"""
        self.logger.info(f"Received signal {signum}, shutting down gracefully...")
        self.shutdown()
    
    def initialize_database(self):
        """Initialize database connection and create tables"""
        try:
            self.logger.info("Initializing database connection...")
            
            # First, try to initialize the engine
            if not db_connection.init_engine():
                raise Exception("Database engine initialization failed")
            
            # Test the connection
            if not db_connection.test_connection():
                raise Exception("Database connection test failed")
            
            # Create tables if they don't exist
            self.logger.info("Creating database tables...")
            Base.metadata.create_all(db_connection.engine)
            
            self.logger.info("Database initialized successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Database initialization failed: {e}")
            self.logger.info("You may need to:")
            self.logger.info("1. Install and start MySQL server")
            self.logger.info("2. Create the database: CREATE DATABASE TB;")
            self.logger.info("3. Update credentials in .env file")
            self.logger.info("4. Or run setup_database.py for automated setup")
            return False
    
    def initialize_trading_engine(self):
        """Initialize the trading engine"""
        try:
            self.logger.info("Initializing trading engine...")
            
            # Create trading engine in demo mode by default
            self.trading_engine = TradingEngine(demo_mode=True)
            
            self.logger.info("Trading engine initialized successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Trading engine initialization failed: {e}")
            return False
    
    def initialize_web_interface(self):
        """Initialize the web dashboard"""
        try:
            self.logger.info("Initializing web interface...")
            
            # Create Flask app with trading engine reference
            self.web_app = create_app(self.trading_engine)
            
            self.logger.info("Web interface initialized successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Web interface initialization failed: {e}")
            return False
    
    def start_web_interface(self):
        """Start the web interface in a separate thread"""
        try:
            def run_flask():
                self.web_app.run(
                    host=WEB_CONFIG['host'],
                    port=WEB_CONFIG['port'],
                    debug=False,  # Don't use debug mode in production
                    use_reloader=False,  # Prevent double initialization
                    threaded=True
                )
            
            self.flask_thread = threading.Thread(target=run_flask, daemon=True)
            self.flask_thread.start()
            
            self.logger.info(f"Web interface started on http://{WEB_CONFIG['host']}:{WEB_CONFIG['port']}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to start web interface: {e}")
            return False
    
    def log_system_event(self, level: str, module: str, message: str, details: str = None):
        """Log system event to database and enhanced logger"""
        try:
            # Use enhanced logger for structured events
            if hasattr(self, 'logger_system'):
                if level.upper() == 'INFO':
                    self.logger_system.log_ml_event('system_event', {
                        'level': level,
                        'module': module,
                        'message': message,
                        'details': details
                    })
                else:
                    # For non-info events, log as trading event
                    self.logger_system.log_trading_event('system_event', 'SYSTEM', {
                        'level': level,
                        'module': module,
                        'message': message,
                        'details': details
                    })
            
            # Also log to database if available
            session = db_connection.get_session()
            
            log_entry = SystemLog(
                level=level,
                module=module,
                message=message,
                details=details
            )
            
            session.add(log_entry)
            session.commit()
            session.close()
            
        except Exception as e:
            self.logger.error(f"Failed to log system event: {e}")
    
    def start(self):
        """Start the complete trading bot system"""
        try:
            self.logger.info("=" * 60)
            self.logger.info("Starting AI Trading Bot System")
            self.logger.info("=" * 60)
            
            # Log startup
            self.log_system_event('INFO', 'main', 'AI Trading Bot system startup initiated')
            
            # 1. Initialize database
            if not self.initialize_database():
                raise Exception("Database initialization failed")
            
            # 2. Initialize trading engine
            if not self.initialize_trading_engine():
                raise Exception("Trading engine initialization failed")
            
            # 3. Initialize web interface
            if not self.initialize_web_interface():
                raise Exception("Web interface initialization failed")
            
            # 4. Start web interface (must start before trading system)
            if not self.start_web_interface():
                raise Exception("Failed to start web interface")
            
            # Give web interface time to start
            time.sleep(2)
            
            # 5. Start trading system
            self.logger.info("Starting complete trading system...")
            self.trading_engine.start_system()
            
            self.running = True
            self.log_system_event('INFO', 'main', 'AI Trading Bot system started successfully')
            
            self.logger.info("=" * 60)
            self.logger.info("🚀 AI Trading Bot is now running!")
            self.logger.info(f"📊 Web Dashboard: http://{WEB_CONFIG['host']}:{WEB_CONFIG['port']}")
            self.logger.info(f"💰 Demo Mode: ${TRADING_CONFIG['demo_balance']}")
            self.logger.info(f"📈 Symbols: {', '.join(TRADING_CONFIG['symbols'])}")
            self.logger.info(f"⏱️  Timeframe: {TRADING_CONFIG['timeframe']}")
            self.logger.info(f"🎯 Confidence Threshold: {TRADING_CONFIG['confidence_threshold']*100}%")
            self.logger.info("=" * 60)
            
            # Keep the application running
            self.main_loop()
            
        except KeyboardInterrupt:
            self.logger.info("Received keyboard interrupt")
            self.shutdown()
        except Exception as e:
            self.logger.error(f"Failed to start trading bot: {e}")
            self.log_system_event('ERROR', 'main', f'System startup failed: {str(e)}')
            self.shutdown()
            sys.exit(1)
    
    def main_loop(self):
        """Main application loop"""
        try:
            while self.running and not self.shutdown_event.is_set():
                # Check system health
                self.check_system_health()
                
                # Wait for shutdown signal
                self.shutdown_event.wait(timeout=60)  # Check every minute
                
        except Exception as e:
            self.logger.error(f"Error in main loop: {e}")
            self.shutdown()
    
    def check_system_health(self):
        """Perform system health checks"""
        try:
            # Check database connection
            if not db_connection.test_connection():
                self.logger.error("Database connection lost")
                self.log_system_event('ERROR', 'health', 'Database connection lost')
            
            # Check trading engine status
            if self.trading_engine:
                status = self.trading_engine.get_system_status()
                if not status.get('is_running', False) and self.running:
                    self.logger.warning("Trading engine not running")
            
            # Log health check (less frequently)
            if datetime.now().minute % 15 == 0:  # Every 15 minutes
                self.logger.debug("System health check completed")
                
        except Exception as e:
            self.logger.error(f"Health check failed: {e}")
    
    def shutdown(self):
        """Graceful shutdown of all components"""
        try:
            self.logger.info("Initiating graceful shutdown...")
            
            self.running = False
            self.shutdown_event.set()
            
            # Stop trading system
            if self.trading_engine:
                self.logger.info("Stopping trading system...")
                self.trading_engine.stop_system()
            
            # Web interface will stop when main thread exits
            
            self.log_system_event('INFO', 'main', 'AI Trading Bot system shutdown completed')
            self.logger.info("Graceful shutdown completed")
            
        except Exception as e:
            self.logger.error(f"Error during shutdown: {e}")
        finally:
            sys.exit(0)

def main():
    """Main entry point"""
    print("""
╔══════════════════════════════════════════════════════════════════════════════╗
║                          AI TRADING BOT SYSTEM                              ║
║                                                                              ║
║  🤖 Advanced AI-powered cryptocurrency trading bot                          ║
║  📊 196+ Technical indicators with RFE feature selection                    ║
║  💹 Advanced TP/SL management with trailing stops                           ║
║  🌐 Real-time web dashboard for monitoring                                  ║
║  🔄 4-hour timeframe with >70% confidence signals                           ║
║  💰 Demo and live trading support                                           ║
║                                                                              ║
║  Created for: amirsofali3/TB repository                                     ║
╚══════════════════════════════════════════════════════════════════════════════╝
    """)
    
    # Create and start the application
    app = TradingBotApplication()
    app.start()

if __name__ == "__main__":
    main()