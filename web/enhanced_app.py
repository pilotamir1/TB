"""
Enhanced web API endpoints with model status, signals, and health monitoring
"""

from flask import Flask, render_template, jsonify, request
from flask_cors import CORS
import logging
from datetime import datetime, timedelta
import json
import os
from typing import Dict, Any, Optional

from database.connection import db_connection
from database.models import *
from config.config_loader import load_config

def create_enhanced_app(trading_engine=None, signal_manager=None, prediction_scheduler=None):
    """Create enhanced Flask application with new endpoints"""
    
    # Load configuration
    config = load_config()
    web_config = config.get('web', {})
    
    app = Flask(__name__)
    app.config['SECRET_KEY'] = web_config.get('secret_key', 'your-secret-key-change-this')
    
    # Enable CORS
    CORS(app)
    
    # Store references
    app.trading_engine = trading_engine
    app.signal_manager = signal_manager
    app.prediction_scheduler = prediction_scheduler
    app.config_data = config
    
    @app.route('/')
    def dashboard():
        """Main dashboard page"""
        return render_template('dashboard.html')
    
    @app.route('/api/health')
    def health_check():
        """Comprehensive health check endpoint"""
        try:
            health_status = {
                'timestamp': datetime.now().isoformat(),
                'status': 'healthy',
                'services': {},
                'data_sources': {},
                'system': {}
            }
            
            # Database health
            try:
                db_healthy = db_connection.test_connection()
                health_status['services']['database'] = {
                    'status': 'healthy' if db_healthy else 'unhealthy',
                    'last_check': datetime.now().isoformat()
                }
            except Exception as e:
                health_status['services']['database'] = {
                    'status': 'error',
                    'error': str(e),
                    'last_check': datetime.now().isoformat()
                }
            
            # Trading engine health
            if app.trading_engine:
                try:
                    engine_status = app.trading_engine.get_system_status()
                    health_status['services']['trading_engine'] = {
                        'status': 'healthy' if engine_status.get('is_running') else 'stopped',
                        'details': engine_status
                    }
                except Exception as e:
                    health_status['services']['trading_engine'] = {
                        'status': 'error',
                        'error': str(e)
                    }
            else:
                health_status['services']['trading_engine'] = {
                    'status': 'not_initialized'
                }
            
            # Data sources health (if available)
            if hasattr(app.trading_engine, 'data_fetcher') and hasattr(app.trading_engine.data_fetcher, 'api_client'):
                try:
                    api_health = app.trading_engine.data_fetcher.api_client.get_health_status()
                    health_status['data_sources'] = api_health
                except Exception as e:
                    health_status['data_sources'] = {
                        'status': 'error',
                        'error': str(e)
                    }
            
            # Prediction scheduler health
            if app.prediction_scheduler:
                try:
                    scheduler_status = app.prediction_scheduler.get_status()
                    health_status['services']['prediction_scheduler'] = {
                        'status': 'healthy' if scheduler_status.get('is_running') else 'stopped',
                        'details': scheduler_status
                    }
                except Exception as e:
                    health_status['services']['prediction_scheduler'] = {
                        'status': 'error',
                        'error': str(e)
                    }
            
            # Signal manager health
            if app.signal_manager:
                try:
                    signal_stats = app.signal_manager.get_statistics()
                    health_status['services']['signal_manager'] = {
                        'status': 'healthy',
                        'details': signal_stats
                    }
                except Exception as e:
                    health_status['services']['signal_manager'] = {
                        'status': 'error',
                        'error': str(e)
                    }
            
            # Overall status
            unhealthy_services = [
                service for service, data in health_status['services'].items()
                if data.get('status') not in ['healthy', 'stopped']
            ]
            
            if unhealthy_services:
                health_status['status'] = 'degraded'
                health_status['unhealthy_services'] = unhealthy_services
            
            return jsonify(health_status)
            
        except Exception as e:
            return jsonify({
                'status': 'error',
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }), 500
    
    @app.route('/api/model/status')
    def model_status():
        """Enhanced model status endpoint"""
        try:
            if not app.trading_engine or not hasattr(app.trading_engine, 'model'):
                return jsonify({
                    'success': False,
                    'error': 'Trading engine or model not available'
                })
            
            model = app.trading_engine.model
            
            if not model or not model.is_trained:
                return jsonify({
                    'success': False,
                    'error': 'No trained model available'
                })
            
            # Get model information
            model_info = model.get_model_info()
            metadata = model_info.get('training_metadata', {})
            
            status_data = {
                'success': True,
                'model_type': model_info.get('model_type', 'unknown'),
                'model_version': model_info.get('model_version', 'unknown'),
                'is_trained': model_info.get('is_trained', False),
                'feature_count': model_info.get('feature_count', 0),
                'training_samples': metadata.get('training_samples', 0),
                'val_accuracy': metadata.get('val_accuracy', 0),
                'train_accuracy': metadata.get('train_accuracy', 0),
                'precision': metadata.get('precision', 0),
                'recall': metadata.get('recall', 0),
                'f1_score': metadata.get('f1_score', 0),
                'roc_auc': metadata.get('roc_auc', 0),
                'last_trained_at': metadata.get('training_timestamp', 'unknown'),
                'training_time_seconds': metadata.get('training_time_seconds', 0),
                'class_distribution': metadata.get('class_distribution', {}),
                'confusion_matrix': metadata.get('confusion_matrix', []),
                'cv_mean_accuracy': metadata.get('cv_mean_accuracy', 0),
                'cv_std_accuracy': metadata.get('cv_std_accuracy', 0)
            }
            
            return jsonify(status_data)
            
        except Exception as e:
            return jsonify({
                'success': False,
                'error': str(e)
            })
    
    @app.route('/api/signals/recent')
    def get_recent_signals():
        """Get recent trading signals"""
        try:
            limit = request.args.get('limit', 50, type=int)
            symbol = request.args.get('symbol')
            hours_back = request.args.get('hours', 24, type=int)
            
            if app.signal_manager:
                # Get signals from signal manager
                signals = app.signal_manager.get_recent_signals(limit=limit, symbol=symbol)
                
                # Filter by time if specified
                if hours_back:
                    cutoff_time = datetime.now() - timedelta(hours=hours_back)
                    signals = [
                        signal for signal in signals
                        if datetime.fromisoformat(signal['timestamp']) > cutoff_time
                    ]
                
                return jsonify({
                    'success': True,
                    'signals': signals,
                    'count': len(signals),
                    'filters': {
                        'symbol': symbol,
                        'hours_back': hours_back,
                        'limit': limit
                    }
                })
            else:
                # Fallback to database query
                session = db_connection.get_session()
                
                query = session.query(TradingSignal)
                
                if symbol:
                    query = query.filter(TradingSignal.symbol == symbol)
                
                if hours_back:
                    cutoff_time = datetime.now() - timedelta(hours=hours_back)
                    query = query.filter(TradingSignal.timestamp > cutoff_time)
                
                signals_db = query.order_by(TradingSignal.timestamp.desc()).limit(limit).all()
                
                signals = []
                for signal in signals_db:
                    signals.append({
                        'id': signal.id,
                        'timestamp': signal.timestamp.isoformat(),
                        'symbol': signal.symbol,
                        'direction': signal.signal_type,
                        'confidence': signal.confidence,
                        'price': signal.price,
                        'model_version': signal.model_version,
                        'status': 'active'
                    })
                
                session.close()
                
                return jsonify({
                    'success': True,
                    'signals': signals,
                    'count': len(signals),
                    'source': 'database'
                })
            
        except Exception as e:
            return jsonify({
                'success': False,
                'error': str(e)
            })
    
    @app.route('/api/signals/<signal_id>/status', methods=['PUT'])
    def update_signal_status(signal_id: str):
        """Update signal status"""
        try:
            data = request.get_json()
            new_status = data.get('status')
            
            if not new_status:
                return jsonify({
                    'success': False,
                    'error': 'Status is required'
                }), 400
            
            if app.signal_manager:
                success = app.signal_manager.update_signal_status(signal_id, new_status)
                
                return jsonify({
                    'success': success,
                    'message': f'Signal {signal_id} status updated to {new_status}' if success else 'Update failed'
                })
            else:
                return jsonify({
                    'success': False,
                    'error': 'Signal manager not available'
                })
            
        except Exception as e:
            return jsonify({
                'success': False,
                'error': str(e)
            })
    
    @app.route('/api/prediction/force')
    def force_prediction():
        """Force immediate prediction for debugging"""
        try:
            symbol = request.args.get('symbol')
            
            if not app.prediction_scheduler:
                return jsonify({
                    'success': False,
                    'error': 'Prediction scheduler not available'
                })
            
            result = app.prediction_scheduler.force_prediction(symbol)
            
            return jsonify(result)
            
        except Exception as e:
            return jsonify({
                'success': False,
                'error': str(e)
            })
    
    @app.route('/api/system/status')
    def system_status():
        """Enhanced system status"""
        try:
            status_data = {
                'timestamp': datetime.now().isoformat(),
                'services': {},
                'statistics': {}
            }
            
            # Trading engine status
            if app.trading_engine:
                try:
                    engine_status = app.trading_engine.get_system_status()
                    if hasattr(app.trading_engine, 'trainer'):
                        training_progress = app.trading_engine.trainer.get_training_progress()
                        engine_status['training_progress'] = training_progress
                    
                    status_data['services']['trading_engine'] = engine_status
                except Exception as e:
                    status_data['services']['trading_engine'] = {'error': str(e)}
            
            # Prediction scheduler status
            if app.prediction_scheduler:
                try:
                    scheduler_status = app.prediction_scheduler.get_status()
                    status_data['services']['prediction_scheduler'] = scheduler_status
                except Exception as e:
                    status_data['services']['prediction_scheduler'] = {'error': str(e)}
            
            # Signal manager statistics
            if app.signal_manager:
                try:
                    signal_stats = app.signal_manager.get_statistics()
                    status_data['statistics']['signals'] = signal_stats
                except Exception as e:
                    status_data['statistics']['signals'] = {'error': str(e)}
            
            return jsonify({
                'success': True,
                'system_status': status_data
            })
            
        except Exception as e:
            return jsonify({
                'success': False,
                'error': str(e)
            })
    
    @app.route('/api/positions')
    def get_positions():
        """Get active positions"""
        try:
            if app.trading_engine and hasattr(app.trading_engine, 'position_manager'):
                positions = app.trading_engine.position_manager.get_active_positions()
                return jsonify({
                    'success': True,
                    'positions': positions
                })
            else:
                return jsonify({
                    'success': False,
                    'error': 'Trading engine or position manager not available'
                })
        except Exception as e:
            return jsonify({'success': False, 'error': str(e)})
    
    @app.route('/api/portfolio')
    def get_portfolio():
        """Get portfolio information including balance and daily P&L"""
        try:
            portfolio_data = {
                'portfolio_value': 0.0,
                'available_balance': 100.0,  # Default demo balance
                'daily_pnl': 0.0,
                'daily_pnl_pct': 0.0,
                'total_positions': 0
            }
            
            if app.trading_engine:
                # Get current balance
                if hasattr(app.trading_engine, 'demo_balance'):
                    portfolio_data['available_balance'] = app.trading_engine.demo_balance
                
                # Get active positions and calculate total P&L
                if hasattr(app.trading_engine, 'position_manager'):
                    summary = app.trading_engine.position_manager.get_position_summary()
                    portfolio_data['total_positions'] = summary.get('total_active_positions', 0)
                    portfolio_data['daily_pnl'] = summary.get('total_unrealized_pnl', 0.0)
                    
                    # Calculate portfolio value
                    portfolio_data['portfolio_value'] = portfolio_data['available_balance'] + portfolio_data['daily_pnl']
                    
                    # Calculate daily P&L percentage
                    if portfolio_data['available_balance'] > 0:
                        portfolio_data['daily_pnl_pct'] = (portfolio_data['daily_pnl'] / portfolio_data['available_balance']) * 100
            
            return jsonify({
                'success': True,
                'portfolio': portfolio_data
            })
        except Exception as e:
            return jsonify({'success': False, 'error': str(e)})
    
    @app.route('/api/trading/start', methods=['POST'])
    def start_trading():
        """Start trading"""
        try:
            if app.trading_engine and hasattr(app.trading_engine, 'start_trading'):
                result = app.trading_engine.start_trading()
                return jsonify({
                    'success': result,
                    'message': 'Trading started' if result else 'Failed to start trading'
                })
            else:
                return jsonify({
                    'success': False,
                    'error': 'Trading engine not available'
                })
        except Exception as e:
            return jsonify({'success': False, 'error': str(e)})
    
    @app.route('/api/trading/stop', methods=['POST'])
    def stop_trading():
        """Stop trading"""
        try:
            if app.trading_engine and hasattr(app.trading_engine, 'stop_trading'):
                result = app.trading_engine.stop_trading()
                return jsonify({
                    'success': result,
                    'message': 'Trading stopped' if result else 'Failed to stop trading'
                })
            else:
                return jsonify({
                    'success': False,
                    'error': 'Trading engine not available'
                })
        except Exception as e:
            return jsonify({'success': False, 'error': str(e)})
    
    @app.route('/api/trades/history')
    def get_trade_history():
        """Get completed trade history"""
        try:
            limit = request.args.get('limit', 50, type=int)
            
            session = db_connection.get_session()
            
            # Query completed positions
            completed_positions = session.query(Position).filter(
                Position.status == 'CLOSED'
            ).order_by(Position.closed_at.desc()).limit(limit).all()
            
            trades = []
            for position in completed_positions:
                trades.append({
                    'id': position.id,
                    'symbol': position.symbol,
                    'side': position.side,
                    'entry_price': position.entry_price,
                    'exit_price': position.current_price,
                    'quantity': position.quantity,
                    'pnl': position.pnl,
                    'pnl_percentage': position.pnl_percentage,
                    'opened_at': position.opened_at.isoformat() if position.opened_at else None,
                    'closed_at': position.closed_at.isoformat() if position.closed_at else None,
                    'duration': str(position.closed_at - position.opened_at) if position.opened_at and position.closed_at else None
                })
            
            session.close()
            
            return jsonify({
                'success': True,
                'trades': trades,
                'count': len(trades)
            })
            
        except Exception as e:
            return jsonify({'success': False, 'error': str(e)})
    
    @app.route('/api/market/overview')
    def market_overview():
        """Get market overview"""
        try:
            if app.trading_engine and hasattr(app.trading_engine, 'data_fetcher'):
                overview = app.trading_engine.data_fetcher.get_market_overview()
                return jsonify({
                    'success': True,
                    'market_data': overview
                })
            else:
                return jsonify({
                    'success': False,
                    'error': 'Market data not available'
                })
        except Exception as e:
            return jsonify({'success': False, 'error': str(e)})
    
    @app.route('/api/logs')
    def get_logs():
        """Get recent logs"""
        try:
            limit = request.args.get('limit', 50, type=int)
            level = request.args.get('level', 'INFO')
            
            session = db_connection.get_session()
            
            query = session.query(SystemLog)
            if level != 'ALL':
                query = query.filter(SystemLog.level == level)
            
            logs = query.order_by(SystemLog.timestamp.desc()).limit(limit).all()
            
            log_data = []
            for log in logs:
                log_data.append({
                    'timestamp': log.timestamp.isoformat(),
                    'level': log.level,
                    'module': log.module,
                    'message': log.message,
                    'details': log.details
                })
            
            session.close()
            
            return jsonify({
                'success': True,
                'logs': log_data
            })
            
        except Exception as e:
            return jsonify({'success': False, 'error': str(e)})
    
    @app.route('/api/config')
    def get_configuration():
        """Get current configuration"""
        try:
            # Return safe configuration (without secrets)
            safe_config = app.config_data.copy()
            
            # Remove sensitive information
            if 'coinex' in safe_config:
                safe_config['coinex'] = {
                    'base_url': safe_config['coinex'].get('base_url'),
                    'sandbox_mode': safe_config['coinex'].get('sandbox_mode'),
                    'api_key': '***' if safe_config['coinex'].get('api_key') else None,
                    'secret_key': '***' if safe_config['coinex'].get('secret_key') else None
                }
            
            if 'database' in safe_config:
                safe_config['database']['password'] = '***' if safe_config['database'].get('password') else None
            
            return jsonify({
                'success': True,
                'config': safe_config
            })
            
        except Exception as e:
            return jsonify({'success': False, 'error': str(e)})
    
    @app.route('/api/adaptive-threshold/status')
    def adaptive_threshold_status():
        """Get adaptive threshold status"""
        try:
            if app.prediction_scheduler and hasattr(app.prediction_scheduler, 'threshold_manager'):
                status = app.prediction_scheduler.threshold_manager.get_status()
                return jsonify({
                    'success': True,
                    'adaptive_threshold': status
                })
            else:
                return jsonify({
                    'success': False,
                    'error': 'Adaptive threshold manager not available'
                })
        except Exception as e:
            return jsonify({'success': False, 'error': str(e)})
    
    @app.route('/api/model/validation')
    def model_validation():
        """Get model validation results comparing claimed vs real performance"""
        try:
            if not app.trading_engine or not hasattr(app.trading_engine, 'model'):
                return jsonify({
                    'success': False,
                    'error': 'Trading engine or model not available'
                })
            
            model = app.trading_engine.model
            
            if not model or not model.is_trained:
                return jsonify({
                    'success': False,
                    'error': 'No trained model available'
                })
            
            # Get recent trade data for validation
            session = db_connection.get_session()
            
            # Get recent completed trades (last 50)
            recent_trades = session.query(Position).filter(
                Position.status == 'CLOSED'
            ).order_by(Position.closed_at.desc()).limit(50).all()
            
            # Convert to validation format
            trades_data = []
            for trade in recent_trades:
                trades_data.append({
                    'pnl': trade.pnl,
                    'pnl_percentage': trade.pnl_percentage,
                    'symbol': trade.symbol,
                    'side': trade.side
                })
            
            session.close()
            
            # Get recent signals data (you can expand this based on your signal storage)
            signals_data = []  # Placeholder - you can implement signal history retrieval
            
            # Validate model performance
            validation_results = model.validate_model_performance(signals_data, trades_data)
            
            return jsonify({
                'success': True,
                'validation': validation_results
            })
            
        except Exception as e:
            return jsonify({
                'success': False,
                'error': str(e)
            })
    def retrain_model():
        """Retrain the AI model in background with status updates"""
        try:
            if not app.trading_engine:
                return jsonify({
                    'success': False,
                    'error': 'Trading engine not available'
                })
            
            # Check if training is already in progress
            if hasattr(app, '_training_in_progress') and app._training_in_progress:
                return jsonify({
                    'success': False,
                    'error': 'Model training already in progress'
                })
            
            # Start retraining in background using concurrent.futures for better handling
            import concurrent.futures
            import threading
            
            def retrain_with_status():
                """Retrain with status broadcasting"""
                try:
                    app._training_in_progress = True
                    app._training_status = {
                        'stage': 'starting',
                        'message': 'Starting model training...',
                        'progress': 0
                    }
                    
                    # If signal manager exists, broadcast training start
                    if app.signal_manager and hasattr(app.signal_manager, 'broadcast_training_status'):
                        app.signal_manager.broadcast_training_status(app._training_status)
                    
                    # Update progress stages
                    stages = [
                        ('data_preparation', 'Preparing training data...', 20),
                        ('feature_selection', 'Selecting features...', 40),
                        ('training', 'Training model...', 70),
                        ('validation', 'Validating model...', 90),
                        ('saving', 'Saving model...', 95),
                        ('completed', 'Training completed successfully!', 100)
                    ]
                    
                    for stage, message, progress in stages:
                        app._training_status = {
                            'stage': stage,
                            'message': message,
                            'progress': progress
                        }
                        
                        if app.signal_manager and hasattr(app.signal_manager, 'broadcast_training_status'):
                            app.signal_manager.broadcast_training_status(app._training_status)
                        
                        # Simulate stage processing (replace with actual training call)
                        if stage == 'training':
                            # Actual training call
                            app.trading_engine.train_model(retrain=True)
                        else:
                            import time
                            time.sleep(0.5)  # Simulate processing time
                    
                    # Training completed - broadcast model metadata update
                    if app.signal_manager and hasattr(app.signal_manager, 'broadcast_model_update'):
                        app.signal_manager.broadcast_model_update()
                    
                except Exception as e:
                    app._training_status = {
                        'stage': 'failed',
                        'message': f'Training failed: {str(e)}',
                        'progress': 0
                    }
                    
                    if app.signal_manager and hasattr(app.signal_manager, 'broadcast_training_status'):
                        app.signal_manager.broadcast_training_status(app._training_status)
                    
                    app.logger.error(f"Error retraining model: {e}")
                finally:
                    app._training_in_progress = False
            
            # Use thread pool for better resource management
            retrain_thread = threading.Thread(target=retrain_with_status, daemon=True)
            retrain_thread.start()
            
            return jsonify({
                'success': True, 
                'message': 'Model retraining started in background'
            })
            
        except Exception as e:
            return jsonify({'success': False, 'error': str(e)})
    
    @app.route('/api/training/status')
    def get_training_status():
        """Get current training status"""
        try:
            if hasattr(app, '_training_status'):
                return jsonify({
                    'success': True,
                    'training': app._training_status
                })
            else:
                return jsonify({
                    'success': True,
                    'training': {
                        'stage': 'idle',
                        'message': 'No training in progress',
                        'progress': 0
                    }
                })
        except Exception as e:
            return jsonify({'success': False, 'error': str(e)})
    
    @app.route('/api/settings/update', methods=['POST'])
    def update_settings():
        """Update system settings and propagate to engine"""
        try:
            data = request.get_json()
            
            if not data:
                return jsonify({
                    'success': False,
                    'error': 'No data provided'
                })
            
            updated_settings = {}
            
            # Handle confidence threshold update
            if 'confidence_threshold' in data:
                new_threshold = float(data['confidence_threshold'])
                if 0.1 <= new_threshold <= 0.99:
                    # Update trading engine if available
                    if app.trading_engine and hasattr(app.trading_engine, 'update_confidence_threshold'):
                        app.trading_engine.update_confidence_threshold(new_threshold)
                        updated_settings['confidence_threshold'] = new_threshold
                    else:
                        return jsonify({
                            'success': False,
                            'error': 'Trading engine not available or does not support threshold updates'
                        })
                else:
                    return jsonify({
                        'success': False,
                        'error': 'Confidence threshold must be between 0.1 and 0.99'
                    })
            
            # Handle demo balance update
            if 'demo_balance' in data:
                new_balance = float(data['demo_balance'])
                if new_balance > 0:
                    if app.trading_engine and hasattr(app.trading_engine, 'update_demo_balance'):
                        app.trading_engine.update_demo_balance(new_balance)
                        updated_settings['demo_balance'] = new_balance
                    else:
                        return jsonify({
                            'success': False,
                            'error': 'Trading engine not available or does not support balance updates'
                        })
                else:
                    return jsonify({
                        'success': False,
                        'error': 'Demo balance must be greater than 0'
                    })
            
            # Broadcast settings update if signal manager exists
            if app.signal_manager and hasattr(app.signal_manager, 'broadcast_settings_update'):
                app.signal_manager.broadcast_settings_update(updated_settings)
            
            return jsonify({
                'success': True,
                'message': 'Settings updated successfully',
                'updated': updated_settings
            })
            
        except ValueError as e:
            return jsonify({
                'success': False,
                'error': f'Invalid value: {str(e)}'
            })
        except Exception as e:
            return jsonify({
                'success': False,
                'error': str(e)
            })
    
    # Error handlers
    @app.errorhandler(404)
    def not_found(error):
        return jsonify({'success': False, 'error': 'Endpoint not found'}), 404
    
    @app.errorhandler(500)
    def internal_error(error):
        return jsonify({'success': False, 'error': 'Internal server error'}), 500
    
    return app