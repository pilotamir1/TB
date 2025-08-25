import pandas as pd
import numpy as np
from typing import Dict, List, Any, Tuple
import logging
import os
from datetime import datetime, timedelta
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix

from ml.model import TradingModel
from ml.feature_selection import FeatureSelector
from indicators.calculator import IndicatorCalculator
from indicators.definitions import IndicatorDefinitions
from database.connection import db_connection
from database.models import Candle, ModelTraining
from config.settings import ML_CONFIG, TRADING_CONFIG, DATA_CONFIG, FEATURE_SELECTION_CONFIG, XGB_PRO_CONFIG
from config.config_loader import get_config_value

class ModelTrainer:
    """
    Comprehensive AI model trainer with feature selection and evaluation
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.calculator = IndicatorCalculator()
        self.definitions = IndicatorDefinitions()
        self.model = None
        self.feature_selector = None
        self.scaler = StandardScaler()
        self.training_progress = {
            'stage': 'initialized',
            'progress': 0,
            'message': 'Model trainer initialized'
        }
    
    def _is_aligned_4h(self, ts: int, tolerance_seconds: int = 120) -> bool:
        """
        بررسی می‌کند timestamp دقیقاً روی مرز کندل 4 ساعته باشد (± تلرانس).
        4h = 14400 ثانیه.
        """
        modv = ts % 14400
        return modv <= tolerance_seconds or (14400 - modv) <= tolerance_seconds
    
    def get_training_progress(self) -> Dict[str, Any]:
        """Get current training progress"""
        return self.training_progress.copy()
    
    def prepare_training_data(self, symbols: List[str] = None, 
                            train_samples: int = None) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
        """
        Prepare training data from database with indicators, returning separate full and selection datasets
        
        Args:
            symbols: List of symbols to include
            train_samples: Number of samples to use for training
            
        Returns:
            Tuple of (X_full, y_full, selection_X, selection_y)
            - X_full, y_full: Full training dataset (up to max_4h_training_candles)
            - selection_X, selection_y: Recent subset for feature selection (last max_4h_selection_candles)
        """
        try:
            self.training_progress.update({
                'stage': 'data_preparation',
                'progress': 10,
                'message': 'Loading historical data from database'
            })
            
            symbols = symbols or TRADING_CONFIG['symbols']
            train_samples = train_samples or ML_CONFIG['training_data_size']
            
            # Get configuration for 4h timeframe limits
            selection_limit = DATA_CONFIG.get('max_4h_selection_candles', 800)
            training_limit = DATA_CONFIG.get('max_4h_training_candles', 2000)
            use_all_history = DATA_CONFIG.get('use_all_history', False)
            
            # If use_all_history is True, override training_limit to unlimited
            if use_all_history:
                training_limit = 0  # 0 means unlimited
            
            self.logger.info(f"Preparing data (timeframe={TRADING_CONFIG['timeframe']}) use_all_history={use_all_history} selection_limit={selection_limit} training_limit={training_limit}")
            self.logger.info(f"Preparing training data for symbols: {symbols}")
            
            # Load data from database
            session = db_connection.get_session()
            
            all_data = []
            total_raw_rows = 0
            total_aligned_rows = 0
            earliest_ts = None
            latest_ts = None
            
            for symbol in symbols:
                self.logger.info(f"Loading data for {symbol}")
                
                # Get candlestick data, ordered by timestamp
                if use_all_history:
                    # Fetch ALL rows for this symbol without LIMIT
                    query = session.query(Candle).filter(
                        Candle.symbol == symbol
                    ).order_by(Candle.timestamp.asc())  # ASC for chronological order when loading all
                else:
                    # Use existing limit-based logic  
                    query = session.query(Candle).filter(
                        Candle.symbol == symbol
                    ).order_by(Candle.timestamp.desc()).limit(train_samples // len(symbols))
                
                candles = query.all()
                raw_count = len(candles)
                total_raw_rows += raw_count
                
                if not candles:
                    self.logger.warning(f"No data found for {symbol}")
                    continue

                # Convert to DataFrame
                symbol_data = pd.DataFrame([{
                    'timestamp': candle.timestamp,
                    'open': candle.open,
                    'high': candle.high,
                    'low': candle.low,
                    'close': candle.close,
                    'volume': candle.volume,
                    'symbol': candle.symbol
                } for candle in candles])
                
                # Sort chronologically for proper indicator calculation
                symbol_data = symbol_data.sort_values('timestamp').reset_index(drop=True)
                
                # Track timestamps for span calculation
                if len(symbol_data) > 0:
                    symbol_earliest = symbol_data['timestamp'].min()
                    symbol_latest = symbol_data['timestamp'].max()
                    if earliest_ts is None or symbol_earliest < earliest_ts:
                        earliest_ts = symbol_earliest
                    if latest_ts is None or symbol_latest > latest_ts:
                        latest_ts = symbol_latest

                # For 4h timeframe, filter to aligned candles
                if TRADING_CONFIG['timeframe'] == '4h':
                    aligned_mask = symbol_data['timestamp'].apply(self._is_aligned_4h)
                    symbol_data = symbol_data[aligned_mask].reset_index(drop=True)
                    aligned_count = len(symbol_data)
                    total_aligned_rows += aligned_count

                    # Apply training limit ONLY if not using all history and limit > 0
                    if not use_all_history and training_limit and training_limit > 0 and len(symbol_data) > training_limit:
                        symbol_data = symbol_data.tail(training_limit).reset_index(drop=True)
                    
                    used_for_training = len(symbol_data)
                    
                    # Calculate span in days for this symbol
                    if len(symbol_data) > 0:
                        span_seconds = symbol_data['timestamp'].max() - symbol_data['timestamp'].min()
                        span_days = span_seconds / (24 * 3600)
                        oldest_ts = symbol_data['timestamp'].min()
                        newest_ts = symbol_data['timestamp'].max()
                        self.logger.info(f"[{symbol}] total_raw={raw_count} aligned_4h={aligned_count} used_for_training={used_for_training} oldest_ts={oldest_ts} newest_ts={newest_ts} span_days={span_days:.1f}")
                    else:
                        self.logger.info(f"[{symbol}] total_raw={raw_count} aligned_4h={aligned_count} used_for_training={used_for_training}")
                else:
                    self.logger.info(f"[{symbol}] raw={raw_count} used_for_training={len(symbol_data)}")
                
                all_data.append(symbol_data)
            
            session.close()
            
            if not all_data:
                raise ValueError("No training data available")

            # Combine all symbols
            combined_data = pd.concat(all_data, ignore_index=True)
            combined_data = combined_data.sort_values(['symbol', 'timestamp']).reset_index(drop=True)
            
            # Performance monitoring: log memory usage after data loading
            try:
                import psutil
                memory_mb = psutil.Process().memory_info().rss / 1024 / 1024
                self.logger.info(f"Memory usage after data loading: {memory_mb:.1f} MB")
            except ImportError:
                self.logger.info("psutil not available - skipping memory monitoring")
            
            # Calculate full dataset summary
            if earliest_ts and latest_ts:
                total_span_days = (latest_ts - earliest_ts) / (24 * 3600)
                self.logger.info(f"FULL DATASET SUMMARY: rows={len(combined_data)} symbols={len(symbols)} total_raw={total_raw_rows} aligned_4h={total_aligned_rows} earliest={earliest_ts} latest={latest_ts} span_days={total_span_days:.1f}")
            else:
                self.logger.info(f"FULL DATASET SUMMARY: rows={len(combined_data)} symbols={len(symbols)} total_raw={total_raw_rows} aligned_4h={total_aligned_rows}")
            
            self.logger.info(f"Loaded {len(combined_data)} total samples")
            
            self.training_progress.update({
                'stage': 'indicator_calculation',
                'progress': 30,
                'message': 'Calculating technical indicators'
            })
            
            # Calculate indicators for each symbol separately to maintain integrity
            processed_data = []
            for symbol in symbols:
                symbol_df = combined_data[combined_data['symbol'] == symbol].copy()
                if len(symbol_df) > 0:
                    self.logger.info(f"Calculating indicators for {symbol} ({len(symbol_df)} samples)")
                    symbol_df_with_indicators = self.calculator.calculate_all_indicators(symbol_df)
                    processed_data.append(symbol_df_with_indicators)
                    
                    # Memory management: free the symbol_df copy
                    del symbol_df
            
            if not processed_data:
                raise ValueError("No processed data available")
            
            # Combine processed data (memory-safe: process in place)
            features_df = pd.concat(processed_data, ignore_index=True)
            
            # Memory management: clear processed_data list and combined_data
            del processed_data
            del combined_data
            
            # Performance monitoring: log memory usage after indicator calculation
            try:
                import psutil
                memory_mb = psutil.Process().memory_info().rss / 1024 / 1024
                self.logger.info(f"Memory usage after indicator calculation: {memory_mb:.1f} MB")
            except ImportError:
                pass
            
            self.training_progress.update({
                'stage': 'label_generation',
                'progress': 50,
                'message': 'Generating trading labels'
            })
            
            # Generate labels across full data
            labels = self._generate_labels(features_df)
            
            # Remove non-feature columns (memory-efficient: use column selection)
            feature_columns = [col for col in features_df.columns 
                             if col not in ['timestamp', 'symbol', 'open', 'high', 'low', 'close', 'volume']]
            
            # Memory-safe feature extraction
            X_full = features_df[feature_columns]
            y_full = labels
            
            # Clean full data
            X_full = self._clean_training_data(X_full, y_full)
            
            # Remove rows where labels are NaN from full data
            valid_idx = ~y_full.isna()
            X_full = X_full[valid_idx]
            y_full = y_full[valid_idx]
            
            # Build selection subset post-indicator/label generation to avoid recomputation
            # Take last selection_limit samples per symbol from the already processed data
            selection_indices = []
            for symbol in symbols:
                symbol_mask = features_df['symbol'] == symbol
                symbol_indices = features_df[symbol_mask].index
                
                if len(symbol_indices) > 0:
                    # Get the last selection_limit indices for this symbol
                    if selection_limit and len(symbol_indices) > selection_limit:
                        symbol_selection_indices = symbol_indices[-selection_limit:]
                    else:
                        symbol_selection_indices = symbol_indices
                    selection_indices.extend(symbol_selection_indices.tolist())
            
            # Create selection subset using the indices (memory-efficient: avoid copy when possible)
            selection_mask = X_full.index.isin(selection_indices)
            selection_X = X_full[selection_mask]
            selection_y = y_full[selection_mask]
            
            # Memory management: can now free features_df since we have X_full and selection sets
            del features_df
            
            # Log summary
            selection_rows = len(selection_X)
            self.logger.info(f"Selection subset built: rows={selection_rows} (~ {selection_limit} per symbol)")
            self.logger.info(f"Final Full Training Set: rows={len(X_full)} features={len(X_full.columns)} | Selection Set: rows={len(selection_X)}")
            
            # Log class distributions
            full_class_dist = y_full.value_counts().to_dict()
            selection_class_dist = selection_y.value_counts().to_dict()
            self.logger.info(f"Class distribution (full): {full_class_dist}")
            self.logger.info(f"Class distribution (selection subset): {selection_class_dist}")
            
            return X_full, y_full, selection_X, selection_y
            
        except Exception as e:
            self.logger.error(f"Error preparing training data: {e}")
            raise
    
    def _generate_labels(self, df: pd.DataFrame) -> pd.Series:
        """
        Generate trading labels based on future price movement
        
        Labels:
        0 = SELL (price will decrease)
        1 = BUY (price will increase) 
        2 = HOLD (minimal price movement)
        """
        labels = []
        
        # Group by symbol to calculate labels properly
        for symbol in df['symbol'].unique():
            symbol_df = df[df['symbol'] == symbol].copy().sort_values('timestamp')
            
            symbol_labels = []
            for i in range(len(symbol_df)):
                if i < len(symbol_df) - 4:  # Need 4 periods ahead for 4h timeframe
                    current_price = symbol_df.iloc[i]['close']
                    future_price = symbol_df.iloc[i + 4]['close']  # 4 periods = 16 hours ahead
                    
                    price_change_pct = (future_price - current_price) / current_price * 100
                    
                    # Define thresholds for signals
                    if price_change_pct > 2.0:  # > 2% increase
                        label = 1  # BUY
                    elif price_change_pct < -2.0:  # > 2% decrease  
                        label = 0  # SELL
                    else:
                        label = 2  # HOLD
                else:
                    label = np.nan  # Can't determine future for last samples
                
                symbol_labels.append(label)
            
            labels.extend(symbol_labels)
        
        return pd.Series(labels)
    
    def _clean_training_data(self, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        """Clean training data"""
        # Remove columns with all NaN
        X = X.dropna(axis=1, how='all')
        
        # Remove columns with too many NaN (>50%)
        nan_threshold = len(X) * 0.5
        X = X.dropna(axis=1, thresh=nan_threshold)
        
        # Fill remaining NaN with forward fill then backward fill (fixed deprecated method)
        X = X.ffill().bfill()
        
        # Remove infinite values
        X = X.replace([np.inf, -np.inf], np.nan)
        X = X.fillna(0)
        
        return X
    
    def train_with_rfe(self, retrain: bool = False) -> Dict[str, Any]:
        """
        Complete training pipeline with RFE feature selection
        
        Args:
            retrain: Whether to retrain from scratch
            
        Returns:
            Training results and metrics
        """
        try:
            self.logger.info("Starting complete model training with RFE")
            
            training_start = datetime.now()
            
            # Record training start
            session = db_connection.get_session()
            training_record = ModelTraining(
                model_version=f"model_{training_start.strftime('%Y%m%d_%H%M%S')}",
                training_start=training_start,
                status='TRAINING'
            )
            session.add(training_record)
            session.commit()
            training_id = training_record.id
            session.close()
            
            self.training_progress.update({
                'stage': 'initialization',
                'progress': 5,
                'message': 'Starting training pipeline'
            })
            
            # Prepare training data
            X_full, y_full, selection_X, selection_y = self.prepare_training_data()
            
            # Update training record with data info
            session = db_connection.get_session()
            training_record = session.query(ModelTraining).get(training_id)
            training_record.total_samples = len(X_full)
            training_record.total_indicators = len(X_full.columns)
            session.commit()
            session.close()
            
            # Check if feature selection is enabled
            feature_selection_enabled = FEATURE_SELECTION_CONFIG.get('enabled', True)
            
            if feature_selection_enabled:
                self.training_progress.update({
                    'stage': 'feature_selection',
                    'progress': 60,
                    'message': 'Performing feature selection'
                })
                
                # Get feature selection configuration
                feature_selection_mode = get_config_value('feature_selection.mode', 'rfe')
                target_features = get_config_value('feature_selection.target_features', 30)
                
                # Log class distributions - note these are already logged in prepare_training_data
                # but we maintain this for backward compatibility with existing logging expectations
                self.logger.info(f"Class distribution (full training): {y_full.value_counts().to_dict()}")
                self.logger.info(f"Class distribution (selection subset): {selection_y.value_counts().to_dict()}")
                
                # Feature selection with config-based mode switching
                self.feature_selector = FeatureSelector(n_features_to_select=target_features)
                
                # Get must-include features (prerequisites and required indicators)
                required_indicators = self.definitions.get_required_indicators()
                must_include = list(required_indicators.keys())
                
                # Use selection subset for feature selection (recent data regime)
                X_recent = selection_X
                y_recent = selection_y
                
                # Perform feature selection based on mode
                if feature_selection_mode == 'dynamic':
                    self.logger.info("Using dynamic feature selection")
                    dynamic_config = get_config_value('feature_selection.dynamic', {})
                    selected_features, selection_info = self.feature_selector.select_features_dynamic(
                        X_recent, y_recent, 
                        must_include=must_include,
                        min_features=dynamic_config.get('min_features', 20),
                        drop_fraction=dynamic_config.get('drop_fraction', 0.05),
                        corr_threshold=dynamic_config.get('corr_threshold', 0.95),
                        tolerance=dynamic_config.get('tolerance', 0.003),
                        metric=dynamic_config.get('metric', 'macro_f1'),
                        cv_splits=dynamic_config.get('cv_splits', 3),
                        max_iterations=dynamic_config.get('max_iterations', 50)
                    )
                elif feature_selection_mode == 'hybrid':
                    self.logger.info("Using hybrid feature selection")
                    selected_features, selection_info = self.feature_selector.select_features_hybrid(
                        X_recent, y_recent, must_include=must_include
                    )
                else:  # default to RFE
                    self.logger.info("Using RFE feature selection")
                    selected_features, selection_info = self.feature_selector.select_features_rfe(
                        X_recent, y_recent, must_include=must_include
                    )
                
                self.logger.info(f"Selected {len(selected_features)} features using {feature_selection_mode} method")
                self.logger.info(f"Selection details: {selection_info}")
                
                # Enhanced logging for dynamic feature selection
                if feature_selection_mode == 'dynamic' and 'history' in selection_info:
                    self.logger.info("=== Dynamic Feature Selection Summary ===")
                    self.logger.info(f"Baseline score: {selection_info.get('baseline_score', 0.0):.4f}")
                    self.logger.info(f"Final score: {selection_info.get('final_score', 0.0):.4f}")
                    self.logger.info(f"Improvement: {selection_info.get('improvement', 0.0):.4f}")
                    self.logger.info(f"Total iterations: {selection_info.get('iterations', 0)}")
                    self.logger.info(f"Correlation removed: {selection_info.get('correlation_removed', 0)} features")
                    
                    # Log iteration details
                    for entry in selection_info['history'][:5]:  # Log first 5 iterations
                        iteration = entry.get('iteration', 0)
                        score = entry.get('score', 0.0)
                        feature_count = entry.get('features_count', 0)
                        action = entry.get('action', 'unknown')
                        self.logger.info(f"  Iteration {iteration}: {score:.4f} score, {feature_count} features, {action}")
                    
                    if len(selection_info['history']) > 5:
                        self.logger.info(f"  ... and {len(selection_info['history']) - 5} more iterations")
                    self.logger.info("===========================================")
            else:
                # Feature selection disabled - use all available features
                self.logger.info("Feature selection disabled via config; using all available features")
                feature_columns = [col for col in X_full.columns 
                                 if col not in ['timestamp', 'symbol', 'open', 'high', 'low', 'close', 'volume']]
                selected_features = feature_columns
                selection_info = {
                    'method': 'disabled_all_features',
                    'total_features': len(selected_features),
                    'note': 'Feature selection bypassed via FEATURE_SELECTION_CONFIG.enabled=False'
                }
                self.logger.info(f"Using all {len(selected_features)} features (feature selection disabled)")
                
                self.training_progress.update({
                    'stage': 'feature_selection_bypassed',
                    'progress': 60,
                    'message': f'Feature selection bypassed - using all {len(selected_features)} features'
                })
            
            # Log comprehensive training setup before starting model training
            features_before_selection = len(X_full.columns)
            features_after_selection = len(selected_features)
            xgb_trees = XGB_PRO_CONFIG.get('n_estimators', 8000)
            
            self.logger.info(f"FULL 4h TRAINING ROWS={len(X_full)} FEATURES(before selection)={features_before_selection} FEATURE_SELECTION_ENABLED={feature_selection_enabled} FEATURES_FINAL={features_after_selection} XGB_TREES={xgb_trees}")
            
            # Filter training data to selected features (apply to full training set)
            X_selected = X_full[selected_features]
            
            self.training_progress.update({
                'stage': 'model_training',
                'progress': 80,
                'message': 'Training AI model'
            })
            
            # Split data (using full training dataset)
            X_train, X_test, y_train, y_test = train_test_split(
                X_selected, y_full, 
                test_size=ML_CONFIG['test_size'],
                random_state=ML_CONFIG['random_state'],
                stratify=y_full
            )
            
            # Scale features
            X_train_scaled = pd.DataFrame(
                self.scaler.fit_transform(X_train),
                columns=X_train.columns,
                index=X_train.index
            )
            
            X_test_scaled = pd.DataFrame(
                self.scaler.transform(X_test),
                columns=X_test.columns,
                index=X_test.index
            )
            
            # Initialize and train model
            self.model = TradingModel(model_type='xgboost_professional')
            self.model.set_confidence_threshold(TRADING_CONFIG['confidence_threshold'])
            
            training_metrics = self.model.train(
                X_train_scaled, y_train, 
                X_test_scaled, y_test,
                optimize_hyperparameters=True
            )
            
            training_end = datetime.now()
            
            # Update training record with results
            session = db_connection.get_session()
            training_record = session.query(ModelTraining).get(training_id)
            training_record.training_end = training_end
            training_record.training_samples = len(X_train)
            training_record.test_samples = len(X_test)
            training_record.selected_indicators = len(selected_features)
            training_record.selected_features = str(selected_features)
            training_record.accuracy = training_metrics.get('val_accuracy', 0.0)
            training_record.precision = training_metrics.get('precision', 0.0)
            training_record.recall = training_metrics.get('recall', 0.0)
            training_record.f1_score = training_metrics.get('f1_score', 0.0)
            training_record.status = 'COMPLETED'
            session.commit()
            session.close()
            
            # Enhance selection_info with metadata for serialization/inference
            enhanced_selection_info = selection_info.copy()
            if 'history' in selection_info:
                # For dynamic selection, add summary to the main info
                enhanced_selection_info['selection_history_summary'] = {
                    'iterations_count': len(selection_info['history']) - 1,
                    'baseline_score': selection_info.get('baseline_score', 0.0),
                    'final_score': selection_info.get('final_score', 0.0),
                    'improvement': selection_info.get('improvement', 0.0),
                    'metric_used': selection_info.get('metric', 'unknown')
                }
                
                # Optional history truncation for storage efficiency
                max_history_length = get_config_value('feature_selection.dynamic.max_history_length', 100)
                if len(selection_info['history']) > max_history_length:
                    self.logger.info(f"Truncating selection history from {len(selection_info['history'])} to {max_history_length} entries")
                    # Keep baseline (first entry) and last N-1 entries
                    truncated_history = [selection_info['history'][0]]  # baseline
                    truncated_history.extend(selection_info['history'][-(max_history_length-1):])  # recent entries
                    enhanced_selection_info['history'] = truncated_history
                    enhanced_selection_info['history_truncated'] = True
                    enhanced_selection_info['original_history_length'] = len(selection_info['history'])
                else:
                    enhanced_selection_info['history_truncated'] = False
            
            self.training_progress.update({
                'stage': 'completed',
                'progress': 100,
                'message': 'Model training completed successfully'
            })
            
            # Save model
            model_path = f"models/{self.model.model_version}.joblib"
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            self.model.save_model(model_path)
            
            result = {
                'model_version': self.model.model_version,
                'training_time': (training_end - training_start).total_seconds(),
                'selected_features': selected_features,
                'selection_info': enhanced_selection_info,
                'training_metrics': training_metrics,
                'model_path': model_path,
                'feature_selection_enabled': feature_selection_enabled  # Add flag to result
            }
            
            # Add note about feature selection if it was disabled
            if not feature_selection_enabled:
                result['feature_selection_note'] = 'Feature selection bypassed via FEATURE_SELECTION_CONFIG.enabled=False'
            
            self.logger.info(f"Training completed successfully: {result}")
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error in training pipeline: {e}")
            
            # Update training record with error
            try:
                session = db_connection.get_session()
                if 'training_id' in locals():
                    training_record = session.query(ModelTraining).get(training_id)
                    training_record.status = 'FAILED'
                    training_record.error_message = str(e)
                    session.commit()
                session.close()
            except:
                pass
            
            self.training_progress.update({
                'stage': 'failed',
                'progress': 0,
                'message': f'Training failed: {str(e)}'
            })
            
            raise
    
    def evaluate_model_performance(self, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, Any]:
        """Evaluate trained model performance"""
        if not self.model or not self.model.is_trained:
            raise ValueError("Model must be trained before evaluation")
        
        try:
            # Scale test data
            X_test_scaled = pd.DataFrame(
                self.scaler.transform(X_test),
                columns=X_test.columns,
                index=X_test.index
            )
            
            # Get predictions
            predictions, confidence_scores = self.model.predict(X_test_scaled)
            
            # Calculate metrics
            from sklearn.metrics import accuracy_score, precision_recall_fscore_support
            
            accuracy = accuracy_score(y_test, predictions)
            precision, recall, f1, _ = precision_recall_fscore_support(y_test, predictions, average='weighted')
            
            # High confidence predictions
            high_conf_mask = confidence_scores >= self.model.confidence_threshold
            high_conf_accuracy = accuracy_score(
                y_test[high_conf_mask], 
                predictions[high_conf_mask]
            ) if high_conf_mask.any() else 0.0
            
            evaluation_metrics = {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'high_confidence_accuracy': high_conf_accuracy,
                'high_confidence_ratio': high_conf_mask.mean(),
                'mean_confidence': confidence_scores.mean(),
                'classification_report': classification_report(y_test, predictions, output_dict=True)
            }
            
            return evaluation_metrics
            
        except Exception as e:
            self.logger.error(f"Error evaluating model: {e}")
            raise
