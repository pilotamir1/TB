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
from config.settings import ML_CONFIG, TRADING_CONFIG

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
    
    def get_training_progress(self) -> Dict[str, Any]:
        """Get current training progress"""
        return self.training_progress.copy()
    
    def prepare_training_data(self, symbols: List[str] = None, 
                            train_samples: int = None) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Prepare training data from database with indicators
        
        Args:
            symbols: List of symbols to include
            train_samples: Number of samples to use for training
            
        Returns:
            Tuple of (features, labels)
        """
        try:
            self.training_progress.update({
                'stage': 'data_preparation',
                'progress': 10,
                'message': 'Loading historical data from database'
            })
            
            symbols = symbols or TRADING_CONFIG['symbols']
            train_samples = train_samples or ML_CONFIG['training_data_size']
            
            self.logger.info(f"Preparing training data for symbols: {symbols}")
            
            # Load data from database
            session = db_connection.get_session()
            
            all_data = []
            for symbol in symbols:
                self.logger.info(f"Loading data for {symbol}")
                
                # Get candlestick data, ordered by timestamp
                query = session.query(Candle).filter(
                    Candle.symbol == symbol
                ).order_by(Candle.timestamp.desc()).limit(train_samples // len(symbols))
                
                candles = query.all()
                
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
                
                symbol_data = symbol_data.sort_values('timestamp').reset_index(drop=True)
                all_data.append(symbol_data)
            
            session.close()
            
            if not all_data:
                raise ValueError("No training data available")
            
            # Combine all symbols
            combined_data = pd.concat(all_data, ignore_index=True)
            combined_data = combined_data.sort_values(['symbol', 'timestamp']).reset_index(drop=True)
            
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
            
            if not processed_data:
                raise ValueError("No processed data available")
            
            # Combine processed data
            features_df = pd.concat(processed_data, ignore_index=True)
            
            self.training_progress.update({
                'stage': 'label_generation',
                'progress': 50,
                'message': 'Generating trading labels'
            })
            
            # Generate labels
            labels = self._generate_labels(features_df)
            
            # Remove non-feature columns
            feature_columns = [col for col in features_df.columns 
                             if col not in ['timestamp', 'symbol', 'open', 'high', 'low', 'close', 'volume']]
            
            X = features_df[feature_columns].copy()
            y = labels
            
            # Clean data
            X = self._clean_training_data(X, y)
            
            # Remove rows where labels are NaN
            valid_idx = ~y.isna()
            X = X[valid_idx]
            y = y[valid_idx]
            
            self.logger.info(f"Training data prepared: {len(X)} samples, {len(X.columns)} features")
            
            return X, y
            
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
        
        # Fill remaining NaN with forward fill then backward fill
        X = X.fillna(method='ffill').fillna(method='bfill')
        
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
            X, y = self.prepare_training_data()
            
            # Update training record with data info
            session = db_connection.get_session()
            training_record = session.query(ModelTraining).get(training_id)
            training_record.total_samples = len(X)
            training_record.total_indicators = len(X.columns)
            session.commit()
            session.close()
            
            self.training_progress.update({
                'stage': 'feature_selection',
                'progress': 60,
                'message': 'Performing RFE feature selection'
            })
            
            # Feature selection
            self.feature_selector = FeatureSelector(n_features_to_select=ML_CONFIG['selected_features'])
            
            # Get must-include features (prerequisites and required indicators)
            required_indicators = self.definitions.get_required_indicators()
            must_include = list(required_indicators.keys())
            
            # Get features to select from recent data for RFE
            recent_samples = min(ML_CONFIG['rfe_sample_size'], len(X))
            X_recent = X.tail(recent_samples)
            y_recent = y.tail(recent_samples)
            
            # Perform feature selection on recent data
            selected_features, selection_info = self.feature_selector.select_features_rfe(
                X_recent, y_recent, must_include=must_include
            )
            
            self.logger.info(f"Selected {len(selected_features)} features: {selection_info}")
            
            # Filter training data to selected features
            X_selected = X[selected_features]
            
            self.training_progress.update({
                'stage': 'model_training',
                'progress': 80,
                'message': 'Training AI model'
            })
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X_selected, y, 
                test_size=ML_CONFIG['test_size'],
                random_state=ML_CONFIG['random_state'],
                stratify=y
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
                'selection_info': selection_info,
                'training_metrics': training_metrics,
                'model_path': model_path
            }
            
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