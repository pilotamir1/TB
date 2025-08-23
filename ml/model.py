import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
import xgboost as xgb
from sklearn.model_selection import GridSearchCV
import joblib
import logging
from typing import Dict, List, Any, Tuple
import os
from datetime import datetime

class TradingModel:
    """
    AI Trading Model for signal generation
    """
    
    def __init__(self, model_type: str = 'random_forest'):
        self.model_type = model_type
        self.model = None
        self.feature_names = []
        self.is_trained = False
        self.model_version = None
        self.confidence_threshold = 0.7
        self.logger = logging.getLogger(__name__)
        
        # Initialize model based on type
        self._initialize_model()
    
    def _initialize_model(self):
        """Initialize the ML model based on specified type"""
        if self.model_type == 'random_forest':
            self.model = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42,
                n_jobs=-1
            )
        elif self.model_type == 'gradient_boosting':
            self.model = GradientBoostingClassifier(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=6,
                random_state=42
            )
        elif self.model_type == 'xgboost_professional':
            # Professional XGBoost configuration with millions of trees equivalent
            self.model = xgb.XGBClassifier(
                n_estimators=5000,  # 5000 trees - professional deep model
                max_depth=12,       # Deep trees for complex patterns
                learning_rate=0.01, # Low learning rate for better generalization
                subsample=0.8,      # Prevent overfitting
                colsample_bytree=0.8,
                colsample_bylevel=0.8,
                min_child_weight=1,
                gamma=0.1,
                reg_alpha=0.1,      # L1 regularization
                reg_lambda=1.0,     # L2 regularization
                random_state=42,
                n_jobs=-1,
                tree_method='hist',  # Efficient tree construction
                enable_categorical=False,
                eval_metric='mlogloss',
                early_stopping_rounds=100,
                verbosity=1
            )
        elif self.model_type == 'logistic_regression':
            self.model = LogisticRegression(
                random_state=42,
                max_iter=1000
            )
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
    
    def train(self, X_train: pd.DataFrame, y_train: pd.Series, 
              X_val: pd.DataFrame = None, y_val: pd.Series = None,
              optimize_hyperparameters: bool = True) -> Dict[str, Any]:
        """
        Train the trading model
        
        Args:
            X_train: Training features
            y_train: Training labels (0=SELL, 1=BUY, 2=HOLD)
            X_val: Validation features (optional)
            y_val: Validation labels (optional)
            optimize_hyperparameters: Whether to perform hyperparameter optimization
        
        Returns:
            Dictionary with training metrics
        """
        try:
            self.feature_names = list(X_train.columns)
            
            if optimize_hyperparameters and self.model_type == 'random_forest':
                # Hyperparameter optimization
                param_grid = {
                    'n_estimators': [50, 100, 200],
                    'max_depth': [6, 10, 15, None],
                    'min_samples_split': [2, 5, 10],
                    'min_samples_leaf': [1, 2, 4]
                }
                
                self.logger.info("Starting hyperparameter optimization...")
                grid_search = GridSearchCV(
                    self.model, param_grid, cv=5, scoring='accuracy',
                    n_jobs=-1, verbose=1
                )
                grid_search.fit(X_train, y_train)
                self.model = grid_search.best_estimator_
                self.logger.info(f"Best parameters: {grid_search.best_params_}")
            elif self.model_type == 'xgboost_professional':
                # Professional XGBoost training with validation
                self.logger.info("Training professional XGBoost model with 5000 trees...")
                if X_val is not None and y_val is not None:
                    # Train with early stopping using validation set
                    self.model.fit(
                        X_train, y_train,
                        eval_set=[(X_val, y_val)],
                        verbose=100  # Print every 100 iterations
                    )
                else:
                    # Train without early stopping
                    self.model.fit(X_train, y_train, verbose=100)
                self.logger.info("Professional XGBoost model training completed")
            else:
                # Regular training
                self.model.fit(X_train, y_train)
            
            # Calculate training metrics
            train_accuracy = self.model.score(X_train, y_train)
            
            metrics = {
                'train_accuracy': train_accuracy,
                'feature_count': len(self.feature_names),
                'training_samples': len(X_train)
            }
            
            if X_val is not None and y_val is not None:
                val_accuracy = self.model.score(X_val, y_val)
                metrics['val_accuracy'] = val_accuracy
                
                # Get predictions with probabilities
                y_pred_proba = self.model.predict_proba(X_val)
                y_pred = self.model.predict(X_val)
                
                # Calculate precision, recall, F1 for each class
                from sklearn.metrics import classification_report, confusion_matrix
                report = classification_report(y_val, y_pred, output_dict=True)
                metrics.update({
                    'precision': report['weighted avg']['precision'],
                    'recall': report['weighted avg']['recall'],
                    'f1_score': report['weighted avg']['f1-score'],
                    'confusion_matrix': confusion_matrix(y_val, y_pred).tolist()
                })
            
            self.is_trained = True
            self.model_version = f"model_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            self.logger.info(f"Model trained successfully. Accuracy: {train_accuracy:.4f}")
            return metrics
            
        except Exception as e:
            self.logger.error(f"Error training model: {e}")
            raise
    
    def predict(self, X: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate trading predictions
        
        Args:
            X: Features for prediction
        
        Returns:
            Tuple of (predictions, confidence_scores)
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        try:
            # Get probability predictions
            probabilities = self.model.predict_proba(X)
            
            # Get class predictions
            predictions = self.model.predict(X)
            
            # Calculate confidence as maximum probability
            confidence_scores = np.max(probabilities, axis=1)
            
            return predictions, confidence_scores
            
        except Exception as e:
            self.logger.error(f"Error making predictions: {e}")
            raise
    
    def predict_single(self, features: Dict[str, float]) -> Dict[str, Any]:
        """
        Predict for a single sample
        
        Args:
            features: Dictionary of feature values
        
        Returns:
            Dictionary with prediction, confidence, and probabilities
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        try:
            # Convert to DataFrame with correct column order
            X = pd.DataFrame([features])[self.feature_names]
            
            # Get probabilities
            probabilities = self.model.predict_proba(X)[0]
            prediction = self.model.predict(X)[0]
            confidence = np.max(probabilities)
            
            # Map prediction to signal
            signal_map = {0: 'SELL', 1: 'BUY', 2: 'HOLD'}
            signal = signal_map.get(prediction, 'HOLD')
            
            return {
                'signal': signal,
                'confidence': confidence,
                'probabilities': {
                    'SELL': probabilities[0],
                    'BUY': probabilities[1] if len(probabilities) > 1 else 0.0,
                    'HOLD': probabilities[2] if len(probabilities) > 2 else 0.0
                },
                'meets_threshold': confidence >= self.confidence_threshold
            }
            
        except Exception as e:
            self.logger.error(f"Error making single prediction: {e}")
            raise
    
    def get_feature_importance(self) -> pd.DataFrame:
        """Get feature importance from trained model"""
        if not self.is_trained:
            raise ValueError("Model must be trained to get feature importance")
        
        if hasattr(self.model, 'feature_importances_'):
            importance_df = pd.DataFrame({
                'feature': self.feature_names,
                'importance': self.model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            return importance_df
        else:
            self.logger.warning("Model does not support feature importance")
            return pd.DataFrame()
    
    def save_model(self, filepath: str) -> bool:
        """Save trained model to file"""
        if not self.is_trained:
            raise ValueError("Model must be trained before saving")
        
        try:
            model_data = {
                'model': self.model,
                'feature_names': self.feature_names,
                'model_type': self.model_type,
                'model_version': self.model_version,
                'confidence_threshold': self.confidence_threshold,
                'is_trained': self.is_trained
            }
            
            joblib.dump(model_data, filepath)
            self.logger.info(f"Model saved to {filepath}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error saving model: {e}")
            return False
    
    def load_model(self, filepath: str) -> bool:
        """Load trained model from file"""
        try:
            if not os.path.exists(filepath):
                self.logger.error(f"Model file not found: {filepath}")
                return False
            
            model_data = joblib.load(filepath)
            
            self.model = model_data['model']
            self.feature_names = model_data['feature_names']
            self.model_type = model_data['model_type']
            self.model_version = model_data['model_version']
            self.confidence_threshold = model_data['confidence_threshold']
            self.is_trained = model_data['is_trained']
            
            self.logger.info(f"Model loaded from {filepath}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error loading model: {e}")
            return False
    
    def set_confidence_threshold(self, threshold: float):
        """Set confidence threshold for signal generation"""
        if 0.0 <= threshold <= 1.0:
            self.confidence_threshold = threshold
            self.logger.info(f"Confidence threshold set to {threshold}")
        else:
            raise ValueError("Confidence threshold must be between 0.0 and 1.0")
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information"""
        return {
            'model_type': self.model_type,
            'model_version': self.model_version,
            'is_trained': self.is_trained,
            'feature_count': len(self.feature_names),
            'confidence_threshold': self.confidence_threshold,
            'feature_names': self.feature_names.copy()
        }