"""
Enhanced Trading Model with CatBoost support, probability calibration, and comprehensive metrics
"""

import pandas as pd
import numpy as np
import logging
import joblib
import json
import os
from datetime import datetime
from typing import Dict, List, Any, Tuple, Optional
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score,
    precision_recall_fscore_support, roc_auc_score
)
from sklearn.calibration import CalibratedClassifierCV
from sklearn.utils.class_weight import compute_class_weight
from sklearn.preprocessing import LabelEncoder

# Try to import advanced models
try:
    import xgboost as xgb
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False

try:
    from catboost import CatBoostClassifier
    HAS_CATBOOST = True
except ImportError:
    HAS_CATBOOST = False

class EnhancedTradingModel:
    """Enhanced AI Trading Model with multiple algorithm support"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.ml_config = config.get('ml', {})
        self.model_type = self.ml_config.get('model_type', 'xgboost')
        
        self.model = None
        self.calibrated_model = None
        self.label_encoder = LabelEncoder()
        self.feature_names = []
        self.is_trained = False
        self.model_version = None
        self.training_metadata = {}
        
        self.logger = logging.getLogger(__name__)
        
        # Initialize model
        self._initialize_model()
    
    def _initialize_model(self):
        """Initialize the ML model based on configuration"""
        if self.model_type == 'catboost' and HAS_CATBOOST:
            self._initialize_catboost()
        elif self.model_type == 'xgboost' and HAS_XGBOOST:
            self._initialize_xgboost()
        elif self.model_type == 'random_forest':
            self._initialize_random_forest()
        else:
            # Fallback to random forest
            self.logger.warning(f"Model type {self.model_type} not available, using Random Forest")
            self.model_type = 'random_forest'
            self._initialize_random_forest()
    
    def _initialize_catboost(self):
        """Initialize CatBoost model"""
        catboost_config = self.config.get('catboost', {})
        
        self.model = CatBoostClassifier(
            iterations=catboost_config.get('iterations', 1500),
            depth=catboost_config.get('depth', 6),
            learning_rate=catboost_config.get('learning_rate', 0.05),
            early_stopping_rounds=catboost_config.get('early_stopping_rounds', 100),
            loss_function=catboost_config.get('loss_function', 'MultiClass'),
            auto_class_weights=catboost_config.get('auto_class_weights', 'Balanced'),
            random_seed=self.ml_config.get('random_state', 42),
            verbose=False,  # Reduce noise
            allow_writing_files=False  # Prevent temp file creation
        )
        
        self.logger.info("Initialized CatBoost model")
    
    def _initialize_xgboost(self):
        """Initialize XGBoost model with enhanced overfitting prevention"""
        xgboost_config = self.config.get('xgboost', {})
        
        self.model = xgb.XGBClassifier(
            n_estimators=xgboost_config.get('n_estimators', 1500),
            max_depth=xgboost_config.get('max_depth', 6),
            learning_rate=xgboost_config.get('learning_rate', 0.05),
            early_stopping_rounds=xgboost_config.get('early_stopping_rounds', 100),
            subsample=xgboost_config.get('subsample', 0.8),
            colsample_bytree=xgboost_config.get('colsample_bytree', 0.8),
            colsample_bylevel=xgboost_config.get('colsample_bylevel', 0.8),  # Additional regularization
            reg_alpha=xgboost_config.get('reg_alpha', 0.1),  # L1 regularization
            reg_lambda=xgboost_config.get('reg_lambda', 1.0),  # L2 regularization
            min_child_weight=xgboost_config.get('min_child_weight', 3),  # Prevent overfitting
            gamma=xgboost_config.get('gamma', 0.1),  # Minimum split loss
            objective='multi:softprob',
            eval_metric='mlogloss',
            random_state=self.ml_config.get('random_state', 42),
            verbosity=0,  # Reduce noise
            n_jobs=-1,  # Use all CPU cores
            tree_method='hist'  # Faster training
        )
        
        self.logger.info("Initialized XGBoost model with enhanced regularization")
    
    def _initialize_random_forest(self):
        """Initialize Random Forest model"""
        self.model = RandomForestClassifier(
            n_estimators=300,
            max_depth=15,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=self.ml_config.get('random_state', 42),
            n_jobs=-1
        )
        
        self.logger.info("Initialized Random Forest model")
    
    def _prepare_labels(self, y: pd.Series) -> np.ndarray:
        """Prepare and encode labels"""
        # Encode string labels to numeric
        y_encoded = self.label_encoder.fit_transform(y)
        return y_encoded
    
    def _compute_class_weights(self, y: np.ndarray) -> Dict[int, float]:
        """Compute class weights for balanced training"""
        unique_classes = np.unique(y)
        class_weights = compute_class_weight('balanced', classes=unique_classes, y=y)
        return dict(zip(unique_classes, class_weights))
    
    def _perform_cross_validation(self, X: pd.DataFrame, y: np.ndarray) -> Dict[str, float]:
        """Perform stratified cross-validation"""
        cv_config = self.ml_config.get('cross_validation', {})
        
        if not cv_config.get('enabled', True):
            return {}
        
        n_folds = cv_config.get('folds', 5)
        cv_strategy = StratifiedKFold(n_splits=n_folds, shuffle=True, 
                                    random_state=self.ml_config.get('random_state', 42))
        
        try:
            # Create a copy of the model for CV
            cv_model = self._create_model_copy()
            
            # Perform cross-validation
            cv_scores = cross_val_score(cv_model, X, y, cv=cv_strategy, scoring='accuracy', n_jobs=-1)
            
            return {
                'cv_mean_accuracy': float(cv_scores.mean()),
                'cv_std_accuracy': float(cv_scores.std()),
                'cv_scores': cv_scores.tolist()
            }
        except Exception as e:
            self.logger.warning(f"Cross-validation failed: {e}")
            return {}
    
    def _create_model_copy(self):
        """Create a copy of the model for cross-validation"""
        if self.model_type == 'catboost':
            return CatBoostClassifier(**self.model.get_params())
        elif self.model_type == 'xgboost':
            return xgb.XGBClassifier(**self.model.get_params())
        else:
            return RandomForestClassifier(**self.model.get_params())
    
    def _calibrate_probabilities(self, X_val: pd.DataFrame, y_val: np.ndarray):
        """Calibrate model probabilities"""
        calibration_config = self.ml_config.get('calibration', {})
        
        if not calibration_config.get('enabled', True):
            self.calibrated_model = self.model
            return
        
        method = calibration_config.get('method', 'isotonic')
        
        try:
            self.logger.info(f"Calibrating probabilities using {method} method")
            
            self.calibrated_model = CalibratedClassifierCV(
                self.model, 
                method=method, 
                cv=3
            )
            
            self.calibrated_model.fit(X_val, y_val)
            
            self.logger.info("Probability calibration completed")
            
        except Exception as e:
            self.logger.warning(f"Probability calibration failed: {e}")
            self.calibrated_model = self.model
    
    def train(self, X_train: pd.DataFrame, y_train: pd.Series, 
              X_val: pd.DataFrame = None, y_val: pd.Series = None,
              optimize_hyperparameters: bool = False) -> Dict[str, Any]:
        """Train the model with comprehensive evaluation"""
        
        training_start = datetime.now()
        self.logger.info(f"Starting {self.model_type} model training...")
        
        # Store feature names
        self.feature_names = X_train.columns.tolist()
        
        # Prepare labels
        y_train_encoded = self._prepare_labels(y_train)
        y_val_encoded = self.label_encoder.transform(y_val) if y_val is not None else None
        
        # Compute class weights
        class_weights = self._compute_class_weights(y_train_encoded)
        self.logger.info(f"Class distribution: {dict(zip(*np.unique(y_train_encoded, return_counts=True)))}")
        self.logger.info(f"Class weights: {class_weights}")
        
        # Perform cross-validation
        cv_results = self._perform_cross_validation(X_train, y_train_encoded)
        
        # Set class weights if supported
        if self.model_type in ['xgboost', 'random_forest']:
            # Convert to class_weight format expected by sklearn
            sklearn_weights = {i: weight for i, weight in class_weights.items()}
            if hasattr(self.model, 'class_weight'):
                self.model.class_weight = sklearn_weights
        
        # Train the model
        try:
            if self.model_type == 'catboost':
                # CatBoost handles class weights automatically with auto_class_weights
                if X_val is not None and y_val_encoded is not None:
                    self.model.fit(
                        X_train, y_train_encoded,
                        eval_set=(X_val, y_val_encoded),
                        verbose=False
                    )
                else:
                    self.model.fit(X_train, y_train_encoded, verbose=False)
                    
            elif self.model_type == 'xgboost':
                # Set sample weights for XGBoost
                sample_weights = np.array([class_weights[label] for label in y_train_encoded])
                
                if X_val is not None and y_val_encoded is not None:
                    val_sample_weights = np.array([class_weights[label] for label in y_val_encoded])
                    self.model.fit(
                        X_train, y_train_encoded,
                        sample_weight=sample_weights,
                        eval_set=[(X_val, y_val_encoded)],
                        sample_weight_eval_set=[val_sample_weights],
                        verbose=False
                    )
                else:
                    self.model.fit(X_train, y_train_encoded, sample_weight=sample_weights)
                    
            else:  # Random Forest
                self.model.fit(X_train, y_train_encoded)
            
            self.logger.info(f"{self.model_type} training completed")
            
        except Exception as e:
            self.logger.error(f"Model training failed: {e}")
            raise
        
        # Calculate training metrics
        y_train_pred = self.model.predict(X_train)
        train_accuracy = accuracy_score(y_train_encoded, y_train_pred)
        
        metrics = {
            'model_type': self.model_type,
            'train_accuracy': float(train_accuracy),
            'feature_count': len(self.feature_names),
            'training_samples': len(X_train),
            'class_distribution': dict(zip(*np.unique(y_train_encoded, return_counts=True))),
            'class_weights': class_weights
        }
        
        # Add cross-validation results
        metrics.update(cv_results)
        
        # Validation metrics
        if X_val is not None and y_val_encoded is not None:
            y_val_pred = self.model.predict(X_val)
            y_val_proba = self.model.predict_proba(X_val)
            
            val_accuracy = accuracy_score(y_val_encoded, y_val_pred)
            precision, recall, f1, _ = precision_recall_fscore_support(
                y_val_encoded, y_val_pred, average='weighted'
            )
            
            # Classification report
            report = classification_report(y_val_encoded, y_val_pred, output_dict=True)
            
            # Confusion matrix
            cm = confusion_matrix(y_val_encoded, y_val_pred)
            
            metrics.update({
                'val_accuracy': float(val_accuracy),
                'precision': float(precision),
                'recall': float(recall),
                'f1_score': float(f1),
                'confusion_matrix': cm.tolist(),
                'classification_report': report
            })
            
            # ROC AUC for multiclass
            try:
                if len(np.unique(y_val_encoded)) > 2:
                    auc_score = roc_auc_score(y_val_encoded, y_val_proba, multi_class='ovr')
                else:
                    auc_score = roc_auc_score(y_val_encoded, y_val_proba[:, 1])
                metrics['roc_auc'] = float(auc_score)
            except Exception as e:
                self.logger.warning(f"Could not calculate ROC AUC: {e}")
            
            # Calibrate probabilities
            self._calibrate_probabilities(X_val, y_val_encoded)
        
        # Get feature importances if available
        if hasattr(self.model, 'feature_importances_'):
            importance_dict = dict(zip(self.feature_names, self.model.feature_importances_))
            metrics['feature_importances'] = importance_dict
        elif self.model_type == 'catboost':
            importance_dict = dict(zip(self.feature_names, self.model.get_feature_importance()))
            metrics['feature_importances'] = importance_dict
        
        # Training time
        training_time = (datetime.now() - training_start).total_seconds()
        metrics['training_time_seconds'] = training_time
        
        # Model version
        self.model_version = datetime.now().strftime("%Y%m%d_%H%M%S")
        metrics['model_version'] = self.model_version
        metrics['training_timestamp'] = training_start.isoformat()
        
        # Git commit hash (if available)
        try:
            import subprocess
            git_hash = subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode('ascii').strip()
            metrics['git_commit_hash'] = git_hash
        except:
            metrics['git_commit_hash'] = 'unknown'
        
        # Store metadata
        self.training_metadata = metrics
        self.is_trained = True
        
        self.logger.info(f"Model training completed in {training_time:.1f}s")
        self.logger.info(f"Train accuracy: {train_accuracy:.4f}")
        if 'val_accuracy' in metrics:
            self.logger.info(f"Validation accuracy: {metrics['val_accuracy']:.4f}")
        
        return metrics
    
    def evaluate_test_set(self, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, Any]:
        """Evaluate model on independent test set"""
        try:
            if not self.is_trained:
                raise ValueError("Model must be trained before evaluation")
            
            # Encode test labels
            y_test_encoded = self.label_encoder.transform(y_test)
            
            # Make predictions
            y_test_pred = self.model.predict(X_test)
            y_test_proba = self.model.predict_proba(X_test)
            
            # Calculate metrics
            test_accuracy = accuracy_score(y_test_encoded, y_test_pred)
            precision, recall, f1, _ = precision_recall_fscore_support(
                y_test_encoded, y_test_pred, average='weighted'
            )
            
            # Confusion matrix
            cm = confusion_matrix(y_test_encoded, y_test_pred)
            
            test_metrics = {
                'test_accuracy': float(test_accuracy),
                'test_precision': float(precision),
                'test_recall': float(recall),
                'test_f1_score': float(f1),
                'test_confusion_matrix': cm.tolist(),
                'test_samples': len(X_test)
            }
            
            # ROC AUC for test set
            try:
                if len(np.unique(y_test_encoded)) > 2:
                    auc_score = roc_auc_score(y_test_encoded, y_test_proba, multi_class='ovr')
                else:
                    auc_score = roc_auc_score(y_test_encoded, y_test_proba[:, 1])
                test_metrics['test_roc_auc'] = float(auc_score)
            except Exception as e:
                self.logger.warning(f"Could not calculate test ROC AUC: {e}")
            
            # Update training metadata with test results
            if hasattr(self, 'training_metadata'):
                self.training_metadata.update(test_metrics)
            
            return test_metrics
            
        except Exception as e:
            self.logger.error(f"Test set evaluation failed: {e}")
            return {'test_accuracy': 0.0, 'test_error': str(e)}
    
    def predict(self, X: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Generate trading predictions with calibrated probabilities"""
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        try:
            # Use calibrated model if available
            model_to_use = self.calibrated_model if self.calibrated_model else self.model
            
            # Get probability predictions
            probabilities = model_to_use.predict_proba(X)
            
            # Get class predictions
            predictions = model_to_use.predict(X)
            
            # Decode predictions back to original labels
            predictions_decoded = self.label_encoder.inverse_transform(predictions)
            
            # Calculate confidence as maximum probability
            confidence_scores = np.max(probabilities, axis=1)
            
            return predictions_decoded, confidence_scores, probabilities
            
        except Exception as e:
            self.logger.error(f"Error making predictions: {e}")
            raise
    
    def predict_single(self, features: Dict[str, float]) -> Dict[str, Any]:
        """Generate prediction for single feature vector"""
        # Create DataFrame from features
        feature_df = pd.DataFrame([features])
        
        # Ensure all required features are present
        for feature_name in self.feature_names:
            if feature_name not in feature_df.columns:
                feature_df[feature_name] = 0.0
        
        # Reorder columns to match training data
        feature_df = feature_df[self.feature_names]
        
        # Make prediction
        predictions, confidence_scores, probabilities = self.predict(feature_df)
        
        return {
            'signal': predictions[0],
            'confidence': float(confidence_scores[0]),
            'probabilities': {
                self.label_encoder.classes_[i]: float(prob)
                for i, prob in enumerate(probabilities[0])
            }
        }
    
    def save_model(self, model_dir: str = "models") -> str:
        """Save the trained model and metadata"""
        if not self.is_trained:
            raise ValueError("No trained model to save")
        
        os.makedirs(model_dir, exist_ok=True)
        
        # Model filename
        if self.model_type == 'catboost':
            model_filename = f"model_{self.model_version}.cbm"
            model_path = os.path.join(model_dir, model_filename)
            self.model.save_model(model_path)
        else:
            model_filename = f"model_{self.model_version}.joblib"
            model_path = os.path.join(model_dir, model_filename)
            joblib.dump(self.calibrated_model or self.model, model_path)
        
        # Save metadata
        metadata_path = os.path.join(model_dir, f"model_metadata_{self.model_version}.json")
        with open(metadata_path, 'w') as f:
            json.dump(self.training_metadata, f, indent=2)
        
        # Save label encoder
        encoder_path = os.path.join(model_dir, f"label_encoder_{self.model_version}.joblib")
        joblib.dump(self.label_encoder, encoder_path)
        
        self.logger.info(f"Model saved: {model_path}")
        self.logger.info(f"Metadata saved: {metadata_path}")
        
        return model_path
    
    def load_model(self, model_path: str) -> bool:
        """Load a trained model"""
        try:
            # Extract model version from path
            self.model_version = os.path.basename(model_path).split('_')[1].split('.')[0]
            model_dir = os.path.dirname(model_path)
            
            # Load model
            if model_path.endswith('.cbm'):
                self.model_type = 'catboost'
                self.model = CatBoostClassifier()
                self.model.load_model(model_path)
                self.calibrated_model = self.model
            else:
                self.model = joblib.load(model_path)
                self.calibrated_model = self.model
            
            # Load metadata
            metadata_path = os.path.join(model_dir, f"model_metadata_{self.model_version}.json")
            if os.path.exists(metadata_path):
                with open(metadata_path, 'r') as f:
                    self.training_metadata = json.load(f)
                    self.feature_names = list(self.training_metadata.get('feature_importances', {}).keys())
                    if not self.feature_names and 'feature_count' in self.training_metadata:
                        # Fallback: generate generic feature names
                        self.feature_names = [f"feature_{i}" for i in range(self.training_metadata['feature_count'])]
            
            # Load label encoder
            encoder_path = os.path.join(model_dir, f"label_encoder_{self.model_version}.joblib")
            if os.path.exists(encoder_path):
                self.label_encoder = joblib.load(encoder_path)
            
            self.is_trained = True
            self.logger.info(f"Model loaded successfully: {model_path}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to load model: {e}")
            return False
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information and metadata"""
        if not self.is_trained:
            return {'error': 'No trained model available'}
        
        return {
            'model_type': self.model_type,
            'model_version': self.model_version,
            'is_trained': self.is_trained,
            'feature_count': len(self.feature_names),
            'training_metadata': self.training_metadata
        }