"""
Enhanced Feature Selection with SHAP-based importance and correlation pruning
"""

import pandas as pd
import numpy as np
import logging
import json
from typing import Dict, List, Any, Tuple, Optional
from datetime import datetime
from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split

# Try to import SHAP
try:
    import shap
    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False

class EnhancedFeatureSelector:
    """Enhanced feature selection with SHAP, permutation importance, and correlation analysis"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.feature_config = config.get('feature_selection', {})
        
        self.method = self.feature_config.get('method', 'shap')
        self.target_features = self.feature_config.get('target_features', 50)
        self.correlation_threshold = self.feature_config.get('correlation_threshold', 0.9)
        self.importance_threshold = self.feature_config.get('importance_threshold', 0.001)
        
        self.selected_features = []
        self.feature_importances = {}
        self.correlation_matrix = None
        self.selection_metadata = {}
        
        self.logger = logging.getLogger(__name__)
    
    def calculate_shap_importance(self, X: pd.DataFrame, y: pd.Series, 
                                 sample_size: int = 1000) -> Dict[str, float]:
        """Calculate SHAP-based feature importance"""
        if not HAS_SHAP:
            self.logger.warning("SHAP not available, falling back to permutation importance")
            return self.calculate_permutation_importance(X, y, sample_size)
        
        try:
            self.logger.info("Calculating SHAP importance...")
            
            # Sample data for faster computation
            if len(X) > sample_size:
                X_sample, _, y_sample, _ = train_test_split(
                    X, y, train_size=sample_size, 
                    stratify=y, random_state=42
                )
            else:
                X_sample, y_sample = X, y
            
            # Train a simple model for SHAP analysis
            model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
            model.fit(X_sample, y_sample)
            
            # Create SHAP explainer
            explainer = shap.TreeExplainer(model)
            
            # Calculate SHAP values (use smaller sample if needed)
            if len(X_sample) > 500:
                shap_sample = X_sample.sample(n=500, random_state=42)
            else:
                shap_sample = X_sample
            
            shap_values = explainer.shap_values(shap_sample)
            
            # Handle multi-class case
            if isinstance(shap_values, list):
                # Average importance across classes
                mean_shap_values = np.mean([np.abs(sv) for sv in shap_values], axis=0)
            else:
                mean_shap_values = np.abs(shap_values)
            
            # Calculate feature importance as mean absolute SHAP value
            feature_importance = np.mean(mean_shap_values, axis=0)
            
            # Create importance dictionary
            importance_dict = dict(zip(X.columns, feature_importance))
            
            self.logger.info("SHAP importance calculation completed")
            return importance_dict
            
        except Exception as e:
            self.logger.error(f"SHAP calculation failed: {e}")
            return self.calculate_permutation_importance(X, y, sample_size)
    
    def calculate_permutation_importance(self, X: pd.DataFrame, y: pd.Series,
                                       sample_size: int = 1000) -> Dict[str, float]:
        """Calculate permutation-based feature importance"""
        try:
            self.logger.info("Calculating permutation importance...")
            
            # Sample data for faster computation
            if len(X) > sample_size:
                X_sample, _, y_sample, _ = train_test_split(
                    X, y, train_size=sample_size, 
                    stratify=y, random_state=42
                )
            else:
                X_sample, y_sample = X, y
            
            # Train model
            model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
            model.fit(X_sample, y_sample)
            
            # Calculate permutation importance
            perm_importance = permutation_importance(
                model, X_sample, y_sample, n_repeats=10, 
                random_state=42, n_jobs=-1
            )
            
            importance_dict = dict(zip(X.columns, perm_importance.importances_mean))
            
            self.logger.info("Permutation importance calculation completed")
            return importance_dict
            
        except Exception as e:
            self.logger.error(f"Permutation importance calculation failed: {e}")
            # Fallback to tree-based importance
            return self.calculate_tree_importance(X, y)
    
    def calculate_tree_importance(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, float]:
        """Calculate tree-based feature importance as fallback"""
        try:
            self.logger.info("Calculating tree-based importance...")
            
            model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
            model.fit(X, y)
            
            importance_dict = dict(zip(X.columns, model.feature_importances_))
            
            self.logger.info("Tree-based importance calculation completed")
            return importance_dict
            
        except Exception as e:
            self.logger.error(f"Tree-based importance calculation failed: {e}")
            # Ultimate fallback - uniform importance
            return {col: 1.0 / len(X.columns) for col in X.columns}
    
    def calculate_correlation_matrix(self, X: pd.DataFrame) -> pd.DataFrame:
        """Calculate correlation matrix"""
        self.logger.info("Calculating correlation matrix...")
        
        # Handle missing values
        X_clean = X.fillna(X.median())
        
        # Calculate correlation matrix
        correlation_matrix = X_clean.corr().abs()
        
        self.correlation_matrix = correlation_matrix
        return correlation_matrix
    
    def remove_correlated_features(self, feature_importance: Dict[str, float], 
                                 correlation_matrix: pd.DataFrame) -> List[str]:
        """Remove highly correlated features based on importance"""
        self.logger.info(f"Removing features with correlation > {self.correlation_threshold}")
        
        features_to_remove = set()
        importance_series = pd.Series(feature_importance)
        
        # Find pairs of highly correlated features
        for i, feature1 in enumerate(correlation_matrix.columns):
            if feature1 in features_to_remove:
                continue
                
            for j, feature2 in enumerate(correlation_matrix.columns[i+1:], i+1):
                if feature2 in features_to_remove:
                    continue
                
                correlation = correlation_matrix.iloc[i, j]
                
                if correlation > self.correlation_threshold:
                    # Remove the feature with lower importance
                    imp1 = importance_series.get(feature1, 0)
                    imp2 = importance_series.get(feature2, 0)
                    
                    if imp1 < imp2:
                        features_to_remove.add(feature1)
                        break  # Move to next feature1
                    else:
                        features_to_remove.add(feature2)
        
        # Get remaining features
        remaining_features = [f for f in feature_importance.keys() if f not in features_to_remove]
        
        self.logger.info(f"Removed {len(features_to_remove)} highly correlated features")
        self.logger.info(f"Remaining features: {len(remaining_features)}")
        
        return remaining_features
    
    def select_top_features(self, feature_importance: Dict[str, float], 
                          remaining_features: List[str]) -> List[str]:
        """Select top features based on importance"""
        # Filter importance for remaining features
        filtered_importance = {f: feature_importance[f] for f in remaining_features}
        
        # Sort by importance
        sorted_features = sorted(
            filtered_importance.items(), 
            key=lambda x: x[1], 
            reverse=True
        )
        
        # Apply importance threshold
        significant_features = [
            f for f, imp in sorted_features 
            if imp >= self.importance_threshold
        ]
        
        # Take top N features
        selected_features = significant_features[:self.target_features]
        
        self.logger.info(f"Selected top {len(selected_features)} features")
        
        return selected_features
    
    def select_features_shap_based(self, X: pd.DataFrame, y: pd.Series,
                                  must_include: List[str] = None) -> Tuple[List[str], Dict[str, Any]]:
        """SHAP-based feature selection with correlation pruning"""
        try:
            must_include = must_include or []
            
            self.logger.info("Starting SHAP-based feature selection...")
            
            # Calculate feature importance
            if self.method == 'shap':
                feature_importance = self.calculate_shap_importance(X, y)
            elif self.method == 'permutation':
                feature_importance = self.calculate_permutation_importance(X, y)
            else:
                feature_importance = self.calculate_tree_importance(X, y)
            
            # Store importance scores
            self.feature_importances = feature_importance
            
            # Calculate correlation matrix
            correlation_matrix = self.calculate_correlation_matrix(X)
            
            # Remove highly correlated features (but preserve must_include)
            remaining_features = self.remove_correlated_features(feature_importance, correlation_matrix)
            
            # Ensure must_include features are preserved
            for feature in must_include:
                if feature in X.columns and feature not in remaining_features:
                    remaining_features.append(feature)
            
            # Select top features
            selected_features = self.select_top_features(feature_importance, remaining_features)
            
            # Ensure must_include features are in final selection
            for feature in must_include:
                if feature in X.columns and feature not in selected_features:
                    selected_features.append(feature)
            
            # Remove duplicates and limit to target count
            selected_features = list(dict.fromkeys(selected_features))[:self.target_features]
            
            # Create selection metadata
            selection_info = {
                'method': self.method,
                'total_features': len(X.columns),
                'selected_count': len(selected_features),
                'must_include_count': len(must_include),
                'correlation_threshold': self.correlation_threshold,
                'importance_threshold': self.importance_threshold,
                'target_features': self.target_features,
                'timestamp': datetime.now().isoformat(),
                'top_features_by_importance': [
                    {'feature': f, 'importance': feature_importance.get(f, 0)}
                    for f in selected_features
                ]
            }
            
            self.selected_features = selected_features
            self.selection_metadata = selection_info
            
            self.logger.info(f"SHAP-based feature selection completed: {len(selected_features)} features selected")
            
            return selected_features, selection_info
            
        except Exception as e:
            self.logger.error(f"SHAP-based feature selection failed: {e}")
            # Fallback to RFE
            return self.select_features_rfe_fallback(X, y, must_include)
    
    def select_features_rfe_fallback(self, X: pd.DataFrame, y: pd.Series,
                                   must_include: List[str] = None) -> Tuple[List[str], Dict[str, Any]]:
        """RFE-based feature selection as fallback"""
        try:
            must_include = must_include or []
            
            self.logger.info("Using RFE as fallback feature selection method...")
            
            # Create estimator
            estimator = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1)
            
            # Features available for RFE (excluding must_include)
            available_features = [f for f in X.columns if f not in must_include]
            X_available = X[available_features]
            
            # Calculate how many features to select with RFE
            rfe_target = max(1, self.target_features - len(must_include))
            rfe_target = min(rfe_target, len(available_features))
            
            if rfe_target > 0:
                # Perform RFE
                rfe = RFE(estimator, n_features_to_select=rfe_target)
                rfe.fit(X_available, y)
                
                # Get selected features
                rfe_selected = X_available.columns[rfe.support_].tolist()
                
                # Store rankings
                self.feature_importances = dict(zip(available_features, rfe.ranking_))
            else:
                rfe_selected = []
            
            # Combine must_include and RFE selected
            selected_features = must_include + rfe_selected
            selected_features = selected_features[:self.target_features]
            
            selection_info = {
                'method': 'rfe_fallback',
                'total_features': len(X.columns),
                'selected_count': len(selected_features),
                'must_include_count': len(must_include),
                'rfe_selected_count': len(rfe_selected),
                'target_features': self.target_features,
                'timestamp': datetime.now().isoformat()
            }
            
            self.selected_features = selected_features
            self.selection_metadata = selection_info
            
            self.logger.info(f"RFE fallback selection completed: {len(selected_features)} features selected")
            
            return selected_features, selection_info
            
        except Exception as e:
            self.logger.error(f"RFE fallback selection failed: {e}")
            # Ultimate fallback - use all features up to target
            all_features = X.columns.tolist()[:self.target_features]
            return all_features, {'method': 'fallback_all', 'selected_count': len(all_features)}
    
    def select_features(self, X: pd.DataFrame, y: pd.Series, 
                       must_include: List[str] = None) -> Tuple[List[str], Dict[str, Any]]:
        """Main feature selection method"""
        must_include = must_include or []
        
        if self.method in ['shap', 'permutation']:
            return self.select_features_shap_based(X, y, must_include)
        else:
            return self.select_features_rfe_fallback(X, y, must_include)
    
    def get_feature_importance_report(self) -> Dict[str, Any]:
        """Get detailed feature importance report"""
        if not self.feature_importances:
            return {}
        
        # Sort features by importance
        sorted_importance = sorted(
            self.feature_importances.items(), 
            key=lambda x: x[1], 
            reverse=True
        )
        
        # Calculate cumulative importance
        total_importance = sum(self.feature_importances.values())
        cumulative_importance = 0
        cumulative_curve = []
        
        for feature, importance in sorted_importance:
            cumulative_importance += importance
            cumulative_curve.append({
                'feature': feature,
                'importance': importance,
                'cumulative_importance': cumulative_importance,
                'cumulative_ratio': cumulative_importance / total_importance if total_importance > 0 else 0
            })
        
        return {
            'method': self.method,
            'total_features': len(self.feature_importances),
            'selected_features': self.selected_features,
            'importance_scores': dict(sorted_importance),
            'cumulative_importance_curve': cumulative_curve,
            'selection_metadata': self.selection_metadata
        }
    
    def save_selection_artifact(self, filepath: str):
        """Save feature selection artifact to JSON file"""
        artifact = {
            'selected_features': self.selected_features,
            'feature_importances': self.feature_importances,
            'selection_metadata': self.selection_metadata,
            'importance_report': self.get_feature_importance_report()
        }
        
        with open(filepath, 'w') as f:
            json.dump(artifact, f, indent=2)
        
        self.logger.info(f"Feature selection artifact saved: {filepath}")
    
    def load_selection_artifact(self, filepath: str) -> bool:
        """Load feature selection artifact from JSON file"""
        try:
            with open(filepath, 'r') as f:
                artifact = json.load(f)
            
            self.selected_features = artifact.get('selected_features', [])
            self.feature_importances = artifact.get('feature_importances', {})
            self.selection_metadata = artifact.get('selection_metadata', {})
            
            self.logger.info(f"Feature selection artifact loaded: {filepath}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to load selection artifact: {e}")
            return False