"""
Enhanced Feature Selection with SHAP-based importance and correlation pruning
"""

import pandas as pd
import numpy as np
import logging
import json
from typing import Dict, List, Any, Tuple, Optional
from datetime import datetime
from sklearn.feature_selection import RFE, SequentialFeatureSelector
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import balanced_accuracy_score, f1_score

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
        
        # Adaptive selection parameters
        self.adaptive_config = self.feature_config.get('adaptive', {})
        self.use_adaptive = self.adaptive_config.get('enabled', False)
        self.initial_top_k = self.adaptive_config.get('initial_top_k', 80)
        self.min_features = self.adaptive_config.get('min_features', 5)
        self.max_features = self.adaptive_config.get('max_features', 50)
        self.cv_folds = self.adaptive_config.get('cv_folds', 5)
        self.early_stopping_patience = self.adaptive_config.get('early_stopping_patience', 3)
        self.early_stopping_epsilon = self.adaptive_config.get('early_stopping_epsilon', 0.002)
        self.scoring_metric = self.adaptive_config.get('scoring_metric', 'balanced_accuracy')
        
        # Base features that should always be included
        self.base_feature_patterns = self.adaptive_config.get('base_features', 
            ['open', 'high', 'low', 'close', 'volume', 'OHLC4'])
        
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
    
    def select_features_adaptive(self, X: pd.DataFrame, y: pd.Series,
                                must_include: List[str] = None) -> Tuple[List[str], Dict[str, Any]]:
        """Advanced adaptive feature selection with multi-stage reduction"""
        try:
            must_include = must_include or []
            
            self.logger.info("Starting adaptive feature selection with multi-stage reduction...")
            
            # Identify base features (OHLCV)
            base_features = self._identify_base_features(X)
            
            # Add any explicitly required features to base features
            all_base_features = list(set(base_features + must_include))
            
            # Get candidate pool (technical indicators minus base features)
            candidate_features = self._get_candidate_features(X, all_base_features)
            
            if len(candidate_features) == 0:
                self.logger.warning("No candidate features found, using only base features")
                selected_features = all_base_features
                selection_info = {
                    'method': 'adaptive_no_candidates',
                    'total_features': len(X.columns),
                    'selected_count': len(selected_features),
                    'base_features': all_base_features,
                    'stages_completed': ['base_only'],
                    'timestamp': datetime.now().isoformat()
                }
                return selected_features, selection_info
            
            # Stage A: Initial importance ranking
            stage_a_features = self._stage_a_initial_importance(X, y, candidate_features)
            
            # Stage B: Correlation pruning
            # Calculate importance for correlation pruning
            X_stage_a = X[stage_a_features]
            if self.method == 'shap':
                importance_scores = self.calculate_shap_importance(X_stage_a, y)
            elif self.method == 'permutation':
                importance_scores = self.calculate_permutation_importance(X_stage_a, y)
            else:
                importance_scores = self.calculate_tree_importance(X_stage_a, y)
            
            stage_b_features = self._stage_b_correlation_pruning(X, stage_a_features, importance_scores)
            
            # Stage C: Sequential Forward Floating Selection with CV
            stage_c_features, stage_c_score = self._stage_c_sffs_with_cv(X, y, stage_b_features, all_base_features)
            
            # Stage D: Early stopping evaluation
            stage_d_features, stage_d_score = self._stage_d_early_stopping_evaluation(X, y, all_base_features, stage_c_features)
            
            # Stage E: Dynamic subset size selection
            stage_e_features, stage_e_score = self._stage_e_dynamic_subset_selection(X, y, all_base_features, stage_d_features)
            
            # Final feature set: base features + best selected candidates
            final_selected = all_base_features + stage_e_features
            
            # Store results
            self.selected_features = final_selected
            self.feature_importances = importance_scores
            
            selection_info = {
                'method': 'adaptive_multi_stage',
                'total_features': len(X.columns),
                'selected_count': len(final_selected),
                'base_features_count': len(all_base_features),
                'candidate_features_count': len(stage_e_features),
                'base_features': all_base_features,
                'selected_candidates': stage_e_features,
                'stages': {
                    'stage_a_initial_count': len(stage_a_features),
                    'stage_b_correlation_count': len(stage_b_features),
                    'stage_c_sffs_count': len(stage_c_features),
                    'stage_c_score': stage_c_score,
                    'stage_d_early_stopping_count': len(stage_d_features),
                    'stage_d_score': stage_d_score,
                    'stage_e_dynamic_count': len(stage_e_features),
                    'stage_e_score': stage_e_score
                },
                'parameters': {
                    'initial_top_k': self.initial_top_k,
                    'correlation_threshold': self.correlation_threshold,
                    'min_features': self.min_features,
                    'max_features': self.max_features,
                    'cv_folds': self.cv_folds,
                    'early_stopping_patience': self.early_stopping_patience,
                    'early_stopping_epsilon': self.early_stopping_epsilon,
                    'scoring_metric': self.scoring_metric
                },
                'final_score': stage_e_score,
                'timestamp': datetime.now().isoformat()
            }
            
            self.selection_metadata = selection_info
            
            self.logger.info(f"Adaptive feature selection completed: {len(final_selected)} features selected")
            self.logger.info(f"Base features: {len(all_base_features)}, Selected candidates: {len(stage_e_features)}")
            self.logger.info(f"Final CV score: {stage_e_score:.4f}")
            
            return final_selected, selection_info
            
        except Exception as e:
            self.logger.error(f"Adaptive feature selection failed: {e}")
            # Fallback to existing SHAP-based method
            return self.select_features_shap_based(X, y, must_include)

    def select_features(self, X: pd.DataFrame, y: pd.Series, 
                       must_include: List[str] = None) -> Tuple[List[str], Dict[str, Any]]:
        """Main feature selection method - chooses between adaptive and traditional methods"""
        must_include = must_include or []
        
        if self.use_adaptive:
            return self.select_features_adaptive(X, y, must_include)
        elif self.method in ['shap', 'permutation']:
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
    
    def _identify_base_features(self, X: pd.DataFrame) -> List[str]:
        """Identify base OHLCV features that should always be included"""
        base_features = []
        
        for feature in X.columns:
            for pattern in self.base_feature_patterns:
                if feature.lower() == pattern.lower() or feature.lower().startswith(pattern.lower()):
                    base_features.append(feature)
                    break
        
        self.logger.info(f"Identified {len(base_features)} base features: {base_features}")
        return base_features
    
    def _get_candidate_features(self, X: pd.DataFrame, base_features: List[str]) -> List[str]:
        """Get candidate features for selection (excluding base features)"""
        candidates = [col for col in X.columns if col not in base_features]
        self.logger.info(f"Found {len(candidates)} candidate features for selection")
        return candidates
    
    def _stage_a_initial_importance(self, X: pd.DataFrame, y: pd.Series, 
                                   candidate_features: List[str]) -> List[str]:
        """Stage A: Initial importance ranking -> keep top K0 candidates"""
        self.logger.info(f"Stage A: Selecting top {self.initial_top_k} features by importance")
        
        X_candidates = X[candidate_features]
        
        # Calculate importance using specified method
        if self.method == 'shap':
            importance_scores = self.calculate_shap_importance(X_candidates, y)
        elif self.method == 'permutation':
            importance_scores = self.calculate_permutation_importance(X_candidates, y)
        else:
            importance_scores = self.calculate_tree_importance(X_candidates, y)
        
        # Sort by importance and take top K0
        sorted_features = sorted(importance_scores.items(), key=lambda x: x[1], reverse=True)
        top_features = [f[0] for f in sorted_features[:self.initial_top_k]]
        
        self.logger.info(f"Stage A completed: {len(top_features)} features selected")
        return top_features
    
    def _stage_b_correlation_pruning(self, X: pd.DataFrame, candidate_features: List[str],
                                   importance_scores: Dict[str, float]) -> List[str]:
        """Stage B: Correlation pruning (|rho| > threshold)"""
        self.logger.info(f"Stage B: Correlation pruning with threshold {self.correlation_threshold}")
        
        X_candidates = X[candidate_features]
        correlation_matrix = self.calculate_correlation_matrix(X_candidates)
        
        # Remove correlated features based on importance
        remaining_features = self.remove_correlated_features(importance_scores, correlation_matrix)
        
        # Filter to only include candidates that were in our input
        remaining_features = [f for f in remaining_features if f in candidate_features]
        
        self.logger.info(f"Stage B completed: {len(remaining_features)} features after correlation pruning")
        return remaining_features
    
    def _stage_c_sffs_with_cv(self, X: pd.DataFrame, y: pd.Series, 
                             candidate_features: List[str], base_features: List[str]) -> Tuple[List[str], float]:
        """Stage C: Sequential Forward Floating Selection with cross-validation"""
        self.logger.info("Stage C: Starting SFFS with cross-validation")
        
        if len(candidate_features) == 0:
            self.logger.warning("No candidate features for SFFS")
            return [], 0.0
        
        # Prepare data with base features always included + candidates
        all_features_for_sffs = base_features + candidate_features
        X_sffs = X[all_features_for_sffs]
        
        # Check if we have enough samples for CV
        min_samples_needed = self.cv_folds * 2  # At least 2 samples per fold
        if len(X_sffs) < min_samples_needed:
            self.logger.warning(f"Insufficient samples for CV ({len(X_sffs)} < {min_samples_needed}), skipping SFFS")
            return candidate_features[:min(len(candidate_features), self.max_features - len(base_features))], 0.0
        
        # Create estimator for SFFS
        estimator = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1)
        
        # Setup cross-validation with adaptive folds
        actual_cv_folds = min(self.cv_folds, len(X_sffs) // 2)
        cv = StratifiedKFold(n_splits=actual_cv_folds, shuffle=True, random_state=42)
        
        # Configure SFFS
        max_features = min(self.max_features - len(base_features), len(candidate_features))
        max_features = max(1, max_features)  # At least 1 feature to select
        
        # Create scoring function with better error handling
        def cv_score_func(estimator, X_subset, y_subset):
            try:
                if self.scoring_metric == 'balanced_accuracy':
                    scores = cross_val_score(estimator, X_subset, y_subset, cv=cv, 
                                           scoring='balanced_accuracy', n_jobs=-1)
                else:  # f1_macro
                    scores = cross_val_score(estimator, X_subset, y_subset, cv=cv, 
                                           scoring='f1_macro', n_jobs=-1)
                # Filter out NaN scores and return mean
                valid_scores = scores[~np.isnan(scores)]
                return valid_scores.mean() if len(valid_scores) > 0 else 0.0
            except Exception as e:
                self.logger.warning(f"CV scoring failed: {e}")
                return 0.0
        
        # Run SFFS with error handling
        try:
            sffs = SequentialFeatureSelector(
                estimator=estimator,
                n_features_to_select=max_features,
                direction='forward',
                scoring=cv_score_func,
                cv=None,  # We handle CV in our scoring function
                n_jobs=1   # We use n_jobs in cross_val_score instead
            )
            
            sffs.fit(X_sffs, y)
            selected_mask = sffs.get_support()
            selected_features_all = X_sffs.columns[selected_mask].tolist()
            
            # Split back into base and selected candidates  
            selected_candidates = [f for f in selected_features_all if f not in base_features]
            
            # Get final score
            final_score = cv_score_func(estimator, X_sffs[selected_features_all], y)
            
            self.logger.info(f"Stage C completed: {len(selected_candidates)} candidate features selected, CV score: {final_score:.4f}")
            return selected_candidates, final_score
            
        except Exception as e:
            self.logger.error(f"SFFS failed: {e}")
            # Fallback: return top features by importance up to max_features
            fallback_features = candidate_features[:max_features]
            self.logger.info(f"SFFS fallback: returning {len(fallback_features)} top features")
            return fallback_features, 0.0
    
    def _stage_d_early_stopping_evaluation(self, X: pd.DataFrame, y: pd.Series,
                                          base_features: List[str], candidate_features: List[str]) -> Tuple[List[str], float]:
        """Stage D: Early stopping with iterative feature evaluation"""
        self.logger.info("Stage D: Early stopping evaluation")
        
        if not candidate_features:
            return [], 0.0
        
        # Check if we have enough samples for CV
        min_samples_needed = self.cv_folds * 2
        if len(X) < min_samples_needed:
            self.logger.warning(f"Insufficient samples for CV in Stage D ({len(X)} < {min_samples_needed}), returning top candidates")
            max_candidates = min(len(candidate_features), self.max_features - len(base_features))
            return candidate_features[:max_candidates], 0.0
        
        # Setup CV and estimator
        actual_cv_folds = min(self.cv_folds, len(X) // 2)
        cv = StratifiedKFold(n_splits=actual_cv_folds, shuffle=True, random_state=42)
        estimator = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1)
        
        # Start with base features only
        current_features = base_features.copy()
        best_score = 0.0
        best_features = current_features.copy()
        patience_counter = 0
        
        # Add features one by one with early stopping
        remaining_candidates = candidate_features.copy()
        
        max_iterations = min(len(candidate_features), self.max_features - len(base_features))
        
        for iteration in range(max_iterations):
            if patience_counter >= self.early_stopping_patience:
                self.logger.info(f"Early stopping triggered at iteration {iteration}")
                break
            
            best_candidate = None
            best_iteration_score = best_score
            
            # Try adding each remaining candidate
            for candidate in remaining_candidates[:]:  # Create copy for safe iteration
                test_features = current_features + [candidate]
                
                if len(test_features) > self.max_features:
                    continue
                
                X_test = X[test_features]
                
                # Calculate CV score with error handling
                try:
                    if self.scoring_metric == 'balanced_accuracy':
                        scores = cross_val_score(estimator, X_test, y, cv=cv, 
                                               scoring='balanced_accuracy', n_jobs=-1)
                    else:
                        scores = cross_val_score(estimator, X_test, y, cv=cv, 
                                               scoring='f1_macro', n_jobs=-1)
                    
                    # Handle NaN scores
                    valid_scores = scores[~np.isnan(scores)]
                    score = valid_scores.mean() if len(valid_scores) > 0 else 0.0
                    
                except Exception as e:
                    self.logger.warning(f"CV failed for candidate {candidate}: {e}")
                    score = 0.0
                
                if score > best_iteration_score:
                    best_iteration_score = score
                    best_candidate = candidate
            
            # Check if we found improvement
            if best_candidate and best_iteration_score > best_score + self.early_stopping_epsilon:
                current_features.append(best_candidate)
                remaining_candidates.remove(best_candidate)
                best_score = best_iteration_score
                best_features = current_features.copy()
                patience_counter = 0
                self.logger.info(f"Iteration {iteration}: Added {best_candidate}, score: {best_score:.4f}")
            else:
                patience_counter += 1
                self.logger.info(f"Iteration {iteration}: No improvement, patience: {patience_counter}")
        
        # Return only the candidate features (not base features)
        selected_candidates = [f for f in best_features if f not in base_features]
        
        self.logger.info(f"Stage D completed: {len(selected_candidates)} features, final score: {best_score:.4f}")
        return selected_candidates, best_score
    
    def _stage_e_dynamic_subset_selection(self, X: pd.DataFrame, y: pd.Series,
                                         base_features: List[str], candidate_features: List[str]) -> Tuple[List[str], float]:
        """Stage E: Dynamic subset size selection (5..50 range, best performing subset)"""
        self.logger.info("Stage E: Dynamic subset size selection")
        
        if not candidate_features:
            return [], 0.0
        
        # Check if we have enough samples for CV
        min_samples_needed = self.cv_folds * 2
        if len(X) < min_samples_needed:
            self.logger.warning(f"Insufficient samples for CV in Stage E ({len(X)} < {min_samples_needed}), returning all candidates")
            max_candidates = min(len(candidate_features), self.max_features - len(base_features))
            return candidate_features[:max_candidates], 0.0
        
        actual_cv_folds = min(self.cv_folds, len(X) // 2)
        cv = StratifiedKFold(n_splits=actual_cv_folds, shuffle=True, random_state=42)
        estimator = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1)
        
        best_score = 0.0
        best_subset = []
        best_size = 0
        
        # Try different subset sizes
        min_candidate_features = max(0, self.min_features - len(base_features))
        max_candidate_features = min(len(candidate_features), self.max_features - len(base_features))
        
        for size in range(min_candidate_features, max_candidate_features + 1):
            if size <= 0:
                continue
                
            # Take top 'size' features from candidates
            test_candidates = candidate_features[:size]
            test_features = base_features + test_candidates
            
            X_test = X[test_features]
            
            # Calculate CV score with error handling
            try:
                if self.scoring_metric == 'balanced_accuracy':
                    scores = cross_val_score(estimator, X_test, y, cv=cv, 
                                           scoring='balanced_accuracy', n_jobs=-1)
                else:
                    scores = cross_val_score(estimator, X_test, y, cv=cv, 
                                           scoring='f1_macro', n_jobs=-1)
                
                # Handle NaN scores
                valid_scores = scores[~np.isnan(scores)]
                score = valid_scores.mean() if len(valid_scores) > 0 else 0.0
                
            except Exception as e:
                self.logger.warning(f"CV failed for size {size}: {e}")
                score = 0.0
            
            if score > best_score:
                best_score = score
                best_subset = test_candidates.copy()
                best_size = len(test_features)
            
            self.logger.info(f"Size {len(test_features)}: {size} candidates, score: {score:.4f}")
        
        self.logger.info(f"Stage E completed: Best subset size {best_size}, score: {best_score:.4f}")
        return best_subset, best_score