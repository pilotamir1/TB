"""
Enhanced target labeling with configurable thresholds and class balancing
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Any, Tuple
from sklearn.utils import resample
from collections import Counter

class EnhancedTargetLabeler:
    """Enhanced target labeling with configurable return thresholds"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.labeling_config = config.get('ml', {}).get('labeling', {})
        self.balance_config = config.get('ml', {}).get('class_balance', {})
        
        # Threshold parameters
        self.up_threshold = self.labeling_config.get('up_threshold', 0.02)  # 2%
        self.down_threshold = self.labeling_config.get('down_threshold', -0.02)  # -2%
        
        # Class balancing parameters
        self.balance_method = self.balance_config.get('method', 'class_weight')
        self.max_class_ratio = self.balance_config.get('max_class_ratio', 0.7)
        
        self.logger = logging.getLogger(__name__)
        
        self.logger.info(f"Target labeler initialized - UP: ≥{self.up_threshold:.3f}, DOWN: ≤{self.down_threshold:.3f}")
    
    def calculate_forward_returns(self, df: pd.DataFrame, periods: int = 1) -> pd.Series:
        """Calculate forward returns for labeling"""
        # Calculate future returns
        future_prices = df['close'].shift(-periods)
        current_prices = df['close']
        
        returns = (future_prices - current_prices) / current_prices
        
        return returns
    
    def label_returns(self, returns: pd.Series) -> pd.Series:
        """Label returns based on thresholds"""
        labels = pd.Series(index=returns.index, dtype=str)
        
        # Apply thresholds
        up_mask = returns >= self.up_threshold
        down_mask = returns <= self.down_threshold
        neutral_mask = (~up_mask) & (~down_mask)
        
        labels[up_mask] = 'UP'
        labels[down_mask] = 'DOWN'
        labels[neutral_mask] = 'NEUTRAL'
        
        # Remove NaN values (last few rows due to shift)
        labels = labels.dropna()
        
        return labels
    
    def create_labels(self, df: pd.DataFrame, periods: int = 1) -> Tuple[pd.DataFrame, pd.Series]:
        """Create labels for the dataset"""
        self.logger.info(f"Creating labels with {periods}-period forward returns...")
        
        # Calculate forward returns
        returns = self.calculate_forward_returns(df, periods)
        
        # Create labels
        labels = self.label_returns(returns)
        
        # Align dataframe with labels (remove last few rows)
        df_aligned = df.loc[labels.index]
        
        # Log class distribution
        class_counts = labels.value_counts()
        total_samples = len(labels)
        
        self.logger.info("Class distribution:")
        for class_name, count in class_counts.items():
            percentage = (count / total_samples) * 100
            self.logger.info(f"  {class_name}: {count} ({percentage:.1f}%)")
        
        return df_aligned, labels
    
    def balance_classes(self, X: pd.DataFrame, y: pd.Series) -> Tuple[pd.DataFrame, pd.Series]:
        """Balance classes based on configuration"""
        if self.balance_method == 'class_weight':
            # Class weights handled in model training
            return X, y
        
        self.logger.info(f"Applying class balancing method: {self.balance_method}")
        
        # Get current class distribution
        class_counts = y.value_counts()
        total_samples = len(y)
        
        # Check if balancing is needed
        max_class_size = class_counts.max()
        max_ratio = max_class_size / total_samples
        
        if max_ratio <= self.max_class_ratio:
            self.logger.info("Classes already balanced within threshold")
            return X, y
        
        # Apply balancing strategy
        if self.balance_method == 'oversample':
            return self._oversample_classes(X, y)
        elif self.balance_method == 'undersample':
            return self._undersample_classes(X, y)
        elif self.balance_method == 'downsample_neutral':
            return self._downsample_neutral(X, y)
        else:
            self.logger.warning(f"Unknown balancing method: {self.balance_method}")
            return X, y
    
    def _oversample_classes(self, X: pd.DataFrame, y: pd.Series) -> Tuple[pd.DataFrame, pd.Series]:
        """Oversample minority classes"""
        class_counts = y.value_counts()
        max_samples = class_counts.max()
        
        balanced_X_parts = []
        balanced_y_parts = []
        
        for class_name in class_counts.index:
            class_mask = y == class_name
            class_X = X[class_mask]
            class_y = y[class_mask]
            
            current_samples = len(class_X)
            
            if current_samples < max_samples:
                # Oversample to match max
                oversampled_X, oversampled_y = resample(
                    class_X, class_y,
                    replace=True,
                    n_samples=max_samples,
                    random_state=42
                )
                balanced_X_parts.append(oversampled_X)
                balanced_y_parts.append(oversampled_y)
            else:
                balanced_X_parts.append(class_X)
                balanced_y_parts.append(class_y)
        
        # Combine all classes
        X_balanced = pd.concat(balanced_X_parts, ignore_index=True)
        y_balanced = pd.concat(balanced_y_parts, ignore_index=True)
        
        # Shuffle the result
        shuffled_indices = np.random.RandomState(42).permutation(len(X_balanced))
        X_balanced = X_balanced.iloc[shuffled_indices].reset_index(drop=True)
        y_balanced = y_balanced.iloc[shuffled_indices].reset_index(drop=True)
        
        self.logger.info(f"Oversampling completed: {len(X)} -> {len(X_balanced)} samples")
        
        return X_balanced, y_balanced
    
    def _undersample_classes(self, X: pd.DataFrame, y: pd.Series) -> Tuple[pd.DataFrame, pd.Series]:
        """Undersample majority classes"""
        class_counts = y.value_counts()
        min_samples = class_counts.min()
        
        balanced_X_parts = []
        balanced_y_parts = []
        
        for class_name in class_counts.index:
            class_mask = y == class_name
            class_X = X[class_mask]
            class_y = y[class_mask]
            
            current_samples = len(class_X)
            
            if current_samples > min_samples:
                # Undersample to match min
                undersampled_X, undersampled_y = resample(
                    class_X, class_y,
                    replace=False,
                    n_samples=min_samples,
                    random_state=42
                )
                balanced_X_parts.append(undersampled_X)
                balanced_y_parts.append(undersampled_y)
            else:
                balanced_X_parts.append(class_X)
                balanced_y_parts.append(class_y)
        
        # Combine all classes
        X_balanced = pd.concat(balanced_X_parts, ignore_index=True)
        y_balanced = pd.concat(balanced_y_parts, ignore_index=True)
        
        # Shuffle the result
        shuffled_indices = np.random.RandomState(42).permutation(len(X_balanced))
        X_balanced = X_balanced.iloc[shuffled_indices].reset_index(drop=True)
        y_balanced = y_balanced.iloc[shuffled_indices].reset_index(drop=True)
        
        self.logger.info(f"Undersampling completed: {len(X)} -> {len(X_balanced)} samples")
        
        return X_balanced, y_balanced
    
    def _downsample_neutral(self, X: pd.DataFrame, y: pd.Series) -> Tuple[pd.DataFrame, pd.Series]:
        """Specifically downsample the NEUTRAL class if it's too dominant"""
        class_counts = y.value_counts()
        
        if 'NEUTRAL' not in class_counts:
            return X, y
        
        neutral_count = class_counts['NEUTRAL']
        other_counts = class_counts.drop('NEUTRAL')
        total_other = other_counts.sum()
        
        # If NEUTRAL is more than X times the size of all others combined
        if neutral_count > total_other * 2:
            # Downsample NEUTRAL to be at most equal to sum of others
            target_neutral_size = min(neutral_count, total_other)
            
            # Separate NEUTRAL and others
            neutral_mask = y == 'NEUTRAL'
            other_mask = ~neutral_mask
            
            X_neutral = X[neutral_mask]
            y_neutral = y[neutral_mask]
            X_others = X[other_mask]
            y_others = y[other_mask]
            
            # Downsample NEUTRAL
            if len(X_neutral) > target_neutral_size:
                X_neutral_downsampled, y_neutral_downsampled = resample(
                    X_neutral, y_neutral,
                    replace=False,
                    n_samples=target_neutral_size,
                    random_state=42
                )
            else:
                X_neutral_downsampled = X_neutral
                y_neutral_downsampled = y_neutral
            
            # Combine
            X_balanced = pd.concat([X_others, X_neutral_downsampled], ignore_index=True)
            y_balanced = pd.concat([y_others, y_neutral_downsampled], ignore_index=True)
            
            # Shuffle
            shuffled_indices = np.random.RandomState(42).permutation(len(X_balanced))
            X_balanced = X_balanced.iloc[shuffled_indices].reset_index(drop=True)
            y_balanced = y_balanced.iloc[shuffled_indices].reset_index(drop=True)
            
            self.logger.info(f"NEUTRAL downsampling completed: {len(X)} -> {len(X_balanced)} samples")
            self.logger.info(f"NEUTRAL reduced from {neutral_count} to {len(y_balanced[y_balanced == 'NEUTRAL'])}")
            
            return X_balanced, y_balanced
        
        return X, y
    
    def create_binary_labels(self, df: pd.DataFrame, periods: int = 1, 
                           exclude_neutral: bool = True) -> Tuple[pd.DataFrame, pd.Series]:
        """Create binary labels (UP/DOWN) by optionally excluding NEUTRAL"""
        df_labeled, labels = self.create_labels(df, periods)
        
        if exclude_neutral:
            # Remove NEUTRAL class
            non_neutral_mask = labels != 'NEUTRAL'
            df_binary = df_labeled[non_neutral_mask]
            labels_binary = labels[non_neutral_mask]
            
            self.logger.info(f"Binary labeling: {len(df_labeled)} -> {len(df_binary)} samples (excluded NEUTRAL)")
            
            # Log new distribution
            class_counts = labels_binary.value_counts()
            total_samples = len(labels_binary)
            
            self.logger.info("Binary class distribution:")
            for class_name, count in class_counts.items():
                percentage = (count / total_samples) * 100
                self.logger.info(f"  {class_name}: {count} ({percentage:.1f}%)")
            
            return df_binary, labels_binary
        else:
            return df_labeled, labels
    
    def get_labeling_stats(self, labels: pd.Series) -> Dict[str, Any]:
        """Get detailed labeling statistics"""
        class_counts = labels.value_counts()
        total_samples = len(labels)
        
        stats = {
            'total_samples': total_samples,
            'num_classes': len(class_counts),
            'class_distribution': {},
            'class_ratios': {},
            'is_balanced': True,
            'dominant_class': None,
            'dominant_class_ratio': 0,
            'labeling_thresholds': {
                'up_threshold': self.up_threshold,
                'down_threshold': self.down_threshold
            }
        }
        
        for class_name, count in class_counts.items():
            ratio = count / total_samples
            stats['class_distribution'][class_name] = int(count)
            stats['class_ratios'][class_name] = float(ratio)
            
            # Check for dominant class
            if ratio > stats['dominant_class_ratio']:
                stats['dominant_class'] = class_name
                stats['dominant_class_ratio'] = float(ratio)
        
        # Check if balanced within threshold
        if stats['dominant_class_ratio'] > self.max_class_ratio:
            stats['is_balanced'] = False
        
        return stats
    
    def validate_labeling_quality(self, df: pd.DataFrame, labels: pd.Series) -> Dict[str, Any]:
        """Validate the quality of labeling"""
        validation_results = {
            'is_valid': True,
            'issues': [],
            'warnings': []
        }
        
        # Check for minimum class representation
        class_counts = labels.value_counts()
        min_samples_per_class = max(10, len(labels) * 0.01)  # At least 1% or 10 samples
        
        for class_name, count in class_counts.items():
            if count < min_samples_per_class:
                validation_results['warnings'].append(
                    f"Class '{class_name}' has only {count} samples (< {min_samples_per_class:.0f})"
                )
        
        # Check for extreme class imbalance
        if len(class_counts) > 1:
            max_ratio = class_counts.max() / len(labels)
            if max_ratio > 0.9:
                validation_results['issues'].append(
                    f"Extreme class imbalance: {class_counts.idxmax()} represents {max_ratio:.1%} of data"
                )
                validation_results['is_valid'] = False
        
        # Check for data leakage (future information)
        if 'close' in df.columns:
            # Simple check: ensure we're not using future prices
            last_valid_close = df['close'].iloc[-len(labels)]
            if pd.notna(last_valid_close):
                validation_results['warnings'].append(
                    "Ensure no data leakage: last labeled sample should not use future information"
                )
        
        # Check threshold effectiveness
        if 'NEUTRAL' in class_counts:
            neutral_ratio = class_counts['NEUTRAL'] / len(labels)
            if neutral_ratio > 0.8:
                validation_results['warnings'].append(
                    f"High NEUTRAL ratio ({neutral_ratio:.1%}) - consider adjusting thresholds"
                )
        
        return validation_results