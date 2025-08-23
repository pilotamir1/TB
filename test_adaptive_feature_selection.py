#!/usr/bin/env python3
"""
Test script for adaptive feature selection functionality
"""

import os
import sys
import pandas as pd
import numpy as np
from sklearn.datasets import make_classification

# Add project root to Python path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

def create_mock_trading_data():
    """Create mock trading data with OHLCV and technical indicators"""
    n_samples = 1000
    n_features = 200  # Approximately 196 technical indicators + OHLCV
    
    # Create base classification dataset
    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=20,
        n_redundant=10,
        n_clusters_per_class=1,
        n_classes=3,
        random_state=42
    )
    
    # Create proper column names
    columns = []
    
    # Base OHLCV features
    base_features = ['open', 'high', 'low', 'close', 'volume', 'OHLC4']
    columns.extend(base_features)
    
    # Technical indicators
    indicators = []
    for i in range(n_features - len(base_features)):
        indicators.append(f'indicator_{i:03d}')
    columns.extend(indicators)
    
    # Create DataFrame
    df = pd.DataFrame(X, columns=columns)
    
    return df, pd.Series(y)

def test_adaptive_feature_selection():
    """Test the adaptive feature selection implementation"""
    print("🔍 Testing Adaptive Feature Selection...")
    
    try:
        from ml.enhanced_feature_selection import EnhancedFeatureSelector
        from config.config_loader import load_config
        
        # Load config and enable adaptive selection
        config = load_config()
        config['feature_selection']['adaptive']['enabled'] = True
        config['feature_selection']['adaptive']['initial_top_k'] = 20  # Smaller for test
        config['feature_selection']['adaptive']['max_features'] = 15   # Smaller for test
        config['feature_selection']['adaptive']['cv_folds'] = 3        # Smaller for speed
        
        # Create test data
        X, y = create_mock_trading_data()
        print(f"Created test data: {X.shape[0]} samples, {X.shape[1]} features")
        
        # Initialize selector
        selector = EnhancedFeatureSelector(config)
        
        # Test adaptive selection
        selected_features, selection_info = selector.select_features(X, y)
        
        # Validate results
        print(f"✅ Selected {len(selected_features)} features")
        print(f"Method: {selection_info['method']}")
        
        if selection_info['method'] == 'adaptive_multi_stage':
            print("✅ Adaptive method was used")
            print(f"Base features: {selection_info['base_features_count']}")
            print(f"Selected candidates: {selection_info['candidate_features_count']}")
            print(f"Final CV score: {selection_info.get('final_score', 'N/A')}")
            
            # Check that base features are included
            base_features = ['open', 'high', 'low', 'close', 'volume', 'OHLC4']
            base_in_selected = [f for f in base_features if f in selected_features]
            print(f"Base features in selection: {len(base_in_selected)}/{len(base_features)}")
            
            # Check feature count is within bounds
            min_features = config['feature_selection']['adaptive']['min_features']
            max_features = config['feature_selection']['adaptive']['max_features']
            
            if min_features <= len(selected_features) <= max_features:
                print(f"✅ Feature count within bounds: {min_features} <= {len(selected_features)} <= {max_features}")
            else:
                print(f"⚠️  Feature count outside bounds: {len(selected_features)} not in [{min_features}, {max_features}]")
            
            return True
        else:
            print(f"⚠️  Expected adaptive method, got: {selection_info['method']}")
            return False
            
    except Exception as e:
        print(f"❌ Adaptive feature selection test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_traditional_vs_adaptive():
    """Compare traditional vs adaptive feature selection"""
    print("\n🔍 Comparing Traditional vs Adaptive Selection...")
    
    try:
        from ml.enhanced_feature_selection import EnhancedFeatureSelector
        from config.config_loader import load_config
        
        # Create test data
        X, y = create_mock_trading_data()
        
        # Test traditional method
        config_traditional = load_config()
        config_traditional['feature_selection']['adaptive']['enabled'] = False
        selector_traditional = EnhancedFeatureSelector(config_traditional)
        
        features_traditional, info_traditional = selector_traditional.select_features(X, y)
        
        # Test adaptive method  
        config_adaptive = load_config()
        config_adaptive['feature_selection']['adaptive']['enabled'] = True
        config_adaptive['feature_selection']['adaptive']['max_features'] = 25
        selector_adaptive = EnhancedFeatureSelector(config_adaptive)
        
        features_adaptive, info_adaptive = selector_adaptive.select_features(X, y)
        
        print(f"Traditional method: {len(features_traditional)} features, method: {info_traditional['method']}")
        print(f"Adaptive method: {len(features_adaptive)} features, method: {info_adaptive['method']}")
        
        # Check that methods are different
        if info_traditional['method'] != info_adaptive['method']:
            print("✅ Different methods used as expected")
            return True
        else:
            print("⚠️  Same method used for both configurations")
            return False
            
    except Exception as e:
        print(f"❌ Comparison test failed: {e}")
        return False

def main():
    """Run all tests"""
    print("============================================================")
    print("ADAPTIVE FEATURE SELECTION TESTS")
    print("============================================================")
    
    # Install dependencies if needed
    try:
        import pandas
        import numpy
        import sklearn
    except ImportError as e:
        print(f"❌ Missing dependencies: {e}")
        return False
    
    tests_passed = 0
    total_tests = 2
    
    # Run tests
    if test_adaptive_feature_selection():
        tests_passed += 1
    
    if test_traditional_vs_adaptive():
        tests_passed += 1
    
    print("\n============================================================")
    print(f"RESULTS: {tests_passed}/{total_tests} tests passed")
    print("============================================================")
    
    if tests_passed == total_tests:
        print("✅ All tests passed!")
        return True
    else:
        print("⚠️  Some tests failed")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)