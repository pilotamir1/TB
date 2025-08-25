#!/usr/bin/env python3
"""
Test script for the combined training pipeline implementation
Validates configuration changes and basic functionality
"""

import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_config_imports():
    """Test that all new configuration imports work"""
    print("Testing configuration imports...")
    
    try:
        from config.settings import FEATURE_SELECTION_CONFIG, LABELING_CONFIG, DATA_CONFIG, XGB_PRO_CONFIG
        print("✓ All configuration imports successful")
        return True
    except ImportError as e:
        print(f"✗ Configuration import failed: {e}")
        return False

def test_feature_selection_config():
    """Test the new FEATURE_SELECTION_CONFIG parameters"""
    print("Testing FEATURE_SELECTION_CONFIG...")
    
    try:
        from config.settings import FEATURE_SELECTION_CONFIG
        
        # Check required parameters
        required_params = ['enabled', 'mode', 'selection_window_4h', 'min_features', 'tolerance', 'max_iterations']
        
        for param in required_params:
            if param not in FEATURE_SELECTION_CONFIG:
                print(f"✗ Missing parameter: {param}")
                return False
        
        # Validate values
        if FEATURE_SELECTION_CONFIG['enabled'] != True:
            print("✗ Feature selection should be enabled")
            return False
            
        if FEATURE_SELECTION_CONFIG['mode'] != 'dynamic':
            print("✗ Mode should be 'dynamic'")
            return False
            
        if FEATURE_SELECTION_CONFIG['selection_window_4h'] != 800:
            print("✗ Selection window should be 800")
            return False
            
        if FEATURE_SELECTION_CONFIG['min_features'] != 20:
            print("✗ Min features should be 20")
            return False
            
        print("✓ FEATURE_SELECTION_CONFIG validation passed")
        return True
        
    except Exception as e:
        print(f"✗ FEATURE_SELECTION_CONFIG test failed: {e}")
        return False

def test_labeling_config():
    """Test the new LABELING_CONFIG parameters"""
    print("Testing LABELING_CONFIG...")
    
    try:
        from config.settings import LABELING_CONFIG
        
        # Check required parameters
        required_params = ['target_distribution', 'initial_up_pct', 'initial_down_pct', 
                          'search_up_range', 'search_down_range', 'optimization_metric']
        
        for param in required_params:
            if param not in LABELING_CONFIG:
                print(f"✗ Missing parameter: {param}")
                return False
        
        # Validate target distribution
        target_dist = LABELING_CONFIG['target_distribution']
        if 'SELL' not in target_dist or 'BUY' not in target_dist or 'HOLD' not in target_dist:
            print("✗ Target distribution missing class labels")
            return False
            
        # Check if distribution sums to 1.0
        total = sum(target_dist.values())
        if abs(total - 1.0) > 0.01:
            print(f"✗ Target distribution should sum to 1.0, got {total}")
            return False
            
        # Validate thresholds
        if LABELING_CONFIG['initial_up_pct'] <= 0:
            print("✗ Initial up percentage should be positive")
            return False
            
        if LABELING_CONFIG['initial_down_pct'] >= 0:
            print("✗ Initial down percentage should be negative")
            return False
            
        print("✓ LABELING_CONFIG validation passed")
        return True
        
    except Exception as e:
        print(f"✗ LABELING_CONFIG test failed: {e}")
        return False

def test_data_config():
    """Test DATA_CONFIG for unlimited history"""
    print("Testing DATA_CONFIG...")
    
    try:
        from config.settings import DATA_CONFIG
        
        if not DATA_CONFIG.get('use_all_history', False):
            print("✗ use_all_history should be True")
            return False
            
        if DATA_CONFIG.get('max_4h_training_candles', -1) != 0:
            print("✗ max_4h_training_candles should be 0 for unlimited")
            return False
            
        print("✓ DATA_CONFIG validation passed")
        return True
        
    except Exception as e:
        print(f"✗ DATA_CONFIG test failed: {e}")
        return False

def test_trainer_imports():
    """Test that trainer can import new configs"""
    print("Testing trainer imports...")
    
    try:
        # This will test the import without actually running the trainer
        from ml.trainer import ModelTrainer
        print("✓ ModelTrainer import successful")
        return True
    except ImportError as e:
        print(f"✗ ModelTrainer import failed: {e}")
        return False

def test_label_calibration_logic():
    """Test the new label calibration logic basic functionality"""
    print("Testing label calibration logic...")
    
    try:
        import pandas as pd
        import numpy as np
        
        # Create mock data
        mock_data = pd.DataFrame({
            'symbol': ['BTCUSDT'] * 10,
            'timestamp': range(1000, 1010),
            'close': [100, 101, 102, 103, 104, 105, 106, 107, 108, 109]
        })
        
        # Test that we can create a trainer instance 
        from ml.trainer import ModelTrainer
        trainer = ModelTrainer()
        
        # Test KL divergence calculation
        actual_dist = {'BUY': 0.3, 'SELL': 0.3, 'HOLD': 0.4}
        target_dist = {'BUY': 0.4, 'SELL': 0.4, 'HOLD': 0.2}
        
        kl_div = trainer._calculate_kl_divergence(actual_dist, target_dist)
        
        if not isinstance(kl_div, (int, float)):
            print("✗ KL divergence should return a number")
            return False
            
        if kl_div < 0:
            print("✗ KL divergence should be non-negative")
            return False
            
        print("✓ Label calibration logic validation passed")
        return True
        
    except Exception as e:
        print(f"✗ Label calibration logic test failed: {e}")
        return False

def main():
    """Run all tests"""
    print("=== Combined Training Pipeline Tests ===\n")
    
    tests = [
        test_config_imports,
        test_feature_selection_config,
        test_labeling_config,
        test_data_config,
        test_trainer_imports,
        test_label_calibration_logic
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
            print()
        except Exception as e:
            print(f"✗ Test {test.__name__} crashed: {e}\n")
    
    print(f"=== Results: {passed}/{total} tests passed ===")
    
    if passed == total:
        print("🎉 All tests passed! Combined pipeline implementation is ready.")
        return True
    else:
        print("❌ Some tests failed. Please check the implementation.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)